"""
Kafka producer: publishes the historical transaction stream onto a real Kafka
topic (`transactions`), replayed in timestamp order at configurable speed.

This is the production-spec streaming-ingestion component from the original
brief. It requires a running Kafka broker (see docker-compose.yml — the
`kafka` service). For local development without Docker, `ingest.py`'s
asyncio-queue replay is used instead and the API defaults to that path; this
producer/consumer pair is the "real Kafka" path for the full docker-compose
deployment.

Usage:
    python data_pipeline/kafka_producer.py --speed 500 --max-rows 5000
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from confluent_kafka import Producer

FEATURES_PATH = Path(__file__).parent.parent / "data" / "creditcard_features.parquet"
TOPIC = os.environ.get("KAFKA_TRANSACTIONS_TOPIC", "transactions")
BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


def delivery_report(err, msg):
    if err is not None:
        print(f"[kafka_producer] delivery failed: {err}")


def run(speed_multiplier: float, max_rows: int):
    producer = Producer({"bootstrap.servers": BOOTSTRAP_SERVERS})

    df = pd.read_parquet(FEATURES_PATH).sort_values(["TX_DATETIME", "TRANSACTION_ID"])
    if max_rows:
        df = df.tail(max_rows).reset_index(drop=True)

    print(f"[kafka_producer] streaming {len(df)} transactions to topic '{TOPIC}' "
          f"on {BOOTSTRAP_SERVERS} at {speed_multiplier}x speed")

    previous_datetime = None
    sent = 0
    for _, row in df.iterrows():
        current_datetime = row["TX_DATETIME"]
        if previous_datetime is not None:
            delta = max(0.0, (current_datetime - previous_datetime).total_seconds())
            time.sleep(min(delta / max(speed_multiplier, 1.0), 0.5))
        previous_datetime = current_datetime

        payload = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in row.to_dict().items()
        }
        payload["TX_DATETIME"] = current_datetime.isoformat()
        producer.produce(
            TOPIC,
            key=str(row["CUSTOMER_ID"]),
            value=json.dumps(payload).encode("utf-8"),
            callback=delivery_report,
        )
        producer.poll(0)
        sent += 1
        if sent % 200 == 0:
            print(f"[kafka_producer] sent {sent} transactions")

    producer.flush()
    print(f"[kafka_producer] done, sent {sent} transactions total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=500.0,
                        help="Replay multiplier relative to historical time")
    parser.add_argument("--max-rows", type=int, default=5000)
    args = parser.parse_args()
    run(args.speed, args.max_rows)
