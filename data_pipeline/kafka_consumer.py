"""
Kafka consumer: reads transactions from the `transactions` topic and scores
each one against the live serving API (POST /predict). Predictions update API
metrics; persistence is handled only by the labeled replay simulator.

Run alongside kafka_producer.py and the serving API + Postgres (all started
together via docker-compose) to see real Kafka-based streaming inference.

Usage:
    python data_pipeline/kafka_consumer.py
"""
import json
import os

import requests
from confluent_kafka import Consumer
from data_pipeline.feature_engineering import FEATURE_COLUMNS

TOPIC = os.environ.get("KAFKA_TRANSACTIONS_TOPIC", "transactions")
BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
API_URL = os.environ.get("API_URL", "http://localhost:8000")


def run():
    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP_SERVERS,
        "group.id": "fraud-scoring-consumer",
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe([TOPIC])
    print(f"[kafka_consumer] subscribed to '{TOPIC}' on {BOOTSTRAP_SERVERS}, scoring via {API_URL}")

    scored = 0
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"[kafka_consumer] error: {msg.error()}")
                continue

            row = json.loads(msg.value().decode("utf-8"))
            payload = {
                "TRANSACTION_ID": int(row["TRANSACTION_ID"]),
                "TX_DATETIME": row["TX_DATETIME"],
                "CUSTOMER_ID": str(row["CUSTOMER_ID"]),
                "TERMINAL_ID": str(row["TERMINAL_ID"]),
                **{c: float(row[c]) for c in FEATURE_COLUMNS},
                "is_new_terminal": bool(row["IS_NEW_TERMINAL"]),
            }

            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=2)
                resp.raise_for_status()
                result = resp.json()
                scored += 1
                if scored % 50 == 0:
                    print(f"[kafka_consumer] scored {scored} transactions "
                          f"(last decision={result.get('decision')})")
            except (requests.RequestException, ValueError) as e:
                print(f"[kafka_consumer] prediction request failed: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        print(f"[kafka_consumer] stopped, scored {scored} transactions total")


if __name__ == "__main__":
    run()
