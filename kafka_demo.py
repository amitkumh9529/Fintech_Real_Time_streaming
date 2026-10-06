import argparse
import json
import os
import random
import uuid
from datetime import datetime, timezone

from confluent_kafka import Consumer, KafkaException, Producer
from dotenv import load_dotenv

load_dotenv()


def produce_transactions(count: int) -> None:
    if count < 1:
        raise ValueError("count must be at least 1")

    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_TOPIC", "transactions")
    producer = Producer({"bootstrap.servers": bootstrap_servers})
    delivery_errors: list[str] = []

    def on_delivery(error, message) -> None:
        if error is not None:
            delivery_errors.append(str(error))
            return
        print(f"Produced transaction message at offset {message.offset()}")

    for _ in range(count):
        transaction_id = str(uuid.uuid4())
        transaction = {
            "transaction_id": transaction_id,
            "amount": round(random.uniform(1, 1000), 2),
            "currency": "USD",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        producer.produce(
            topic,
            key=transaction_id,
            value=json.dumps(transaction),
            on_delivery=on_delivery,
        )
        producer.poll(0)

    pending = producer.flush(10)
    if pending:
        raise TimeoutError(f"{pending} Kafka message(s) were not delivered within 10 seconds")
    if delivery_errors:
        raise RuntimeError("Kafka message delivery failed: " + "; ".join(delivery_errors))


def consume_transactions(max_messages: int | None) -> None:
    if max_messages is not None and max_messages < 1:
        raise ValueError("max_messages must be at least 1")

    consumer = Consumer(
        {
            "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            "group.id": os.getenv("KAFKA_CONSUMER_GROUP", "fintech-demo-consumer"),
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    topic = os.getenv("KAFKA_TOPIC", "transactions")
    consumed = 0

    try:
        consumer.subscribe([topic])
        while max_messages is None or consumed < max_messages:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error() is not None:
                raise KafkaException(message.error())

            value = message.value()
            if value is None:
                raise ValueError(
                    f"Received a null transaction at {message.topic()} "
                    f"[{message.partition()}] offset {message.offset()}"
                )
            transaction = json.loads(value.decode("utf-8"))
            print(json.dumps(transaction, indent=2))
            consumer.commit(message=message, asynchronous=False)
            consumed += 1
    finally:
        consumer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce and consume demo Kafka transactions.")
    commands = parser.add_subparsers(dest="command", required=True)

    produce_parser = commands.add_parser("produce", help="publish demo transactions")
    produce_parser.add_argument("--count", type=int, default=5)

    consume_parser = commands.add_parser("consume", help="read transactions")
    consume_parser.add_argument(
        "--max-messages",
        type=int,
        help="stop after this many messages; by default, keep consuming",
    )

    args = parser.parse_args()
    if args.command == "produce":
        produce_transactions(args.count)
    else:
        consume_transactions(args.max_messages)


if __name__ == "__main__":
    main()
