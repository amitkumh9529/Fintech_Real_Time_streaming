# Fintech real-time streaming

## Run Kafka locally

The development broker runs Apache Kafka in KRaft mode as a single node. Docker
Compose is required.

```powershell
docker compose up -d
uv sync
```

Kafka listens on `localhost:9092`. The broker stores its data in the
`kafka_data` Docker volume, which is retained by `docker compose down`.

Copy `.env.example` to `.env` if you want to override the local Kafka settings.
The defaults are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka bootstrap address |
| `KAFKA_TOPIC` | `transactions` | Topic used by the demo |
| `KAFKA_CONSUMER_GROUP` | `fintech-demo-consumer` | Consumer group ID |

## Try the Python producer and consumer

In one terminal, start the consumer:

```powershell
uv run python kafka_demo.py consume --max-messages 5
```

In a second terminal, publish five sample transactions:

```powershell
uv run python kafka_demo.py produce --count 5
```

Omit `--max-messages` to keep the consumer running. Stop the broker with
`docker compose down`. This broker is for local development only; it has no
authentication or TLS configured and should not be exposed to untrusted
networks.