"""
Ingestion / streaming simulation layer.

Per project requirements we use a lightweight in-process async queue instead
of Kafka -- this is enough to realistically simulate "transactions arriving
over time" for the demo (live feed in the dashboard, drift-monitoring batches)
without requiring a broker/cluster to operate the project.

Two entry points:
  - `load_transactions()`      : loads the engineered feature parquet into memory
  - `stream_transactions()`    : async producer that replays transactions in
                                  timestamp order at a configurable speed multiplier,
                                  pushing them onto an asyncio.Queue
"""
import asyncio
from pathlib import Path

import pandas as pd

FEATURES_PATH = Path(__file__).parent.parent / "data" / "creditcard_features.parquet"


def load_transactions(path: Path = FEATURES_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.sort_values(["TX_DATETIME", "TRANSACTION_ID"]).reset_index(drop=True)


async def stream_transactions(
    queue: "asyncio.Queue",
    speed_multiplier: float = 500.0,
    max_rows: int | None = 5000,
):
    """
    Replays transactions in timestamp order onto `queue`.

    speed_multiplier: how much faster than the historical timeline to replay.
    max_rows: cap the number of rows replayed.
    """
    df = load_transactions()
    if max_rows is not None:
        df = df.tail(max_rows).reset_index(drop=True)

    previous_datetime = None
    for _, row in df.iterrows():
        current_datetime = row["TX_DATETIME"]
        if previous_datetime is not None:
            delta = max(0.0, (current_datetime - previous_datetime).total_seconds())
            await asyncio.sleep(min(delta / max(speed_multiplier, 1.0), 0.5))
        previous_datetime = current_datetime
        await queue.put(row.to_dict())


if __name__ == "__main__":
    async def _demo():
        q: asyncio.Queue = asyncio.Queue()
        producer = asyncio.create_task(stream_transactions(q, speed_multiplier=2000, max_rows=200))
        consumed = 0
        while consumed < 200:
            item = await q.get()
            consumed += 1
            if consumed % 50 == 0:
                print(f"[ingest demo] consumed {consumed} transactions, last amount={item['TX_AMOUNT']:.2f}")
        await producer

    asyncio.run(_demo())
