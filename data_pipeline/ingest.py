"""
Ingestion / streaming simulation layer.

Per project requirements we use a lightweight in-process async queue instead
of Kafka -- this is enough to realistically simulate "transactions arriving
over time" for the demo (live feed in the dashboard, drift-monitoring batches)
without requiring a broker/cluster to operate the project.

Two entry points:
  - `load_transactions()`      : loads the engineered feature parquet into memory
  - `stream_transactions()`    : async producer that replays rows in source-ID
                                  order at a configurable rate,
                                  pushing them onto an asyncio.Queue
"""
import asyncio
from pathlib import Path

import pandas as pd

FEATURES_PATH = Path(__file__).parent.parent / "data" / "creditcard_features.parquet"


def load_transactions(path: Path = FEATURES_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.sort_values("id").reset_index(drop=True)


async def stream_transactions(
    queue: "asyncio.Queue",
    rows_per_second: float = 500.0,
    max_rows: int | None = 5000,
):
    """
    Replays rows in source-ID order onto `queue`. The source ID is not a timestamp.

    rows_per_second: approximate replay rate.
    max_rows: cap the number of rows replayed.
    """
    df = load_transactions()
    if max_rows is not None:
        df = df.tail(max_rows).reset_index(drop=True)

    for _, row in df.iterrows():
        await asyncio.sleep(1 / max(rows_per_second, 1.0))
        await queue.put(row.to_dict())


if __name__ == "__main__":
    async def _demo():
        q: asyncio.Queue = asyncio.Queue()
        producer = asyncio.create_task(stream_transactions(q, rows_per_second=2000, max_rows=200))
        consumed = 0
        while consumed < 200:
            item = await q.get()
            consumed += 1
            if consumed % 50 == 0:
                print(f"[ingest demo] consumed {consumed} transactions, last amount={item['Amount']:.2f}")
        await producer

    asyncio.run(_demo())
