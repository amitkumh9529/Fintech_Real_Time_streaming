FROM python:3.14-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    openjdk-17-jre-headless \
    procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data_pipeline /app/data_pipeline
COPY ml_pipeline /app/ml_pipeline

ENV PYTHONPATH=/app

# Default: run the full pipeline end to end (Spark features -> train -> select champion -> evaluate)
CMD ["bash", "-c", "python3 data_pipeline/feature_engineering_spark.py && python3 ml_pipeline/train.py --trials 25 && python3 ml_pipeline/select_champion.py && python3 ml_pipeline/evaluate.py"]
