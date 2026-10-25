FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==3.15.0

EXPOSE 5000
