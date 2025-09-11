FROM python:3.11-slim

ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install \
      gradio==5.44.1 \
      yandex-cloud-ml-sdk==0.15.0 \
      pandas==2.2.3 \
      dotenv \
      openai

COPY src/ /app/src
COPY app.py /app/app.py

EXPOSE 7860

CMD ["python3", "app.py"]
