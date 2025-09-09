docker run -it \
  --env-file .env \
  -p 7860:7860 \
  -v $(pwd)/data.csv:/app/data.csv \
  exam-app
