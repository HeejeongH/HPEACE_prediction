FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
# 학습된 모델 산출물과, main.py가 import하는 feature_engineering.py를 함께 복사
COPY result/production_models /app/model_artifacts
COPY src/feature_engineering.py /app/project_src/feature_engineering.py

ENV PRODUCTION_MODEL_DIR=/app/model_artifacts
ENV PROJECT_SRC_DIR=/app/project_src
ENV CORS_ORIGINS=*

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
