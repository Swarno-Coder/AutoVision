FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port 8000 & streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0"]