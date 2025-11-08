# ------------ Frontend Dockerfile ------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY frontend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY frontend/streamlit_app.py /app/streamlit_app.py
COPY frontend/.streamlit /app/.streamlit

# Tell Streamlit where the backend lives (Docker internal name)
ENV API_URL=http://backend:8000

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
