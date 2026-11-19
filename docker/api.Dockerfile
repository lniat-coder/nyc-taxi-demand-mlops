FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TAXI_ROOT=/app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
# Code and configuration only: no data, no secrets, no model inside the image.
# The model is pulled from the MLflow Registry at startup (@production alias).
COPY params.yaml policy.yaml pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
RUN useradd --create-home app && mkdir -p /app/logs && chown app /app/logs
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "taxi.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
