FROM python:3.12-slim

WORKDIR /app

# Install dependencies first, separately from the app code, so Docker's
# layer cache skips this (slow) step on rebuilds that only touch src/.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
