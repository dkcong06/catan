FROM python:3.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --root-user-action=ignore -r requirements.txt
COPY . .

CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
