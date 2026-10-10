FROM node:24-alpine AS viewer-checks

WORKDIR /app
COPY scripts/board_viewer.html scripts/board_viewer.html
COPY scripts/discard_viewer.html scripts/discard_viewer.html
COPY tests/board_viewer.test.cjs tests/board_viewer.test.cjs
COPY tests/discard_viewer.test.cjs tests/discard_viewer.test.cjs
CMD ["node", "--test", "tests/board_viewer.test.cjs", "tests/discard_viewer.test.cjs"]

FROM python:3.14-slim-bookworm AS checks

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --root-user-action=ignore -r requirements.txt
COPY . .

CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
