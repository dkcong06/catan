FROM node:24-alpine AS viewer-checks

WORKDIR /app
COPY scripts/board_viewer.html scripts/board_viewer.html
COPY scripts/discard_viewer.html scripts/discard_viewer.html
COPY tests/board_viewer.test.cjs tests/board_viewer.test.cjs
COPY tests/discard_viewer.test.cjs tests/discard_viewer.test.cjs
CMD ["node", "--test", "tests/board_viewer.test.cjs", "tests/discard_viewer.test.cjs"]

FROM python:3.14-slim-bookworm AS dependencies

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

FROM dependencies AS learning-dependencies
COPY requirements-learning.txt .
RUN python -m pip install --no-cache-dir --root-user-action=ignore --index-url https://download.pytorch.org/whl/cpu -r requirements-learning.txt

FROM learning-dependencies AS learning-checks
COPY . .
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]

# Default image keeps engine and collector checks independent of PyTorch.
FROM dependencies AS checks
COPY . .
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
