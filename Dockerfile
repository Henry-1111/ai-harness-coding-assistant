FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates git openssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system harness && useradd --system --gid harness --home-dir /app harness
RUN mkdir -p /data && chown harness:harness /data

# Resolve third-party dependencies in a layer that source-only edits do not invalidate.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip mkdir -p src/ai_harness \
    && touch src/ai_harness/__init__.py \
    && printf '# AI Harness\n' > README.md \
    && pip install .

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps --force-reinstall .

USER harness
EXPOSE 8080
CMD ["ai-harness", "serve", "--host", "0.0.0.0", "--port", "8080"]
