FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN groupadd --system harness && useradd --system --gid harness --home-dir /app harness
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

USER harness
EXPOSE 8080
CMD ["ai-harness", "serve", "--host", "0.0.0.0", "--port", "8080"]
