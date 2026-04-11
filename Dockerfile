FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 1000 orion \
    && useradd -u 1000 -g orion -m -s /bin/bash orion

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY orion/ ./orion/
COPY tasks/ ./tasks/
COPY app/ ./app/
COPY env.py .
COPY inference.py .
COPY server.py .
COPY server/ ./server/
COPY openenv.yaml .

RUN chown -R orion:orion /app

ENV PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/health >/dev/null || exit 1

USER orion

COPY scripts/ ./scripts/
RUN python -c "from scripts.preseed_bandit import preseed_bandit; preseed_bandit()"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]