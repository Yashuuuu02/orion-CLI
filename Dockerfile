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
COPY env.py .
COPY inference.py .
COPY server.py .
COPY server/ ./server/
COPY openenv.yaml .

RUN chown -R orion:orion /app

ENV PORT=7860

EXPOSE 7860

USER orion

CMD ["python", "server.py"]