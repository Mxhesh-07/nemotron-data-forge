# Container for running the pipeline as a Nebius Serverless Job.
# See: https://docs.nebius.com/serverless/quickstart/jobs
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Default command; override the arguments when you launch the job.
# Secrets must be provided as environment variables (NEBIUS_API_KEY, TAVILY_API_KEY).
ENTRYPOINT ["python", "scripts/serverless_job.py"]
CMD ["--topic", "Indian tax law for freelancers", "--out", "/outputs"]
