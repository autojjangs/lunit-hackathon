# Submission image. Build must finish in under 5 minutes on the eval VM,
# so: slim base, pinned wheels, no source builds, no model downloads.
FROM python:3.12-slim

WORKDIR /app

COPY submission/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/healthbench_harness/ /app/healthbench_harness/
COPY submission/ /app/submission/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LUNIT_FM_API_URL=https://model.hackathon.lunit.io \
    LUNIT_FM_MODEL=Lunit/L2-preview \
    LUNIT_MCP_URL=https://mcp.hackathon.lunit.io/mcp

ENV L2_MAX_CONCURRENCY=15 \
    MCP_MAX_CONCURRENT_SESSIONS=8

EXPOSE 8000
CMD ["uvicorn", "submission.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
