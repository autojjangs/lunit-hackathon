# Submission image. Build must finish in under 5 minutes on the eval VM,
# so: slim base, pinned wheels, no source builds, no model downloads.
FROM python:3.13-slim

WORKDIR /app

COPY submission/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# The runtime is one self-contained package. The MCP tool schemas sit beside it
# so a container start never depends on a live tools/list round-trip.
COPY submission/ /app/submission/
COPY mcp_tools.json /app/submission/mcp_tools.json

ARG LUNIT_FM_API_KEY=lunit_mHCixQhPzR--PRq4FEmZnx1yzdtQLCegz_E_SL58b20
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LUNIT_FM_API_KEY=${LUNIT_FM_API_KEY} \
    LUNIT_MCP_URL=https://mcp.hackathon.lunit.io/mcp

EXPOSE 8000
CMD ["uvicorn", "submission.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
