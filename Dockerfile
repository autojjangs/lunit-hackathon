# Submission image. Build must finish in under 5 minutes on the eval VM,
# so: slim base, pinned wheels, no source builds, no model downloads.
FROM python:3.13-slim

WORKDIR /app

COPY submission/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY system/ /app/system/
COPY submission/ /app/submission/
COPY mcp_tools.json /app/mcp_tools.json

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LUNIT_FM_API_URL=https://model.hackathon.lunit.io \
    LUNIT_FM_API_KEY=lunit_mHCixQhPzR--PRq4FEmZnx1yzdtQLCegz_E_SL58b20 \
    LUNIT_FM_MODEL=Lunit/L2-preview \
    LUNIT_MCP_URL=https://mcp.hackathon.lunit.io/mcp

EXPOSE 8000
CMD ["uvicorn", "submission.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
