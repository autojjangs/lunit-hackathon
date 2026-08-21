FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py pipeline.py ./

ARG LUNIT_FM_API_KEY=lunit_mHCixQhPzR--PRq4FEmZnx1yzdtQLCegz_E_SL58b20
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LUNIT_FM_API_KEY=${LUNIT_FM_API_KEY}

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
