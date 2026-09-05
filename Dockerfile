# Application image for the zsttSystem online service.
#
# The offline pipeline runs from the same image (see the `pipeline` profile in
# docker-compose.yml), so a reviewer can reproduce parsing, indexing and the
# API with one checkout and one `docker compose up`.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep the sentence-transformer download on a mounted volume instead of
    # baking a 400 MB model into the image.
    HF_HOME=/models

WORKDIR /app

# Dependencies first so that source edits do not invalidate the wheel layer.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY maintenance ./maintenance
COPY eval ./eval
COPY run_pipeline.py ./

RUN useradd --create-home --uid 10001 zstt \
    && mkdir -p /app/outputs /models \
    && chown -R zstt:zstt /app /models
USER zstt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
