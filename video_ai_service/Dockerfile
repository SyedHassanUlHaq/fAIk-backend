FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ffmpeg/ffprobe: used by helpers/video_helper.py and tasks/video_scan_task.py
# libgl1/libglib2.0-0: required by opencv-python at import time
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Model weights (checkpoints/, microsoft/, repositories/validation_tool/checkpoints/)
# are git-ignored and expected to be bind-mounted at runtime, not baked into the image
# (see infra/docker-compose.yml). Mount them to these paths when running the container.
RUN mkdir -p checkpoints microsoft repositories/validation_tool/checkpoints sessions

EXPOSE 8000

# Default command runs the FastAPI app. For the Celery worker, override with:
#   celery -A celery_app worker --loglevel=info
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
