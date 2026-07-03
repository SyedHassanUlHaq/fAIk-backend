from celery import Celery
from config.project_config import REDIS_URL

celery_app = Celery(
    "faik",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks.scan_task"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)
