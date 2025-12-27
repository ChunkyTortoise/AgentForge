"""
Celery configuration for distributed task processing.
"""
from celery import Celery
from core.config import settings

# Initialize Celery
# Note: broker is for message passing, backend is for result storage
celery_app = Celery(
    "agentforge",
    broker=settings.redis_url,
    backend=settings.redis_url
)

# Optional configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300, # 5 minute limit for swarms
)
