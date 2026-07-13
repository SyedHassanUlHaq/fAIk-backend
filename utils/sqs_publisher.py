"""
SQS publisher for the core service.
Replaces Celery task dispatch — call enqueue_scan() where process_scan.delay() was called.
"""

import json
import os

import boto3

_sqs = None


def _client():
    global _sqs
    if _sqs is None:
        _sqs = boto3.client("sqs", region_name=os.environ["AWS_REGION"])
    return _sqs


QUEUE_URLS = {
    "video":  os.environ.get("SQS_URL_VIDEO", ""),
    "tamper": os.environ.get("SQS_URL_LIPSYNC", ""),   # scene/tamper → lipsync queue
    "audio":  os.environ.get("SQS_URL_AUDIO", ""),
}


def enqueue_scan(scan_id: str, s3_key: str, scan_type: str) -> str:
    """
    Publish a scan job to the appropriate SQS queue.
    Returns the SQS MessageId.
    """
    queue_url = QUEUE_URLS.get(scan_type)
    if not queue_url:
        raise ValueError(f"No SQS queue configured for scan_type '{scan_type}'")

    response = _client().send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps({
            "scan_id": scan_id,
            "s3_key":  s3_key,
            "scan_type": scan_type,
        }),
    )
    return response["MessageId"]
