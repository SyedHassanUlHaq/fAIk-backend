"""
SQS polling worker — shared template for all three ML worker containers.
Each container sets WORKER_TYPE and SQS_QUEUE_URL via environment variables.

Deploy this file into each worker repo. The model inference logic
(run_inference) is the only thing that differs between workers.
"""

import json
import os
import time
import logging
import subprocess

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

QUEUE_URL    = os.environ["SQS_QUEUE_URL"]
WORKER_TYPE  = os.environ["WORKER_TYPE"]          # video | audio | lipsync
RESULT_URL   = os.environ["SQS_RESULT_QUEUE_URL"] # core service reads results from here
AWS_REGION   = os.environ.get("AWS_REGION", "ap-southeast-1")
IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT_SECONDS", "600"))  # 10 min

sqs = boto3.client("sqs", region_name=AWS_REGION)


# ---------------------------------------------------------------------------
# Inference stub — replace with actual model call in each worker repo
# ---------------------------------------------------------------------------

def run_inference(s3_key: str, scan_type: str) -> dict:
    """
    Download file from S3, run the model, return result dict.
    Each worker repo implements this with its own model.

    Must return:
    {
        "score":   float,   # 0.0 – 1.0
        "verdict": str,     # "ai" | "authentic" | "tampered"
        "details": dict,    # model-specific extra data
    }
    """
    raise NotImplementedError("Implement run_inference() in the worker repo")


# ---------------------------------------------------------------------------
# Result callback — publishes back to core service via result queue
# ---------------------------------------------------------------------------

def publish_result(scan_id: str, result: dict):
    sqs.send_message(
        QueueUrl=RESULT_URL,
        MessageBody=json.dumps({
            "scan_id":     scan_id,
            "worker_type": WORKER_TYPE,
            "score":       result["score"],
            "verdict":     result["verdict"],
            "details":     result.get("details", {}),
        }),
    )


# ---------------------------------------------------------------------------
# Self-terminate after idle (scale-to-zero)
# ---------------------------------------------------------------------------

def self_terminate():
    log.info(f"Idle for {IDLE_TIMEOUT}s — self-terminating instance.")
    instance_id = subprocess.check_output(
        ["curl", "-s", "http://169.254.169.254/latest/meta-data/instance-id"]
    ).decode().strip()
    boto3.client("ec2", region_name=AWS_REGION).terminate_instances(
        InstanceIds=[instance_id]
    )


# ---------------------------------------------------------------------------
# Main poll loop
# ---------------------------------------------------------------------------

def poll():
    log.info(f"Worker started — type={WORKER_TYPE}  queue={QUEUE_URL}")
    idle_since = time.time()

    while True:
        response = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,      # long-polling — reduces empty-receive cost
            VisibilityTimeout=300,   # 5 min — message reappears if worker crashes
        )

        messages = response.get("Messages", [])

        if not messages:
            if time.time() - idle_since >= IDLE_TIMEOUT:
                self_terminate()
                return
            continue

        idle_since = time.time()
        message = messages[0]
        body = json.loads(message["Body"])
        scan_id  = body["scan_id"]
        s3_key   = body["s3_key"]
        scan_type = body["scan_type"]

        log.info(f"Processing scan {scan_id}  s3_key={s3_key}")

        try:
            result = run_inference(s3_key, scan_type)
            publish_result(scan_id, result)

            # Delete only on success — failure leaves message for retry
            sqs.delete_message(
                QueueUrl=QUEUE_URL,
                ReceiptHandle=message["ReceiptHandle"],
            )
            log.info(f"Done {scan_id}  score={result['score']:.3f}")

        except Exception as exc:
            log.error(f"Inference failed for {scan_id}: {exc}")
            # Message stays in queue and reappears after visibility timeout


if __name__ == "__main__":
    poll()
