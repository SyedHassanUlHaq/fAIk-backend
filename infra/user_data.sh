#!/bin/bash
# GPU instance user-data script.
# Runs once at boot. Downloads model weights from S3, then starts Docker containers.
# Set these variables before baking into your Launch Template.

set -euxo pipefail

BUCKET="YOUR_BUCKET_NAME"
WORKDIR="/home/ubuntu"

# ---------------------------------------------------------------------------
# 1. Download model weights from S3
#    Runs in parallel to minimise cold-start time.
# ---------------------------------------------------------------------------
mkdir -p $WORKDIR/checkpoints
mkdir -p $WORKDIR/saved_models

aws s3 cp s3://$BUCKET/models/fused_best.pt \
    $WORKDIR/checkpoints/fused_best.pt &

aws s3 cp s3://$BUCKET/models/raft-sintel.pth \
    $WORKDIR/checkpoints/raft-sintel.pth &

aws s3 sync s3://$BUCKET/models/xclip-base-patch16 \
    $WORKDIR/saved_models/xclip-base-patch16/ &

aws s3 sync s3://$BUCKET/models/nomic-embed-vision-v1.5 \
    $WORKDIR/saved_models/nomic-embed-vision-v1.5/ &

aws s3 sync s3://$BUCKET/models/nomic-bert-2048 \
    $WORKDIR/saved_models/nomic-bert-2048/ &

aws s3 sync s3://$BUCKET/models/faster-whisper-small.en \
    $WORKDIR/saved_models/faster-whisper-small.en/ &

# Wait for all downloads to complete before starting containers
wait
echo "[boot] All model weights downloaded."

# ---------------------------------------------------------------------------
# 2. Pull latest worker code from S3 (or bake repos into AMI instead)
# ---------------------------------------------------------------------------
aws s3 sync s3://$BUCKET/worker-repos/repo-video   $WORKDIR/repo-video/
aws s3 sync s3://$BUCKET/worker-repos/repo-audio   $WORKDIR/repo-audio/
aws s3 sync s3://$BUCKET/worker-repos/repo-lipsync $WORKDIR/repo-lipsync/
aws s3 cp   s3://$BUCKET/scripts/worker.py          $WORKDIR/worker.py

# ---------------------------------------------------------------------------
# 3. Start all three worker containers
# ---------------------------------------------------------------------------
cd $WORKDIR
docker compose up -d --build
echo "[boot] Worker containers started."
