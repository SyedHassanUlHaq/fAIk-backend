#!/bin/bash
# ============================================================
# 5dot AWS infrastructure setup
# Run once to create all SQS queues, IAM roles, Launch Templates,
# Auto Scaling Groups, and CloudWatch alarms.
#
# Services:
#   core_service          — FastAPI, PostgreSQL, auth, billing
#   video_ai_service      — RAFT + XCLIP + DeMamba
#   ai_audio_service      — faster-whisper / WavLM + Nes2Net
#   scene_detection       — nomic-embed-vision-v1.5
#   lipsync_service       — lipsync model (TBD)
#
# Fill in the variables below before running.
# ============================================================

set -euo pipefail

# ---------- CONFIGURE THESE ----------
AMI_ID="ami-XXXXXXXXXXXXXXXXX"   # GPU AMI (Docker + NVIDIA toolkit installed)
KEY_NAME="5dot-key.pem"
SECURITY_GROUP_ID="sg-05bb33eaf125e1d75"
SUBNET_ID="subnet-09b4acea93ecf055b"
BUCKET="5dot-production"
REGION="ap-southeast-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
# -------------------------------------

echo "==> Account: $ACCOUNT_ID  Region: $REGION"

# ============================================================
# 1. SQS QUEUES
#    One job queue per ML service + one result queue for
#    workers to publish results back to core service.
# ============================================================
echo "==> Creating SQS queues..."

for QUEUE in deepfake-video deepfake-audio deepfake-scene deepfake-lipsync deepfake-results; do
    aws sqs create-queue \
        --queue-name $QUEUE \
        --attributes VisibilityTimeout=300,MessageRetentionPeriod=86400 \
        --region $REGION
    echo "    Created: $QUEUE"
done

VIDEO_URL=$(aws sqs get-queue-url   --queue-name deepfake-video    --region $REGION --query QueueUrl --output text)
AUDIO_URL=$(aws sqs get-queue-url   --queue-name deepfake-audio    --region $REGION --query QueueUrl --output text)
SCENE_URL=$(aws sqs get-queue-url   --queue-name deepfake-scene    --region $REGION --query QueueUrl --output text)
LIPSYNC_URL=$(aws sqs get-queue-url --queue-name deepfake-lipsync  --region $REGION --query QueueUrl --output text)
RESULT_URL=$(aws sqs get-queue-url  --queue-name deepfake-results  --region $REGION --query QueueUrl --output text)

VIDEO_ARN=$(aws sqs get-queue-attributes   --queue-url $VIDEO_URL   --attribute-names QueueArn --query Attributes.QueueArn --output text --region $REGION)
AUDIO_ARN=$(aws sqs get-queue-attributes   --queue-url $AUDIO_URL   --attribute-names QueueArn --query Attributes.QueueArn --output text --region $REGION)
SCENE_ARN=$(aws sqs get-queue-attributes   --queue-url $SCENE_URL   --attribute-names QueueArn --query Attributes.QueueArn --output text --region $REGION)
LIPSYNC_ARN=$(aws sqs get-queue-attributes --queue-url $LIPSYNC_URL --attribute-names QueueArn --query Attributes.QueueArn --output text --region $REGION)

echo "    Video   : $VIDEO_URL"
echo "    Audio   : $AUDIO_URL"
echo "    Scene   : $SCENE_URL"
echo "    Lipsync : $LIPSYNC_URL"
echo "    Results : $RESULT_URL"

# ============================================================
# 2. IAM — GPU WORKER ROLE
#    Attached to all ML worker EC2 instances.
#    Allows: consume job queues, publish to results queue,
#            read models from S3, self-terminate on idle.
# ============================================================
echo "==> Creating IAM role for GPU workers..."

aws iam create-role \
    --role-name 5dot-gpu-worker-role \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{
        "Effect":"Allow",
        "Principal":{"Service":"ec2.amazonaws.com"},
        "Action":"sts:AssumeRole"
      }]
    }' 2>/dev/null || echo "    Role already exists, skipping."

aws iam put-role-policy \
    --role-name 5dot-gpu-worker-role \
    --policy-name 5dot-gpu-worker-policy \
    --policy-document "{
      \"Version\":\"2012-10-17\",
      \"Statement\":[
        {
          \"Effect\":\"Allow\",
          \"Action\":[\"sqs:ReceiveMessage\",\"sqs:DeleteMessage\",\"sqs:GetQueueAttributes\"],
          \"Resource\":[\"$VIDEO_ARN\",\"$AUDIO_ARN\",\"$SCENE_ARN\",\"$LIPSYNC_ARN\"]
        },
        {
          \"Effect\":\"Allow\",
          \"Action\":[\"sqs:SendMessage\"],
          \"Resource\":\"arn:aws:sqs:$REGION:$ACCOUNT_ID:deepfake-results\"
        },
        {
          \"Effect\":\"Allow\",
          \"Action\":[\"s3:GetObject\"],
          \"Resource\":\"arn:aws:s3:::$BUCKET/models/*\"
        },
        {
          \"Effect\":\"Allow\",
          \"Action\":[\"s3:PutObject\",\"s3:GetObject\"],
          \"Resource\":\"arn:aws:s3:::$BUCKET/scans/*\"
        },
        {
          \"Effect\":\"Allow\",
          \"Action\":[\"ec2:TerminateInstances\"],
          \"Resource\":\"*\",
          \"Condition\":{\"StringEquals\":{\"ec2:ResourceTag/Role\":\"5dot-gpu-worker\"}}
        },
        {
          \"Effect\":\"Allow\",
          \"Action\":[\"cloudwatch:PutMetricData\"],
          \"Resource\":\"*\"
        }
      ]
    }"

aws iam create-instance-profile \
    --instance-profile-name 5dot-gpu-worker-profile 2>/dev/null || true
aws iam add-role-to-instance-profile \
    --instance-profile-name 5dot-gpu-worker-profile \
    --role-name 5dot-gpu-worker-role 2>/dev/null || true

echo "    IAM role ready: 5dot-gpu-worker-role"

# ============================================================
# 3. IAM — CORE SERVICE ROLE
#    Attached to the core service EC2 instance.
#    Allows: publish to job queues, consume results queue,
#            full S3 access for scans/reports/thumbnails,
#            SES for OTP emails.
# ============================================================
echo "==> Creating IAM role for core service..."

aws iam create-role \
    --role-name 5dot-core-service-role \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{
        "Effect":"Allow",
        "Principal":{"Service":"ec2.amazonaws.com"},
        "Action":"sts:AssumeRole"
      }]
    }' 2>/dev/null || echo "    Role already exists, skipping."

aws iam put-role-policy \
    --role-name 5dot-core-service-role \
    --policy-name 5dot-core-service-policy \
    --policy-document "{
      \"Version\":\"2012-10-17\",
      \"Statement\":[
        {
          \"Effect\":\"Allow\",
          \"Action\":[\"sqs:SendMessage\",\"sqs:GetQueueAttributes\"],
          \"Resource\":[\"$VIDEO_ARN\",\"$AUDIO_ARN\",\"$SCENE_ARN\",\"$LIPSYNC_ARN\"]
        },
        {
          \"Effect\":\"Allow\",
          \"Action\":[\"sqs:ReceiveMessage\",\"sqs:DeleteMessage\",\"sqs:GetQueueAttributes\"],
          \"Resource\":\"arn:aws:sqs:$REGION:$ACCOUNT_ID:deepfake-results\"
        },
        {
          \"Effect\":\"Allow\",
          \"Action\":[\"s3:PutObject\",\"s3:GetObject\",\"s3:DeleteObject\"],
          \"Resource\":[
            \"arn:aws:s3:::$BUCKET/scans/*\",
            \"arn:aws:s3:::$BUCKET/thumbnails/*\",
            \"arn:aws:s3:::$BUCKET/reports/*\"
          ]
        },
        {
          \"Effect\":\"Allow\",
          \"Action\":[\"ses:SendEmail\",\"ses:SendRawEmail\"],
          \"Resource\":\"*\"
        }
      ]
    }"

aws iam create-instance-profile \
    --instance-profile-name 5dot-core-service-profile 2>/dev/null || true
aws iam add-role-to-instance-profile \
    --instance-profile-name 5dot-core-service-profile \
    --role-name 5dot-core-service-role 2>/dev/null || true

echo "    IAM role ready: 5dot-core-service-role"

# ============================================================
# 4. LAUNCH TEMPLATE  (GPU workers)
# ============================================================
echo "==> Creating Launch Template..."

USER_DATA=$(base64 -w 0 infra/user_data.sh)

aws ec2 create-launch-template \
    --launch-template-name 5dot-gpu-worker \
    --version-description "v1" \
    --launch-template-data "{
      \"ImageId\": \"$AMI_ID\",
      \"InstanceType\": \"g4dn.xlarge\",
      \"KeyName\": \"$KEY_NAME\",
      \"SecurityGroupIds\": [\"$SECURITY_GROUP_ID\"],
      \"IamInstanceProfile\": {\"Name\": \"5dot-gpu-worker-profile\"},
      \"TagSpecifications\": [{
        \"ResourceType\": \"instance\",
        \"Tags\": [{\"Key\": \"Role\", \"Value\": \"5dot-gpu-worker\"}]
      }],
      \"InstanceMarketOptions\": {
        \"MarketType\": \"spot\",
        \"SpotOptions\": {
          \"SpotInstanceType\": \"one-time\",
          \"InstanceInterruptionBehavior\": \"terminate\"
        }
      },
      \"UserData\": \"$USER_DATA\",
      \"BlockDeviceMappings\": [{
        \"DeviceName\": \"/dev/sda1\",
        \"Ebs\": {\"VolumeSize\": 100, \"VolumeType\": \"gp3\"}
      }]
    }" \
    --region $REGION 2>/dev/null || echo "    Launch template already exists."

LT_ID=$(aws ec2 describe-launch-templates \
    --filters Name=launch-template-name,Values=5dot-gpu-worker \
    --query 'LaunchTemplates[0].LaunchTemplateId' --output text --region $REGION)

echo "    Launch Template: $LT_ID"

# ============================================================
# 5. AUTO SCALING GROUPS  (one per ML service)
# ============================================================
echo "==> Creating Auto Scaling Groups..."

for WORKER in video audio scene lipsync; do
    aws autoscaling create-auto-scaling-group \
        --auto-scaling-group-name 5dot-worker-$WORKER \
        --launch-template "LaunchTemplateId=$LT_ID,Version=1" \
        --min-size 0 \
        --max-size 5 \
        --desired-capacity 0 \
        --vpc-zone-identifier $SUBNET_ID \
        --health-check-type EC2 \
        --health-check-grace-period 300 \
        --default-cooldown 300 \
        --tags "Key=WorkerType,Value=$WORKER,PropagateAtLaunch=true" \
              "Key=Role,Value=5dot-gpu-worker,PropagateAtLaunch=true" \
        --region $REGION 2>/dev/null || echo "    ASG 5dot-worker-$WORKER already exists."

    echo "    ASG created: 5dot-worker-$WORKER"
done

# ============================================================
# 6. SCALING POLICIES  (step scaling per ASG)
# ============================================================
echo "==> Creating scaling policies..."

for WORKER in video audio scene lipsync; do
    aws autoscaling put-scaling-policy \
        --auto-scaling-group-name 5dot-worker-$WORKER \
        --policy-name 5dot-$WORKER-scale-up \
        --policy-type StepScaling \
        --adjustment-type ChangeInCapacity \
        --metric-aggregation-type Maximum \
        --step-adjustments \
            "MetricIntervalLowerBound=0,MetricIntervalUpperBound=5,ScalingAdjustment=1" \
            "MetricIntervalLowerBound=5,MetricIntervalUpperBound=10,ScalingAdjustment=2" \
            "MetricIntervalLowerBound=10,ScalingAdjustment=3" \
        --region $REGION > /dev/null

    echo "    Scaling policy created: 5dot-$WORKER-scale-up"
done

# ============================================================
# 7. CLOUDWATCH ALARMS  (trigger scale-up when queue > 0)
# ============================================================
echo "==> Creating CloudWatch alarms..."

for WORKER in video audio scene lipsync; do
    QUEUE_NAME="deepfake-$WORKER"
    POLICY_ARN=$(aws autoscaling describe-policies \
        --auto-scaling-group-name 5dot-worker-$WORKER \
        --policy-names 5dot-$WORKER-scale-up \
        --query 'ScalingPolicies[0].PolicyARN' --output text --region $REGION)

    aws cloudwatch put-metric-alarm \
        --alarm-name "5dot-$WORKER-queue-depth" \
        --alarm-description "Scale up $WORKER workers when queue has messages" \
        --namespace AWS/SQS \
        --metric-name ApproximateNumberOfMessagesVisible \
        --dimensions Name=QueueName,Value=$QUEUE_NAME \
        --statistic Maximum \
        --period 60 \
        --evaluation-periods 1 \
        --threshold 0 \
        --comparison-operator GreaterThanThreshold \
        --alarm-actions $POLICY_ARN \
        --treat-missing-data notBreaching \
        --region $REGION

    echo "    Alarm created: 5dot-$WORKER-queue-depth"
done

# ============================================================
# DONE — print env vars to copy into .env files
# ============================================================
echo ""
echo "============================================================"
echo "  Core service .env"
echo "============================================================"
echo "  AWS_REGION=$REGION"
echo "  SQS_URL_VIDEO=$VIDEO_URL"
echo "  SQS_URL_AUDIO=$AUDIO_URL"
echo "  SQS_URL_SCENE=$SCENE_URL"
echo "  SQS_URL_LIPSYNC=$LIPSYNC_URL"
echo "  SQS_RESULT_QUEUE_URL=$RESULT_URL"
echo ""
echo "============================================================"
echo "  GPU instance docker-compose.yml environment"
echo "============================================================"
echo "  SQS_URL_VIDEO=$VIDEO_URL"
echo "  SQS_URL_AUDIO=$AUDIO_URL"
echo "  SQS_URL_SCENE=$SCENE_URL"
echo "  SQS_URL_LIPSYNC=$LIPSYNC_URL"
echo "  SQS_RESULT_QUEUE_URL=$RESULT_URL"
echo "  AWS_REGION=$REGION"
echo "============================================================"
