# fAIk Backend — Infrastructure & Architecture Documentation

> **Audience:** Project manager, technical leads, DevOps.  
> **Purpose:** Justifies every infrastructure decision — what we use, why we use it, and how it scales.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Why This Architecture](#2-why-this-architecture)
3. [Microservices Decision](#3-microservices-decision)
4. [Component Breakdown](#4-component-breakdown)
5. [Celery & Redis — Async Job System](#5-celery--redis--async-job-system)
6. [ML Model Queue Strategy](#6-ml-model-queue-strategy)
7. [AWS Infrastructure](#7-aws-infrastructure)
8. [Auto-Scaling Strategy](#8-auto-scaling-strategy)
9. [File Storage & CDN](#9-file-storage--cdn)
10. [Database](#10-database)
11. [Security](#11-security)
12. [CI/CD Pipeline](#12-cicd-pipeline)
13. [Monitoring & Alerting](#13-monitoring--alerting)
14. [Cost Estimate](#14-cost-estimate)
15. [Deployment Order](#15-deployment-order)

---

## 1. System Overview

fAIk is a mobile application (iOS + Android via Expo/React Native) that lets users upload audio and video files and receive an AI-generated authenticity verdict. The backend exposes a REST API consumed exclusively by the mobile app.

### Core Capabilities

| Capability | Description |
|---|---|
| **Deepfake video detection** | Detects GAN-based face swaps and lip-sync manipulation |
| **Scene / tamper detection** | Finds hard cuts, splices, and re-encode signatures in video |
| **AI audio detection** | Identifies TTS and voice-cloned audio |
| **Lipsync analysis** | Detects audio-video sync drift as a deepfake signal |
| **Forensic PDF reports** | Generates a downloadable evidence report (Pro/Team plans) |
| **Subscription billing** | Free / Pro / Team plans via Stripe |
| **Push notifications** | Real-time scan completion alerts via Expo Push |

### Traffic Profile

- Mobile app published on App Store and Google Play
- User acquisition driven by paid marketing → **bursty, unpredictable traffic**
- Scans are compute-heavy (GPU) but asynchronous — users are not blocked waiting
- API calls (auth, history, profile) are lightweight and frequent
- Peak scan load is independent of peak API load

---

## 2. Why This Architecture

### The Core Problem

ML inference is slow and GPU-dependent. A naive approach — running inference synchronously inside the API server — has fatal flaws:

| Problem | Impact |
|---|---|
| API server blocks for 4–15 s per scan | All other users' requests queue up |
| GPU required on the API server | Every API replica needs an expensive GPU instance |
| One slow scan type blocks all others | Audio backlog delays video scans |
| Can't scale API and inference independently | Pay for GPU even when no scans are running |

### The Solution: Decouple API from Inference

```
Mobile App → API Server (lightweight, no GPU)
                  ↓  enqueues job
             Redis Queue
                  ↓  worker picks up job
             ML Worker (GPU, runs inference)
                  ↓  writes result to DB
             PostgreSQL
                  ↓  sends push notification
             Mobile App
```

The API server's only job on a scan request is to accept the file, save it to S3, write a record to the database, and put a job on a queue. It returns in milliseconds. The heavy work happens in a completely separate process on a completely separate (GPU) machine.

---

## 3. Microservices Decision

### Decision: Partial — Service Separation at the Worker Level

Full microservices (each ML model as its own HTTP service behind an API gateway) is **not recommended at this stage**. The overhead — separate service discovery, inter-service auth, network hops, individual deployment pipelines — is not justified for the current team size and product maturity.

However, **the single most important split has already been made**: the API layer and the ML workers are separate deployables. Beyond that, we separate workers **by model type**, not by HTTP service boundary.

### What We Separate and Why

| Separation | Method | Why |
|---|---|---|
| API vs. ML inference | Separate processes / machines | API needs no GPU; inference needs GPU |
| Video vs. audio vs. tamper inference | Separate Celery queues + worker pools | Independent scaling, no cross-model blocking |
| PDF generation | Separate Celery queue + CPU-only workers | No GPU needed; shouldn't share GPU resources |
| Database | Managed RDS | Separate lifecycle, backups, failover |
| File storage | S3 | Decoupled from compute entirely |

### When to Revisit

Consider moving to true microservices (separate HTTP services) when:
- The team grows to 3+ backend engineers
- Different models need to be deployed on different cadences
- External partners need direct API access to individual models
- Inference latency SLAs require dedicated load balancing per model

---

## 4. Component Breakdown

```
┌─────────────────────────────────────────────────────────────────────┐
│                          MOBILE APP                                 │
│                    (iOS / Android — Expo)                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTPS
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       AWS INFRASTRUCTURE                             │
│                                                                      │
│  Route 53 (DNS)                                                      │
│       ↓                                                              │
│  CloudFront CDN ─────────────────────────────────── S3 Bucket       │
│  (thumbnails, PDFs,                                 (scan files,    │
│   static presigned URLs)                             thumbnails,    │
│       ↓                                              PDF reports)   │
│  Application Load Balancer                                           │
│       ↓                                                              │
│  ┌─────────────────────────────┐                                     │
│  │   ECS Fargate (API Layer)   │  ← No GPU, t3.medium               │
│  │   FastAPI application       │  ← 2–20 tasks, auto-scaled         │
│  │   Handles: auth, scans,     │  ← Stateless, ephemeral            │
│  │   users, plans, webhooks    │                                     │
│  └─────────────┬───────────────┘                                     │
│                │ enqueue jobs                                        │
│                ▼                                                     │
│  ┌─────────────────────────────┐                                     │
│  │   ElastiCache Redis         │  ← Celery broker + result backend  │
│  │   4 queues (one per model)  │  ← cache.t3.medium                 │
│  └──────┬───────┬──────┬───────┘                                     │
│         │       │      │       │                                     │
│         ▼       ▼      ▼       ▼                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│  │  Video   │ │  Tamper  │ │  Audio   │ │   PDF    │               │
│  │  Worker  │ │  Worker  │ │  Worker  │ │  Worker  │               │
│  │ deepfake │ │  scene   │ │    AI    │ │reportlab │               │
│  │ lipsync  │ │   cut    │ │  audio   │ │ CPU only │               │
│  │g4dn.xlg  │ │g4dn.xlg  │ │g4dn.xlg  │ │ Fargate  │               │
│  │  Spot    │ │  Spot    │ │  Spot    │ │          │               │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘               │
│                           ↓                                          │
│  ┌─────────────────────────────┐                                     │
│  │   RDS PostgreSQL            │  ← Multi-AZ, db.t3.medium          │
│  │   Users, scans, feedback,   │  ← Automated backups, failover     │
│  │   subscriptions, tokens     │                                     │
│  └─────────────────────────────┘                                     │
│                                                                      │
│  Secrets Manager  │  CloudWatch  │  ACM (SSL)  │  ECR (images)      │
└─────────────────────────────────────────────────────────────────────-┘
                               │ push notification
                               ▼
                       Expo Push Service
                      (routes to APNs / FCM)
```

---

## 5. Celery & Redis — Async Job System

### What is Celery?

Celery is a distributed task queue. When a user submits a scan, the API does not run the ML model. Instead it creates a **task** — a serialised job message placed on a queue in Redis. A separate worker process (on a separate machine) picks up the task, runs the ML model, and writes the result to the database.

### Why Celery?

| Requirement | How Celery Satisfies It |
|---|---|
| Non-blocking API responses | Task is submitted in milliseconds; API returns immediately |
| Retries on failure | `max_retries=2` with exponential back-off — transient GPU errors don't lose scans |
| Independent worker scaling | Workers are separate processes; scale independently of the API |
| Priority queues | Different queues for different model types |
| Visibility | Task state (queued → processing → complete) tracked in Redis and DB |
| Monitoring | Flower dashboard gives real-time visibility of all workers and tasks |

### Why Redis as the Broker?

Redis is used as both the **broker** (where tasks are queued) and the **result backend** (where task return values are stored).

| Alternative | Why Not |
|---|---|
| RabbitMQ | More complex to operate; Redis is already in the stack for caching |
| Amazon SQS | Does not support result backend; extra cost; Celery SQS support is less mature |
| In-memory (no broker) | No persistence — tasks lost on worker crash |

Redis is fast (in-memory), simple to operate via ElastiCache, and handles our expected task volume (thousands/day) with ease.

### Celery Configuration

```
Broker:  redis://elasticache-endpoint:6379/0
Backend: redis://elasticache-endpoint:6379/0
Queues:  video_queue, tamper_queue, audio_queue, pdf_queue
```

Tasks are serialised as JSON (not pickle) for security — no arbitrary Python execution from the broker.

---

## 6. ML Model Queue Strategy

Each ML model runs in its own Celery worker group, consuming its own dedicated queue.

### Queue Map

| Queue | Models Running | Instance Type | GPU | Spot? |
|---|---|---|---|---|
| `video_queue` | Deepfake detection + lipsync analysis | g4dn.xlarge | NVIDIA T4 (16 GB) | Yes + 1 on-demand fallback |
| `tamper_queue` | Scene cut detection, re-encode analysis | g4dn.xlarge | NVIDIA T4 (16 GB) | Yes + 1 on-demand fallback |
| `audio_queue` | AI audio / voice clone detection | g4dn.xlarge | NVIDIA T4 (16 GB) | Yes + 1 on-demand fallback |
| `pdf_queue` | Forensic PDF generation (CPU only) | Fargate t3.small | None | N/A |

### Why Separate Queues?

**Problem with a single queue:** If 50 video scans are queued and one audio scan comes in, that audio scan waits behind all 50 video scans even though audio and video workers are completely different processes with no resource conflict.

**With separate queues:** Each queue has its own workers. A backlog of video scans has zero effect on audio scan latency. Each queue scales independently based on its own depth.

### Task Routing

```python
# API enqueues to the correct queue based on scan type
process_scan.apply_async(
    args=[str(scan.id)],
    queue=f"{scan.scan_type}_queue"   # video_queue | tamper_queue | audio_queue
)
```

### Concurrency Per Worker

Each GPU worker runs one task at a time (`--concurrency=1`). Running multiple GPU-heavy inference tasks in parallel on the same GPU causes OOM errors and slower throughput than sequential processing. The auto-scaling group adds **more instances** rather than more concurrent tasks per instance.

---

## 7. AWS Infrastructure

### Service Selection Rationale

#### ECS Fargate (API Layer)

**What:** Serverless container platform — you provide a Docker image; AWS manages the underlying servers.

**Why Fargate over EC2 for the API:**
- No server patching, capacity planning, or OS management
- Scales from 2 to 20 replicas in under 60 seconds based on CPU/memory
- Pay per task-second — no idle EC2 cost at night
- Perfect for stateless HTTP servers

**Why not Lambda:** Cold starts add latency to auth and profile requests. Lambda also doesn't support long-lived WebSocket connections (needed for future real-time progress).

#### EC2 Auto Scaling Groups (ML Workers)

**What:** Groups of GPU EC2 instances that automatically add or remove instances based on queue depth.

**Why EC2 (not Fargate) for workers:**
- Fargate does not support GPU instances
- EC2 gives direct control over CUDA drivers and GPU memory
- Spot Instances are available for EC2, saving 60–70% on GPU costs

**Instance: g4dn.xlarge**
- 4 vCPU, 16 GB RAM, 1× NVIDIA T4 GPU (16 GB VRAM)
- $0.526/hr on-demand → ~$0.16/hr Spot (70% saving)
- Sufficient for all four model types

#### RDS PostgreSQL (Multi-AZ)

**What:** Managed PostgreSQL with automatic failover to a standby replica in a different availability zone.

**Why Multi-AZ:** If the primary database instance fails (hardware fault, AZ outage), RDS automatically promotes the standby within 60–120 seconds. For a paid mobile app, database downtime is unacceptable.

**Why not DynamoDB:** The data model is relational (users → scans → feedback → subscriptions with foreign keys). PostgreSQL is the right tool. DynamoDB would require denormalising the schema and complicates Stripe webhook handling.

#### ElastiCache Redis

**What:** Managed Redis cluster.

**Why managed:** ElastiCache handles patching, backups, failover, and Multi-AZ replication. A self-hosted Redis on EC2 requires manual management and is a single point of failure.

**Cluster mode:** Single node with a read replica is sufficient. Redis is not the bottleneck — task volume is in the thousands per day, not millions.

#### S3

**What:** Object storage for all uploaded files, thumbnails, and generated PDFs.

**Why S3:**
- Infinitely scalable — no capacity planning
- 99.999999999% durability
- Pre-signed URLs allow the mobile app to download files directly without routing through the API server
- Lifecycle policies auto-delete uploaded scan files after 24 hours (privacy)
- Costs fractions of a cent per GB per month

#### CloudFront CDN

**What:** AWS's global content delivery network, placed in front of S3.

**Why:** Pre-signed S3 URLs serve files from a single region. CloudFront caches thumbnails and PDF reports at 400+ edge locations globally. A user in Pakistan downloading a thumbnail gets it from a nearby edge node, not from us-east-1.

**Also used for:** SSL termination, DDoS protection (Shield Standard, free), and WAF integration (optional, for rate limiting).

#### AWS Secrets Manager

**What:** Encrypted secret storage for all environment variables (database URL, Stripe keys, JWT secrets, etc.)

**Why not `.env` files:** Secrets in `.env` files in Docker images or on EC2 instances are a security risk. Secrets Manager rotates credentials automatically and logs every access in CloudTrail.

---

## 8. Auto-Scaling Strategy

### API Layer (ECS Fargate)

**Trigger:** CPU utilisation > 60% for 2 consecutive minutes  
**Scale out:** Add 2 tasks  
**Scale in:** CPU < 30% for 10 minutes → remove 1 task  
**Minimum:** 2 tasks (for availability)  
**Maximum:** 20 tasks  
**Cooldown:** 60 seconds between scale events

API tasks are stateless and start in ~15 seconds (FastAPI startup). This is fast enough to handle traffic spikes from marketing campaigns.

### ML Worker Groups (EC2 Auto Scaling)

**Trigger:** Redis queue depth (custom CloudWatch metric published by a small Lambda every 30 seconds)

| Metric | Action |
|---|---|
| Queue depth > 0 for 1 minute | Add 1 GPU instance |
| Queue depth > 5 for 2 minutes | Add 2 GPU instances |
| Queue depth = 0 for 5 minutes | Terminate 1 instance |
| Queue depth = 0 for 10 minutes | Scale to 0 (all instances terminated) |

**Minimum: 0 instances** — GPU instances cost money even when idle. Scale to zero when no scans are queued.

**Warm-up time:** ~2–3 minutes for a new GPU instance to boot, pull the Docker image, and load ML models into VRAM. This is acceptable because scans are async — the user is not staring at a spinner waiting for the instance to start. Their scan is in the queue and will be processed as soon as the worker is ready.

**Spot Instance strategy:**
- Primary: Spot Instances (60–70% cheaper)
- Fallback: 1 On-Demand instance per ASG always running during business hours
- Interruption handling: Celery's `acks_late=True` ensures a task interrupted by a Spot reclaim is re-queued automatically

### Database (RDS)

RDS scales vertically (instance size upgrade). Horizontal read scaling via Read Replicas can be added later if analytics queries start competing with write traffic.

**Current:** db.t3.medium (2 vCPU, 4 GB RAM) — sufficient for ~10,000 users  
**Next:** db.r6g.large (2 vCPU, 16 GB RAM) when query latency exceeds 50 ms consistently

---

## 9. File Storage & CDN

### S3 Bucket Structure

```
faik-storage/
├── scans/
│   └── {scan_id}.mp4          ← uploaded files (deleted after 24 h)
├── thumbnails/
│   └── {scan_id}.jpg          ← video frame at 2 s (kept 7 days)
└── reports/
    └── {scan_id}_forensic.pdf ← forensic reports (kept 30 days)
```

### Lifecycle Policies

| Prefix | Action | After |
|---|---|---|
| `scans/` | Delete object | 24 hours |
| `thumbnails/` | Delete object | 7 days |
| `reports/` | Move to Glacier | 30 days, delete after 1 year |

These are set as S3 Lifecycle Rules — no application code needed.

### Access Pattern

Files are never served directly through the API server. The API generates **pre-signed URLs** (time-limited S3 URLs) and returns them in API responses. The mobile app downloads directly from S3/CloudFront. This keeps the API server lean and eliminates bandwidth costs on ECS.

---

## 10. Database

### Schema Summary

| Table | Purpose |
|---|---|
| `users` | Account, plan, push token, accuracy stats |
| `scans` | Scan lifecycle, ML results (JSONB), S3 keys |
| `feedback` | User corrections for model retraining |
| `subscriptions` | Stripe subscription lifecycle |
| `refresh_tokens` | JWT refresh token store (hashed, revocable) |
| `payments` | Stripe payment records |

### Backup Strategy

- **Automated daily snapshots** retained for 7 days (RDS default)
- **Point-in-time recovery** enabled — restore to any second within the retention window
- **Multi-AZ standby** provides instant failover (not a backup, but protects against instance failure)

### Connection Pooling

ECS Fargate tasks connect to RDS directly using SQLAlchemy's connection pool. As task count grows, the number of DB connections grows with it. At scale, add **RDS Proxy** between ECS and RDS — it pools connections at the proxy layer, preventing the "too many clients" error.

---

## 11. Security

| Layer | Measure |
|---|---|
| **Network** | VPC with private subnets for RDS, ElastiCache, workers. Only the ALB is public-facing |
| **API** | JWT access tokens (1-hour TTL) + refresh tokens (30-day TTL, stored hashed in DB) |
| **Transport** | HTTPS enforced via ACM SSL certificate on CloudFront and ALB |
| **Secrets** | All credentials in AWS Secrets Manager, injected as env vars at container start |
| **S3** | Bucket policy blocks all public access. Files accessible only via pre-signed URLs |
| **Celery** | Tasks serialised as JSON (not pickle) to prevent remote code execution via broker |
| **Database** | RDS in private subnet, security group allows inbound only from ECS and worker security groups |
| **File uploads** | 250 MB limit enforced at API. File extension allowlist. Files validated before ML processing |
| **Rate limiting** | CloudFront WAF rules limit requests per IP. API returns `429 SCAN_LIMIT_REACHED` at plan quota |
| **Stripe webhooks** | Signature verified with `STRIPE_WEBHOOK_SECRET` before processing any billing event |

---

## 12. CI/CD Pipeline

```
Developer pushes to main branch
          ↓
    GitHub Actions
          ↓
    1. Run tests (pytest)
    2. Build Docker image
    3. Push to ECR
          ↓
    4a. Deploy API → ECS (rolling update, zero downtime)
    4b. Deploy workers → update Launch Template AMI / trigger instance refresh
          ↓
    5. Run database migrations (alembic upgrade head)
          ↓
    Deployment complete (~5 min total)
```

**Zero-downtime deployments:** ECS rolling updates replace tasks one at a time, with health checks ensuring new tasks are serving traffic before old ones are terminated.

**Rollback:** ECS stores previous task definition revisions. A rollback is a single command: `aws ecs update-service --task-definition previous-revision`.

---

## 13. Monitoring & Alerting

### CloudWatch Dashboards

| Metric | Source | Alert Threshold |
|---|---|---|
| API response time (p95) | ALB | > 500 ms for 5 min |
| API 5xx error rate | ALB | > 1% for 2 min |
| Celery queue depth | Custom Lambda metric | > 20 tasks for 10 min |
| Worker task failure rate | CloudWatch Logs | > 5% failures/hr |
| RDS CPU | RDS | > 80% for 5 min |
| RDS storage | RDS | < 20% free |
| Redis memory | ElastiCache | > 80% used |
| ECS task count | ECS | < 2 tasks running |

### Logging

All services write structured JSON logs to **CloudWatch Logs**. Log groups:
- `/faik/api` — FastAPI access and error logs
- `/faik/workers/video` — video worker logs
- `/faik/workers/tamper` — tamper worker logs
- `/faik/workers/audio` — audio worker logs
- `/faik/workers/pdf` — PDF worker logs

Log retention: 30 days (cost-controlled).

### Error Tracking

Integrate **Sentry** (`sentry-sdk` already in `requirements.txt`) for exception tracking with stack traces across both the API and Celery workers.

---

## 14. Cost Estimate

### Monthly Baseline (low traffic, ~500 scans/day)

| Service | Config | Monthly Cost |
|---|---|---|
| ECS Fargate (API) | 2 tasks × t3.medium equivalent | ~$40 |
| RDS PostgreSQL | db.t3.medium, Multi-AZ | ~$70 |
| ElastiCache Redis | cache.t3.medium | ~$30 |
| Application Load Balancer | — | ~$20 |
| CloudFront + S3 | ~50 GB transfer, ~10 GB storage | ~$15 |
| GPU Workers (Spot) | g4dn.xlarge, ~4 hrs/day avg | ~$20 |
| Route 53 | 1 hosted zone | ~$1 |
| Secrets Manager | ~10 secrets | ~$5 |
| ECR | ~5 GB storage | ~$1 |
| CloudWatch | Logs + metrics | ~$10 |
| **Total** | | **~$212/mo** |

### At Scale (~10,000 scans/day)

| Service | Change | Estimated Cost |
|---|---|---|
| ECS Fargate | 6–10 tasks | ~$120 |
| RDS | Upgrade to db.r6g.large | ~$175 |
| ElastiCache | Upgrade to cache.r6g.large | ~$120 |
| GPU Workers | ~16 hrs/day across queues | ~$80 |
| CloudFront + S3 | ~500 GB transfer | ~$55 |
| Other | — | ~$30 |
| **Total** | | **~$580/mo** |

At the Pro plan price of $9/month, **65 Pro subscribers cover the baseline cost.** At 10,000 scans/day the app is well into profitability.

---

## 15. Deployment Order

Follow this order to avoid dependency issues:

| Step | Action | Notes |
|---|---|---|
| 1 | Create VPC with public/private subnets | Foundation for all other services |
| 2 | Create RDS PostgreSQL | Database must exist before running migrations |
| 3 | Create ElastiCache Redis | Must exist before running workers |
| 4 | Create S3 bucket + lifecycle policies | Must exist before API can accept uploads |
| 5 | Set up ECR repositories | API and worker images go here |
| 6 | Configure Secrets Manager | Store all .env values |
| 7 | Build and push Docker images | Via GitHub Actions or manually |
| 8 | Run `alembic upgrade head` | Apply database schema |
| 9 | Deploy ECS Fargate (API) | Verify health endpoint responds |
| 10 | Set up CloudFront distribution | Point to ALB |
| 11 | Configure Route 53 | Point `api.fivedot.app` to CloudFront |
| 12 | Deploy GPU worker Auto Scaling Groups | One per ML model queue |
| 13 | Configure CloudWatch alarms | Auto-scaling triggers + alerting |
| 14 | Configure Stripe webhooks | Point to `https://api.fivedot.app/v1/webhooks/stripe` |
| 15 | End-to-end smoke test | Upload a scan, verify notification received |

---

*Document version: 1.0 — June 2026*  
*Author: Backend team*
