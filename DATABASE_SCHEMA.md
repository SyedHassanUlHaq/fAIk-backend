# Database Schema

PostgreSQL database. All timestamps are timezone-aware (`TIMESTAMPTZ`). UUIDs use the `uuid` extension (`gen_random_uuid()` / Python `uuid.uuid4()`).

---

## Tables

- [users](#users)
- [refresh_tokens](#refresh_tokens)
- [scans](#scans)
- [feedback](#feedback)
- [subscriptions](#subscriptions)

---

## `users`

Primary identity table. Supports both email/password and OAuth (Google, Apple) accounts — `hashed_password` is nullable for OAuth-only users.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | `SERIAL` | NO | auto-increment | Primary key |
| `email` | `VARCHAR` | NO | — | Unique. Used for login and identity |
| `first_name` | `VARCHAR` | YES | — | From signup or OAuth profile |
| `last_name` | `VARCHAR` | YES | — | From signup or OAuth profile |
| `name` | `VARCHAR` | YES | — | Display name (overrides first+last if set) |
| `hashed_password` | `VARCHAR` | YES | — | bcrypt hash. NULL for OAuth-only accounts |
| `avatar_color` | `VARCHAR` | YES | `#E91E8C` | Hex colour for avatar background |
| `plan` | `VARCHAR` | NO | `free` | Active plan: `free` \| `pro` \| `team` |
| `scans_used_this_month` | `INTEGER` | NO | `0` | Incremented on each completed scan |
| `plan_reset_date` | `TIMESTAMPTZ` | YES | — | When the monthly scan counter resets |
| `accuracy_rate` | `FLOAT` | NO | `94.2` | Displayed accuracy stat |
| `push_token` | `VARCHAR` | YES | — | Expo push token (`ExponentPushToken[...]`) |
| `created_at` | `TIMESTAMPTZ` | NO | `now()` | Account creation timestamp |

**Indexes:** `email` (unique)

**Relationships:**
- One-to-many → `refresh_tokens`, `scans`, `feedback`
- One-to-one → `subscription`

---

## `refresh_tokens`

Stores hashed JWT refresh tokens. Tokens are rotated on every use — the old row is marked `revoked = true` and a new row is inserted.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | `UUID` | NO | `uuid4()` | Primary key |
| `user_id` | `INTEGER` | NO | — | FK → `users.id` (CASCADE DELETE) |
| `token_hash` | `VARCHAR` | NO | — | SHA-256 hash of the raw token. Unique |
| `expires_at` | `TIMESTAMPTZ` | NO | — | Token expiry (30 days from issue) |
| `revoked` | `BOOLEAN` | NO | `false` | Set to `true` on use or signout |
| `created_at` | `TIMESTAMPTZ` | NO | `now()` | Issue timestamp |

**Indexes:** `token_hash` (unique), `user_id`

---

## `scans`

Central table for all media analysis jobs. A scan moves through statuses: `queued → processing → complete | failed`. Raw ML results are stored in the `result_data` JSONB column.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | `UUID` | NO | `uuid4()` | Primary key |
| `user_id` | `INTEGER` | NO | — | FK → `users.id` (CASCADE DELETE) |
| `filename` | `VARCHAR` | NO | — | Original filename or URL title |
| `file_key` | `VARCHAR` | YES | — | S3 object key for the uploaded file |
| `url_source` | `VARCHAR` | YES | — | Source URL for URL-based scans (yt-dlp) |
| `file_size` | `INTEGER` | YES | — | File size in bytes |
| `duration` | `FLOAT` | YES | — | Media duration in seconds |
| `bitrate` | `VARCHAR` | YES | — | e.g. `"3200 kbps"` (audio stream) |
| `scan_type` | `VARCHAR` | NO | — | `video` \| `tamper` \| `audio` |
| `result_type` | `VARCHAR` | YES | — | `deepfakeVideo` \| `editTamper` \| `aiVoice` \| `authentic` \| `tampered` |
| `verdict` | `VARCHAR` | YES | — | `ai` \| `tampered` \| `authentic` |
| `score` | `INTEGER` | YES | — | Confidence score 0–100 |
| `status` | `VARCHAR` | NO | `queued` | `queued` \| `processing` \| `complete` \| `failed` |
| `progress` | `INTEGER` | NO | `0` | Progress 0–100, updated by Celery worker |
| `current_stage` | `VARCHAR` | YES | — | Current pipeline stage label |
| `thumbnail_key` | `VARCHAR` | YES | — | S3 key for extracted video thumbnail |
| `result_data` | `JSONB` | YES | — | Model-specific results (see below) |
| `error_message` | `VARCHAR` | YES | — | Set on `failed` status |
| `created_at` | `TIMESTAMPTZ` | NO | `now()` | Scan submission time |
| `completed_at` | `TIMESTAMPTZ` | YES | — | Inference completion time |

**Indexes:** `user_id`, `status`, `created_at`

### `result_data` structure

#### `scan_type = "video"` (AI deepfake detection)

```json
{
  "plainEnglishExplanation": "string",
  "thumbnailUrl": "pre-signed S3 URL",
  "segments": [
    {
      "startSeconds": 0,
      "endSeconds": 5,
      "label": "Face GAN signature",
      "type": "gan | lip_sync",
      "severity": "high | medium | low",
      "timelinePositionPercent": 0,
      "timelineWidthPercent": 25
    }
  ],
  "forensicPdfUrl": "pre-signed S3 URL (added after PDF generation)"
}
```

#### `scan_type = "tamper"` (scene/tampering detection)

```json
{
  "editCount": 3,
  "editSummary": "string",
  "tamperLevel": "high | medium | low",
  "edits": [
    {
      "number": 1,
      "timeSeconds": 3.2,
      "label": "Hard cut | Splice from another clip | Re-encode detected",
      "tag": "cut | splice | recode",
      "severity": "high | medium | low"
    }
  ],
  "forensicPdfUrl": "pre-signed S3 URL (added after PDF generation)"
}
```

#### `scan_type = "audio"` (AI voice detection — stub)

```json
{
  "tagline": "string",
  "waveformBars": [],
  "evidence": []
}
```

---

## `feedback`

User-submitted corrections to a scan verdict. Used to collect training signal.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | `UUID` | NO | `uuid4()` | Primary key |
| `scan_id` | `UUID` | NO | — | FK → `scans.id` (CASCADE DELETE) |
| `user_id` | `INTEGER` | NO | — | FK → `users.id` (CASCADE DELETE) |
| `correct_verdict` | `VARCHAR` | NO | — | User's belief: `authentic` \| `ai` \| `unsure` |
| `reasons` | `JSONB` | YES | — | Array of reason strings e.g. `["voice_sounds_real"]` |
| `detail` | `TEXT` | YES | — | Free-text explanation from user |
| `allow_anonymized_copy` | `BOOLEAN` | NO | `false` | Consent to use for model retraining |
| `created_at` | `TIMESTAMPTZ` | NO | `now()` | Submission timestamp |

---

## `subscriptions`

One row per user. Tracks Stripe billing state. The `users.plan` column is the source of truth for access control — this table holds the raw Stripe data that drives it.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | `UUID` | NO | `uuid4()` | Primary key |
| `user_id` | `INTEGER` | NO | — | FK → `users.id` (CASCADE DELETE). Unique |
| `stripe_subscription_id` | `VARCHAR` | YES | — | Stripe `sub_xxx` ID. Unique |
| `stripe_customer_id` | `VARCHAR` | YES | — | Stripe `cus_xxx` ID |
| `plan_id` | `VARCHAR` | NO | — | `free` \| `pro` \| `team` |
| `status` | `VARCHAR` | NO | — | `active` \| `cancelled` \| `trialing` \| `past_due` |
| `trial_ends_at` | `TIMESTAMPTZ` | YES | — | Trial period end |
| `current_period_end` | `TIMESTAMPTZ` | YES | — | Current billing period end |
| `created_at` | `TIMESTAMPTZ` | NO | `now()` | Row creation timestamp |

---

## Entity Relationship Diagram

```
users
 ├── refresh_tokens   (1 : many)   — auth sessions
 ├── scans            (1 : many)   — analysis jobs
 │    └── feedback    (1 : many)   — verdict corrections
 └── subscriptions    (1 : 1)      — Stripe billing
```

---

## Cascade Behaviour

All foreign keys use `ON DELETE CASCADE`. Deleting a user removes all their tokens, scans, feedback, and subscription automatically.
