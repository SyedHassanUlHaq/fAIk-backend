# fivedot — Backend API Specification

> For the backend developer. Covers every API endpoint, request/response shape, data model, and behavioural contract the mobile app depends on.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Base URL & Auth Headers](#2-base-url--auth-headers)
3. [Authentication](#3-authentication)
4. [User Profile](#4-user-profile)
5. [Scans — Upload & Submit](#5-scans--upload--submit)
6. [Scans — Real-time Analysis Progress](#6-scans--real-time-analysis-progress)
7. [Scans — Result](#7-scans--result)
8. [Scan History](#8-scan-history)
9. [Feedback / Corrections](#9-feedback--corrections)
10. [Forensic PDF Report](#10-forensic-pdf-report)
11. [Plans & Subscriptions](#11-plans--subscriptions)
12. [Weekly Stats (Dashboard)](#12-weekly-stats-dashboard)
13. [Full Data Models](#13-full-data-models)
14. [Enums & Constants](#14-enums--constants)
15. [Error Format](#15-error-format)
16. [Pipeline Stages](#16-pipeline-stages)

---

## 1. Overview

fivedot is a mobile app (Expo / React Native) that lets users upload or record audio and video files and receive an AI-authenticity verdict. There are three detection modes:

| Mode | What it detects | `scanType` value |
|---|---|---|
| **Audio** | AI-synthesised / cloned voice (TTS, voice cloning) | `"audio"` |
| **Video** | Deepfake face-swap, lip-sync manipulation, GAN artefacts | `"video"` |
| **Tamper** | Cut-and-splice edits, re-encoding jumps, timeline discontinuities | `"tamper"` |

Every scan returns a **0–100 confidence score** and one of five `resultType` values the app uses to decide which result screen to render. Evidence items, timeline segments, and plain-English explanations are attached to the result.

The app also supports:
- Feedback/correction submissions (used to retrain the model)
- Forensic PDF report generation (Pro plan)
- Usage-based subscription plans (Free / Pro / Team)

---

## 2. Base URL & Auth Headers

```
Base URL: https://api.fivedot.app/v1
```

All authenticated endpoints require a Bearer token:

```
Authorization: Bearer <jwt_access_token>
Content-Type: application/json          # for JSON bodies
Content-Type: multipart/form-data       # for file uploads
```

Tokens are JWTs. Access token TTL: **1 hour**. Refresh token TTL: **30 days**.

---

## 3. Authentication

### 3.1 Sign Up

```
POST /auth/signup
```

**Request:**
```json
{
  "email": "user@example.com",
  "password": "min8chars"
}
```

**Response `201`:**
```json
{
  "accessToken": "eyJ...",
  "refreshToken": "eyJ...",
  "user": { /* User object — see §13.1 */ }
}
```

---

### 3.2 Sign In

```
POST /auth/signin
```

**Request:**
```json
{
  "email": "user@example.com",
  "password": "min8chars"
}
```

**Response `200`:** same shape as Sign Up.

---

### 3.3 OAuth — Google

```
POST /auth/google
```

**Request:**
```json
{
  "idToken": "<Google ID token from expo-auth-session>"
}
```

**Response `200`:** same shape as Sign Up.

---

### 3.4 OAuth — Apple

```
POST /auth/apple
```

**Request:**
```json
{
  "identityToken": "<Apple identity token>",
  "fullName": { "givenName": "Jamie", "familyName": "Marks" }
}
```

**Response `200`:** same shape as Sign Up.

---

### 3.5 Refresh Token

```
POST /auth/refresh
```

**Request:**
```json
{
  "refreshToken": "eyJ..."
}
```

**Response `200`:**
```json
{
  "accessToken": "eyJ...",
  "refreshToken": "eyJ..."
}
```

---

### 3.6 Sign Out

```
POST /auth/signout
```

Invalidates the refresh token server-side. No request body required (token is in the `Authorization` header).

**Response `204`:** no body.

---

### 3.7 Forgot Password

```
POST /auth/forgot-password
```

**Request:**
```json
{
  "email": "user@example.com"
}
```

**Response `200`:**
```json
{
  "message": "Reset email sent if account exists."
}
```

---

## 4. User Profile

### 4.1 Get Current User

```
GET /users/me
```

**Response `200`:**
```json
{
  "id": "usr_01HX...",
  "email": "jamie@example.com",
  "name": "Jamie Marks",
  "avatarInitials": "JM",
  "avatarColor": "#E91E8C",
  "plan": "pro",
  "modelVersion": "v3.2",
  "scansUsedThisMonth": 347,
  "scansLimitThisMonth": 500,
  "planResetDate": "2026-06-30T00:00:00Z",
  "accuracyRate": 94.2,
  "createdAt": "2025-11-01T09:15:00Z"
}
```

> `plan` is one of `"free"` | `"pro"` | `"team"`.  
> `scansLimitThisMonth` is `null` for the Team plan (unlimited).

---

### 4.2 Update Profile

```
PATCH /users/me
```

**Request (any subset):**
```json
{
  "name": "Jamie M.",
  "avatarColor": "#7C3AED"
}
```

**Response `200`:** updated `User` object.

---

## 5. Scans — Upload & Submit

### 5.1 Upload a File

```
POST /scans
Content-Type: multipart/form-data
```

**Form fields:**

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | binary | yes | mp4, mov, wav, m4a, mp3, opus. Max 250 MB |
| `scanType` | string | yes | `"audio"` \| `"video"` \| `"tamper"` |
| `filename` | string | no | Display name if different from the file's name |

**Response `202`:**
```json
{
  "scanId": "scan_01HX...",
  "status": "queued",
  "estimatedSeconds": 4,
  "uploadedAt": "2026-06-02T10:00:00Z"
}
```

The app immediately navigates to the Analyzing screen using `scanId`. Analysis happens asynchronously — poll or subscribe via WebSocket (§6).

---

### 5.2 Submit a URL (YouTube / X)

```
POST /scans/url
```

**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=...",
  "scanType": "video"
}
```

**Response `202`:** same as §5.1.

---

## 6. Scans — Real-time Analysis Progress

The Analyzing screen shows a live progress percentage and a pipeline checklist. Two approaches are supported; use whichever suits your infrastructure:

### Option A — WebSocket (preferred)

```
WS wss://api.fivedot.app/v1/scans/<scanId>/progress
```

The server emits JSON messages:

```json
{
  "scanId": "scan_01HX...",
  "progress": 78,
  "currentStage": "frame_coherence",
  "stages": {
    "decoding_stream":    { "status": "complete" },
    "spectral_analysis":  { "status": "complete" },
    "frame_coherence":    { "status": "running" },
    "cross_check_model":  { "status": "pending" }
  },
  "estimatedSecondsRemaining": 2
}
```

When complete, one final message is sent with `"progress": 100` and `"status": "complete"`, then the socket closes. On failure, `"status": "failed"` is sent.

### Option B — Polling

```
GET /scans/<scanId>/status
```

Returns the same shape as the WebSocket message above. The app polls every ~500 ms while on the Analyzing screen.

### Stage names (in order)

| `currentStage` value | Display label |
|---|---|
| `decoding_stream` | Decoding stream |
| `spectral_analysis` | Spectral analysis |
| `frame_coherence` | Frame coherence |
| `cross_check_model` | Cross-checking model v3.2 |

---

## 7. Scans — Result

```
GET /scans/<scanId>
```

**Response `200`:** a `ScanResult` object. The shape varies by `resultType` — see §13.2 for the full model.

### 7.1 Audio result (`resultType: "aiVoice" | "authentic" | "tampered"`)

```json
{
  "scanId": "scan_01HX...",
  "userId": "usr_01HX...",
  "filename": "voicemail-clip.m4a",
  "fileSize": 1240000,
  "duration": 18,
  "bitrate": "320 kbps",
  "scanType": "audio",
  "resultType": "aiVoice",
  "verdict": "ai",
  "score": 92,
  "tagline": "This voice was likely synthesized by an AI model.",
  "status": "complete",
  "createdAt": "2026-06-02T10:00:00Z",
  "completedAt": "2026-06-02T10:00:04Z",
  "waveformBars": [0.4, 0.7, 0.9, 0.6, 1.0, 0.8, 0.5],
  "evidence": [
    {
      "label": "Spectral artifacts",
      "description": "Harmonic banding consistent with neural vocoders.",
      "scoreContribution": 34,
      "icon": "activity"
    },
    {
      "label": "Breath & pause cues",
      "description": "Atypical micro-pauses; breaths missing before plosives.",
      "scoreContribution": 22,
      "icon": "wind"
    },
    {
      "label": "Model fingerprint",
      "description": "Patterns match a known synthesis family (XTTS-class).",
      "scoreContribution": 36,
      "icon": "cpu"
    }
  ]
}
```

> `waveformBars`: array of 30–50 floats in `[0, 1]`. The app renders these as vertical bars in the waveform player. Pre-compute on the backend so the client doesn't have to decode the audio.

> `scoreContribution`: how many points this evidence item contributes to the overall `score`. The app displays it as `+34`, `+22`, `+36`.

> `tagline`: the plain-English one-liner displayed below the confidence gauge. Backend should generate this based on `resultType` and `score`.

---

### 7.2 Deepfake video result (`resultType: "deepfakeVideo"`)

```json
{
  "scanId": "scan_01HX...",
  "filename": "interview-clip-04.mp4",
  "fileSize": 14900000,
  "duration": 131,
  "scanType": "video",
  "resultType": "deepfakeVideo",
  "verdict": "ai",
  "score": 92,
  "status": "complete",
  "createdAt": "2026-06-02T10:00:00Z",
  "completedAt": "2026-06-02T10:00:08Z",
  "thumbnailUrl": "https://cdn.fivedot.app/thumbnails/scan_01HX.jpg",
  "plainEnglishExplanation": "The face shows GAN-typical micro-flicker around the eyes, and the lip movement runs ~80 ms behind the audio — both strong cues for a face-swap or lip-sync model. One cut between 1:29 and 1:37 also shows a re-encode signature.",
  "segments": [
    {
      "startSeconds": 7,
      "endSeconds": 15,
      "label": "Lip-sync drift",
      "type": "lip_sync",
      "severity": "high",
      "timelinePositionPercent": 5,
      "timelineWidthPercent": 6
    },
    {
      "startSeconds": 44,
      "endSeconds": 53,
      "label": "Face GAN signature",
      "type": "gan",
      "severity": "high",
      "timelinePositionPercent": 33,
      "timelineWidthPercent": 7
    },
    {
      "startSeconds": 89,
      "endSeconds": 96,
      "label": "Cut + re-encode",
      "type": "cut",
      "severity": "medium",
      "timelinePositionPercent": 67,
      "timelineWidthPercent": 5
    }
  ]
}
```

> `timelinePositionPercent` and `timelineWidthPercent`: used by the app to position coloured blocks on the timeline bar absolutely. Range `0–100` as percentages of total video duration.

> `severity`: `"high"` renders red, `"medium"` renders amber.

> `type`: one of `"lip_sync"` | `"gan"` | `"cut"` | `"splice"` | `"recode"`.

---

### 7.3 Edit/tamper result (`resultType: "editTamper"`)

```json
{
  "scanId": "scan_01HX...",
  "filename": "press-statement.mov",
  "duration": 131,
  "scanType": "tamper",
  "resultType": "editTamper",
  "verdict": "tampered",
  "score": 64,
  "status": "complete",
  "editCount": 4,
  "editSummary": "Audio likely authentic. Video has been cut and re-encoded.",
  "tamperLevel": "medium",
  "edits": [
    {
      "number": 1,
      "timeSeconds": 23,
      "label": "Hard cut · 1f gap",
      "tag": "cut",
      "severity": "high"
    },
    {
      "number": 2,
      "timeSeconds": 44,
      "label": "Splice from clip B",
      "tag": "splice",
      "severity": "high"
    },
    {
      "number": 3,
      "timeSeconds": 68,
      "label": "Re-encode · CRF jump",
      "tag": "recode",
      "severity": "medium"
    },
    {
      "number": 4,
      "timeSeconds": 93,
      "label": "Hard cut",
      "tag": "cut",
      "severity": "high"
    }
  ]
}
```

> `tag`: one of `"cut"` | `"splice"` | `"recode"`. The app renders these as small chips next to each edit row.  
> `tamperLevel`: `"low"` | `"medium"` | `"high"` — shown in the badge next to the score.  
> `severity: "high"` → red marker on timeline; `"medium"` → amber marker.

---

## 8. Scan History

### 8.1 List Scans

```
GET /scans?page=1&limit=20&verdict=ai&scanType=video
```

**Query params:**

| Param | Type | Default | Notes |
|---|---|---|---|
| `page` | integer | `1` | |
| `limit` | integer | `20` | Max 50 |
| `verdict` | string | — | Filter: `"ai"` \| `"authentic"` \| `"tampered"` |
| `scanType` | string | — | Filter: `"audio"` \| `"video"` \| `"tamper"` |

**Response `200`:**
```json
{
  "items": [
    {
      "scanId": "scan_01HX...",
      "filename": "interview-clip-04.mp4",
      "duration": 131,
      "scanType": "video",
      "resultType": "deepfakeVideo",
      "verdict": "ai",
      "score": 92,
      "completedAt": "2026-06-02T10:00:04Z"
    }
  ],
  "total": 47,
  "page": 1,
  "limit": 20,
  "hasMore": true
}
```

> The app currently renders `timeAgo` labels (e.g. "2 min ago"). Compute these on the client from `completedAt`. Return raw ISO timestamps from the API.

---

### 8.2 Delete a Scan

```
DELETE /scans/<scanId>
```

**Response `204`:** no body.

---

## 9. Feedback / Corrections

Submitted when a user taps **"Wrong?"** on a result screen and fills in the correction form.

```
POST /feedback
```

**Request:**
```json
{
  "scanId": "scan_01HX...",
  "correctVerdict": "authentic",
  "reasons": ["I made the recording", "Familiar voice"],
  "detail": "I recorded this on my phone last Tuesday during the press conference.",
  "allowAnonymizedCopy": true
}
```

| Field | Type | Values |
|---|---|---|
| `correctVerdict` | string | `"authentic"` \| `"ai"` \| `"unsure"` |
| `reasons` | string[] | Free list — pass through exactly as sent |
| `allowAnonymizedCopy` | boolean | User consent for 30-day model retraining use |

**Response `201`:**
```json
{
  "feedbackId": "fb_01HX...",
  "message": "Thank you. We'll review within 48 hours."
}
```

---

## 10. Forensic PDF Report

Available on **Pro and Team plans only**.

```
POST /scans/<scanId>/forensic-pdf
```

No request body.

**Response `202`:**
```json
{
  "jobId": "pdf_01HX...",
  "status": "generating",
  "estimatedSeconds": 8
}
```

Poll for completion:

```
GET /scans/<scanId>/forensic-pdf/status
```

**Response `200`:**
```json
{
  "status": "complete",
  "downloadUrl": "https://cdn.fivedot.app/reports/scan_01HX_forensic.pdf",
  "expiresAt": "2026-06-03T10:00:00Z"
}
```

> `downloadUrl` is a pre-signed URL valid for 24 hours. The app opens it in the system browser / share sheet.

---

## 11. Plans & Subscriptions

### 11.1 List Plans

```
GET /plans
```

**Response `200`:**
```json
{
  "plans": [
    {
      "id": "free",
      "name": "Free",
      "price": 0,
      "currency": "USD",
      "billingPeriod": "forever",
      "scansPerMonth": 50,
      "features": [
        "Audio + video AI detection",
        "720p video, 5 min max",
        "Basic plain-English explanations",
        "On-device processing"
      ],
      "maxVideoResolution": "720p",
      "maxVideoDurationSeconds": 300,
      "forensicPdf": false,
      "priorityQueue": false,
      "apiAccess": false
    },
    {
      "id": "pro",
      "name": "Pro",
      "price": 9,
      "currency": "USD",
      "billingPeriod": "month",
      "scansPerMonth": 500,
      "features": [
        "Tamper / cut detection",
        "4K video, 30 min max",
        "Forensic PDF reports",
        "Priority queue · 2× speed",
        "Advanced evidence details"
      ],
      "maxVideoResolution": "4K",
      "maxVideoDurationSeconds": 1800,
      "forensicPdf": true,
      "priorityQueue": true,
      "apiAccess": false,
      "trialDays": 7
    },
    {
      "id": "team",
      "name": "Team",
      "price": 24,
      "currency": "USD",
      "billingPeriod": "month",
      "scansPerMonth": null,
      "seats": 5,
      "features": [
        "5 seats included",
        "API access · 10k calls/mo",
        "SSO + audit log",
        "Dedicated model tuning",
        "Priority support"
      ],
      "forensicPdf": true,
      "priorityQueue": true,
      "apiAccess": true,
      "apiCallsPerMonth": 10000
    }
  ]
}
```

---

### 11.2 Subscribe / Upgrade

```
POST /subscriptions
```

**Request:**
```json
{
  "planId": "pro",
  "paymentMethodId": "pm_stripe_..."
}
```

**Response `200`:**
```json
{
  "subscriptionId": "sub_01HX...",
  "planId": "pro",
  "status": "active",
  "trialEndsAt": "2026-06-09T00:00:00Z",
  "currentPeriodEnd": "2026-07-02T00:00:00Z"
}
```

---

### 11.3 Cancel Subscription

```
DELETE /subscriptions/<subscriptionId>
```

**Response `200`:**
```json
{
  "message": "Subscription cancelled. Access continues until 2026-07-02T00:00:00Z."
}
```

---

## 12. Weekly Stats (Dashboard)

Used on the History / Dashboard screen.

```
GET /stats/weekly
```

**Response `200`:**
```json
{
  "totalScans": 47,
  "flaggedScans": 12,
  "accuracyRate": 94.2,
  "accuracyDelta": 1.2,
  "modelVersion": "v3.2",
  "modelBenchmarkAccuracy": 88.4,
  "crossCheckedCases": 240,
  "dailyBreakdown": [
    { "day": "Mon", "scans": 4 },
    { "day": "Tue", "scans": 7 },
    { "day": "Wed", "scans": 5 },
    { "day": "Thu", "scans": 9 },
    { "day": "Fri", "scans": 6 },
    { "day": "Sat", "scans": 12 },
    { "day": "Sun", "scans": 4 }
  ]
}
```

> `dailyBreakdown` always contains exactly 7 entries, Mon–Sun of the current week.  
> `accuracyDelta`: positive = improved vs previous week. Used to render `"+1.2 vs last"`.

---

## 13. Full Data Models

### 13.1 User

```typescript
interface User {
  id: string;                      // "usr_01HX..."
  email: string;
  name: string;
  avatarInitials: string;          // e.g. "JM" — first letters of name
  avatarColor: string;             // hex, used for avatar gradient
  plan: "free" | "pro" | "team";
  modelVersion: string;            // e.g. "v3.2"
  scansUsedThisMonth: number;
  scansLimitThisMonth: number | null; // null = unlimited (Team plan)
  planResetDate: string;           // ISO 8601
  accuracyRate: number;            // 0–100
  createdAt: string;               // ISO 8601
}
```

---

### 13.2 ScanResult (union)

```typescript
// Shared base
interface ScanResultBase {
  scanId: string;
  userId: string;
  filename: string;
  fileSize: number;        // bytes
  duration: number;        // seconds
  scanType: ScanType;
  resultType: ResultType;
  verdict: VerdictType;
  score: number;           // 0–100
  status: ScanStatus;
  createdAt: string;
  completedAt: string | null;
}

// Audio result (aiVoice | authentic | tampered)
interface AudioScanResult extends ScanResultBase {
  scanType: "audio";
  resultType: "aiVoice" | "authentic" | "tampered";
  bitrate: string;           // e.g. "320 kbps"
  tagline: string;
  waveformBars: number[];    // 30–50 values, each 0–1
  evidence: EvidenceItem[];
}

// Deepfake video result
interface DeepfakeVideoResult extends ScanResultBase {
  scanType: "video";
  resultType: "deepfakeVideo";
  thumbnailUrl: string;
  plainEnglishExplanation: string;
  segments: VideoSegment[];
}

// Edit/tamper result
interface EditTamperResult extends ScanResultBase {
  scanType: "tamper";
  resultType: "editTamper";
  editCount: number;
  editSummary: string;
  tamperLevel: "low" | "medium" | "high";
  edits: EditItem[];
}
```

---

### 13.3 EvidenceItem

```typescript
interface EvidenceItem {
  label: string;            // e.g. "Spectral artifacts"
  description: string;      // one-line explanation shown under label
  scoreContribution: number; // integer, shown as "+34"
  icon: string;             // Feather icon name
}
```

---

### 13.4 VideoSegment

```typescript
interface VideoSegment {
  startSeconds: number;
  endSeconds: number;
  label: string;                          // e.g. "Lip-sync drift"
  type: "lip_sync" | "gan" | "cut" | "splice" | "recode";
  severity: "high" | "medium" | "low";
  timelinePositionPercent: number;        // 0–100
  timelineWidthPercent: number;           // 0–100
}
```

---

### 13.5 EditItem

```typescript
interface EditItem {
  number: number;                           // 1-based index
  timeSeconds: number;                      // position in the clip
  label: string;                            // e.g. "Hard cut · 1f gap"
  tag: "cut" | "splice" | "recode";
  severity: "high" | "medium" | "low";
}
```

---

### 13.6 ScanListItem (history row)

```typescript
interface ScanListItem {
  scanId: string;
  filename: string;
  duration: number;          // seconds
  scanType: ScanType;
  resultType: ResultType;
  verdict: "ai" | "authentic" | "tampered";
  score: number;
  completedAt: string;       // ISO 8601 — client computes "2 min ago"
}
```

---

## 14. Enums & Constants

```typescript
type ScanType   = "audio" | "video" | "tamper";

type ResultType = "aiVoice" | "authentic" | "tampered" | "deepfakeVideo" | "editTamper";

type VerdictType = "ai" | "authentic" | "tampered";

type ScanStatus = "queued" | "processing" | "complete" | "failed";

type PlanId = "free" | "pro" | "team";

type TamperLevel = "low" | "medium" | "high";

type SegmentType = "lip_sync" | "gan" | "cut" | "splice" | "recode";

type EditTag = "cut" | "splice" | "recode";
```

### Verdict → ResultType mapping

The app decides which result screen to show from `resultType`, not `verdict`. The mapping the backend must follow:

| `scanType` | `verdict` | `resultType` |
|---|---|---|
| `audio` | `ai` | `aiVoice` |
| `audio` | `authentic` | `authentic` |
| `audio` | `tampered` | `tampered` |
| `video` | `ai` | `deepfakeVideo` |
| `video` | `authentic` | `authentic` |
| `tamper` | `tampered` | `editTamper` |
| `tamper` | `authentic` | `authentic` |

---

## 15. Error Format

All error responses follow this envelope:

```json
{
  "error": {
    "code": "SCAN_NOT_FOUND",
    "message": "No scan with id scan_01HX was found.",
    "statusCode": 404
  }
}
```

### Common error codes

| Code | HTTP | Meaning |
|---|---|---|
| `UNAUTHORIZED` | 401 | Missing or invalid token |
| `FORBIDDEN` | 403 | Plan does not include this feature |
| `SCAN_NOT_FOUND` | 404 | Scan ID doesn't exist or belongs to another user |
| `FILE_TOO_LARGE` | 413 | Exceeds plan file size limit |
| `UNSUPPORTED_FORMAT` | 415 | File type not accepted |
| `SCAN_LIMIT_REACHED` | 429 | Monthly scan quota exhausted |
| `PROCESSING_FAILED` | 500 | ML pipeline error |

---

## 16. Pipeline Stages

The Analyzing screen shows a 4-step checklist that checks off as each stage completes. The `currentStage` field in the progress payload maps to these display labels:

| Backend `currentStage` | UI label | Visible after progress % |
|---|---|---|
| `decoding_stream` | Decoding stream | 15% |
| `spectral_analysis` | Spectral analysis | 40% |
| `frame_coherence` | Frame coherence | 65% |
| `cross_check_model` | Cross-checking model v3.2 | 90% |

The client marks each step **done** when `progress` passes its threshold, not when `currentStage` changes. This keeps the UI smooth even if stage transitions arrive slightly out of order.

---

## Notes for the Backend Developer

- **File storage**: Files uploaded for scanning should be deleted from cloud storage after 24 hours (privacy note is shown in the UI). Feedback-consented files may be kept anonymised for up to 30 days.
- **Waveform pre-computation**: The app expects `waveformBars` (30–50 normalised values) ready in the result payload. Compute these server-side on audio/video files so the mobile client doesn't have to decode audio.
- **Thumbnail**: For video scans, generate a frame thumbnail at ~2 s and store at `thumbnailUrl`. The app shows it on the deepfake result screen.
- **`tagline` field**: The app displays a plain-English sentence below the confidence gauge for audio results. Generate this on the backend based on `resultType` and `score` so it can be A/B tested or localised independently.
- **Model version**: Expose `modelVersion` (e.g. `"v3.2"`) on both the User object and weekly stats. The app shows it in the home screen hero card and the Analyzing screen footnote.
- **Plan enforcement**: Return `403 FORBIDDEN` with code `PLAN_REQUIRED` if a Free user requests a Forensic PDF. The app currently does not pre-check plan limits client-side — it relies on the API to gate.
- **Scan type for the Tamper tab**: When the user selects the "Tamper" tab in the Upload screen, `scanType` is sent as `"tamper"`. The backend should run both audio-cut detection and video re-encode detection and return `resultType: "editTamper"`.
