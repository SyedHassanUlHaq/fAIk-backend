#!/usr/bin/env python3
"""
Generate a sample forensic PDF with dummy data to preview the report layout.

Usage:
    .venv/bin/python generate_sample_pdf.py
Outputs:
    sample_report_video.pdf
    sample_report_tamper.pdf
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

from tasks.pdf_task import _build_pdf


def _mock_user():
    return SimpleNamespace(
        name="Hassan Ahmed",
        first_name="Hassan",
        last_name="Ahmed",
        email="hassan@faik.ai",
        plan="pro",
        created_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
    )


def _video_scan():
    return SimpleNamespace(
        id=uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890"),
        filename="interview_clip_deepfake.mp4",
        url_source=None,
        file_size=50_331_648,          # 48 MB
        duration=20.4,
        bitrate="3200 kbps",
        scan_type="video",
        result_type="deepfakeVideo",
        verdict="ai",
        score=87,
        status="complete",
        created_at=datetime(2026, 6, 18, 14, 30, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 6, 18, 14, 30, 47, tzinfo=timezone.utc),
        thumbnail_key=None,
        result_data={
            "plainEnglishExplanation": (
                "The video analysis detected an overall AI probability of 87%. "
                "Strong GAN artefacts and temporal inconsistencies were found across "
                "multiple segments. The facial regions show characteristic deepfake "
                "signatures including unnatural blinking patterns and lip-sync drift "
                "in three distinct time windows. The optical flow analysis reveals "
                "abnormal motion vectors around the mouth and eye regions consistent "
                "with face-swapping or re-animation techniques."
            ),
            "segments": [
                {
                    "startSeconds": 0, "endSeconds": 5,
                    "label": "Face GAN signature",
                    "type": "gan", "severity": "high",
                    "timelinePositionPercent": 0, "timelineWidthPercent": 25,
                },
                {
                    "startSeconds": 5, "endSeconds": 10,
                    "label": "Lip-sync drift",
                    "type": "lip_sync", "severity": "medium",
                    "timelinePositionPercent": 25, "timelineWidthPercent": 25,
                },
                {
                    "startSeconds": 10, "endSeconds": 15,
                    "label": "Lip-sync drift",
                    "type": "lip_sync", "severity": "medium",
                    "timelinePositionPercent": 50, "timelineWidthPercent": 25,
                },
                {
                    "startSeconds": 15, "endSeconds": 20,
                    "label": "Face GAN signature",
                    "type": "gan", "severity": "high",
                    "timelinePositionPercent": 75, "timelineWidthPercent": 25,
                },
            ],
        },
        user=_mock_user(),
    )


def _tamper_scan():
    return SimpleNamespace(
        id=uuid.UUID("b2c3d4e5-f6a7-8901-bcde-f12345678901"),
        filename="news_segment_edited.mp4",
        url_source="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        file_size=28_311_552,          # 27 MB
        duration=18.7,
        bitrate="1800 kbps",
        scan_type="tamper",
        result_type="editTamper",
        verdict="tampered",
        score=65,
        status="complete",
        created_at=datetime(2026, 6, 18, 15, 0, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 6, 18, 15, 0, 12, tzinfo=timezone.utc),
        thumbnail_key=None,
        result_data={
            "editCount": 3,
            "editSummary": "3 edits detected. Likely tampered.",
            "tamperLevel": "high",
            "edits": [
                {
                    "number": 1, "timeSeconds": 3.2,
                    "label": "Hard cut",
                    "tag": "cut", "severity": "medium",
                },
                {
                    "number": 2, "timeSeconds": 9.8,
                    "label": "Splice from another clip",
                    "tag": "splice", "severity": "high",
                },
                {
                    "number": 3, "timeSeconds": 15.5,
                    "label": "Re-encode detected",
                    "tag": "recode", "severity": "medium",
                },
            ],
        },
        user=_mock_user(),
    )


if __name__ == "__main__":
    for scan, filename in [
        (_video_scan(), "sample_report_video.pdf"),
        (_tamper_scan(), "sample_report_tamper.pdf"),
    ]:
        pdf_bytes = _build_pdf(scan)
        with open(filename, "wb") as f:
            f.write(pdf_bytes)
        print(f"Saved: {filename}  ({len(pdf_bytes) // 1024} KB)")
