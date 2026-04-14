# FSHD Early Screening API

FastAPI backend for FSHD early screening with two clearly separated analysis lanes:

- `LLM lane`: image-based screening through an OpenAI-compatible vision model
- `CV lane`: MediaPipe + rule-based scoring for the three required facial actions

The current repo does not lock you to a single vendor. The LLM lane reads model and base URL from environment variables. In the current project baseline, the default example is aligned to the local OpenClaw/OpenAI-compatible setup using `openai/qwen3.5-plus`.

No plaintext secret is stored in the repository. All credentials must be provided through environment variables.

## Endpoints

- `POST /api/inference`
  - Legacy single-image endpoint kept for backward compatibility
  - Returns `status`, `probability`, `advice`, `image_url`

- `POST /api/analyze`
  - LLM screening endpoint
  - Accepts `files`
  - Returns a structured screening result with:
    - `report_id`
    - `source`
    - `risk_level`
    - `confidence`
    - `completeness`
    - `missing_required_actions`
    - `actions`
    - `key_findings`
    - `recommendations`
    - `disclaimer`
    - `analysis_valid`
    - `invalid_reason`

- `POST /api/analyze/cv`
  - CV screening endpoint
  - Accepts exactly three files:
    - `close_eye_force`
    - `pout`
    - `puff_cheek`
  - Returns a structured CV result with:
    - `risk_score`
    - `risk_level`
    - `action_scores`
    - `action_metrics`
    - `top_abnormal_features`
    - `visual_regions`
    - `short_advice`

## Quick Start

```bash
git clone https://github.com/sealofyou/fshd-early-screening.git
cd fshd-early-screening
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

## Environment Variables

Use `.env.example` as the template.

Required for real LLM inference:

- `OPENAI_API_KEY` or `SILICONFLOW_API_KEY`

Recommended:

- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `OPENAI_TWO_PASS`

For local mock LLM testing:

- `SCREENING_ALLOW_MOCK=true`

Minimal example:

```env
OPENAI_API_KEY=
OPENAI_BASE_URL=http://127.0.0.1:42060/v1
OPENAI_MODEL=openai/qwen3.5-plus
OPENAI_TWO_PASS=true
SCREENING_ALLOW_MOCK=false

SILICONFLOW_API_KEY=
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

DATABASE_URL=sqlite:///fshd.db
```

## Request Examples

Base URL:

```text
http://127.0.0.1:8000
```

### 1. Legacy Single-Image Endpoint

Request:

```bash
curl -X POST "http://127.0.0.1:8000/api/inference" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@example.jpg"
```

Example response:

```json
{
  "status": "success",
  "probability": 0.72,
  "advice": "Risk is elevated. Recommend specialist review. For early screening support only. Not a clinical diagnosis.",
  "image_url": null
}
```

### 2. LLM Screening Endpoint

Request:

```bash
curl -X POST "http://127.0.0.1:8000/api/analyze" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "files=@close_eye.jpg" ^
  -F "files=@pout.jpg" ^
  -F "files=@puff_cheek.jpg"
```

Example response:

```json
{
  "report_id": "LLM-20260415-1024",
  "source": "llm_rule_v1",
  "risk_level": "medium",
  "confidence": 68,
  "completeness": "complete",
  "missing_required_actions": [],
  "actions": [
    {
      "action": "close_eye_force",
      "status": "abnormal",
      "note": "Incomplete forceful eye closure or visible eyelash exposure."
    },
    {
      "action": "pout",
      "status": "abnormal",
      "note": "Insufficient lip pursing or asymmetric mouth shape."
    },
    {
      "action": "puff_cheek",
      "status": "normal",
      "note": ""
    }
  ],
  "key_findings": [
    {
      "action": "close_eye_force",
      "status": "abnormal",
      "note": "Incomplete forceful eye closure or visible eyelash exposure."
    }
  ],
  "recommendations": [
    "Medium-risk pattern detected. Recommend retake and specialist review.",
    "Complete the required actions with better lighting if possible."
  ],
  "disclaimer": "For early screening support only. Not a clinical diagnosis.",
  "analysis_valid": true,
  "invalid_reason": null
}
```

### 3. CV Screening Endpoint

Request:

```bash
curl -X POST "http://127.0.0.1:8000/api/analyze/cv" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "close_eye_force=@close_eye.jpg" ^
  -F "pout=@pout.jpg" ^
  -F "puff_cheek=@puff_cheek.jpg"
```

Example response:

```json
{
  "report_id": "CV-20260415-2048",
  "source": "cv_rule_v1",
  "incomplete": false,
  "risk_score": 28,
  "risk_level": "low",
  "action_scores": [
    {
      "action_type": "close_eye_force",
      "action_label": "force_close_eye",
      "score": 55,
      "feature_values": {
        "palpebral_gap_left_norm": 0.12,
        "palpebral_gap_right_norm": 0.08,
        "palpebral_gap_asym": 0.04
      },
      "feature_scores": {
        "palpebral_gap_left_norm": 100,
        "palpebral_gap_right_norm": 70,
        "palpebral_gap_asym": 100
      },
      "quality_flags": []
    }
  ],
  "action_metrics": {
    "close_eye_force": {
      "feature_values": {
        "palpebral_gap_left_norm": 0.12,
        "palpebral_gap_right_norm": 0.08,
        "palpebral_gap_asym": 0.04
      },
      "feature_scores": {
        "palpebral_gap_left_norm": 100,
        "palpebral_gap_right_norm": 70,
        "palpebral_gap_asym": 100
      },
      "score": 55,
      "quality_flags": []
    }
  },
  "missing_actions": [],
  "top_abnormal_features": [
    {
      "action_type": "close_eye_force",
      "feature": "palpebral_gap_left_norm",
      "value": 0.12,
      "score": 100,
      "severity": "high",
      "region_key": "left_eyelid",
      "region_label": "left eyelid"
    }
  ],
  "visual_regions": [
    {
      "action_type": "close_eye_force",
      "region_key": "left_eyelid",
      "region_label": "left eyelid"
    }
  ],
  "short_advice": "Current CV risk is low. Keep standard follow-up and recheck if symptoms change.",
  "quality_flags": [],
  "disclaimer": "For early screening support only. Not a clinical diagnosis."
}
```

### 4. Python Example

```python
import requests

base_url = "http://127.0.0.1:8000"

with open("close_eye.jpg", "rb") as f1, open("pout.jpg", "rb") as f2, open("puff_cheek.jpg", "rb") as f3:
    response = requests.post(
        f"{base_url}/api/analyze/cv",
        files={
            "close_eye_force": ("close_eye.jpg", f1, "image/jpeg"),
            "pout": ("pout.jpg", f2, "image/jpeg"),
            "puff_cheek": ("puff_cheek.jpg", f3, "image/jpeg"),
        },
        timeout=60,
    )

print(response.status_code)
print(response.json())
```

## Common Notes

- `/api/analyze` is the LLM lane.
- `/api/analyze/cv` is the CV lane.
- These two endpoints are intentionally separated so frontend or orchestration layers can choose one lane explicitly.
- No plaintext secret should be committed. Use `.env` or deployment platform secrets only.
- The original upstream repo used SiliconFlow-oriented wording and examples. This branch has been adjusted so the runtime model is environment-driven instead of being presented as fixed to one vendor.

## Verification

Run API contract tests:

```bash
python -m unittest tests.test_api_contracts
```

Open docs:

```text
http://127.0.0.1:8000/docs
```

## Notes

- The CV lane depends on `numpy`, `opencv-python`, and `mediapipe`.
- CV scoring config lives in `app/cv_risk_config.json`.
- This repo intentionally keeps LLM and CV interfaces separate so frontend or orchestration layers can choose one lane explicitly.
