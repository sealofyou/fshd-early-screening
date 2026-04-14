from __future__ import annotations

import base64
import json
import mimetypes
import random
from datetime import datetime
from typing import Any

from fastapi import UploadFile
from openai import OpenAI

from ..core.config import settings

REQUIRED_ACTIONS = ["close_eye_force", "puff_cheek", "pout"]
ALL_ACTIONS = REQUIRED_ACTIONS + [
    "natural_close_eye",
    "smile",
    "raise_eyebrow",
    "frown",
    "show_teeth",
    "neutral_face",
    "scapular_winging",
    "arm_front_raise",
    "arm_side_raise",
]

ACTION_NOTES = {
    "close_eye_force": "Incomplete forceful eye closure or visible eyelash exposure.",
    "puff_cheek": "Weak cheek inflation or air leakage on one side.",
    "pout": "Insufficient lip pursing or asymmetric mouth shape.",
    "natural_close_eye": "Incomplete natural eye closure.",
    "smile": "Insufficient mouth corner elevation.",
    "raise_eyebrow": "Limited eyebrow lift or asymmetry.",
    "frown": "Weak glabellar contraction.",
    "show_teeth": "Limited tooth exposure or asymmetric smile.",
    "neutral_face": "Static facial asymmetry.",
    "scapular_winging": "Visible scapular winging.",
    "arm_front_raise": "Limited front raise or asymmetry.",
    "arm_side_raise": "Limited side raise or asymmetry.",
}


class ScreeningUnavailableError(RuntimeError):
    pass


def _extract_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def _image_to_data_url(filename: str, data: bytes) -> str:
    mime, _ = mimetypes.guess_type(filename)
    if not mime:
        mime = "image/jpeg"
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _client() -> OpenAI:
    api_key = settings.OPENAI_API_KEY or settings.SILICONFLOW_API_KEY
    base_url = settings.OPENAI_BASE_URL or settings.SILICONFLOW_BASE_URL
    if not api_key:
        raise ScreeningUnavailableError("Missing OPENAI_API_KEY or SILICONFLOW_API_KEY")
    return OpenAI(api_key=api_key, base_url=base_url)


def _mock_action(filename: str, index: int) -> dict[str, str]:
    name = filename.lower()
    if "close" in name or "eye" in name:
        action = "close_eye_force"
    elif "puff" in name or "cheek" in name:
        action = "puff_cheek"
    elif "pout" in name or "duck" in name:
        action = "pout"
    else:
        action = ALL_ACTIONS[index % len(ALL_ACTIONS)]

    status = "abnormal" if index % 3 == 0 else "normal" if index % 3 == 1 else "uncertain"
    note = ACTION_NOTES.get(action, "") if status == "abnormal" else ""
    if status == "uncertain":
        note = "Image quality or action completeness is insufficient."
    return {"action": action, "status": status, "note": note}


def _mock_response(files: list[UploadFile]) -> dict[str, Any]:
    actions = [_mock_action(file.filename or f"image_{idx}.jpg", idx) for idx, file in enumerate(files)]
    missing_required = [
        action for action in REQUIRED_ACTIONS if action not in {item["action"] for item in actions}
    ]
    abnormal_count = sum(1 for item in actions if item["status"] == "abnormal")
    risk_level = "high" if abnormal_count >= 2 else "medium" if abnormal_count == 1 else "low"
    key_findings = [item for item in actions if item["status"] == "abnormal"][:3] or actions[:2]
    return {
        "report_id": f"LLM-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}",
        "source": "mock",
        "risk_level": risk_level,
        "confidence": max(45, min(95, 65 + abnormal_count * 12 - len(missing_required) * 6)),
        "completeness": "complete" if not missing_required else "incomplete",
        "missing_required_actions": missing_required,
        "actions": actions,
        "key_findings": key_findings,
        "recommendations": _recommendations(risk_level),
        "disclaimer": "For early screening support only. Not a clinical diagnosis.",
        "analysis_valid": True,
        "invalid_reason": None,
    }


def _recommendations(level: str) -> list[str]:
    if level == "high":
        return [
            "High-risk pattern detected. Recommend in-person neurology review.",
            "Discuss gene testing and neuromuscular specialist follow-up.",
        ]
    if level == "medium":
        return [
            "Medium-risk pattern detected. Recommend retake and specialist review.",
            "Complete the required actions with better lighting if possible.",
        ]
    return [
        "Current visual risk is low.",
        "Retest if symptoms change or image quality improves.",
    ]


def _normalize_action(item: dict[str, Any]) -> dict[str, str]:
    action = str(item.get("action", "")).strip()
    status = str(item.get("status", "")).strip().lower()
    note = str(item.get("note", "")).strip()

    if action not in ALL_ACTIONS:
        action = "unknown"

    if status not in {"normal", "abnormal", "uncertain", "missing"}:
        status = "uncertain"

    if status == "abnormal" and not note:
        note = ACTION_NOTES.get(action, "")
    if status == "uncertain" and not note:
        note = "Image quality or action completeness is insufficient."
    if status == "missing" and not note:
        note = "Required action was not provided."

    return {"action": action, "status": status, "note": note}


def _normalize_response(payload: dict[str, Any], fallback_actions: list[dict[str, str]]) -> dict[str, Any]:
    actions = [_normalize_action(item) for item in payload.get("actions", fallback_actions)]
    missing_required = [
        action
        for action in REQUIRED_ACTIONS
        if not any(item["action"] == action and item["status"] != "missing" for item in actions)
    ]
    abnormal_count = sum(1 for item in actions if item["status"] == "abnormal")

    risk_level = str(payload.get("risk_level", "")).strip().lower()
    if risk_level not in {"low", "medium", "high"}:
        risk_level = "high" if abnormal_count >= 2 else "medium" if abnormal_count == 1 else "low"

    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = max(45, min(95, 65 + abnormal_count * 12 - len(missing_required) * 6))

    key_findings = payload.get("key_findings")
    if not isinstance(key_findings, list) or not key_findings:
        key_findings = [item for item in actions if item["status"] == "abnormal"][:3] or actions[:2]

    recommendations = payload.get("recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        recommendations = _recommendations(risk_level)

    return {
        "report_id": payload.get(
            "report_id",
            f"LLM-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}",
        ),
        "source": "llm_rule_v1",
        "risk_level": risk_level,
        "confidence": int(confidence),
        "completeness": "complete" if not missing_required else "incomplete",
        "missing_required_actions": missing_required,
        "actions": actions,
        "key_findings": key_findings,
        "recommendations": recommendations,
        "disclaimer": payload.get("disclaimer", "For early screening support only. Not a clinical diagnosis."),
        "analysis_valid": True,
        "invalid_reason": None,
    }


def _infer_per_image(filename: str, data: bytes) -> dict[str, Any]:
    client = _client()
    prompt = (
        "You are an FSHD early-screening assistant. "
        "Given one image, identify the action and whether it appears normal, abnormal, uncertain, or missing. "
        "Return strict JSON only with fields: action, status, note. "
        f"Allowed actions: {', '.join(ALL_ACTIONS)}."
    )
    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image for FSHD screening action recognition."},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(filename, data)}},
                ],
            },
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content or ""
    return _extract_json(content)


def _infer_aggregate(actions: list[dict[str, str]]) -> dict[str, Any]:
    client = _client()
    prompt = (
        "You are an FSHD early-screening assistant. "
        "Given pre-identified action results, return strict JSON only with fields: "
        "risk_level(low/medium/high), confidence(0-100), key_findings(array), recommendations(array), disclaimer."
    )
    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps({"actions": actions}, ensure_ascii=False)},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content or ""
    return _extract_json(content)


async def run_screening_pipeline(files: list[UploadFile]) -> dict[str, Any]:
    if settings.SCREENING_ALLOW_MOCK:
        return _mock_response(files)

    images: list[tuple[str, bytes]] = []
    for file in files:
        images.append((file.filename or "image.jpg", await file.read()))

    fallback_actions = [_mock_action(name, idx) for idx, (name, _) in enumerate(images)]

    try:
        if settings.OPENAI_TWO_PASS:
            actions = []
            for filename, data in images:
                actions.append(_normalize_action(_infer_per_image(filename, data)))
            payload = _infer_aggregate(actions)
            payload["actions"] = actions
        else:
            raise ScreeningUnavailableError("Single-pass LLM mode is not enabled in this repo yet.")
        return _normalize_response(payload, fallback_actions)
    except Exception as exc:
        raise ScreeningUnavailableError(f"LLM screening unavailable: {exc}") from exc
