from __future__ import annotations

import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import mediapipe as mp  # type: ignore
except ImportError:  # pragma: no cover
    mp = None

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None

CV_ACTION_FIELDS = ("close_eye_force", "pout", "puff_cheek")
CV_ACTION_LABELS = {
    "close_eye_force": "force_close_eye",
    "pout": "pout",
    "puff_cheek": "puff_cheek",
}
CV_CONFIG_PATH = Path(__file__).resolve().parents[1] / "cv_risk_config.json"
CV_KEYPOINT_INDEX = {
    "left_eye_upper_eyelid": 159,
    "left_eye_lower_eyelid": 145,
    "right_eye_upper_eyelid": 386,
    "right_eye_lower_eyelid": 374,
    "mouth_corner_left": 61,
    "mouth_corner_right": 291,
    "upper_lip_top": 13,
    "nose_base": 2,
}
CV_LEFT_CHEEK_POLYGON = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152]
CV_RIGHT_CHEEK_POLYGON = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152]


def _default_cv_risk_config() -> dict[str, Any]:
    return {
        "feature_rules": {
            "close_eye_force": {
                "palpebral_gap_left_norm": {
                    "thresholds": [0.020, 0.035, 0.055],
                    "direction": "higher_abnormal",
                    "weight": 0.45,
                    "region_key": "left_eyelid",
                    "region_label": "left eyelid",
                },
                "palpebral_gap_right_norm": {
                    "thresholds": [0.020, 0.035, 0.055],
                    "direction": "higher_abnormal",
                    "weight": 0.45,
                    "region_key": "right_eyelid",
                    "region_label": "right eyelid",
                },
                "palpebral_gap_asym": {
                    "thresholds": [0.008, 0.016, 0.028],
                    "direction": "higher_abnormal",
                    "weight": 0.10,
                    "region_key": "eyelid_symmetry",
                    "region_label": "eyelid symmetry",
                },
            },
            "pout": {
                "lip_corner_distance_norm": {
                    "thresholds": [0.38, 0.48, 0.58],
                    "direction": "higher_abnormal",
                    "weight": 0.25,
                    "region_key": "mouth_corner_span",
                    "region_label": "mouth corner span",
                },
                "lip_nose_distance_norm": {
                    "thresholds": [0.24, 0.20, 0.16],
                    "direction": "lower_abnormal",
                    "weight": 0.05,
                    "region_key": "upper_lip_projection",
                    "region_label": "upper lip projection",
                },
                "lip_asym_proxy_norm": {
                    "thresholds": [0.020, 0.036, 0.055],
                    "direction": "higher_abnormal",
                    "weight": 0.70,
                    "region_key": "mouth_symmetry",
                    "region_label": "mouth symmetry",
                },
            },
            "puff_cheek": {
                "cheek_area_left_norm": {
                    "thresholds": [0.40, 0.36, 0.32],
                    "direction": "lower_abnormal",
                    "weight": 0.15,
                    "region_key": "left_cheek",
                    "region_label": "left cheek",
                },
                "cheek_area_right_norm": {
                    "thresholds": [0.40, 0.36, 0.32],
                    "direction": "lower_abnormal",
                    "weight": 0.15,
                    "region_key": "right_cheek",
                    "region_label": "right cheek",
                },
                "cheek_area_ratio": {
                    "thresholds": [1.06, 1.12, 1.22],
                    "direction": "higher_abnormal",
                    "weight": 0.70,
                    "region_key": "cheek_symmetry",
                    "region_label": "cheek symmetry",
                },
            },
        },
        "action_weights": {
            "close_eye_force": 0.40,
            "puff_cheek": 0.35,
            "pout": 0.25,
        },
        "risk_level_thresholds": {
            "low_max": 34,
            "medium_max": 64,
        },
        "short_advice": {
            "low": "Current CV risk is low. Keep standard follow-up and recheck if symptoms change.",
            "medium": "Current CV risk is medium. Recommend retake and clinical review with neuromuscular specialist.",
            "high": "Current CV risk is high. Recommend in-person neurology evaluation and gene testing discussion.",
            "incomplete": "Input is incomplete or low quality. Please retake the required three actions clearly.",
        },
    }


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    keys = set(base.keys()) | set(override.keys())
    for key in keys:
        base_value = base.get(key)
        override_value = override.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = _merge_dict(base_value, override_value)
        elif key in override:
            merged[key] = override_value
        else:
            merged[key] = base_value
    return merged


def _load_cv_risk_config() -> dict[str, Any]:
    default_config = _default_cv_risk_config()
    if not CV_CONFIG_PATH.exists():
        return default_config
    try:
        raw = json.loads(CV_CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return default_config
        return _merge_dict(default_config, raw)
    except Exception:
        return default_config


CV_RISK_CONFIG = _load_cv_risk_config()


def _cv_point(lm: Any, width: int, height: int) -> tuple[float, float]:
    return lm.x * width, lm.y * height


def _cv_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _cv_polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _cv_score_feature(value: float | None, rule: dict[str, Any]) -> int | None:
    if value is None:
        return None
    t1, t2, t3 = [float(item) for item in rule.get("thresholds", [])]
    direction = str(rule.get("direction", "higher_abnormal")).strip()
    if direction == "lower_abnormal":
        if value >= t1:
            return 0
        if value >= t2:
            return 35
        if value >= t3:
            return 70
        return 100
    if value <= t1:
        return 0
    if value <= t2:
        return 35
    if value <= t3:
        return 70
    return 100


def _cv_severity(score: int) -> str:
    if score >= 85:
        return "high"
    if score >= 60:
        return "medium"
    return "low"


def _cv_risk_level(score: int | None, incomplete: bool) -> str:
    if incomplete or score is None:
        return "incomplete"
    thresholds = CV_RISK_CONFIG.get("risk_level_thresholds", {})
    low_max = int(thresholds.get("low_max", 34))
    medium_max = int(thresholds.get("medium_max", 64))
    if score <= low_max:
        return "low"
    if score <= medium_max:
        return "medium"
    return "high"


def _cv_short_advice(level: str) -> str:
    advice_map = CV_RISK_CONFIG.get("short_advice", {})
    advice = advice_map.get(level)
    return advice if isinstance(advice, str) and advice.strip() else advice_map["incomplete"]


def extract_landmark_map(image_bytes: bytes) -> tuple[dict[int, tuple[float, float]], list[str]]:
    quality_flags: list[str] = []
    if cv2 is None or np is None or mp is None:
        return {}, ["cv_dependency_missing"]

    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    if image_np.size == 0:
        return {}, ["empty_image_bytes"]

    image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return {}, ["imdecode_failed"]

    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_face_mesh = mp.solutions.face_mesh
    result = None
    used_confidence: float | None = None
    for confidence in (0.5, 0.3):
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=confidence,
        ) as face_mesh:
            result = face_mesh.process(image_rgb)
        if result.multi_face_landmarks:
            used_confidence = confidence
            break

    if not result or not result.multi_face_landmarks:
        return {}, ["no_face_detected"]
    if used_confidence == 0.3:
        quality_flags.append("face_detected_with_confidence_0.3")

    landmarks = result.multi_face_landmarks[0].landmark
    points: dict[int, tuple[float, float]] = {}
    for idx, lm in enumerate(landmarks):
        points[idx] = _cv_point(lm, width, height)
    return points, quality_flags


def extract_action_metrics(action_type: str, filename: str, image_bytes: bytes) -> tuple[dict[str, float], list[str]]:
    del filename
    points, quality_flags = extract_landmark_map(image_bytes)
    if not points:
        return {}, quality_flags

    required_idx = [CV_KEYPOINT_INDEX[item] for item in CV_KEYPOINT_INDEX]
    if any(idx not in points for idx in required_idx):
        return {}, quality_flags + ["missing_required_landmarks"]

    left_eye_center = (
        (points[CV_KEYPOINT_INDEX["left_eye_upper_eyelid"]][0] + points[CV_KEYPOINT_INDEX["left_eye_lower_eyelid"]][0]) / 2.0,
        (points[CV_KEYPOINT_INDEX["left_eye_upper_eyelid"]][1] + points[CV_KEYPOINT_INDEX["left_eye_lower_eyelid"]][1]) / 2.0,
    )
    right_eye_center = (
        (points[CV_KEYPOINT_INDEX["right_eye_upper_eyelid"]][0] + points[CV_KEYPOINT_INDEX["right_eye_lower_eyelid"]][0]) / 2.0,
        (points[CV_KEYPOINT_INDEX["right_eye_upper_eyelid"]][1] + points[CV_KEYPOINT_INDEX["right_eye_lower_eyelid"]][1]) / 2.0,
    )
    face_scale = _cv_distance(left_eye_center, right_eye_center)
    if face_scale < 1e-6:
        return {}, quality_flags + ["invalid_face_scale"]

    features: dict[str, float] = {}
    left_eye_upper = points[CV_KEYPOINT_INDEX["left_eye_upper_eyelid"]]
    left_eye_lower = points[CV_KEYPOINT_INDEX["left_eye_lower_eyelid"]]
    right_eye_upper = points[CV_KEYPOINT_INDEX["right_eye_upper_eyelid"]]
    right_eye_lower = points[CV_KEYPOINT_INDEX["right_eye_lower_eyelid"]]
    mouth_left = points[CV_KEYPOINT_INDEX["mouth_corner_left"]]
    mouth_right = points[CV_KEYPOINT_INDEX["mouth_corner_right"]]
    upper_lip = points[CV_KEYPOINT_INDEX["upper_lip_top"]]
    nose_base = points[CV_KEYPOINT_INDEX["nose_base"]]

    if action_type == "close_eye_force":
        left_gap = abs(left_eye_upper[1] - left_eye_lower[1]) / face_scale
        right_gap = abs(right_eye_upper[1] - right_eye_lower[1]) / face_scale
        features["palpebral_gap_left_norm"] = float(left_gap)
        features["palpebral_gap_right_norm"] = float(right_gap)
        features["palpebral_gap_asym"] = float(abs(left_gap - right_gap))

    if action_type == "pout":
        features["lip_corner_distance_norm"] = float(_cv_distance(mouth_left, mouth_right) / face_scale)
        features["lip_nose_distance_norm"] = float(abs(upper_lip[1] - nose_base[1]) / face_scale)
        lip_left_nose = _cv_distance(mouth_left, nose_base) / face_scale
        lip_right_nose = _cv_distance(mouth_right, nose_base) / face_scale
        features["lip_asym_proxy_norm"] = float(abs(lip_left_nose - lip_right_nose))

    if action_type == "puff_cheek":
        left_points = [points[idx] for idx in CV_LEFT_CHEEK_POLYGON if idx in points]
        right_points = [points[idx] for idx in CV_RIGHT_CHEEK_POLYGON if idx in points]
        if len(left_points) < 3 or len(right_points) < 3:
            quality_flags.append("missing_cheek_polygon_points")
        else:
            norm_scale = face_scale * face_scale
            left_area_norm = _cv_polygon_area(left_points) / norm_scale
            right_area_norm = _cv_polygon_area(right_points) / norm_scale
            min_area = min(left_area_norm, right_area_norm)
            features["cheek_area_left_norm"] = float(left_area_norm)
            features["cheek_area_right_norm"] = float(right_area_norm)
            features["cheek_area_ratio"] = float(max(left_area_norm, right_area_norm) / (min_area + 1e-6))

    if not features:
        quality_flags.append("no_action_features")
    return features, quality_flags


def score_cv_action(action_type: str, feature_values: dict[str, float], quality_flags: list[str]) -> dict[str, Any]:
    feature_rules = CV_RISK_CONFIG.get("feature_rules", {}).get(action_type, {})
    feature_scores: dict[str, int] = {}
    weighted_total = 0.0
    weight_sum = 0.0

    for feature_name, rule in feature_rules.items():
        value = feature_values.get(feature_name)
        score = _cv_score_feature(value, rule)
        if score is None:
            quality_flags.append(f"missing_feature:{feature_name}")
            continue
        weight = float(rule.get("weight", 1.0))
        feature_scores[feature_name] = score
        weighted_total += score * weight
        weight_sum += weight

    action_score = int(round(weighted_total / weight_sum)) if weight_sum > 0 else None
    if action_score is None:
        quality_flags.append("no_scorable_features")

    return {
        "action_type": action_type,
        "action_label": CV_ACTION_LABELS.get(action_type, action_type),
        "score": action_score,
        "feature_values": feature_values,
        "feature_scores": feature_scores,
        "quality_flags": sorted(set(quality_flags)),
    }


def _collect_top_abnormal_features(action_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for action_item in action_scores:
        action_type = action_item.get("action_type")
        feature_scores = action_item.get("feature_scores", {})
        feature_values = action_item.get("feature_values", {})
        action_rule = CV_RISK_CONFIG.get("feature_rules", {}).get(action_type, {})
        for feature_name, score in feature_scores.items():
            if not isinstance(score, int) or score < 35:
                continue
            rule = action_rule.get(feature_name, {})
            ranked.append(
                {
                    "action_type": action_type,
                    "feature": feature_name,
                    "value": feature_values.get(feature_name),
                    "score": score,
                    "severity": _cv_severity(score),
                    "region_key": rule.get("region_key", feature_name),
                    "region_label": rule.get("region_label", feature_name.replace("_", " ")),
                }
            )
    ranked.sort(key=lambda item: item.get("score", 0), reverse=True)
    return ranked[:3]


def _collect_visual_regions(top_abnormal_features: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    regions: list[dict[str, str]] = []
    for item in top_abnormal_features:
        action_type = str(item.get("action_type", ""))
        region_key = str(item.get("region_key", ""))
        region_label = str(item.get("region_label", region_key))
        signature = (action_type, region_key)
        if not action_type or not region_key or signature in seen:
            continue
        seen.add(signature)
        regions.append({"action_type": action_type, "region_key": region_key, "region_label": region_label})
    return regions


def run_cv_scoring(action_inputs: dict[str, tuple[str, bytes]]) -> dict[str, Any]:
    action_scores: list[dict[str, Any]] = []
    action_metrics: dict[str, dict[str, Any]] = {}
    global_quality_flags: list[str] = []

    for action_type in CV_ACTION_FIELDS:
        filename, image_bytes = action_inputs[action_type]
        feature_values, quality_flags = extract_action_metrics(action_type, filename, image_bytes)
        action_result = score_cv_action(action_type, feature_values, quality_flags)
        action_scores.append(action_result)
        action_metrics[action_type] = {
            "feature_values": action_result["feature_values"],
            "feature_scores": action_result["feature_scores"],
            "score": action_result["score"],
            "quality_flags": action_result["quality_flags"],
        }
        global_quality_flags.extend(action_result["quality_flags"])

    action_weights = CV_RISK_CONFIG.get("action_weights", {})
    weighted_sum = 0.0
    weight_sum = 0.0
    missing_actions: list[str] = []
    for item in action_scores:
        action_type = item.get("action_type")
        action_score = item.get("score")
        action_weight = float(action_weights.get(action_type, 0.0))
        if isinstance(action_score, int) and action_weight > 0:
            weighted_sum += action_score * action_weight
            weight_sum += action_weight
        else:
            missing_actions.append(str(action_type))

    risk_score: int | None = int(round(weighted_sum / weight_sum)) if weight_sum > 0 else None
    if len(missing_actions) == 1 and risk_score is not None:
        risk_score = int(round(risk_score * 0.85))
        global_quality_flags.append("confidence_penalty_0.15")

    incomplete = len(missing_actions) >= 2 or risk_score is None
    if incomplete:
        risk_score = None
        global_quality_flags.append("incomplete_required_actions")

    risk_level = _cv_risk_level(risk_score, incomplete)
    top_abnormal_features = _collect_top_abnormal_features(action_scores)
    visual_regions = _collect_visual_regions(top_abnormal_features)
    return {
        "report_id": f"CV-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}",
        "source": "cv_rule_v1",
        "incomplete": incomplete,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "action_scores": action_scores,
        "action_metrics": action_metrics,
        "missing_actions": missing_actions,
        "top_abnormal_features": top_abnormal_features,
        "visual_regions": visual_regions,
        "short_advice": _cv_short_advice(risk_level),
        "quality_flags": sorted(set(global_quality_flags)),
        "disclaimer": "For early screening support only. Not a clinical diagnosis.",
    }
