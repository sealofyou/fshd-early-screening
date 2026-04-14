from __future__ import annotations

import base64
import os
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..core.config import settings
from ..models.database import Base, InferenceRecord
from ..schemas.response import InferenceResponse
from ..services.cv_scoring import run_cv_scoring
from ..services.llm_screening import ScreeningUnavailableError, run_screening_pipeline
from ..services.vision_client import VisionModelClient

router = APIRouter(prefix="/api")


def get_db():
    engine = create_engine(settings.DATABASE_URL)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_local()
    try:
        yield db
    finally:
        db.close()


def init_db():
    engine = create_engine(settings.DATABASE_URL)
    Base.metadata.create_all(bind=engine)


@router.post("/inference", response_model=InferenceResponse)
async def inference(file: UploadFile = File(...), db: Session = Depends(get_db)):
    allowed_ext = {".jpg", ".jpeg", ".png"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail="Only JPG/PNG files are supported")

    file_content = await file.read()
    client = VisionModelClient()
    try:
        result = client.infer_from_base64(base64.b64encode(file_content).decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vision model request failed: {exc}") from exc

    record = InferenceRecord(
        file_id=str(uuid4()),
        original_filename=file.filename,
        risk_probability=result.risk_probability,
        advice=result.advice,
        raw_response=result.dict(),
        image_url=None,
    )
    db.add(record)
    db.commit()

    return InferenceResponse(
        status="success",
        probability=result.risk_probability,
        advice=f"{result.advice} For early screening support only. Not a clinical diagnosis.",
        image_url=None,
    )


@router.post("/analyze")
async def analyze(files: list[UploadFile] = File(...)) -> dict:
    try:
        return await run_screening_pipeline(files)
    except ScreeningUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/analyze/cv")
async def analyze_cv(
    close_eye_force: UploadFile = File(...),
    pout: UploadFile = File(...),
    puff_cheek: UploadFile = File(...),
) -> dict:
    action_inputs = {
        "close_eye_force": (
            close_eye_force.filename or "close_eye_force.jpg",
            await close_eye_force.read(),
        ),
        "pout": (
            pout.filename or "pout.jpg",
            await pout.read(),
        ),
        "puff_cheek": (
            puff_cheek.filename or "puff_cheek.jpg",
            await puff_cheek.read(),
        ),
    }
    return run_cv_scoring(action_inputs)
