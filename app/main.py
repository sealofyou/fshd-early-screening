from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api.inference import init_db, router as inference_router

app = FastAPI(title="FSHD Early Screening API")

uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")
app.include_router(inference_router)


@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
