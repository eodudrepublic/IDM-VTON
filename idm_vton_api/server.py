from __future__ import annotations

import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .pipeline import IDMVTONService, PROJECT_ROOT

Category = Literal["upper_body", "lower_body", "dresses"]
OUTPUT_ROOT = Path(os.environ.get("IDM_VTON_API_OUTPUT_DIR", PROJECT_ROOT / "api_outputs"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="IDM-VTON API",
    description="UI-independent API wrapper for local IDM-VTON inference.",
    version="0.1.0",
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")

_service: IDMVTONService | None = None


def get_service() -> IDMVTONService:
    global _service
    if _service is None:
        model_path = os.environ.get("IDM_VTON_MODEL", "yisol/IDM-VTON")
        _service = IDMVTONService(model_path=model_path)
    return _service


async def read_image(upload: UploadFile, field_name: str) -> Image.Image:
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")
    try:
        return Image.open(BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} is not a valid image") from exc


def make_job_dir(prefix: str) -> Path:
    job_id = f"{int(time.time())}_{uuid4().hex[:8]}"
    job_dir = OUTPUT_ROOT / prefix / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def output_url(path: Path) -> str:
    return "/outputs/" + path.relative_to(OUTPUT_ROOT).as_posix()


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": _service is not None,
        "output_root": str(OUTPUT_ROOT),
    }


@app.post("/tryon")
async def tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: Category = Form("upper_body"),
    garment_description: str = Form("clothing"),
    auto_crop: bool = Form(False),
    denoise_steps: int = Form(30),
    seed: int = Form(42),
) -> dict[str, object]:
    person = await read_image(person_image, "person_image")
    garment = await read_image(garment_image, "garment_image")

    try:
        result = get_service().tryon_once(
            person_image=person,
            garment_image=garment,
            garment_description=garment_description,
            category=category,
            auto_crop=auto_crop,
            denoise_steps=denoise_steps,
            seed=seed,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    job_dir = make_job_dir("single")
    image_path = job_dir / "result.png"
    mask_path = job_dir / "mask.png"
    metadata_path = job_dir / "metadata.json"

    result.image.save(image_path)
    result.mask.save(mask_path)
    metadata_path.write_text(
        json.dumps(
            {
                "category": result.category,
                "garment_description": garment_description,
                "auto_crop": auto_crop,
                "denoise_steps": denoise_steps,
                "seed": result.seed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "job_id": job_dir.name,
        "category": result.category,
        "seed": result.seed,
        "result_image_url": output_url(image_path),
        "mask_image_url": output_url(mask_path),
        "metadata_url": output_url(metadata_path),
    }


@app.post("/tryon/outfit")
async def tryon_outfit(
    person_image: UploadFile = File(...),
    upper_image: UploadFile = File(...),
    lower_image: UploadFile = File(...),
    upper_description: str = Form("upper garment"),
    lower_description: str = Form("lower garment"),
    auto_crop: bool = Form(False),
    denoise_steps: int = Form(30),
    seed: int = Form(42),
) -> dict[str, object]:
    person = await read_image(person_image, "person_image")
    upper = await read_image(upper_image, "upper_image")
    lower = await read_image(lower_image, "lower_image")

    try:
        results = get_service().tryon_outfit(
            person_image=person,
            upper_image=upper,
            lower_image=lower,
            upper_description=upper_description,
            lower_description=lower_description,
            auto_crop=auto_crop,
            denoise_steps=denoise_steps,
            seed=seed,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    job_dir = make_job_dir("outfit")
    upper_path = job_dir / "upper_result.png"
    final_path = job_dir / "final_result.png"
    upper_mask_path = job_dir / "upper_mask.png"
    lower_mask_path = job_dir / "lower_mask.png"
    metadata_path = job_dir / "metadata.json"

    results["upper"].image.save(upper_path)
    results["final"].image.save(final_path)
    results["upper"].mask.save(upper_mask_path)
    results["final"].mask.save(lower_mask_path)
    metadata_path.write_text(
        json.dumps(
            {
                "upper_description": upper_description,
                "lower_description": lower_description,
                "auto_crop": auto_crop,
                "denoise_steps": denoise_steps,
                "upper_seed": results["upper"].seed,
                "lower_seed": results["final"].seed,
                "strategy": "sequential_upper_then_lower",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "job_id": job_dir.name,
        "strategy": "sequential_upper_then_lower",
        "upper_seed": results["upper"].seed,
        "lower_seed": results["final"].seed,
        "upper_result_url": output_url(upper_path),
        "final_result_url": output_url(final_path),
        "upper_mask_url": output_url(upper_mask_path),
        "lower_mask_url": output_url(lower_mask_path),
        "metadata_url": output_url(metadata_path),
    }
