#!/usr/bin/env python
"""Launch a separate Gradio demo for upper/lower outfit try-on validation."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from pathlib import Path
from uuid import uuid4

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageFilter

from idm_vton_api.pipeline import IDMVTONService


NETWORK_LOG_JS = """
() => {
  if (window.__idmVtonNetworkLoggerInstalled) {
    console.log("[IDM-VTON 7860][browser] network logger already installed");
    return;
  }
  window.__idmVtonNetworkLoggerInstalled = true;
  console.log("[IDM-VTON 7860][browser] network logger installed");

  const interesting = (url) => {
    const text = String(url || "");
    return text.includes("/info") || text.includes("/queue") || text.includes("/run") || text.includes("/api");
  };

  const originalFetch = window.fetch;
  window.fetch = async (...args) => {
    const raw = args[0];
    const url = typeof raw === "string" ? raw : raw && raw.url;
    const method = (args[1] && args[1].method) || (raw && raw.method) || "GET";
    const t0 = performance.now();
    if (interesting(url)) {
      console.log(`[IDM-VTON 7860][fetch:start] ${method} ${url}`);
    }
    try {
      const response = await originalFetch(...args);
      if (interesting(url)) {
        console.log(`[IDM-VTON 7860][fetch:done] ${method} ${url} status=${response.status} duration_ms=${Math.round(performance.now() - t0)}`);
      }
      return response;
    } catch (error) {
      if (interesting(url)) {
        console.error(`[IDM-VTON 7860][fetch:error] ${method} ${url}`, error);
      }
      throw error;
    }
  };

  const OriginalEventSource = window.EventSource;
  if (OriginalEventSource) {
    window.EventSource = function(url, config) {
      console.log(`[IDM-VTON 7860][eventsource:create] ${url}`);
      const source = new OriginalEventSource(url, config);
      source.addEventListener("open", () => console.log(`[IDM-VTON 7860][eventsource:open] ${url}`));
      source.addEventListener("message", (event) => {
        const data = String(event.data || "");
        if (data.includes("process_starts") || data.includes("process_completed") || data.includes("estimation") || data.includes("progress") || data.includes("msg")) {
          console.log(`[IDM-VTON 7860][eventsource:message] ${url}`, data.slice(0, 500));
        }
      });
      source.addEventListener("error", (event) => console.warn(`[IDM-VTON 7860][eventsource:error] ${url}`, event));
      return source;
    };
    window.EventSource.prototype = OriginalEventSource.prototype;
  }
}
"""

OUTFIT_CLICK_JS = """
(person, upper, lower, upper_desc, lower_desc, strategy, lower_mask_refinement, lower_mask_expand, auto_crop, steps, seed) => {
  console.log("[IDM-VTON 7860][ui] Try upper+lower clicked", {
    hasPerson: Boolean(person),
    hasUpper: Boolean(upper),
    hasLower: Boolean(lower),
    upper_desc,
    lower_desc,
    strategy,
    lower_mask_refinement,
    lower_mask_expand,
    auto_crop,
    steps,
    seed
  });
  console.log("[IDM-VTON 7860][ui] submitting Gradio queue request for /tryon_outfit");
  return [person, upper, lower, upper_desc, lower_desc, strategy, lower_mask_refinement, lower_mask_expand, auto_crop, steps, seed];
}
"""

SINGLE_CLICK_JS = """
(person, garment, desc, category, auto_crop, steps, seed) => {
  console.log("[IDM-VTON 7860][ui] Try single garment clicked", {
    hasPerson: Boolean(person),
    hasGarment: Boolean(garment),
    desc,
    category,
    auto_crop,
    steps,
    seed
  });
  console.log("[IDM-VTON 7860][ui] submitting Gradio queue request for /tryon_single_category");
  return [person, garment, desc, category, auto_crop, steps, seed];
}
"""

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "api_outputs" / "gradio_outfit"
LOG_PATH = Path("/tmp/idm_vton_outfit_demo_7860.log")

_service = None
_service_loaded_at = None


def patch_gradio_filedata_schema() -> None:
    """Work around Gradio 4.24 /info failing on boolean additionalProperties."""
    import gradio_client.utils as client_utils

    original = client_utils.json_schema_to_python_type
    if getattr(original, "_idm_vton_schema_patch", False):
        return

    def normalize(schema):
        if isinstance(schema, dict):
            normalized = {}
            for key, value in schema.items():
                if key == "additionalProperties" and isinstance(value, bool):
                    normalized[key] = {} if value else {"type": "null"}
                else:
                    normalized[key] = normalize(value)
            return normalized
        if isinstance(schema, list):
            return [normalize(item) for item in schema]
        return schema

    def patched(schema):
        return original(normalize(schema))

    patched._idm_vton_schema_patch = True
    client_utils.json_schema_to_python_type = patched


patch_gradio_filedata_schema()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--model", default="yisol/IDM-VTON")
    parser.add_argument("--device", default="")
    return parser.parse_args()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def cuda_state() -> str:
    if not torch.cuda.is_available():
        return "cuda=unavailable"
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    return f"cuda_allocated_gb={allocated:.2f} cuda_reserved_gb={reserved:.2f}"


def model_status_text(note: str = "") -> str:
    loaded = _service is not None
    status = "loaded" if loaded else "not_loaded"
    wait_hint = "first request should skip model loading" if loaded else "first request will load the model before inference"
    lines = [
        f"[{now()}] model_status={status}",
        f"pid={os.getpid()}",
        f"loaded_at={_service_loaded_at or 'n/a'}",
        f"{cuda_state()}",
        wait_hint,
    ]
    if note:
        lines.append(note)
    return "\n".join(lines)


def log_event(job_id: str, stage: str, message: str = "") -> None:
    print(f"[IDM-VTON-7860][{now()}][pid={os.getpid()}][job={job_id}][stage={stage}] {message} {cuda_state()}", flush=True)


def status_text(job_id: str, stage: str, message: str) -> str:
    return f"[{now()}] job={job_id}\nstage={stage}\n{message}\nmodel_loaded={_service is not None}\nserver_log={LOG_PATH}"


def cleanup_cuda(job_id: str) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_event(job_id, "cleanup", "gc.collect() and torch.cuda.empty_cache() completed")


def get_service(model_path: str, job_id: str):
    global _service, _service_loaded_at
    if _service is None:
        log_event(job_id, "model_load.begin", f"model_path={model_path}")
        _service = IDMVTONService(model_path=model_path)
        _service_loaded_at = now()
        log_event(job_id, "model_load.done", "model service is ready")
    else:
        log_event(job_id, "model_load.reuse", "reusing already loaded model service")
    return _service


def refresh_model_status():
    return model_status_text()


def preload_model():
    job_id = f"preload-{int(time.time())}-{uuid4().hex[:6]}"
    yield model_status_text("preload requested; loading model now")
    try:
        get_service(os.environ.get("IDM_VTON_MODEL", "yisol/IDM-VTON"), job_id)
        yield model_status_text("preload completed; next try-on should start faster")
    except Exception as exc:
        log_event(job_id, "model_preload.error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        cleanup_cuda(job_id)
        yield model_status_text(f"preload failed: {type(exc).__name__}: {exc}")


def unload_model():
    global _service, _service_loaded_at
    job_id = f"unload-{int(time.time())}-{uuid4().hex[:6]}"
    if _service is not None:
        log_event(job_id, "model_unload.begin", "dropping model service reference")
        old_service = _service
        _service = None
        _service_loaded_at = None
        del old_service
    cleanup_cuda(job_id)
    return model_status_text("model unloaded")


def require_image(image, name: str) -> Image.Image:
    if image is None:
        raise ValueError(f"{name} image is required.")
    return image.convert("RGB")


def expand_mask(mask: Image.Image, expand_px: int, blur_px: int = 4) -> Image.Image:
    mask_l = mask.convert("L")
    expand_px = max(0, int(expand_px))
    if expand_px > 0:
        mask_l = mask_l.filter(ImageFilter.MaxFilter(expand_px * 2 + 1))
    if blur_px > 0:
        mask_l = mask_l.filter(ImageFilter.GaussianBlur(blur_px))
    return mask_l


def lower_prompt_is_skirt_like(description: str) -> bool:
    text = (description or "").lower()
    skirt_words = (
        "skirt",
        "pleated",
        "mini skirt",
        "midi skirt",
        "maxi skirt",
        "dress skirt",
        "치마",
        "스커트",
        "플리츠",
    )
    return any(word in text for word in skirt_words)


def fill_horizontal_mask_gaps(mask: Image.Image) -> Image.Image:
    arr = np.array(mask.convert("L")) > 127
    filled = arr.copy()
    height, width = arr.shape
    min_span = max(16, int(width * 0.04))
    max_span = int(width * 0.86)

    for y in range(height):
        xs = np.flatnonzero(arr[y])
        if xs.size < 2:
            continue
        left = int(xs[0])
        right = int(xs[-1])
        span = right - left + 1
        if min_span <= span <= max_span:
            filled[y, left : right + 1] = True

    filled_mask = Image.fromarray((filled * 255).astype("uint8"), mode="L")
    return filled_mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(2))


def refine_lower_mask(mask: Image.Image, mode: str, lower_description: str) -> tuple[Image.Image, str]:
    selected = mode or "auto"
    if selected == "auto":
        selected = "skirt_fill" if lower_prompt_is_skirt_like(lower_description) else "none"
    if selected == "skirt_fill":
        return fill_horizontal_mask_gaps(mask), "skirt_fill"
    return mask.convert("L"), "none"


def composite_with_mask(base: Image.Image, overlay: Image.Image, mask: Image.Image, expand_px: int) -> tuple[Image.Image, Image.Image]:
    base_img = base.convert("RGB")
    overlay_img = overlay.convert("RGB")
    if overlay_img.size != base_img.size:
        overlay_img = overlay_img.resize(base_img.size)
    used_mask = expand_mask(mask.resize(base_img.size), expand_px=expand_px)
    return Image.composite(overlay_img, base_img, used_mask), used_mask


def save_job(prefix: str, images: dict[str, Image.Image], metadata: dict[str, object]) -> tuple[str, dict[str, str]]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    job_id = f"{int(time.time())}_{uuid4().hex[:8]}"
    job_dir = OUTPUT_ROOT / prefix / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    for name, image in images.items():
        path = job_dir / f"{name}.png"
        image.save(path)
        paths[name] = str(path)

    metadata_path = job_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["metadata"] = str(metadata_path)
    return job_id, paths


def friendly_error(exc: BaseException) -> str:
    message = str(exc)
    if "out of memory" in message.lower() or "MemoryAllocation" in message:
        return (
            "CUDA out-of-memory occurred. Close other GPU jobs or restart this demo process to clear partial model allocations. "
            "Upper+lower outfit may use two diffusion passes, so it is slower and memory pressure is higher than one upper-body try-on."
        )
    return "Unexpected error. Check the server log for the full traceback."


def run_single(person_image, garment_image, garment_description, category, auto_crop, denoise_steps, seed):
    job_id = f"single-{int(time.time())}-{uuid4().hex[:6]}"
    stage = "request.received"
    log_event(job_id, stage, f"category={category} steps={denoise_steps} seed={seed}")
    yield None, None, status_text(job_id, stage, "Request received by server.")

    try:
        stage = "input.validate"
        person = require_image(person_image, "Person")
        garment = require_image(garment_image, "Garment")
        log_event(job_id, stage, f"person_size={person.size} garment_size={garment.size}")
        yield None, None, status_text(job_id, stage, "Input images validated.")

        stage = "model.load"
        yield None, None, status_text(job_id, stage, "Loading or reusing IDM-VTON model. First load can take a while.")
        service = get_service(os.environ.get("IDM_VTON_MODEL", "yisol/IDM-VTON"), job_id)

        stage = "single.tryon"
        log_event(job_id, stage, "starting single try-on")
        yield None, None, status_text(job_id, stage, "Running single try-on diffusion.")
        result = service.tryon_once(
            person_image=person,
            garment_image=garment,
            garment_description=garment_description or "clothing",
            category=category,
            auto_crop=auto_crop,
            denoise_steps=int(denoise_steps),
            seed=int(seed),
        )

        stage = "save"
        job_dir_id, paths = save_job(
            "single",
            {"result": result.image, "mask": result.mask},
            {
                "category": result.category,
                "garment_description": garment_description,
                "auto_crop": auto_crop,
                "denoise_steps": int(denoise_steps),
                "seed": result.seed,
            },
        )
        log_event(job_id, stage, f"saved job_dir_id={job_dir_id}")
        yield result.image, result.mask, status_text(job_id, "done", f"Done. result={paths['result']}\nmask={paths['mask']}")
    except Exception as exc:
        log_event(job_id, f"{stage}.error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        cleanup_cuda(job_id)
        yield None, None, status_text(job_id, f"{stage}.error", f"{type(exc).__name__}: {exc}\n{friendly_error(exc)}")


def run_outfit(
    person_image,
    upper_image,
    lower_image,
    upper_description,
    lower_description,
    strategy,
    lower_mask_refinement,
    lower_mask_expand,
    auto_crop,
    denoise_steps,
    seed,
):
    job_id = f"outfit-{int(time.time())}-{uuid4().hex[:6]}"
    stage = "request.received"
    strategy = strategy or "fixed_original_masks"
    lower_mask_refinement = lower_mask_refinement or "auto"
    lower_mask_expand = int(lower_mask_expand or 0)
    log_event(
        job_id,
        stage,
        (
            f"strategy={strategy} steps={denoise_steps} seed={seed} auto_crop={auto_crop} "
            f"lower_mask_refinement={lower_mask_refinement} lower_mask_expand={lower_mask_expand}"
        ),
    )
    yield None, None, None, None, status_text(job_id, stage, "Request received by server.")

    upper_result = None
    lower_result = None
    final_image = None
    used_lower_mask = None
    try:
        stage = "input.validate"
        person = require_image(person_image, "Person")
        upper = require_image(upper_image, "Upper garment")
        lower = require_image(lower_image, "Lower garment")
        log_event(job_id, stage, f"person_size={person.size} upper_size={upper.size} lower_size={lower.size}")
        yield None, None, None, None, status_text(job_id, stage, "Input images validated.")

        stage = "model.load"
        yield None, None, None, None, status_text(job_id, stage, "Loading or reusing IDM-VTON model. Use Refresh model status to see whether it is already loaded.")
        service = get_service(os.environ.get("IDM_VTON_MODEL", "yisol/IDM-VTON"), job_id)

        upper_seed = int(seed)
        lower_seed = upper_seed + 1 if upper_seed >= 0 else -1

        if strategy == "fixed_original_masks":
            stage = "condition.capture"
            log_event(job_id, stage, "capturing upper/lower masks and densepose on original person")
            yield None, None, None, None, status_text(
                job_id,
                stage,
                "Capturing upper mask, lower mask, and DensePose from the original person image.",
            )
            upper_condition_mask = service.build_original_mask(person, "upper_body", auto_crop=auto_crop)
            lower_condition_mask_raw = service.build_original_mask(person, "lower_body", auto_crop=auto_crop)
            lower_condition_mask, applied_refinement = refine_lower_mask(
                lower_condition_mask_raw,
                lower_mask_refinement,
                lower_description,
            )
            densepose_condition = service.build_original_densepose(person, auto_crop=auto_crop)
            yield (
                None,
                None,
                upper_condition_mask,
                lower_condition_mask,
                status_text(job_id, stage, f"Original conditions captured. lower_mask_refinement={applied_refinement}"),
            )

            stage = "lower.tryon.fixed_original_mask"
            log_event(job_id, stage, f"starting lower_body pass with original lower mask refinement={applied_refinement}")
            yield None, None, upper_condition_mask, lower_condition_mask, status_text(
                job_id,
                stage,
                "Running lower-body pass with the lower mask captured from the original image.",
            )
            lower_result = service.tryon_once(
                person,
                lower,
                lower_description or "lower garment",
                "lower_body",
                auto_crop,
                int(denoise_steps),
                lower_seed,
                mask_override=lower_condition_mask,
                densepose_override=densepose_condition,
            )
            yield lower_result.image, None, upper_condition_mask, lower_result.mask, status_text(
                job_id,
                stage,
                "Lower-body pass completed. Starting upper-body pass with the original upper mask.",
            )

            cleanup_cuda(job_id)
            stage = "upper.tryon.fixed_original_mask"
            log_event(job_id, stage, "starting upper_body pass with original upper mask")
            upper_result = service.tryon_once(
                person,
                upper,
                upper_description or "upper garment",
                "upper_body",
                auto_crop,
                int(denoise_steps),
                upper_seed,
                mask_override=upper_condition_mask,
                densepose_override=densepose_condition,
            )
            yield lower_result.image, upper_result.image, upper_result.mask, lower_result.mask, status_text(
                job_id,
                stage,
                "Upper-body pass completed. Compositing fixed lower result over fixed upper result.",
            )

            stage = "compose.fixed_lower_over_upper"
            final_image, used_lower_mask = composite_with_mask(upper_result.image, lower_result.image, lower_result.mask, lower_mask_expand)
            log_event(job_id, stage, f"composited with lower_mask_expand={lower_mask_expand} lower_mask_refinement={applied_refinement}")

        elif strategy == "upper_then_lower":
            stage = "upper.tryon"
            log_event(job_id, stage, "starting upper_body pass")
            yield None, None, None, None, status_text(job_id, stage, "Running upper-body pass first. Lower mask will be computed on upper-pass result.")
            upper_result = service.tryon_once(person, upper, upper_description or "upper garment", "upper_body", auto_crop, int(denoise_steps), upper_seed)
            yield upper_result.image, upper_result.image, upper_result.mask, None, status_text(job_id, stage, "Upper-body pass completed. Starting lower-body pass on upper-pass result.")

            cleanup_cuda(job_id)
            stage = "lower.tryon.after_upper"
            lower_result = service.tryon_once(upper_result.image, lower, lower_description or "lower garment", "lower_body", False, int(denoise_steps), lower_seed)
            final_image = lower_result.image
            used_lower_mask = lower_result.mask

        elif strategy == "lower_then_upper":
            stage = "lower.tryon"
            log_event(job_id, stage, "starting lower_body pass first")
            yield None, None, None, None, status_text(job_id, stage, "Running lower-body pass first. This often gives a larger lower mask than upper-then-lower.")
            lower_result = service.tryon_once(person, lower, lower_description or "lower garment", "lower_body", auto_crop, int(denoise_steps), lower_seed)
            used_lower_mask = lower_result.mask
            yield lower_result.image, None, None, used_lower_mask, status_text(job_id, stage, "Lower-body pass completed. Starting upper-body pass on lower-pass result.")

            cleanup_cuda(job_id)
            stage = "upper.tryon.after_lower"
            upper_result = service.tryon_once(lower_result.image, upper, upper_description or "upper garment", "upper_body", False, int(denoise_steps), upper_seed)
            final_image = upper_result.image

        else:
            strategy = "parallel_composite"
            stage = "lower.tryon.original"
            log_event(job_id, stage, "starting lower_body pass on original person")
            yield None, None, None, None, status_text(job_id, stage, "Running lower-body pass on original person to avoid tiny masks after upper pass.")
            lower_result = service.tryon_once(person, lower, lower_description or "lower garment", "lower_body", auto_crop, int(denoise_steps), lower_seed)
            yield lower_result.image, None, None, lower_result.mask, status_text(job_id, stage, "Lower-body pass completed. Starting upper-body pass on original person.")

            cleanup_cuda(job_id)
            stage = "upper.tryon.original"
            log_event(job_id, stage, "starting upper_body pass on original person")
            upper_result = service.tryon_once(person, upper, upper_description or "upper garment", "upper_body", auto_crop, int(denoise_steps), upper_seed)
            yield lower_result.image, upper_result.image, upper_result.mask, lower_result.mask, status_text(job_id, stage, "Upper-body pass completed. Compositing lower result over upper result with lower mask.")

            stage = "compose.lower_over_upper"
            final_image, used_lower_mask = composite_with_mask(upper_result.image, lower_result.image, lower_result.mask, lower_mask_expand)
            log_event(job_id, stage, f"composited with lower_mask_expand={lower_mask_expand}")

        stage = "save"
        images = {
            "upper_result": upper_result.image,
            "lower_result": lower_result.image,
            "final_result": final_image,
            "upper_mask": upper_result.mask,
            "lower_mask": used_lower_mask,
        }
        if lower_result.mask is not used_lower_mask:
            images["lower_mask_raw"] = lower_result.mask
        job_dir_id, paths = save_job(
            "outfit",
            images,
            {
                "upper_description": upper_description,
                "lower_description": lower_description,
                "auto_crop": auto_crop,
                "denoise_steps": int(denoise_steps),
                "upper_seed": upper_result.seed,
                "lower_seed": lower_result.seed,
                "strategy": strategy,
                "lower_mask_refinement": lower_mask_refinement,
                "lower_mask_expand": lower_mask_expand,
            },
        )
        log_event(job_id, stage, f"saved job_dir_id={job_dir_id}")
        status = (
            f"Done.\nstrategy={strategy}\nupper_result={paths['upper_result']}\n"
            f"lower_result={paths['lower_result']}\nfinal_result={paths['final_result']}\n"
            f"upper_mask={paths['upper_mask']}\nlower_mask={paths['lower_mask']}"
        )
        yield final_image, upper_result.image, upper_result.mask, used_lower_mask, status_text(job_id, "done", status)
    except Exception as exc:
        log_event(job_id, f"{stage}.error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        cleanup_cuda(job_id)
        final_img = final_image
        upper_img = upper_result.image if upper_result else None
        upper_mask = upper_result.mask if upper_result else None
        lower_mask = used_lower_mask or (lower_result.mask if lower_result else None)
        yield final_img, upper_img, upper_mask, lower_mask, status_text(job_id, f"{stage}.error", f"{type(exc).__name__}: {exc}\n{friendly_error(exc)}")


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="IDM-VTON Outfit Validation", js=NETWORK_LOG_JS) as demo:
        gr.Markdown("## IDM-VTON Outfit Validation")

        with gr.Row():
            model_status = gr.Textbox(label="Model status", value=model_status_text(), lines=6)
        with gr.Row():
            refresh_button = gr.Button("Refresh model status")
            preload_button = gr.Button("Preload model")
            unload_button = gr.Button("Unload model")
        refresh_button.click(fn=refresh_model_status, inputs=None, outputs=model_status, api_name="model_status")
        preload_button.click(fn=preload_model, inputs=None, outputs=model_status, api_name="preload_model", show_progress="full")
        unload_button.click(fn=unload_model, inputs=None, outputs=model_status, api_name="unload_model")

        with gr.Tab("Upper + Lower Outfit"):
            with gr.Row():
                person = gr.Image(label="Person", type="pil")
                upper = gr.Image(label="Upper garment", type="pil")
                lower = gr.Image(label="Lower garment", type="pil")
            with gr.Row():
                upper_desc = gr.Textbox(label="Upper description", value="long sleeve knit")
                lower_desc = gr.Textbox(label="Lower description", value="white pleated skirt")
            with gr.Row():
                strategy = gr.Radio(
                    label="Outfit strategy",
                    choices=["upper_then_lower", "lower_then_upper", "parallel_composite", "fixed_original_masks"],
                    value="upper_then_lower",
                )
                lower_mask_refinement = gr.Radio(
                    label="Lower mask refinement",
                    choices=["auto", "none", "skirt_fill"],
                    value="auto",
                )
                lower_mask_expand = gr.Slider(label="Lower mask expansion", minimum=0, maximum=40, value=0, step=1)
            gr.Markdown(
                "- `upper_then_lower`: 상의를 먼저 합성한 뒤 그 결과 위에 하의를 합성합니다. 일반 전신 사진에서 가장 자연스러운 경우가 많습니다.\n"
                "- `lower_then_upper`: 하의를 먼저 합성한 뒤 그 결과 위에 상의를 합성합니다. 하의 형태를 우선 보존하고 싶을 때 비교용으로 사용합니다.\n"
                "- `parallel_composite`: 원본 사진 기준으로 상의와 하의를 각각 합성한 뒤, 하의 mask로 두 결과를 합성합니다.\n"
                "- `fixed_original_masks`: 원본 사진에서 상의/하의 mask와 DensePose를 먼저 고정한 뒤 각각 합성하고 조합합니다."
            )
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    auto_crop = gr.Checkbox(label="Auto crop", value=False)
                    steps = gr.Number(label="Denoising steps", value=20, minimum=10, maximum=50, step=1)
                    seed = gr.Number(label="Seed", value=42, minimum=-1, maximum=2147483647, step=1)
            outfit_button = gr.Button("Try upper + lower")
            with gr.Row():
                final_out = gr.Image(label="Final result", type="pil")
                upper_out = gr.Image(label="Upper pass result", type="pil")
            with gr.Row():
                upper_mask = gr.Image(label="Upper mask", type="pil")
                lower_mask = gr.Image(label="Lower mask", type="pil")
            outfit_status = gr.Textbox(label="Status", lines=7)
            outfit_button.click(
                fn=run_outfit,
                inputs=[
                    person,
                    upper,
                    lower,
                    upper_desc,
                    lower_desc,
                    strategy,
                    lower_mask_refinement,
                    lower_mask_expand,
                    auto_crop,
                    steps,
                    seed,
                ],
                outputs=[final_out, upper_out, upper_mask, lower_mask, outfit_status],
                api_name="tryon_outfit",
                js=OUTFIT_CLICK_JS,
                show_progress="full",
            )

        with gr.Tab("Single Category"):
            with gr.Row():
                single_person = gr.Image(label="Person", type="pil")
                single_garment = gr.Image(label="Garment", type="pil")
            with gr.Row():
                single_category = gr.Radio(
                    label="Category",
                    choices=["upper_body", "lower_body", "dresses"],
                    value="upper_body",
                )
                single_desc = gr.Textbox(label="Garment description", value="clothing")
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    single_auto_crop = gr.Checkbox(label="Auto crop", value=False)
                    single_steps = gr.Number(label="Denoising steps", value=20, minimum=10, maximum=50, step=1)
                    single_seed = gr.Number(label="Seed", value=42, minimum=-1, maximum=2147483647, step=1)
            single_button = gr.Button("Try single garment")
            with gr.Row():
                single_result = gr.Image(label="Result", type="pil")
                single_mask = gr.Image(label="Mask", type="pil")
            single_status = gr.Textbox(label="Status", lines=7)
            single_button.click(
                fn=run_single,
                inputs=[
                    single_person,
                    single_garment,
                    single_desc,
                    single_category,
                    single_auto_crop,
                    single_steps,
                    single_seed,
                ],
                outputs=[single_result, single_mask, single_status],
                api_name="tryon_single_category",
                js=SINGLE_CLICK_JS,
                show_progress="full",
            )

    return demo.queue()


def main() -> int:
    args = parse_args()
    os.environ["IDM_VTON_MODEL"] = args.model
    if args.device:
        os.environ["IDM_VTON_DEVICE"] = args.device

    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
