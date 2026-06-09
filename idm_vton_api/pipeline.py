from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRADIO_ROOT = PROJECT_ROOT / "gradio_demo"

for path in (PROJECT_ROOT, GRADIO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from diffusers import AutoencoderKL, DDPMScheduler  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline  # noqa: E402
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModelRef  # noqa: E402
from src.unet_hacked_tryon import UNet2DConditionModel  # noqa: E402
from transformers import (  # noqa: E402
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

import apply_net  # noqa: E402
from preprocess.humanparsing.run_parsing import Parsing  # noqa: E402
from preprocess.openpose.run_openpose import OpenPose  # noqa: E402
from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy  # noqa: E402
from utils_mask import get_mask_location  # noqa: E402

Category = Literal["upper_body", "lower_body", "dresses"]


def _cuda_state() -> str:
    if not torch.cuda.is_available():
        return "cuda=unavailable"
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    return f"cuda_allocated_gb={allocated:.2f} cuda_reserved_gb={reserved:.2f}"


def _service_log(stage: str, message: str = "") -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[IDM-VTONService][{stamp}][pid={os.getpid()}][stage={stage}] {message} {_cuda_state()}", flush=True)


@dataclass(frozen=True)
class TryonResult:
    image: Image.Image
    mask: Image.Image
    seed: int
    category: Category


class IDMVTONService:
    """Loads IDM-VTON once and exposes UI-independent try-on methods."""

    def __init__(
        self,
        model_path: str = "yisol/IDM-VTON",
        device: str | None = None,
        gpu_id: int = 0,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("IDM-VTON requires CUDA for the current local setup.")

        self.model_path = model_path
        self.device = device or "cuda"
        self.gpu_id = gpu_id
        self.torch_dtype = torch.float16
        self.vae_processor = VaeImageProcessor(vae_scale_factor=8)
        self.tensor_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self._load_models()

    def _load_models(self) -> None:
        base_path = self.model_path
        _service_log("load.begin", f"model_path={base_path} device={self.device}")

        _service_log("load.unet.begin")
        self.unet = UNet2DConditionModel.from_pretrained(
            base_path,
            subfolder="unet",
            torch_dtype=self.torch_dtype,
        )
        self.unet.requires_grad_(False)
        _service_log("load.unet.done")

        _service_log("load.tokenizers.begin")
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
        self.tokenizer_two = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
        _service_log("load.tokenizers.done")

        _service_log("load.text_encoders.begin")
        self.text_encoder_one = CLIPTextModel.from_pretrained(
            base_path,
            subfolder="text_encoder",
            torch_dtype=self.torch_dtype,
        )
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            base_path,
            subfolder="text_encoder_2",
            torch_dtype=self.torch_dtype,
        )
        _service_log("load.text_encoders.done")

        _service_log("load.image_encoder.begin")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_path,
            subfolder="image_encoder",
            torch_dtype=self.torch_dtype,
        )
        _service_log("load.image_encoder.done")

        _service_log("load.vae.begin")
        self.vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae", torch_dtype=self.torch_dtype)
        _service_log("load.vae.done")

        _service_log("load.unet_encoder.begin")
        self.unet_encoder = UNet2DConditionModelRef.from_pretrained(
            base_path,
            subfolder="unet_encoder",
            torch_dtype=self.torch_dtype,
        )

        _service_log("load.unet_encoder.done")

        _service_log("load.parsing.begin")
        self.parsing_model = Parsing(self.gpu_id)
        _service_log("load.parsing.done")

        _service_log("load.openpose.begin")
        self.openpose_model = OpenPose(self.gpu_id)
        self.openpose_model.preprocessor.body_estimation.model.to(self.device)
        _service_log("load.openpose.done")

        _service_log("load.pipeline.begin")
        self.pipe = TryonPipeline.from_pretrained(
            base_path,
            unet=self.unet,
            vae=self.vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            scheduler=self.noise_scheduler,
            image_encoder=self.image_encoder,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.unet_encoder = self.unet_encoder
        self.pipe.to(self.device)
        self.pipe.unet_encoder.to(self.device)

        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet_encoder.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        _service_log("load.done", "all model components loaded")

    def build_original_mask(self, person_image: Image.Image, category: Category, auto_crop: bool = False) -> Image.Image:
        if category not in {"upper_body", "lower_body", "dresses"}:
            raise ValueError("category must be one of: upper_body, lower_body, dresses")
        human_img_orig = person_image.convert("RGB")
        human_img, _crop_box = self._prepare_human_image(human_img_orig, auto_crop=auto_crop)
        _service_log("condition.mask.begin", f"category={category} auto_crop={auto_crop}")
        mask, _mask_gray = self._build_auto_mask(human_img, category)
        _service_log("condition.mask.done", f"category={category}")
        return mask

    def build_original_densepose(self, person_image: Image.Image, auto_crop: bool = False) -> Image.Image:
        human_img_orig = person_image.convert("RGB")
        human_img, _crop_box = self._prepare_human_image(human_img_orig, auto_crop=auto_crop)
        _service_log("condition.densepose.begin", f"auto_crop={auto_crop}")
        densepose_img = self._build_densepose(human_img)
        _service_log("condition.densepose.done")
        return densepose_img

    def tryon_once(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        garment_description: str,
        category: Category = "upper_body",
        auto_crop: bool = False,
        denoise_steps: int = 30,
        seed: int = 42,
        mask_override: Image.Image | None = None,
        densepose_override: Image.Image | None = None,
    ) -> TryonResult:
        _service_log("tryon_once.begin", f"category={category} desc={garment_description!r} steps={denoise_steps} seed={seed} auto_crop={auto_crop}")
        if category not in {"upper_body", "lower_body", "dresses"}:
            raise ValueError("category must be one of: upper_body, lower_body, dresses")

        denoise_steps = int(denoise_steps)
        seed = random.randint(0, 2**31 - 1) if int(seed) < 0 else int(seed)

        human_img_orig = person_image.convert("RGB")
        garment_img = garment_image.convert("RGB").resize((768, 1024))
        human_img, crop_box = self._prepare_human_image(human_img_orig, auto_crop=auto_crop)
        _service_log("tryon_once.prepare.done", f"person_size={human_img.size} garment_size={garment_img.size} crop={crop_box is not None}")

        if mask_override is None:
            _service_log("tryon_once.mask.begin", f"category={category}")
            mask, mask_gray = self._build_auto_mask(human_img, category)
            _service_log("tryon_once.mask.done", f"category={category}")
        else:
            _service_log("tryon_once.mask.override", f"category={category}")
            mask = mask_override.convert("L").resize((768, 1024))
            masked_tensor = (1 - transforms.ToTensor()(mask)) * self.tensor_transform(human_img)
            mask_gray = to_pil_image((masked_tensor + 1.0) / 2.0)

        if densepose_override is None:
            _service_log("tryon_once.densepose.begin")
            densepose_img = self._build_densepose(human_img)
            _service_log("tryon_once.densepose.done")
        else:
            _service_log("tryon_once.densepose.override")
            densepose_img = densepose_override.convert("RGB").resize((768, 1024))

        prompt = f"model is wearing {garment_description}"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            cloth_prompt = f"a photo of {garment_description}"
            if not isinstance(cloth_prompt, list):
                cloth_prompt = [cloth_prompt] * 1
            (
                cloth_prompt_embeds,
                _negative_cloth_prompt_embeds,
                _pooled_cloth_prompt_embeds,
                _negative_pooled_cloth_prompt_embeds,
            ) = self.pipe.encode_prompt(
                cloth_prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=negative_prompt,
            )

            pose_img = self.tensor_transform(densepose_img).unsqueeze(0).to(self.device, self.torch_dtype)
            garment_tensor = self.tensor_transform(garment_img).unsqueeze(0).to(self.device, self.torch_dtype)
            generator = torch.Generator(self.device).manual_seed(seed)

            _service_log("tryon_once.diffusion.begin", f"category={category} steps={denoise_steps}")
            with torch.cuda.amp.autocast():
                output = self.pipe(
                    prompt_embeds=prompt_embeds.to(self.device, self.torch_dtype),
                    negative_prompt_embeds=negative_prompt_embeds.to(self.device, self.torch_dtype),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(self.device, self.torch_dtype),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(self.device, self.torch_dtype),
                    num_inference_steps=denoise_steps,
                    generator=generator,
                    strength=1.0,
                    pose_img=pose_img.to(self.device, self.torch_dtype),
                    text_embeds_cloth=cloth_prompt_embeds.to(self.device, self.torch_dtype),
                    cloth=garment_tensor.to(self.device, self.torch_dtype),
                    mask_image=mask,
                    image=human_img,
                    height=1024,
                    width=768,
                    ip_adapter_image=garment_img.resize((768, 1024)),
                    guidance_scale=2.0,
                )[0][0]
            _service_log("tryon_once.diffusion.done", f"category={category}")

        result_img = output
        if crop_box is not None:
            result_img = self._paste_back_to_original(human_img_orig, output, crop_box)
            mask = self._paste_back_to_original(human_img_orig, mask.convert("RGB"), crop_box).convert("L")

        _service_log("tryon_once.done", f"category={category} seed={seed}")
        return TryonResult(image=result_img, mask=mask, seed=seed, category=category)

    def tryon_outfit(
        self,
        person_image: Image.Image,
        upper_image: Image.Image,
        lower_image: Image.Image,
        upper_description: str,
        lower_description: str,
        auto_crop: bool = False,
        denoise_steps: int = 30,
        seed: int = 42,
    ) -> dict[str, TryonResult]:
        first = self.tryon_once(
            person_image=person_image,
            garment_image=upper_image,
            garment_description=upper_description,
            category="upper_body",
            auto_crop=auto_crop,
            denoise_steps=denoise_steps,
            seed=seed,
        )
        second_seed = first.seed + 1 if int(seed) >= 0 else -1
        second = self.tryon_once(
            person_image=first.image,
            garment_image=lower_image,
            garment_description=lower_description,
            category="lower_body",
            auto_crop=False,
            denoise_steps=denoise_steps,
            seed=second_seed,
        )
        return {"upper": first, "final": second}

    def _prepare_human_image(self, image: Image.Image, auto_crop: bool) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
        if not auto_crop:
            return image.resize((768, 1024)), None

        width, height = image.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        right = (width + target_width) // 2
        bottom = (height + target_height) // 2
        crop_box = (left, top, right, bottom)
        return image.crop(crop_box).resize((768, 1024)), crop_box

    def _paste_back_to_original(
        self,
        original: Image.Image,
        generated: Image.Image,
        crop_box: tuple[int, int, int, int],
    ) -> Image.Image:
        left, top, right, bottom = crop_box
        generated = generated.resize((right - left, bottom - top))
        composed = original.copy()
        composed.paste(generated, crop_box)
        return composed

    def _build_auto_mask(self, human_img: Image.Image, category: Category) -> tuple[Image.Image, Image.Image]:
        keypoints = self.openpose_model(human_img.resize((384, 512)))
        model_parse, _ = self.parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location("hd", category, model_parse, keypoints)
        mask = mask.resize((768, 1024))
        masked_tensor = (1 - transforms.ToTensor()(mask)) * self.tensor_transform(human_img)
        mask_gray = to_pil_image((masked_tensor + 1.0) / 2.0)
        return mask, mask_gray

    def _build_densepose(self, human_img: Image.Image) -> Image.Image:
        cfg_path = PROJECT_ROOT / "configs" / "densepose_rcnn_R_50_FPN_s1x.yaml"
        model_path = PROJECT_ROOT / "ckpt" / "densepose" / "model_final_162be9.pkl"
        human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
        args = apply_net.create_argument_parser().parse_args(
            (
                "show",
                str(cfg_path),
                str(model_path),
                "dp_segm",
                "-v",
                "--opts",
                "MODEL.DEVICE",
                self.device,
            )
        )
        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:, :, ::-1]
        return Image.fromarray(pose_img).resize((768, 1024))
