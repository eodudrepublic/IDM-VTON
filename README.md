# IDM-VTON Fork: Person + Upper/Lower Outfit Try-on API

This repository is a fork of [`yisol/IDM-VTON`](https://github.com/yisol/IDM-VTON), the official implementation of **IDM-VTON: Improving Diffusion Models for Authentic Virtual Try-on in the Wild**.

The upstream project focuses on research code, dataset inference, training, and a Gradio demo. This fork keeps the original model pipeline and paper references, but changes the repository goal toward a local service that can receive:

- one original person image
- one upper garment image
- one lower garment image

and return virtual try-on outputs through an API or validation demo.

<div align="center">

<a href="https://idm-vton.github.io"><img src="https://img.shields.io/badge/Project-Page-green"></a>
<a href="https://arxiv.org/abs/2403.05139"><img src="https://img.shields.io/badge/Paper-Arxiv-red"></a>
<a href="https://huggingface.co/yisol/IDM-VTON"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a>

</div>

![teaser2](assets/teaser2.png)
![teaser](assets/teaser.png)

## 포크 목표

이 레포지토리의 목표는 IDM-VTON을 더 큰 코디/의상 생성 시스템의 try-on 엔진으로 사용하는 것입니다. 현재 우선순위는 모델 학습이 아니라, 업로드된 인물 사진에 상의와 하의 이미지를 합성할 수 있는 안정적인 로컬 API를 제공하는 것입니다.

이 포크에서 추가한 주요 기능은 다음과 같습니다.

- 현재 ARM64/CUDA 13.0 워크스테이션용 로컬 환경 파일
- 체크포인트 다운로드 및 설치 검증 스크립트
- 원본 Gradio UI와 분리된 API 래퍼
- 단일 의류 합성을 위한 `/tryon` 엔드포인트
- 상의+하의 합성을 위한 `/tryon/outfit` 엔드포인트
- outfit 합성 전략을 비교하기 위한 별도 Gradio 검증 데모
- 모델 상태, preload, unload, queue, mask, DensePose, diffusion 단계 로그

원본 `demo.py`, `gradio_demo/`, 학습 스크립트, 추론 스크립트, citation/license 정보는 추적 가능성을 위해 유지합니다.

## 레포지토리 구조

이 포크에서 작업하거나 새로 추가한 주요 파일은 다음과 같습니다.

```text
IDM-VTON/
|-- demo.py                         # 원본 Gradio 데모의 로컬 실행용 조정본
|-- demo_outfit.py                  # 상의/하의 outfit 검증용 Gradio 데모
|-- idm_vton_api/
|   |-- pipeline.py                 # UI와 분리된 IDM-VTON 서비스 래퍼
|   |-- server.py                   # FastAPI 엔드포인트
|-- scripts/
|   |-- download_demo_checkpoints.py
|   |-- check_install.py
|-- environment.idm-vton-aarch64.yaml
|-- LOCAL_DEMO.md
```

다음 파일과 디렉토리는 Git에 포함하지 않습니다.

```text
.hf_cache/
ckpt/
api_outputs/
*.pth
*.pkl
*.onnx
*.safetensors
*.bin
```

다운로드한 체크포인트는 커밋하지 않습니다. GitHub는 100 MB 초과 파일을 거부하므로, 체크포인트는 각 실행 환경에서 helper script로 복원하는 방식으로 관리합니다.

## 환경 구성

원본 `environment.yaml`은 `linux-64`, Python 3.10, PyTorch 2.0.1, CUDA 11.8 기준입니다. 이 포크는 `linux-aarch64`와 NVIDIA GB10/CUDA 13.0 환경에서 준비했기 때문에 별도 환경 파일을 사용합니다.

```bash
conda env create -f environment.idm-vton-aarch64.yaml
conda activate idm-vton
```

이미 환경이 만들어져 있다면 다음만 실행합니다.

```bash
conda activate idm-vton
```

Hugging Face 캐시는 프로젝트 내부에 두는 것을 기준으로 합니다.

```bash
export HF_HOME=$PWD/.hf_cache
```

## 체크포인트

데모 실행에 필요한 전처리 체크포인트는 다음 명령으로 다운로드합니다.

```bash
python scripts/download_demo_checkpoints.py
```

로컬 체크포인트 구조는 다음과 같아야 합니다.

```text
ckpt/
|-- densepose/
|   |-- model_final_162be9.pkl
|-- humanparsing/
|   |-- parsing_atr.onnx
|   |-- parsing_lip.onnx
|-- openpose/
|   |-- ckpts/
|       |-- body_pose_model.pth
|-- ip_adapter/
|   |-- ip-adapter-plus_sdxl_vit-h.bin
|-- image_encoder/
```

IDM-VTON 본 모델은 기본적으로 Hugging Face에서 로드합니다.

```text
yisol/IDM-VTON
```

로컬 모델 경로를 사용하려면 다음 환경변수를 지정합니다.

```bash
export IDM_VTON_MODEL=/path/to/local/model
```

## 설치 검증

```bash
python scripts/check_install.py
```

검증 항목은 다음을 포함합니다.

- CUDA 사용 가능 여부
- PyTorch/torchvision import 가능 여부
- 현재 플랫폼용 `gradio_demo/detectron2/_C...so` 존재 여부
- IDM-VTON 주요 모듈 import 가능 여부

## API 서버

API 서버 실행 명령은 다음과 같습니다.

```bash
conda activate idm-vton
export HF_HOME=$PWD/.hf_cache
python -m uvicorn idm_vton_api.server:app --host 127.0.0.1 --port 7861
```

상태 확인:

```bash
curl http://127.0.0.1:7861/health
```

### 단일 의류 합성 API

`/tryon`은 인물 사진 1장과 의류 이미지 1장을 입력받아 선택한 카테고리의 의류를 합성합니다.

```bash
curl -X POST http://127.0.0.1:7861/tryon \
  -F person_image=@path/to/person.jpg \
  -F garment_image=@path/to/garment.jpg \
  -F category=upper_body \
  -F garment_description="short sleeve shirt" \
  -F denoise_steps=20 \
  -F seed=42
```

지원 카테고리:

```text
upper_body
lower_body
dresses
```

### 상의+하의 합성 API

`/tryon/outfit`은 인물 사진 1장, 상의 이미지 1장, 하의 이미지 1장을 입력받아 outfit 합성 결과를 생성합니다.

```bash
curl -X POST http://127.0.0.1:7861/tryon/outfit \
  -F person_image=@path/to/person.jpg \
  -F upper_image=@path/to/upper.jpg \
  -F lower_image=@path/to/lower.jpg \
  -F upper_description="long sleeve knit" \
  -F lower_description="white pleated skirt" \
  -F denoise_steps=20 \
  -F seed=42
```

현재 API 구현은 상의를 먼저 합성한 뒤, 그 상의 합성 결과 위에 하의를 합성하는 방식입니다. 이 경로는 상위 시스템에서 IDM-VTON을 호출하기 위한 기본 통합 인터페이스로 사용합니다.

결과 파일은 다음 경로 아래에 저장됩니다.

```text
api_outputs/
```

API 응답에서는 다음 경로 형태로 결과 파일 URL을 반환합니다.

```text
/outputs/...
```

## 검증용 Gradio 데모

검증용 데모는 원본 데모와 별도로 동작합니다. 단일 카테고리 합성과 상의/하의 outfit 합성 전략을 비교하기 위해 사용합니다.

```bash
conda activate idm-vton
export HF_HOME=$PWD/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python demo_outfit.py --host 127.0.0.1 --port 7862
```

접속 주소:

```text
http://127.0.0.1:7862
```

Outfit 데모는 다음 전략을 지원합니다.

- `upper_then_lower`: 상의를 먼저 합성한 뒤, 상의 합성 결과 위에 하의를 합성합니다. 일반 검증에서 가장 안정적이어서 현재 기본값으로 사용합니다.
- `lower_then_upper`: 하의를 먼저 합성한 뒤, 하의 합성 결과 위에 상의를 합성합니다.
- `parallel_composite`: 원본 인물 사진 기준으로 상의와 하의를 각각 합성한 뒤, 하의 mask로 두 결과를 합성합니다.
- `fixed_original_masks`: 원본 인물 사진에서 상의/하의 mask와 DensePose를 먼저 캡처하고, 이 조건을 두 pass에 재사용한 뒤 합성합니다.

추가 제어 항목은 다음과 같습니다.

- `Model status`: 모델이 로드되어 있는지, 아직 lazy load 상태인지 표시합니다.
- `Preload model`: 첫 try-on 요청 전에 모델을 미리 로드합니다.
- `Unload model`: 로드된 모델 참조를 해제하고 CUDA cache를 정리합니다.
- `Lower mask expansion`: 합성에 사용할 하의 mask를 확장합니다.
- `Lower mask refinement`: 스커트 계열 하의에 사용할 수 있는 `skirt_fill` 보정을 제공합니다.

## 원본 Gradio 데모

원본 상의 중심 Gradio 데모도 유지되어 있습니다.

```bash
conda activate idm-vton
export HF_HOME=$PWD/.hf_cache
python demo.py --host 127.0.0.1 --port 7860
```

접속 주소:

```text
http://127.0.0.1:7860
```

## 원본 데이터셋 및 학습/추론 메모

원본 프로젝트는 VITON-HD와 DressCode 워크플로우를 지원합니다. 아래 내용은 원본 연구 학습/데이터셋 추론 경로를 재현해야 하는 경우를 위해 유지합니다.

### VITON-HD

[VITON-HD](https://github.com/shadow2496/VITON-HD)에서 데이터셋을 다운로드합니다.

예상 디렉토리 구조:

```text
train/
|-- image/
|-- image-densepose/
|-- agnostic-mask/
|-- cloth/
|-- vitonhd_train_tagged.json

test/
|-- image/
|-- image-densepose/
|-- agnostic-mask/
|-- cloth/
|-- vitonhd_test_tagged.json
```

### DressCode

[DressCode](https://github.com/aimagelab/dress-code)에서 데이터셋을 다운로드합니다.

예상 디렉토리 구조:

```text
DressCode/
|-- dresses/
|   |-- images/
|   |-- image-densepose/
|   |-- dc_caption.txt
|-- lower_body/
|   |-- images/
|   |-- image-densepose/
|   |-- dc_caption.txt
|-- upper_body/
|   |-- images/
|   |-- image-densepose/
|   |-- dc_caption.txt
```

DressCode 추론은 다음 카테고리를 받습니다.

```text
--category upper_body
--category lower_body
--category dresses
```

### 학습

원본 학습 경로는 `train_xl.py`와 `train_xl.sh`를 사용합니다.

```bash
accelerate launch train_xl.py \
  --gradient_checkpointing --use_8bit_adam \
  --output_dir=result \
  --train_batch_size=6 \
  --data_dir=DATA_DIR
```

### 데이터셋 추론

VITON-HD:

```bash
accelerate launch inference.py \
  --width 768 --height 1024 \
  --num_inference_steps 30 \
  --output_dir result \
  --unpaired \
  --data_dir DATA_DIR \
  --seed 42 \
  --test_batch_size 2 \
  --guidance_scale 2.0
```

DressCode:

```bash
accelerate launch inference_dc.py \
  --width 768 --height 1024 \
  --num_inference_steps 30 \
  --output_dir result \
  --unpaired \
  --data_dir DATA_DIR \
  --seed 42 \
  --test_batch_size 2 \
  --guidance_scale 2.0 \
  --category upper_body
```

## 디버깅

자주 사용하는 확인 명령은 다음과 같습니다.

```bash
ss -ltnp sport = :7860
ss -ltnp sport = :7861
ss -ltnp sport = :7862
nvidia-smi
tail -f /tmp/idm_vton_outfit_demo_7862.log
```

Gradio `/info` 문제는 다음 명령으로 확인합니다.

```bash
curl -I http://127.0.0.1:7862
curl -s http://127.0.0.1:7862/info
```

## Upstream Acknowledgements

This fork keeps the upstream IDM-VTON credits:

- [ZeroGPU](https://huggingface.co/zero-gpu-explorers)
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
- [OOTDiffusion](https://github.com/levihsu/OOTDiffusion)
- [DCI-VTON](https://github.com/bcmi/DCI-VTON-Virtual-Try-On)
- [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing)
- [DensePose](https://github.com/facebookresearch/DensePose)

## Citation

```bibtex
@article{choi2024improving,
  title={Improving Diffusion Models for Authentic Virtual Try-on in the Wild},
  author={Choi, Yisol and Kwak, Sangkyung and Lee, Kyungmin and Choi, Hyungwon and Shin, Jinwoo},
  journal={arXiv preprint arXiv:2403.05139},
  year={2024}
}
```

## License

The original code and checkpoints are under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). This fork does not change the upstream license terms.
