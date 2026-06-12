# IDM-VTON API 명세서 (7862)

## 개요

현재 `127.0.0.1:7862`에는 `demo_outfit.py`가 아니라 FastAPI 기반 IDM-VTON API 서버가 실행된다.

```text
Base URL: http://127.0.0.1:7862
Server: idm_vton_api.server:app
Runtime env: IDM-VTON/.conda/idm-vton
Output root: IDM-VTON/api_outputs
```

실행 명령 예시:

```bash
cd /home/eslab/Team25_generative_model/IDM-VTON
PYTHONNOUSERSITE=1 \
HF_HOME=/home/eslab/Team25_generative_model/IDM-VTON/.hf_cache \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/eslab/Team25_generative_model/IDM-VTON/.conda/idm-vton/bin/python \
  -m uvicorn idm_vton_api.server:app --host 127.0.0.1 --port 7862
```

## 공통 동작

- 입력 형식은 `multipart/form-data`다.
- 이미지 입력은 `UploadFile`로 전달한다.
- 출력 이미지는 서버 로컬의 `IDM-VTON/api_outputs/` 아래에 저장된다.
- 응답에는 결과 파일을 조회할 수 있는 `/outputs/...` URL이 포함된다.
- 첫 추론 요청에서 모델이 lazy load된다. 첫 요청은 모델 로딩 때문에 오래 걸릴 수 있다.
- 현재 `/tryon/outfit`은 `sequential_upper_then_lower` 전략만 사용한다.
- `denoise_steps` 값을 낮추면 빠르지만 품질이 낮아질 수 있다.

## GET /health

서버 상태와 모델 로드 여부를 확인한다.

### 요청

```bash
curl http://127.0.0.1:7862/health
```

### 응답 예시

```json
{
  "status": "ok",
  "model_loaded": true,
  "output_root": "/home/eslab/Team25_generative_model/IDM-VTON/api_outputs"
}
```

### 필드

| 필드 | 타입 | 설명 |
|---|---:|---|
| `status` | string | 서버 상태. 정상일 때 `ok` |
| `model_loaded` | boolean | IDM-VTON 모델 객체가 현재 프로세스에 로드되어 있는지 여부 |
| `output_root` | string | 결과 파일 저장 루트 |

## POST /tryon

인물 사진 1장과 의류 이미지 1장을 입력받아 단일 의류 try-on을 수행한다.

### 요청 필드

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---:|---|---|
| `person_image` | file | 예 | 없음 | 원본 인물 이미지 |
| `garment_image` | file | 예 | 없음 | 합성할 의류 이미지 |
| `category` | string | 아니오 | `upper_body` | `upper_body`, `lower_body`, `dresses` 중 하나 |
| `garment_description` | string | 아니오 | `clothing` | 의류 텍스트 설명 |
| `auto_crop` | boolean | 아니오 | `false` | 인물 이미지를 3:4 비율로 자동 crop할지 여부 |
| `denoise_steps` | integer | 아니오 | `30` | diffusion step 수 |
| `seed` | integer | 아니오 | `42` | 난수 seed. 음수면 내부에서 random seed 사용 |

### 요청 예시

```bash
curl -X POST http://127.0.0.1:7862/tryon \
  -F person_image=@IDM-VTON/gradio_demo/example/human/00034_00.jpg \
  -F garment_image=@data/catalog/dataset_clothes/case28_female_ectomorph_warm_balanced/upper.jpg \
  -F category=upper_body \
  -F garment_description="upper garment" \
  -F denoise_steps=20 \
  -F seed=42
```

### 성공 응답 예시

```json
{
  "job_id": "1781250000_ab12cd34",
  "category": "upper_body",
  "seed": 42,
  "result_image_url": "/outputs/single/1781250000_ab12cd34/result.png",
  "mask_image_url": "/outputs/single/1781250000_ab12cd34/mask.png",
  "metadata_url": "/outputs/single/1781250000_ab12cd34/metadata.json"
}
```

### 응답 필드

| 필드 | 타입 | 설명 |
|---|---:|---|
| `job_id` | string | 결과 저장 디렉토리 이름 |
| `category` | string | 적용된 카테고리 |
| `seed` | integer | 실제 사용된 seed |
| `result_image_url` | string | 최종 결과 이미지 URL |
| `mask_image_url` | string | 사용된 mask 이미지 URL |
| `metadata_url` | string | metadata JSON URL |

## POST /tryon/outfit

인물 사진 1장, 상의 이미지 1장, 하의 이미지 1장을 입력받아 상의+하의 outfit try-on을 수행한다.

현재 구현은 상의를 먼저 합성하고, 그 결과 이미지 위에 하의를 합성한다.

```text
strategy = sequential_upper_then_lower
upper seed = seed
lower seed = seed + 1
```

### 요청 필드

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---:|---|---|
| `person_image` | file | 예 | 없음 | 원본 인물 이미지 |
| `upper_image` | file | 예 | 없음 | 합성할 상의 이미지 |
| `lower_image` | file | 예 | 없음 | 합성할 하의 이미지 |
| `upper_description` | string | 아니오 | `upper garment` | 상의 텍스트 설명 |
| `lower_description` | string | 아니오 | `lower garment` | 하의 텍스트 설명 |
| `auto_crop` | boolean | 아니오 | `false` | 인물 이미지를 3:4 비율로 자동 crop할지 여부 |
| `denoise_steps` | integer | 아니오 | `30` | 각 pass의 diffusion step 수 |
| `seed` | integer | 아니오 | `42` | 상의 pass seed. 하의 pass는 `seed + 1` 사용 |

### 요청 예시

```bash
curl -X POST http://127.0.0.1:7862/tryon/outfit \
  -F person_image=@IDM-VTON/gradio_demo/example/human/00034_00.jpg \
  -F upper_image=@data/catalog/dataset_clothes/case28_female_ectomorph_warm_balanced/upper.jpg \
  -F lower_image=@data/catalog/dataset_clothes/case28_female_ectomorph_warm_balanced/lower.jpg \
  -F upper_description="upper garment" \
  -F lower_description="lower garment" \
  -F denoise_steps=10 \
  -F seed=42
```

### 실제 검증 응답 예시

2026-06-12에 위 요청으로 HTTP 200을 확인했다.

```json
{
  "job_id": "1781254076_e50fabf7",
  "strategy": "sequential_upper_then_lower",
  "upper_seed": 42,
  "lower_seed": 43,
  "upper_result_url": "/outputs/outfit/1781254076_e50fabf7/upper_result.png",
  "final_result_url": "/outputs/outfit/1781254076_e50fabf7/final_result.png",
  "upper_mask_url": "/outputs/outfit/1781254076_e50fabf7/upper_mask.png",
  "lower_mask_url": "/outputs/outfit/1781254076_e50fabf7/lower_mask.png",
  "metadata_url": "/outputs/outfit/1781254076_e50fabf7/metadata.json"
}
```

### 응답 필드

| 필드 | 타입 | 설명 |
|---|---:|---|
| `job_id` | string | 결과 저장 디렉토리 이름 |
| `strategy` | string | 현재는 `sequential_upper_then_lower` |
| `upper_seed` | integer | 상의 pass에 사용된 seed |
| `lower_seed` | integer | 하의 pass에 사용된 seed |
| `upper_result_url` | string | 상의 합성 중간 결과 URL |
| `final_result_url` | string | 상의+하의 최종 결과 URL |
| `upper_mask_url` | string | 상의 pass mask URL |
| `lower_mask_url` | string | 하의 pass mask URL |
| `metadata_url` | string | metadata JSON URL |

## GET /outputs/...

`/tryon`, `/tryon/outfit` 응답에 포함된 결과 파일 URL을 조회한다.

### 예시

```bash
curl -O http://127.0.0.1:7862/outputs/outfit/1781254076_e50fabf7/final_result.png
```

## 오류 응답

### 400 Bad Request

입력 파일이 비어 있거나 이미지로 열 수 없을 때 발생한다.

예시:

```json
{
  "detail": "person_image is not a valid image"
}
```

### 500 Internal Server Error

모델 로딩, CUDA OOM, DensePose/OpenPose/human parsing, diffusion 실행 중 예외가 발생하면 반환된다.

예시:

```json
{
  "detail": "CUDA out of memory..."
}
```

## 운영 메모

- 7862 API 서버와 7860 Gradio 검증 데모는 별도 프로세스다.
- 7862 API 서버는 현재 모델이 로드된 상태일 수 있으며 GPU 메모리를 크게 점유한다.
- 7860 `demo_outfit.py`는 접속만으로는 모델을 로드하지 않는다. 버튼을 눌러 추론을 시작하면 별도 프로세스에서 모델을 로드한다.
- 두 프로세스가 동시에 모델을 로드하면 GPU 메모리 부족이 발생할 수 있다.

## 현재 실행 상태 확인 명령

```bash
curl http://127.0.0.1:7862/health
ss -ltnp sport = :7862
nvidia-smi
```
