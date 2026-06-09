# IDM-VTON Local Demo Notes

## Environment summary

- Workdir: `/home/eslab/Team25_generative_model/IDM-VTON`
- Machine platform: `linux-aarch64`
- GPU: NVIDIA GB10, driver CUDA capability reported by `nvidia-smi` as CUDA 13.0
- Base Python before setup: Anaconda Python 3.13.9
- Isolated env created: `idm-vton`

The upstream `environment.yaml` targets `linux-64` with Python 3.10 and PyTorch 2.0.1 + CUDA 11.8. This machine is ARM64/GB10, so the local env uses Python 3.12 and PyTorch/torchvision CUDA 13 wheels instead. `torchaudio` was omitted because no matching ARM64 `2.12.0` wheel is available and the Gradio demo does not use it.

## One-time setup already done

```bash
cd /home/eslab/Team25_generative_model/IDM-VTON
conda activate idm-vton
```

The env is configured with `PYTHONNOUSERSITE=1` so it does not load packages from `~/.local`.

Demo preprocessing checkpoints are downloaded under `ckpt/`:

```bash
python scripts/download_demo_checkpoints.py
```

The vendored detectron2 extension was rebuilt for this ARM64/CUDA 13 machine:

```bash
env FORCE_CUDA=1 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.1 python gradio_demo/setup.py build_ext --inplace
```

Model cache is local to the project when launched with `HF_HOME=$PWD/.hf_cache`.

## Verify install

```bash
cd /home/eslab/Team25_generative_model/IDM-VTON
conda activate idm-vton
python scripts/check_install.py
```

Expected highlights:

- `torch 2.12.0+cu130`
- `torchvision 0.27.0+cu130`
- `cuda available: True`
- `gpu: NVIDIA GB10`
- `detectron2._C ... _C.cpython-312-aarch64-linux-gnu.so`

## Run demo

Local-only:

```bash
cd /home/eslab/Team25_generative_model/IDM-VTON
conda activate idm-vton
export HF_HOME=$PWD/.hf_cache
python demo.py --host 127.0.0.1 --port 7860
```

LAN/remote access:

```bash
python demo.py --host 0.0.0.0 --port 7860
```

Public Gradio share link:

```bash
python demo.py --host 127.0.0.1 --port 7860 --share
```

Open:

```text
http://127.0.0.1:7860
```

## UI flow

1. Upload/select a human image in the left panel.
2. Upload/select a garment image in the middle panel.
3. Enter a garment description, for example `short sleeve round neck t-shirt`.
4. Keep auto mask enabled for the standard demo path.
5. Click `Try-on`.

First generation may be slow because the pipeline moves components to GPU and runs OpenPose, human parsing, DensePose, and SDXL inference.
