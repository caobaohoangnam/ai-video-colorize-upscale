# AI Video Restoration Suite

Restore, colorize, and upscale old or low-quality video footage using AI — runs on consumer-grade GPUs (4GB VRAM).

Bundles three pipelines that run back-to-back on every frame: **denoising/deblurring** (self-trained model) → **black-and-white colorization** (auto-detected) → **super-resolution upscaling** (Real-ESRGAN). Includes scene-aware processing (night / fire / outdoor / animation) and temporal color smoothing to prevent flicker.

---

## Key Features

### 1. Fixing Degraded Video (Blur, Softness, Noise)
- A U-Net + residual-block network, trained from scratch (`models/network.py`), analyzes **5 consecutive frames at once** (temporal window) for more stable denoising than single-frame processing.
- Residual learning: the model only learns the "correction," which is added back to the original frame — preserving more detail and reducing over-smoothing.
- Automatically downsizes large frames before feeding them to the model to run safely on 4GB GPUs, then restores the original size afterward.
- Benchmarked with a dedicated evaluation suite (`evaluate.py`): **PSNR ≈ 24.3 dB, SSIM ≈ 0.83** on a test video set (see `evaluation_report.csv` and `evaluation_report_charts.png`).

### 2. Color Correction & Black-and-White Colorization
- Automatically detects black-and-white footage (average saturation sampling across multiple frames) and colorizes it using a colorization model (Zhang et al., run through OpenCV DNN — no GPU VRAM cost).
- Automatically skips colorization on animated content to preserve the original art style.
- **Scene-aware color grading**: detects night, fire/explosion, outdoor, and animation scenes and applies a distinct color adjustment profile to each (e.g. cool tones for night scenes, orange/red boost in bright areas for fire scenes).
- **Temporal color smoothing** (EMA on the A/B color channels in LAB space) to prevent color flicker between consecutive frames — automatically resets on scene cuts.
- Post-processing color correction: automatic white balance (Gray World), saturation boost, auto-contrast — all toggleable via config.

### 3. Resolution Upscaling
- Uses Real-ESRGAN (`realesr-general-x4v3` variant — lightweight, optimized for low-VRAM GPUs).
- Pick the target resolution directly: **720p, 1080p (Full HD), 1440p (2K), 2160p (4K)** — automatically recalculates width to preserve the original aspect ratio, and accepts input at any resolution (144p, 240p, 360p, 480p, 720p...).
- Always upscales at the maximum factor first, then precision-resizes to the target resolution, avoiding blur when jumping several quality tiers at once (e.g. 144p → 1440p).

### Also Included
- Automatic scene-cut detection to reset temporal buffers, preventing color/detail bleed from one scene into the next.
- Automatically re-attaches the original audio track after processing.
- All parameters are tunable through a single config file (`configs/config.yaml`) — no code changes required.
- Includes a training script (`train.py`) to retrain the denoising model on your own clean video dataset.

---

## System Requirements

- Python 3.10+
- NVIDIA GPU recommended (4GB VRAM minimum); CPU execution works but is much slower
- FFmpeg (used for audio muxing)
- See `requirements.txt` for full dependency details (PyTorch, OpenCV, Real-ESRGAN, BasicSR, PySceneDetect...)

## Installation

```bash
pip install -r requirements.txt
```

Model weights are **not included in this Git repo** (too large for GitHub's limits). You'll need to obtain them before running:
- `weights/video_restoration_net.pth` — self-trained denoising/deblurring model. **[Download here](https://github.com/caobaohoangnam/ai-video-colorize-upscale/releases/download/v1.0.0/video_restoration_net.pth)** (or the [best checkpoint](https://github.com/caobaohoangnam/ai-video-colorize-upscale/releases/download/v1.0.0/video_restoration_net_best.pth), trained for 150 epochs — see the [v1.0.0 release](https://github.com/caobaohoangnam/ai-video-colorize-upscale/releases/tag/v1.0.0)).
- `weights/realesr-general-x4v3.pth` — Real-ESRGAN super-resolution model. Download from [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases) (the `realesr-general-x4v3.pth` file).
- `weights/colorization/` — black-and-white colorization model, **downloaded automatically on first run** (no manual download needed).

## Usage

```bash
python main.py
```

The program lists videos found in `data/input/`, lets you pick one along with your desired output resolution. Results are saved to `data/output/`.

For deeper customization (disable colorization, adjust color smoothing strength, enable auto-contrast, change the temporal window size...), edit `configs/config.yaml` directly.

## Retraining the Denoising Model

Place high-quality video files in `data/train/clean/`, then run:

```bash
python train.py
```

All training parameters (batch size, learning rate, epoch count, loss weights...) live in `configs/config.yaml`.

## Quality Evaluation

```bash
python evaluate.py
```

Exports per-frame PSNR / SSIM / Delta-E to `evaluation_report.csv` along with a summary chart at `evaluation_report_charts.png`.

---

## Project Structure

```
main.py                  Main entry point — runs the full video restoration pipeline
train.py                 Retrains the denoising model
evaluate.py              Evaluates output quality (PSNR/SSIM/Delta-E)
configs/config.yaml      All configuration parameters
models/network.py        Denoising model architecture (VideoRestorationNet)
utils/
  video_utils.py         Video I/O, scene-cut detection, audio muxing
  upscale_utils.py        Real-ESRGAN integration
  colorize_utils.py        Black-and-white video colorization
  color_utils.py           Post-processing color correction + temporal smoothing
  scene_utils.py           Scene-type analysis and color hints
  dataset.py / augment.py  Training data preparation
weights/                 Model weights
data/                    Input/output videos, training data
```

## License & Third-Party Notices

This project uses three open-source pretrained components: Real-ESRGAN (BSD-3-Clause), BasicSR (Apache 2.0), and the Zhang et al. colorization model (BSD-2-Clause). All three permit commercial use, provided the original copyright notices are retained upon redistribution. Full license texts are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) — include this file when selling or distributing the product.
