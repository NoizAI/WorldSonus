# WorldSonus: Bringing Sound to Worlds

[![arXiv](https://img.shields.io/badge/arXiv-2610.08760-b31b1b?style=flat-square)](https://arxiv.org/abs/2610.08760)
[![Demo Page](https://img.shields.io/badge/Demo-Page-2563eb?style=flat-square)](https://noizai.github.io/WorldSonus/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue?style=flat-square)](https://huggingface.co/FF2416/WorldSonus)
[![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey?style=flat-square)](LICENSE)

**WorldSonus brings synchronized, controllable stereo sound to visual worlds.**
Our causal autoregressive diffusion model generates audio in 100 ms chunks,
combining fine-grained visual motion with text instructions that can change
during a sequence.

## Demo

https://github.com/user-attachments/assets/22400b47-b292-476a-89a1-eeaf33372ba2

## Overview

![WorldSonus architecture](docs/media/architecture.png)

## Installation

Requires Python 3.10+, a CUDA-capable GPU, FFmpeg and ffprobe. The setup script
installs PyTorch with CUDA 12.4 wheels; GPU drivers must already be installed.

```bash
git clone https://github.com/NoizAI/WorldSonus.git
cd WorldSonus
bash scripts/setup.sh .venv
source .venv/bin/activate
pip install -e '.[features]'
```

## Pretrained models

Weights are available on [Hugging Face](https://huggingface.co/FF2416/WorldSonus).

```bash
python scripts/download_model.py --output assets
python scripts/download_encoders.py --output assets
```

DINOv3 and T5Gemma 2 retain their upstream licenses and access conditions.
Accept their terms and authenticate with Hugging Face if required.

## Video-to-audio generation

Pass a video and an audio description directly. Feature extraction, audio
generation, and decoding run automatically:

```bash
python -m worldsonus.infer \
  --video input.mp4 \
  --prompt "Water splashes as someone runs through a shallow river. No speech." \
  --output output.wav
```

The output is a stereo 48 kHz WAV file. Use `--start 5 --seconds 10` to select
a window, or omit `--prompt` for video-only conditioning.

## Streaming generation

Add `--stream` to generate audio incrementally:

```bash
python -m worldsonus.infer \
  --video input.mp4 --prompt "A train passes beside a river." \
  --stream --output output.wav
```

For immediate playback with FFplay, stream raw PCM instead of waiting for a file:

```bash
python -m worldsonus.infer \
  --video input.mp4 --prompt "A train passes beside a river." \
  --stream --pcm-stdout \
  | ffplay -f f32le -ar 48000 -ac 2 -i - -nodisp -autoexit
```

The first run takes longer to load models and compile kernels. Use
`--no-compile` to disable compilation if needed.

## License

WorldSonus's original contributions are released under [CC BY-NC 4.0](LICENSE),
for non-commercial use with attribution. Third-party components retain their
own licenses; see [Third-party notices](THIRD_PARTY_NOTICES.md).

## Acknowledgments

We thank [DINOv3](https://github.com/facebookresearch/dinov3),
[T5Gemma](https://huggingface.co/google/t5gemma-2-270m-270m), and
[stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools)
for their open research and tools.
