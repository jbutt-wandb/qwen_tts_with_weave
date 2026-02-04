# Weave TTS

Text-to-speech experimentation using Qwen3-TTS models with Weave observability.

## Features

- **Voice Cloning**: Clone voices from reference audio samples
- **Voice Design**: Generate speech with custom voice characteristics using text descriptions
- **Experiment Tracking**: Track TTS experiments and outputs with Weave

## Requirements

- Python >=3.12
- HuggingFace account with access to Qwen3-TTS models
- OpenAI API key (for audio transcription in voice cloning)
- Weights & Biases account (for Weave)

## Setup

1. Install dependencies using uv:
```bash
uv sync
```

2. Install the qwen-tts library via pip:
```bash
uv pip install -U qwen-tts
```

2. Copy `.env_example` to `.env` and add your API keys:
```bash
cp .env_example .env
```

Required environment variables:
- `HUGGINGFACEHUB_API_TOKEN`: HuggingFace API token
- `OPENAI_API_KEY`: OpenAI API key
- `WANDB_ACCOUNT`: Your Weights & Biases account name
- `WEAVE_PROJECT`: Your Weave project name

## Usage

### Voice Cloning

Open `voice_clone_weave.ipynb` to clone voices from reference audio. Place reference audio files in `audio/input/` and run the notebook to generate cloned speech with different text.

### Voice Design

Open `voice_design_weave.ipynb` to generate speech with custom voice characteristics. Describe the desired voice (e.g., "playful, mature female voice") and the model will synthesize matching audio.

## Models

The project uses two Qwen3-TTS model types:
- **Base (Voice Clone)**: 1.7B parameter model for voice cloning
- **VoiceDesign**: 1.7B parameter model for text-based voice generation

Models are automatically downloaded from HuggingFace on first use.

## Weave Project
To see the Weave project to see traces check this [project](https://wandb.ai/wandb_emea/jb_qwen_tts_weave/weave/traces?view=traces_2026-02-02_21-56-32-535).