# Weave TTS

Text-to-speech experimentation using Qwen3-TTS models with Weave observability.

## Features

- **Voice Design**: Generate speech with custom voice characteristics from a natural-language description
- **Voice Cloning**: Clone voices from a reference audio clip
- **Marimo UI**: One interactive app that exposes both flows side-by-side
- **Experiment Tracking**: Every generation is wrapped with `@weave.op` and shows up as a trace in your Weave project. This is mandatory so we will require an output project and account as environment variables.

## Requirements

- Python >=3.12
- HuggingFace account with access to Qwen3-TTS models
- OpenAI API key (for `gpt-4o-transcribe` in voice cloning)
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

3. Copy `.env_example` to `.env` and add your API keys:
```bash
cp .env_example .env
```

Required environment variables:
- `HUGGINGFACEHUB_API_TOKEN` — HuggingFace API token (must have access to the Qwen3-TTS models)
- `OPENAI_API_KEY` — OpenAI API key
- `WANDB_ACCOUNT` — your Weights & Biases account name
- `WEAVE_PROJECT` — your Weave project name

## Usage

### Marimo app (recommended)

`app.py` is a single-file [marimo](https://marimo.io) notebook that wraps both flows into one UI. Launch it from the repo root:

```bash
# Interactive editor (lets you change cells, inspect state, etc.)
uv run marimo edit app.py

# Read-only "app" view (no cell editing)
uv run marimo run app.py
```

The first generation in each tab triggers a one-time download/load of the corresponding 1.7B Qwen3-TTS model (slow); every subsequent generation in the same session is fast.

#### Voice Design tab

Type the target text and a natural-language voice description, pick a language, and click **Generate voice design**. Output is saved to `audio/designed_audio/generated_audio.wav` and played inline.

![Voice Design tab](docs/screenshots/voice_design.png)

#### Voice Cloning tab

Drag a reference audio clip into the upload area, type the target text, and click **Clone voice**. By default the reference clip is transcribed automatically with `gpt-4o-transcribe`; uncheck the box to paste your own transcript instead. Output is saved to `audio/cloned_audio/generated_cloned_audio.wav` and played inline.

![Voice Cloning tab](docs/screenshots/voice_cloning.png)

### Original notebooks

The two Jupyter notebooks the marimo app was built from are still in the repo for reference and ad-hoc experimentation:

- `voice_design_weave.ipynb` — voice design with the `VoiceDesign` model
- `voice_clone_weave.ipynb` — voice cloning with the `Base` model

They share the same `@weave.op` function names as the marimo app, so traces from notebooks and the app are interchangeable in the Weave UI.

## Models

The project uses two Qwen3-TTS model types:
- **Base (Voice Clone)**: 1.7B parameter model for voice cloning
- **VoiceDesign**: 1.7B parameter model for text-based voice generation

Models are automatically downloaded from HuggingFace on first use.

## Weave Project
To see the Weave project to see traces check this [project](https://wandb.ai/wandb-smle/jb_qwen_tts/weave/traces).
