"""Audio I/O helpers and Whisper transcription, shared across notebooks and the marimo app."""

from __future__ import annotations

import datetime

import librosa
import numpy as np
import soundfile as sf
import weave
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def normalize_audio(x, eps: float = 1e-12, clip: bool = True):
    """Normalize audio to float32 in [-1, 1]. Mono-mixes multi-channel input."""
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return y


def write_sound_to_file(output_filename, audio_file: np.ndarray, sample_rate: int) -> None:
    sf.write(output_filename, audio_file, sample_rate, subtype="PCM_16")


@weave.op(
    call_display_name=lambda call: f"{call.func_name}-{int(datetime.datetime.now().timestamp())}"
)
def transcribe_audio_from_file(filename: str, model_name: str = "openai/whisper-small") -> str:
    """Local HuggingFace Whisper transcription (no OpenAI API dependency)."""
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None

    audio, sr = librosa.load(filename, sr=16000)
    input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features

    predicted_ids = model.generate(language="en", input_features=input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]
