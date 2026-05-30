"""Weave Model classes for Qwen3-TTS voice design and voice cloning.

Canonical source of truth — the notebooks keep their own copies for self-contained demos.
"""

from __future__ import annotations

import datetime
import os
import wave
from pathlib import Path
from typing import Any

import librosa
import torch
import weave
from huggingface_hub import snapshot_download
from pydantic import PrivateAttr
from qwen_tts import Qwen3TTSModel
from weave import Model

from .audio import normalize_audio, transcribe_audio_from_file, write_sound_to_file


def drop_audio_array(output):
    """Hide audio_array from the Weave trace; runtime return value is unchanged."""
    if isinstance(output, dict) and "audio_array" in output:
        return {k: v for k, v in output.items() if k != "audio_array"}
    return output


def open_input_audio_for_trace(inputs):
    """Replace input_audio string path with an opened wave file so Weave renders
    it as a playable audio widget in the trace."""
    path = inputs.get("input_audio")
    if isinstance(path, str):
        try:
            return {**inputs, "input_audio": wave.open(path, "rb")}
        except Exception:
            return inputs
    return inputs


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")


class VoiceDesignModel(Model):
    """Qwen3-TTS VoiceDesign head wrapped as a Weave Model."""

    model_size: str = "1.7B"
    model_type: str = "VoiceDesign"

    _qwen_model: Any = PrivateAttr(default=None)

    def _get_model_path(self, model_type: str, model_size: str) -> str:
        return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")

    def _ensure_loaded(self):
        if self._qwen_model is None:
            print(
                f"Loading Qwen3-TTS-12Hz-{self.model_size}-{self.model_type} (first call only)..."
            )
            self._qwen_model = Qwen3TTSModel.from_pretrained(
                self._get_model_path(self.model_type, self.model_size),
                device_map=_device(),
                dtype=torch.bfloat16,
                token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
            )
        return self._qwen_model

    @weave.op(
        call_display_name="VoiceDesign",
        postprocess_output=drop_audio_array,
    )
    def predict(self, text: str, language: str, voice_description: str):
        if not text or not text.strip():
            return None, "Error: Text is required."
        if not voice_description or not voice_description.strip():
            return None, "Error: Voice description is required."

        try:
            wavs, sr = self._ensure_loaded().generate_voice_design(
                text=text.strip(),
                language=language,
                instruct=voice_description.strip(),
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
            out_dir = Path("./audio/designed_audio/")
            out_dir.mkdir(parents=True, exist_ok=True)
            output_filename = out_dir / "generated_audio.wav"
            write_sound_to_file(output_filename, wavs[0], sr)
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

        return {
            "audio": wave.open(str(output_filename), "rb"),
            "sample_rate": sr,
            "audio_array": wavs[0],
        }


class VoiceCloneModel(Model):
    """Qwen3-TTS Base head (voice cloning) wrapped as a Weave Model.

    Differs from the notebook copy by exposing an optional `ref_text` parameter
    on `predict`: when None (the default), the reference is auto-transcribed via
    `transcribe_audio_from_file`; when a string is supplied, that's used as the
    reference transcript directly (no transcribe op fires).
    """

    model_size: str = "1.7B"
    model_type: str = "Base"

    _qwen_model: Any = PrivateAttr(default=None)

    def _get_model_path(self, model_type: str, model_size: str) -> str:
        return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")

    def _ensure_loaded(self):
        if self._qwen_model is None:
            print(
                f"Loading Qwen3-TTS-12Hz-{self.model_size}-{self.model_type} (first call only)..."
            )
            self._qwen_model = Qwen3TTSModel.from_pretrained(
                self._get_model_path(self.model_type, self.model_size),
                device_map=_device(),
                dtype=torch.bfloat16,
                token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
            )
        return self._qwen_model

    def generate_voice_clone(self, ref_audio, ref_text, target_text, max_new_tokens=2048):
        if not target_text or not target_text.strip():
            return None, "Error: Target text is required."
        if ref_audio is None:
            return None, "Error: Reference audio is required."
        if not ref_text or not ref_text.strip():
            return None, "Error: Reference text is required."

        try:
            wavs, sr = self._ensure_loaded().generate_voice_clone(
                text=target_text.strip(),
                language="Auto",
                ref_audio=ref_audio,
                ref_text=ref_text.strip(),
                x_vector_only_mode=False,
                max_new_tokens=max_new_tokens,
            )
            out_dir = Path("./audio/cloned_audio/")
            out_dir.mkdir(parents=True, exist_ok=True)
            output_filename = out_dir / "generated_cloned_audio.wav"
            write_sound_to_file(output_filename, wavs[0], sr)
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

        return {
            "generated_audio": wave.open(str(output_filename), "rb"),
            "sample_rate": sr,
            "audio_array": wavs[0],
        }

    @weave.op(
        call_display_name="VoiceClone",
        postprocess_inputs=open_input_audio_for_trace,
        postprocess_output=drop_audio_array,
    )
    def predict(
        self,
        input_audio: str,
        target_text: str,
        max_new_tokens: int = 2048,
        ref_text: str | None = None,
    ):
        # Auto-transcribe only when no override is supplied (nested @weave.op call).
        if ref_text is None:
            ref_text = transcribe_audio_from_file(input_audio).strip()

        audio_data, sample_rate = librosa.load(input_audio, sr=None)
        ref_audio = (normalize_audio(audio_data), sample_rate)

        return self.generate_voice_clone(
            ref_audio=ref_audio,
            ref_text=ref_text,
            target_text=target_text,
            max_new_tokens=max_new_tokens,
        )
