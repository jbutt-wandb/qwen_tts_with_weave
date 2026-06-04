"""Weave Model classes for Qwen3-TTS voice design and voice cloning.

Canonical source of truth — the notebooks keep their own copies for self-contained demos.
"""

from __future__ import annotations

import hashlib
import os
import wave
from pathlib import Path
from typing import Any

import torch
import weave
from huggingface_hub import snapshot_download
from pydantic import PrivateAttr
from qwen_tts import Qwen3TTSModel
from weave import Model

from .audio import transcribe_audio_from_file, write_sound_to_file


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

    Uses a single-slot cache keyed by sha256 of the reference-audio bytes:
    consecutive clones of the same reference skip both Whisper transcription
    and speaker-embedding encoding. Content-hashed (not path-keyed) because
    the marimo upload flow overwrites a stable path (`audio/input/_uploaded.{ext}`)
    on every upload — path alone can't distinguish references.

    `predict` accepts an optional `ref_text`: when None (default), the reference
    is auto-transcribed; when a string is supplied, that's the reference transcript
    directly.
    """

    model_size: str = "1.7B"
    model_type: str = "Base"

    _qwen_model: Any = PrivateAttr(default=None)

    _cache_key: Any = PrivateAttr(default=None)         # (content_sha256, ref_text_arg)
    _cached_transcript: Any = PrivateAttr(default=None) # str — auto-transcribed or override
    _cached_prompt: Any = PrivateAttr(default=None)     # list of prompt_items

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

    def _audio_content_key(self, input_audio: str) -> str:
        """Fingerprint the reference audio by hashing its raw bytes."""
        return hashlib.sha256(Path(input_audio).read_bytes()).hexdigest()

    @weave.op(
        call_display_name="build_voice_clone_prompt",
        postprocess_inputs=open_input_audio_for_trace,
    )
    def _build_voice_clone_prompt(self, input_audio: str, ref_text: str):
        """Encode the reference clip into prompt_items. Its own @weave.op so
        cache hits/misses are visible in the trace tree."""
        return self._ensure_loaded().create_voice_clone_prompt(
            ref_audio=input_audio,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )

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
        **kwargs,
    ):
        if not target_text or not target_text.strip():
            return None, "Error: Target text is required."

        content_hash = self._audio_content_key(input_audio)
        cache_key = (content_hash, ref_text)

        if self._cache_key == cache_key:
            print(
                f"[VoiceClone] Cache HIT for {input_audio!r} "
                f"(content={content_hash[:12]}…, ref_text={'override' if ref_text else 'auto'}). "
                "Reusing cached transcript + voice-clone prompt."
            )
            transcript = self._cached_transcript
            prompt = self._cached_prompt
        else:
            print(
                f"[VoiceClone] Cache MISS for {input_audio!r} "
                f"(content={content_hash[:12]}…, ref_text={'override' if ref_text else 'auto'})."
            )
            if ref_text is None:
                print("[VoiceClone]   → Transcribing reference audio with Whisper…")
                transcript = transcribe_audio_from_file(input_audio).strip()
            else:
                print("[VoiceClone]   → Using caller-supplied reference transcript.")
                transcript = ref_text.strip()
            print("[VoiceClone]   → Building voice-clone prompt (encoding speaker embedding)…")
            prompt = self._build_voice_clone_prompt(input_audio, transcript)
            self._cache_key = cache_key
            self._cached_transcript = transcript
            self._cached_prompt = prompt
            print("[VoiceClone]   → Cache populated.")

        print("[VoiceClone] Generating target audio…")
        try:
            wavs, sr = self._ensure_loaded().generate_voice_clone(
                text=[target_text.strip()],
                language=["Auto"],
                voice_clone_prompt=prompt,
                x_vector_only_mode=[False],
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

        out_dir = Path("./audio/cloned_audio/")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_filename = out_dir / "generated_cloned_audio.wav"
        write_sound_to_file(output_filename, wavs[0], sr)
        print(f"[VoiceClone] Generation complete → {output_filename}")

        return {
            "generated_audio": wave.open(str(output_filename), "rb"),
            "sample_rate": sr,
            "audio_array": wavs[0],
        }
