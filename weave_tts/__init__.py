"""Shared utilities and Weave Model classes for the weave_tts repo.

Public surface (the marimo app and external scripts can rely on these names):
    - VoiceDesignModel, VoiceCloneModel: Weave Model subclasses with lazy weight loading
    - run_design, run_clone: marimo work-cell bodies
    - normalize_audio, write_sound_to_file, transcribe_audio_from_file: audio helpers

The three notebooks (`voice_design_weave.ipynb`, `voice_clone_weave.ipynb`,
`qwen_tts_weave.ipynb`) keep their own copies of the model classes for
self-contained demos and may drift from this module over time.
"""

from .audio import normalize_audio, transcribe_audio_from_file, write_sound_to_file
from .config import init_weave_from_env
from .models import VoiceCloneModel, VoiceDesignModel
from .work import run_clone, run_design

__all__ = [
    "VoiceCloneModel",
    "VoiceDesignModel",
    "init_weave_from_env",
    "normalize_audio",
    "run_clone",
    "run_design",
    "transcribe_audio_from_file",
    "write_sound_to_file",
]
