"""Marimo work-cell bodies — invoked from app.py's gen cells.

Each function takes the relevant widgets (and the Weave Model instance) and
returns a single marimo `Html` element ready to be slotted into the panel
`mo.vstack`. Encapsulates the run-button gate, spinner, error handling, and
the success widget construction so the marimo cells stay one-liners.
"""

from __future__ import annotations

import marimo as mo

from .models import VoiceCloneModel, VoiceDesignModel


def run_design(
    run_button,
    text_widget,
    language_widget,
    instruct_widget,
    model: VoiceDesignModel,
):
    """Voice Design work-cell body. Returns the output widget for the panel."""
    if not run_button.value:
        return mo.md("")

    try:
        with mo.status.spinner(
            title="Generating designed audio",
            subtitle="Loading model + synthesizing speech...",
        ):
            result = model.predict(
                text=text_widget.value,
                language=language_widget.value,
                voice_description=instruct_widget.value,
            )
    except Exception as e:
        return mo.callout(mo.md(f"**Error:** {type(e).__name__}: {e}"), kind="danger")

    if isinstance(result, tuple) and result[0] is None:
        return mo.callout(mo.md(f"**{result[1]}**"), kind="danger")

    return mo.vstack(
        [
            mo.md("### 🎧 Generated audio"),
            mo.audio("audio/designed_audio/generated_audio.wav"),
            mo.md("_Saved to `audio/designed_audio/generated_audio.wav`._"),
        ]
    )


def run_clone(
    run_button,
    ref_path: str | None,
    target_widget,
    max_tokens_widget,
    model: VoiceCloneModel,
    ref_text: str | None = None,
):
    """Voice Clone work-cell body. `ref_text=None` triggers auto-transcribe inside
    predict; a non-None string is used as the manual reference transcript."""
    if not run_button.value:
        return mo.md("")
    if not ref_path:
        return mo.callout(
            mo.md("**Upload a reference audio clip first.**"), kind="warn"
        )

    try:
        subtitle = (
            "Auto-transcribing + loading model + synthesizing..."
            if ref_text is None
            else "Loading model + synthesizing..."
        )
        with mo.status.spinner(title="Cloning voice", subtitle=subtitle):
            result = model.predict(
                input_audio=ref_path,
                target_text=target_widget.value,
                max_new_tokens=int(max_tokens_widget.value),
                ref_text=ref_text,
            )
    except Exception as e:
        return mo.callout(mo.md(f"**Error:** {type(e).__name__}: {e}"), kind="danger")

    if isinstance(result, tuple) and result[0] is None:
        return mo.callout(mo.md(f"**{result[1]}**"), kind="danger")

    return mo.vstack(
        [
            mo.md("### 🎧 Cloned audio"),
            mo.audio("audio/cloned_audio/generated_cloned_audio.wav"),
            mo.md("_Saved to `audio/cloned_audio/generated_cloned_audio.wav`._"),
        ]
    )
