import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    from pathlib import Path

    from dotenv import load_dotenv
    from huggingface_hub import login

    from weave_tts import (
        VoiceCloneModel,
        VoiceDesignModel,
        init_weave_from_env,
        run_clone,
        run_design,
    )

    load_dotenv(override=True)

    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if hf_token:
        login(token=hf_token)

    weave_enabled = init_weave_from_env()
    wandb_account = os.getenv("WANDB_ACCOUNT")
    weave_project = os.getenv("WEAVE_PROJECT")

    # Lazy-loaded inside each class; instantiation is cheap.
    voice_design = VoiceDesignModel()
    voice_clone = VoiceCloneModel()

    return (
        Path,
        run_clone,
        run_design,
        voice_clone,
        voice_design,
        wandb_account,
        weave_enabled,
        weave_project,
    )


@app.cell
def _(mo):
    mo.md("""
# Qwen TTS

Voice design and voice cloning powered by Qwen3-TTS. Every generation is wrapped with
`@weave.op`, so each run shows up as a trace in your Weave project alongside the notebook history.
""")
    return


# ─── Voice Design form ───────────────────────────────────────────────────────


@app.cell
def _(mo):
    design_text = mo.ui.text_area(
        value="You wouldn't suspect little old me, would you?",
        label="Target text",
        rows=2,
        full_width=True,
    )
    design_instruct = mo.ui.text_area(
        value=(
            "Use a playful, seductive and mature female voice. "
            "The vocal signature should be female but deeper."
        ),
        label="Voice description",
        rows=3,
        full_width=True,
    )
    design_language = mo.ui.dropdown(
        options=["Auto", "en", "zh", "ja", "ko", "fr", "de", "es", "it", "ru", "pt"],
        value="Auto",
        label="Language",
    )
    design_run = mo.ui.run_button(
        label="🎙  Generate voice design",
        kind="success",
        full_width=True,
    )
    return design_instruct, design_language, design_run, design_text


@app.cell
def _(
    design_instruct,
    design_language,
    design_run,
    design_text,
    run_design,
    voice_design,
):
    design_output = run_design(
        design_run, design_text, design_language, design_instruct, voice_design
    )
    return (design_output,)


@app.cell
def _(design_instruct, design_language, design_output, design_run, design_text, mo):
    design_panel = mo.vstack(
        [
            mo.md("#### What should the voice say, and what should it sound like?"),
            design_text,
            design_instruct,
            design_language,
            design_run,
            mo.md("---"),
            design_output,
        ]
    )
    return (design_panel,)


# ─── Voice Cloning form ──────────────────────────────────────────────────────


@app.cell
def _(mo):
    clone_upload = mo.ui.file(
        filetypes=[".wav"],
        kind="area",
        label="Upload your reference clip (.wav only)",
    )
    return (clone_upload,)


@app.cell
def _(Path, clone_upload):
    ref_path = None
    if clone_upload.value:
        _u = clone_upload.value[0]
        _ext = Path(_u.name).suffix or ".wav"
        _out = Path("audio/input") / f"_uploaded{_ext}"
        _out.parent.mkdir(parents=True, exist_ok=True)
        _out.write_bytes(_u.contents)
        ref_path = str(_out)
    return (ref_path,)


@app.cell
def _(mo, ref_path):
    if ref_path:
        ref_preview = mo.vstack(
            [
                mo.md(f"**Reference:** `{ref_path}`"),
                mo.audio(ref_path),
            ]
        )
    else:
        ref_preview = mo.md("_Upload a reference audio clip to begin._")
    return (ref_preview,)


@app.cell
def _(mo):
    clone_target = mo.ui.text_area(
        value=(
            "Batman here... I use Weights and Biases to help me watch over Gotham City. "
            "Whether it's the Joker, the penguin or Two-Face, Weights and Biases has you covered."
        ),
        label="Target text",
        rows=3,
        full_width=True,
    )
    clone_autotranscribe = mo.ui.checkbox(
        value=True,
        label="Auto-transcribe reference with Whisper",
    )
    clone_ref_text = mo.ui.text_area(
        value="",
        label="Reference transcript",
        rows=2,
        full_width=True,
    )
    # Hidden by request — re-enable by uncommenting this checkbox and returning it,
    # then re-adding it to the panel composition and threading through to run_clone.
    # clone_xvector = mo.ui.checkbox(
    #     value=False,
    #     label="Use x-vector only (ignore reference text entirely)",
    # )
    clone_max_tokens = mo.ui.slider(
        start=512,
        stop=4096,
        step=128,
        value=2048,
        label="max_new_tokens",
        show_value=True,
    )
    clone_run = mo.ui.run_button(
        label="🎙  Clone voice",
        kind="success",
        full_width=True,
    )
    return (
        clone_autotranscribe,
        clone_max_tokens,
        clone_ref_text,
        clone_run,
        clone_target,
    )


@app.cell
def _(
    clone_autotranscribe,
    clone_max_tokens,
    clone_ref_text,
    clone_run,
    clone_target,
    ref_path,
    run_clone,
    voice_clone,
):
    _ref_text_override = (
        None if clone_autotranscribe.value else clone_ref_text.value.strip()
    )
    clone_output = run_clone(
        clone_run,
        ref_path,
        clone_target,
        clone_max_tokens,
        voice_clone,
        ref_text=_ref_text_override,
    )
    return (clone_output,)


@app.cell
def _(
    clone_autotranscribe,
    clone_max_tokens,
    clone_output,
    clone_ref_text,
    clone_run,
    clone_target,
    clone_upload,
    mo,
    ref_preview,
):
    _items = [
        mo.md("#### Reference audio"),
        clone_upload,
        ref_preview,
        mo.md("---"),
        mo.md("#### Target text"),
        clone_target,
        mo.md("#### Reference-text options"),
        clone_autotranscribe,
    ]
    if not clone_autotranscribe.value:
        _items.append(clone_ref_text)
    _items += [
        mo.md("#### Generation options"),
        clone_max_tokens,
        clone_run,
        mo.md("---"),
        clone_output,
    ]
    clone_panel = mo.vstack(_items)
    return (clone_panel,)


# ─── Tabs ────────────────────────────────────────────────────────────────────


@app.cell
def _(clone_panel, design_panel, mo):
    mo.ui.tabs({"Voice Design": design_panel, "Voice Cloning": clone_panel})
    return


@app.cell
def _(mo, wandb_account, weave_enabled, weave_project):
    if weave_enabled and wandb_account and weave_project:
        _footer = mo.md(
            f"---\nView traces in "
            f"[Weave](https://wandb.ai/{wandb_account}/{weave_project}/weave/traces)."
        )
    elif not weave_enabled:
        _footer = mo.md(
            "---\n_Weave tracing disabled — set `WEAVE_ENABLED=true` in `.env` to log traces._"
        )
    else:
        _footer = mo.md(
            "---\n_Set `WANDB_ACCOUNT` and `WEAVE_PROJECT` in `.env` to enable Weave links._"
        )
    _footer
    return


if __name__ == "__main__":
    app.run()
