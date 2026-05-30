import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import datetime
    import os
    import wave
    from pathlib import Path

    import librosa
    import numpy as np
    import torch
    import weave
    from dotenv import load_dotenv
    from huggingface_hub import login, snapshot_download
    from openai import OpenAI
    from qwen_tts import Qwen3TTSModel

    load_dotenv(override=True)

    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if hf_token:
        login(token=hf_token)

    wandb_account = os.getenv("WANDB_ACCOUNT")
    weave_project = os.getenv("WEAVE_PROJECT")
    if wandb_account and weave_project:
        weave.init(f"{wandb_account}/{weave_project}")

    client = OpenAI()
    return (
        Path,
        Qwen3TTSModel,
        client,
        datetime,
        librosa,
        np,
        os,
        snapshot_download,
        torch,
        wandb_account,
        wave,
        weave,
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


@app.cell
def _(np):
    import soundfile as sf

    def normalize_audio(x, eps=1e-12, clip=True):
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

    def write_sound_to_file(output_filename, audio_file, sample_rate):
        sf.write(output_filename, audio_file, sample_rate, subtype="PCM_16")

    return normalize_audio, write_sound_to_file


@app.cell
def _(Qwen3TTSModel, os, snapshot_download, torch):
    _design_holder = [None]
    _clone_holder = [None]

    def _device():
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

    def get_model_path(model_type, model_size):
        return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")

    def get_voice_design_model():
        if _design_holder[0] is None:
            _design_holder[0] = Qwen3TTSModel.from_pretrained(
                get_model_path("VoiceDesign", "1.7B"),
                device_map=_device(),
                dtype=torch.bfloat16,
                token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
            )
        return _design_holder[0]

    def get_voice_clone_model():
        if _clone_holder[0] is None:
            _clone_holder[0] = Qwen3TTSModel.from_pretrained(
                get_model_path("Base", "1.7B"),
                device_map=_device(),
                dtype=torch.bfloat16,
                token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
            )
        return _clone_holder[0]

    return get_voice_clone_model, get_voice_design_model


@app.cell
def _(
    Path,
    client,
    datetime,
    get_voice_clone_model,
    get_voice_design_model,
    librosa,
    normalize_audio,
    wave,
    weave,
    write_sound_to_file,
):
    @weave.op(
        call_display_name=lambda call: f"{call.func_name}-{int(datetime.datetime.now().timestamp())}"
    )
    def transcribe_audio_from_file(filename: str, model: str = "gpt-4o-transcribe") -> str:
        with open(filename, "rb") as audio_file:
            return client.audio.transcriptions.create(
                model=model, file=audio_file, response_format="text"
            )

    @weave.op(
        call_display_name=lambda call: f"{call.func_name}-{int(datetime.datetime.now().timestamp())}"
    )
    def generate_voice_design(text, language, voice_description):
        if not text or not text.strip():
            return None, "Error: Text is required."
        if not voice_description or not voice_description.strip():
            return None, "Error: Voice description is required."
        try:
            wavs, sr = get_voice_design_model().generate_voice_design(
                text=text.strip(),
                language=language,
                instruct=voice_description.strip(),
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
            generated_audio_dir = Path("./audio/designed_audio/")
            generated_audio_dir.mkdir(parents=True, exist_ok=True)
            output_filename = generated_audio_dir / "generated_audio.wav"
            write_sound_to_file(output_filename, wavs[0], sr)
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"
        return {
            "audio": wave.open(str(output_filename), "rb"),
            "sample_rate": sr,
            "audio_array": wavs[0],
        }

    @weave.op(
        call_display_name=lambda call: f"{call.func_name}-{int(datetime.datetime.now().timestamp())}"
    )
    def generate_voice_clone(
        ref_audio, ref_text, target_text, language, use_xvector_only, max_new_tokens=2048
    ):
        if not target_text or not target_text.strip():
            return None, "Error: Target text is required."
        if ref_audio is None:
            return None, "Error: Reference audio is required."
        if not use_xvector_only and (not ref_text or not ref_text.strip()):
            return None, "Error: Reference text is required when 'Use x-vector only' is not enabled."
        try:
            wavs, sr = get_voice_clone_model().generate_voice_clone(
                text=target_text.strip(),
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text.strip() if ref_text else None,
                x_vector_only_mode=use_xvector_only,
                max_new_tokens=max_new_tokens,
            )
            generated_audio_dir = Path("./audio/cloned_audio/")
            generated_audio_dir.mkdir(parents=True, exist_ok=True)
            output_filename = generated_audio_dir / "generated_cloned_audio.wav"
            write_sound_to_file(output_filename, wavs[0], sr)
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"
        return {
            "generated_audio": wave.open(str(output_filename), "rb"),
            "sample_rate": sr,
            "audio_array": wavs[0],
        }

    @weave.op(
        call_display_name=lambda call: f"{call.func_name}-{int(datetime.datetime.now().timestamp())}"
    )
    def clone_voice(input_audio_filename, clone_target_text, max_new_tokens=2048):
        transcribed_audio = transcribe_audio_from_file(input_audio_filename)
        audio_data, sample_rate = librosa.load(input_audio_filename, sr=None)
        clone_ref_audio = (normalize_audio(audio_data), sample_rate)
        return generate_voice_clone(
            ref_audio=clone_ref_audio,
            ref_text=transcribed_audio.strip(),
            target_text=clone_target_text,
            language="Auto",
            use_xvector_only=False,
            max_new_tokens=max_new_tokens,
        )

    return (
        generate_voice_clone,
        generate_voice_design,
        transcribe_audio_from_file,
    )


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
    generate_voice_design,
    get_voice_design_model,
    mo,
):
    if design_run.value:
        print("[design] Stage 1/3: Loading VoiceDesign model (first run only, may take minutes)...")
        with mo.status.spinner(
            title="Generating designed audio",
            subtitle="Stage 1/3 — Loading VoiceDesign model (first run only, may take minutes)...",
        ) as _sp:
            get_voice_design_model()

            print("[design] Stage 2/3: Synthesizing speech...")
            _sp.update(subtitle="Stage 2/3 — Synthesizing speech...")
            design_result = generate_voice_design(
                design_text.value, design_language.value, design_instruct.value
            )

            print("[design] Stage 3/3: Writing WAV to audio/designed_audio/...")
            _sp.update(subtitle="Stage 3/3 — Writing WAV to disk...")

        if isinstance(design_result, tuple) and design_result[0] is None:
            print(f"[design] FAILED: {design_result[1]}")
            design_output = mo.callout(mo.md(f"**{design_result[1]}**"), kind="danger")
        else:
            print("[design] Done — audio saved to audio/designed_audio/generated_audio.wav")
            design_output = mo.vstack(
                [
                    mo.md("### 🎧 Generated audio"),
                    mo.audio("audio/designed_audio/generated_audio.wav"),
                    mo.md("_Saved to `audio/designed_audio/generated_audio.wav`._"),
                ]
            )
    else:
        design_output = mo.md("")
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
        filetypes=[".wav", ".opus", ".mp3", ".flac", ".m4a"],
        kind="area",
        label="Upload your reference clip",
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
        label="Auto-transcribe reference with gpt-4o-transcribe",
    )
    clone_ref_text = mo.ui.text_area(
        value="",
        label="Reference transcript",
        rows=2,
        full_width=True,
    )
    # Hidden by request — re-enable by uncommenting this checkbox and returning it from
    # the cell, then add it back to the panel composition and the work cell below.
    # clone_xvector = mo.ui.checkbox(
    #     value=False,
    #     label="Use x-vector only (ignore reference text entirely)",
    # )
    clone_max_tokens = mo.ui.slider(
        start=512, stop=4096, step=128, value=2048,
        label="max_new_tokens", show_value=True,
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
    generate_voice_clone,
    get_voice_clone_model,
    librosa,
    mo,
    normalize_audio,
    ref_path,
    transcribe_audio_from_file,
):
    if clone_run.value:
        if not ref_path:
            clone_output = mo.callout(
                mo.md("**Upload a reference audio clip first.**"), kind="warn"
            )
        else:
            try:
                # Stage counts depend on whether transcription runs.
                _total = 6 if clone_autotranscribe.value else 5
                _n = 0

                with mo.status.spinner(
                    title="Cloning voice",
                    subtitle="Preparing...",
                ) as _sp:
                    _n += 1
                    _msg = f"Stage {_n}/{_total} — Loading reference audio with librosa..."
                    print(f"[clone] {_msg}")
                    _sp.update(subtitle=_msg)
                    _audio_data, _sample_rate = librosa.load(ref_path, sr=None)

                    _n += 1
                    _msg = f"Stage {_n}/{_total} — Normalizing reference audio..."
                    print(f"[clone] {_msg}")
                    _sp.update(subtitle=_msg)
                    _ref_audio = (normalize_audio(_audio_data), _sample_rate)

                    if clone_autotranscribe.value:
                        _n += 1
                        _msg = f"Stage {_n}/{_total} — Transcribing reference with gpt-4o-transcribe..."
                        print(f"[clone] {_msg}")
                        _sp.update(subtitle=_msg)
                        _ref_text_used = transcribe_audio_from_file(ref_path).strip()
                    else:
                        _ref_text_used = clone_ref_text.value.strip()

                    _n += 1
                    _msg = f"Stage {_n}/{_total} — Loading Voice Clone model (first run only, may take minutes)..."
                    print(f"[clone] {_msg}")
                    _sp.update(subtitle=_msg)
                    get_voice_clone_model()

                    _n += 1
                    _msg = f"Stage {_n}/{_total} — Synthesizing cloned voice..."
                    print(f"[clone] {_msg}")
                    _sp.update(subtitle=_msg)
                    _result = generate_voice_clone(
                        ref_audio=_ref_audio,
                        ref_text=_ref_text_used,
                        target_text=clone_target.value,
                        language="Auto",
                        use_xvector_only=False,  # (re-enable by uncommenting clone_xvector above)
                        max_new_tokens=int(clone_max_tokens.value),
                    )

                    _n += 1
                    _msg = f"Stage {_n}/{_total} — Writing WAV to disk..."
                    print(f"[clone] {_msg}")
                    _sp.update(subtitle=_msg)

                if isinstance(_result, tuple) and _result[0] is None:
                    print(f"[clone] FAILED: {_result[1]}")
                    clone_output = mo.callout(mo.md(f"**{_result[1]}**"), kind="danger")
                else:
                    print("[clone] Done — audio saved to audio/cloned_audio/generated_cloned_audio.wav")
                    clone_output = mo.vstack(
                        [
                            mo.md("### 🎧 Cloned audio"),
                            mo.audio("audio/cloned_audio/generated_cloned_audio.wav"),
                            mo.md(
                                "**Reference transcript used:** "
                                f"_{_ref_text_used or '(none)'}_"
                            ),
                            mo.md("_Saved to `audio/cloned_audio/generated_cloned_audio.wav`._"),
                        ]
                    )
            except Exception as e:
                print(f"[clone] ERROR: {type(e).__name__}: {e}")
                clone_output = mo.callout(
                    mo.md(f"**Error:** {type(e).__name__}: {e}"), kind="danger"
                )
    else:
        clone_output = mo.md("")
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
def _(mo, wandb_account, weave_project):
    if wandb_account and weave_project:
        _footer = mo.md(
            f"---\nView traces in "
            f"[Weave](https://wandb.ai/{wandb_account}/{weave_project}/weave/traces)."
        )
    else:
        _footer = mo.md(
            "---\n_Set `WANDB_ACCOUNT` and `WEAVE_PROJECT` in `.env` to enable Weave links._"
        )
    _footer
    return


if __name__ == "__main__":
    app.run()
