"""Runtime configuration helpers for the weave_tts package."""

from __future__ import annotations

import os

import weave


def init_weave_from_env() -> bool:
    """Initialize Weave tracing based on env vars.

    Reads:
        WEAVE_ENABLED (default "true"): set to "false"/"0"/"no" to disable tracing.
        WANDB_ACCOUNT, WEAVE_PROJECT: project to init when enabled.

    Returns True if `weave.init` was called (tracing is on), False if
    Weave is disabled — private mode. `@weave.op` functions still execute but
    log nothing, and the "Traces will not be logged" warning is suppressed via
    `WEAVE_DISABLED=true`.
    """
    if os.environ.get("WEAVE_ENABLED", "true").strip().lower() in (
        "false",
        "0",
        "no",
        "",
    ):
        os.environ["WEAVE_DISABLED"] = "true"
        return False

    account = os.environ.get("WANDB_ACCOUNT")
    project = os.environ.get("WEAVE_PROJECT")
    if not (account and project):
        os.environ["WEAVE_DISABLED"] = "true"
        return False

    weave.init(f"{account}/{project}")
    return True
