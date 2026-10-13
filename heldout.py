"""Opt-in switch for the held-out check (exp_heldout/PROTOCOL.md): `HELDOUT=1 python ...` makes the measurement pipelines read new text instead of the calibration text.
Off by default, in which case every pipeline behaves exactly as it did for the published results."""
import os
import numpy as np

HELD = bool(os.environ.get("HELDOUT"))
SUFFIX = "_heldout" if HELD else ""
APPS_RANGE = (600, 1200) if HELD else (0, 600)          # APPS problems read for prose/code: held-out sequences use problems 600-1199


def starts(length, window, k):
    """start positions of k windows of `window` tokens over a text of `length` tokens: evenly spaced (calibration) or interleaved between those positions (held-out)."""
    return (np.linspace(0, length - window, 2 * k + 1)[1::2] if HELD else np.linspace(0, length - window, k)).astype(int)


def drop_notes(texts):
    if HELD: texts.pop("notes", None)
    return texts
