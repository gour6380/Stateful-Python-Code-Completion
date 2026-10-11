"""Standard-library-only MPS environment setup, called before importing torch."""

from __future__ import annotations

import os
import sys


def _flag(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be unset, '0', or '1'; found {value!r}")
    return value == "1"


def validate_environment(prefer_metal: bool) -> dict[str, bool]:
    if type(prefer_metal) is not bool:
        raise ValueError("prefer_metal must be a boolean")
    fast_math = _flag("PYTORCH_MPS_FAST_MATH")
    fallback = _flag("PYTORCH_ENABLE_MPS_FALLBACK")
    if fast_math or fallback:
        raise RuntimeError(
            "MPS fast math and implicit CPU fallback must remain disabled; no override was made"
        )
    actual = _flag("PYTORCH_MPS_PREFER_METAL")
    if actual != prefer_metal:
        raise RuntimeError(
            "Metal preference differs from the execution profile; restart and run configure_environment before torch imports"
        )
    return {"prefer_metal": actual, "fast_math": fast_math, "cpu_fallback": fallback}


def configure_environment(prefer_metal: bool = False) -> dict[str, bool]:
    """Set a Metal preference before torch imports; never repair unsafe flags.

    The no-argument default preserves the original API. Notebook/service callers
    must pass the selected configuration's ``training.prefer_metal`` explicitly,
    including when reopening a run created with different execution settings.
    """
    if type(prefer_metal) is not bool:
        raise ValueError("prefer_metal must be a boolean")
    if _flag("PYTORCH_MPS_FAST_MATH") or _flag("PYTORCH_ENABLE_MPS_FALLBACK"):
        raise RuntimeError(
            "MPS fast math and implicit CPU fallback must remain disabled; no override was made"
        )
    actual = _flag("PYTORCH_MPS_PREFER_METAL")
    if "torch" in sys.modules and actual != prefer_metal:
        raise RuntimeError(
            "Torch is already imported; restart before changing the Metal preference"
        )
    # An idempotent call after import is safe: it does not claim to change a cached setting.
    os.environ["PYTORCH_MPS_PREFER_METAL"] = "1" if prefer_metal else "0"
    return validate_environment(prefer_metal)
