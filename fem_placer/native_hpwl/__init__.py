"""Optional native HPWL helpers built from a minimal C extension.

Importing this module is best-effort: if the compiled extension is not
present, ``HAS_NATIVE_HPWL`` is False and callers should fall back to the
pure Python implementation.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple, List

try:
    from . import _hpwl as _native
    HAS_NATIVE_HPWL = True
except Exception:  # pragma: no cover - best-effort import
    _native = None
    HAS_NATIVE_HPWL = False

def hpwl_stats(coords: Sequence[Sequence[float]]) -> Tuple[float, float, float, float, float]:
    """Compute HPWL plus bounding-box stats for a single net using native code."""
    if not HAS_NATIVE_HPWL or _native is None:
        raise RuntimeError("Native HPWL extension is unavailable")
    return _native.hpwl_stats(coords)

def hpwl_stats_batch(batch: Iterable[Sequence[Sequence[float]]]) -> List[Tuple[float, float, float, float, float]]:
    """Compute HPWL stats for many nets at once."""
    if not HAS_NATIVE_HPWL or _native is None:
        raise RuntimeError("Native HPWL extension is unavailable")
    return _native.hpwl_stats_batch(list(batch))

__all__ = ["HAS_NATIVE_HPWL", "hpwl_stats", "hpwl_stats_batch"]
