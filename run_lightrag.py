"""Wrapper: run LightRAG v1.5.0 from the .venv (fixes Windows GBK emoji crash)."""
import sys

# ---- Patch ascii_colors to survive Windows GBK encoding ----
import ascii_colors.core  # type: ignore[import-untyped]

_orig_print = ascii_colors.core.ASCIIColors.print


def _safe_print(t, color, file=None, end="\n", **kw):
    if file is None:
        file = sys.stderr
    try:
        _orig_print(t, color, file=file, end=end, **kw)
    except UnicodeEncodeError:
        cleaned = t.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        file.write(cleaned)
        if end:
            file.write(end)


ascii_colors.core.ASCIIColors.print = staticmethod(_safe_print)
# ------------------------------------------------------------

from lightrag.api.lightrag_server import main

main()
