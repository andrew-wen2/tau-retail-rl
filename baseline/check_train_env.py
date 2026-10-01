"""Report which DeltaNet implementation the trainer will run, and fail if it is the slow one.

Plan requirement 1: without flash-linear-attention and causal-conv1d, Transformers falls back to
a pure-torch DeltaNet path (~50 s/step) with no error. This makes that fallback loud. Run it from
setup.sh, and call `deltanet_impl()` at trainer start so every run's log records the answer.

Usage: .venv-train/bin/python check_train_env.py
"""

from __future__ import annotations

import importlib
import sys

# Where Transformers keeps the Qwen3.5 DeltaNet layer. The module name has not been checked
# against an installed transformers>=5.2; the first one that imports wins, and a miss is reported
# rather than guessed around.
MODELING_CANDIDATES = (
    "transformers.models.qwen3_5.modeling_qwen3_5",
    "transformers.models.qwen3_next.modeling_qwen3_next",
)


# The four DeltaNet functions Transformers can swap for kernels (modeling_qwen3_5.py).
DELTANET_FUNCS = ("causal_conv1d_fn", "causal_conv1d_update",
                  "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")


def _resolved_impl(fn) -> str | None:
    """The module of the implementation a kernel-fallback wrapper resolved to."""
    if fn is None:
        return None
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            v = cell.cell_contents
        except ValueError:
            continue
        if callable(v) and v is not getattr(fn, "__wrapped__", None):
            mod = getattr(v, "__module__", "") or ""
            if mod.startswith(("fla", "causal_conv1d", "kernels")):
                return mod
    return getattr(fn, "__module__", None)


def _version(mod: str) -> str | None:
    try:
        m = importlib.import_module(mod)
    except Exception as e:  # a broken CUDA build raises more than ImportError
        return f"MISSING ({type(e).__name__}: {e})"
    return getattr(m, "__version__", "installed")


def deltanet_impl() -> dict[str, object]:
    """Which DeltaNet path Transformers will take. `fast` is the only acceptable answer."""
    out: dict[str, object] = {
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "peft": _version("peft"),
        "fla": _version("fla"),
        "causal_conv1d": _version("causal_conv1d"),
    }
    for name in MODELING_CANDIDATES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        out["modeling"] = name
        flag = getattr(mod, "is_fast_path_available", None)
        if flag is not None:  # qwen3_next computes one flag at import time
            out["fast"] = bool(flag)
        else:
            # qwen3_5 (transformers 5.x) wraps each DeltaNet function in
            # use_kernel_func_from_hub_with_fallback, which resolves the kernel package at
            # import and keeps the choice in the wrapper's closure. Read it back from there.
            impls = {fn: _resolved_impl(getattr(mod, fn, None)) for fn in DELTANET_FUNCS}
            out["kernels"] = impls
            out["fast"] = all(v not in (None, name) for v in impls.values())
        break
    else:
        out["modeling"] = None
        out["fast"] = None
    return out


def main() -> int:
    import torch

    info = deltanet_impl()
    for k, v in info.items():
        print(f"  {k:<14} {v}")
    print(f"  {'cuda':<14} {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'})")

    if info["modeling"] is None:
        print("FAIL: no Qwen3.5 modeling module found; check MODELING_CANDIDATES against "
              "the installed transformers", file=sys.stderr)
        return 1
    if info["fast"] is False:
        print("FAIL: DeltaNet will run the pure-torch fallback (~50 s/step). "
              "fla or causal_conv1d did not import; see the versions above.", file=sys.stderr)
        return 1
    if info["fast"] is None:
        print(f"WARN: {info['modeling']} has no is_fast_path_available flag; confirm the "
              "kernel path by hand before the first training step.", file=sys.stderr)
    print("train env ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
