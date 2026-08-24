"""Build and load the Mojo C ABI."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJO_EMCEE_LIB") or os.path.join(
    ROOT, "dist", "libmojo-emcee.so"
)

I = ctypes.c_int64

_SIGNATURES = {
    "mem_stretch_proposal": ([I] * 9, I),
    "mem_accept_proposals": ([I] * 10, I),
    "mem_autocorrelation_1d": ([I] * 6, I),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJO_EMCEE_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    manifest = os.path.join(ROOT, "pixi.toml")
    if os.path.exists(pixi) and os.path.exists(manifest):
        return [pixi, "run", "--manifest-path", manifest, "mojo"]
    raise BuildError("mojo not found; set MOJO_EMCEE_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_EMCEE_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not os.path.isdir(SRC):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"no Mojo sources at {SRC} and no shared library at {LIB}")
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= max(os.path.getmtime(s) for s in sources):
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        os.path.join(SRC, "emcee.mojo"),
        "-o",
        LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_loaded = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _loaded


def parallel_lib() -> ctypes.CDLL:
    """Return the kernel library without the removed Mojo AsyncRT bootstrap."""
    return lib()


def f64(value, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind == "c":
        raise TypeError("complex values cannot be represented as float64")
    if copy:
        return np.array(value, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.float64)


def i64(value, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in "iu":
        raise TypeError("integer indices are required")
    limits = np.iinfo(np.int64)
    if source.size and (
        np.min(source) < limits.min or np.max(source) > limits.max
    ):
        raise OverflowError("integer index cannot be represented as int64")
    if copy:
        return np.array(value, dtype=np.int64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.int64)


def addr(value: np.ndarray) -> int:
    return value.ctypes.data


def _array(
    name: str,
    value: np.ndarray,
    dtype: np.dtype,
    shape: tuple[int, ...],
    *,
    writable: bool = False,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {np.dtype(dtype).name}")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not value.flags.aligned:
        raise ValueError(f"{name} must be aligned")
    if writable and not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    if value.size and value.ctypes.data == 0:
        raise ValueError(f"{name} has a null data pointer")
    return value


def stretch_proposal(
    sample: np.ndarray,
    complement: np.ndarray,
    scales: np.ndarray,
    partners: np.ndarray,
    proposal: np.ndarray,
    factors: np.ndarray,
) -> None:
    if sample.ndim != 2:
        raise ValueError("sample must be two-dimensional")
    ns, ndim = sample.shape
    if ns <= 0 or ndim <= 0:
        raise ValueError("sample dimensions must be positive")
    if complement.ndim != 2 or complement.shape[1:] != (ndim,):
        raise ValueError(f"complement must have shape (n, {ndim})")
    nc = len(complement)
    if nc <= 0:
        raise ValueError("complement must not be empty")
    _array("sample", sample, np.dtype(np.float64), (ns, ndim))
    _array("complement", complement, np.dtype(np.float64), (nc, ndim))
    _array("scales", scales, np.dtype(np.float64), (ns,))
    _array("partners", partners, np.dtype(np.int64), (ns,))
    _array(
        "proposal", proposal, np.dtype(np.float64), (ns, ndim), writable=True
    )
    _array("factors", factors, np.dtype(np.float64), (ns,), writable=True)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("scales must be finite and positive")
    if np.any(partners < 0) or np.any(partners >= nc):
        raise IndexError("partner index out of bounds")
    status = lib().mem_stretch_proposal(
        addr(sample),
        addr(complement),
        addr(scales),
        addr(partners),
        addr(proposal),
        addr(factors),
        ns,
        nc,
        ndim,
    )
    if status != 0:
        raise RuntimeError(f"Mojo stretch kernel failed with status {status}")


def accept_proposals(
    coords: np.ndarray,
    log_prob: np.ndarray,
    proposal: np.ndarray,
    proposal_log_prob: np.ndarray,
    factors: np.ndarray,
    log_uniform: np.ndarray,
    indices: np.ndarray,
    accepted: np.ndarray,
) -> int:
    if coords.ndim != 2:
        raise ValueError("coords must be two-dimensional")
    nwalkers, ndim = coords.shape
    if nwalkers <= 0 or ndim <= 0:
        raise ValueError("coordinate dimensions must be positive")
    if proposal.ndim != 2 or proposal.shape[1:] != (ndim,):
        raise ValueError(f"proposal must have shape (n, {ndim})")
    ns = len(proposal)
    _array(
        "coords",
        coords,
        np.dtype(np.float64),
        (nwalkers, ndim),
        writable=True,
    )
    _array(
        "log_prob",
        log_prob,
        np.dtype(np.float64),
        (nwalkers,),
        writable=True,
    )
    _array("proposal", proposal, np.dtype(np.float64), (ns, ndim))
    _array(
        "proposal_log_prob",
        proposal_log_prob,
        np.dtype(np.float64),
        (ns,),
    )
    _array("factors", factors, np.dtype(np.float64), (ns,))
    _array("log_uniform", log_uniform, np.dtype(np.float64), (ns,))
    _array("indices", indices, np.dtype(np.int64), (ns,))
    _array(
        "accepted",
        accepted,
        np.dtype(np.int64),
        (nwalkers,),
        writable=True,
    )
    if np.any(indices < 0) or np.any(indices >= nwalkers):
        raise IndexError("walker index out of bounds")
    if len(np.unique(indices)) != ns:
        raise ValueError("walker indices must be unique")
    status = lib().mem_accept_proposals(
        addr(coords),
        addr(log_prob),
        addr(proposal),
        addr(proposal_log_prob),
        addr(factors),
        addr(log_uniform),
        addr(indices),
        addr(accepted),
        ns,
        ndim,
    )
    if status < 0:
        raise RuntimeError(f"Mojo acceptance kernel failed with status {status}")
    return status


def autocorrelation_1d(
    values: np.ndarray,
    result: np.ndarray,
    work_real: np.ndarray,
    work_imag: np.ndarray,
) -> None:
    n = len(values)
    fft_size = len(work_real)
    _array("values", values, np.dtype(np.float64), (n,))
    _array("result", result, np.dtype(np.float64), (n,), writable=True)
    _array(
        "work_real",
        work_real,
        np.dtype(np.float64),
        (fft_size,),
        writable=True,
    )
    _array(
        "work_imag",
        work_imag,
        np.dtype(np.float64),
        (fft_size,),
        writable=True,
    )
    if n <= 0:
        raise ValueError("values must not be empty")
    if fft_size < 2 * n or fft_size & (fft_size - 1):
        raise ValueError("work size must be a power of two at least twice n")
    status = parallel_lib().mem_autocorrelation_1d(
        addr(values),
        addr(result),
        addr(work_real),
        addr(work_imag),
        n,
        fft_size,
    )
    if status != 0:
        raise RuntimeError(
            f"Mojo autocorrelation kernel failed with status {status}"
        )


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
