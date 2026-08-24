from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ._lib import autocorrelation_1d, f64

logger = logging.getLogger(__name__)

_PARALLEL_WORK_THRESHOLD = 262_144


def next_pow_two(n):
    value = 1
    while value < n:
        value <<= 1
    return value


def function_1d(x):
    values = f64(np.atleast_1d(x))
    if values.ndim != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    if len(values) == 0:
        raise ValueError("cannot autocorrelate an empty series")
    fft_size = 2 * next_pow_two(len(values))
    work_real = np.empty(fft_size, dtype=np.float64)
    work_imag = np.empty(fft_size, dtype=np.float64)
    result = work_real[: len(values)]
    autocorrelation_1d(values, result, work_real, work_imag)
    return result


def auto_window(taus, c):
    mask = np.arange(len(taus)) < c * taus
    if np.any(mask):
        return np.argmin(mask)
    return len(taus) - 1


def integrated_time(x, c=5, tol=50, quiet=False, has_walkers=True):
    values = np.atleast_1d(x)
    if values.ndim == 1:
        values = values[:, np.newaxis, np.newaxis]
    if values.ndim == 2:
        if has_walkers:
            values = values[:, :, np.newaxis]
        else:
            values = values[:, np.newaxis, :]
    if values.ndim != 3:
        raise ValueError("invalid dimensions")

    n_t, n_w, n_d = values.shape
    tau_est = np.empty(n_d)
    for d in range(n_d):
        acf = np.zeros(n_t)
        if n_w > 1 and n_t * n_w >= _PARALLEL_WORK_THRESHOLD:
            workers = min(n_w, os.cpu_count() or 1)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                correlations = executor.map(
                    function_1d,
                    (values[:, walker, d] for walker in range(n_w)),
                )
                for correlation in correlations:
                    acf += correlation
        else:
            for walker in range(n_w):
                acf += function_1d(values[:, walker, d])
        acf /= n_w
        taus = 2.0 * np.cumsum(acf) - 1.0
        tau_est[d] = taus[auto_window(taus, c)]

    flag = tol * tau_est > n_t
    if np.any(flag):
        msg = (
            "The chain is shorter than {0} times the integrated "
            "autocorrelation time for {1} parameter(s). Use this estimate "
            "with caution and run a longer chain!\n"
        ).format(tol, np.sum(flag))
        msg += "N/{0} = {1:.0f};\ntau: {2}".format(tol, n_t / tol, tau_est)
        if not quiet:
            raise AutocorrError(tau_est, msg)
        logger.warning(msg)
    return tau_est


class AutocorrError(Exception):
    def __init__(self, tau, *args, **kwargs):
        self.tau = tau
        super().__init__(*args, **kwargs)


__all__ = [
    "AutocorrError",
    "auto_window",
    "function_1d",
    "integrated_time",
    "next_pow_two",
]
