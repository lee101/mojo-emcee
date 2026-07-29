from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import emcee  # noqa: E402
import mojo_emcee as memcee  # noqa: E402


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def machine():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def stretch_case():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=(32_768, 64))
    complement = [rng.normal(size=(32_768, 64))]

    def ours():
        return memcee.moves.StretchMove().get_proposal(
            sample, complement, np.random.RandomState(4)
        )

    def theirs():
        return emcee.moves.StretchMove().get_proposal(
            sample, complement, np.random.RandomState(4)
        )

    q1, f1 = ours()
    q2, f2 = theirs()
    assert np.allclose(q1, q2, atol=2e-14)
    assert np.allclose(f1, f2, atol=1e-8)
    return ours, theirs


def autocorrelation_case():
    values = np.random.default_rng(1).normal(size=262_144)
    ours = lambda: memcee.autocorr.function_1d(values)
    theirs = lambda: emcee.autocorr.function_1d(values)
    assert np.allclose(ours(), theirs(), atol=1e-12, rtol=1e-12)
    return ours, theirs


def sampler_case():
    rng = np.random.default_rng(2)
    nwalkers, ndim = 4096, 32
    positions = rng.normal(size=(nwalkers, ndim))

    def log_prob(points):
        return -0.5 * np.sum(points * points, axis=1)

    random_state = np.random.RandomState(7).get_state()

    def ours():
        sampler = memcee.EnsembleSampler(
            nwalkers, ndim, log_prob, vectorize=True
        )
        sampler.random_state = random_state
        return sampler.run_mcmc(positions, 40, store=False)

    def theirs():
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob, vectorize=True
        )
        sampler.random_state = random_state
        return sampler.run_mcmc(positions, 40, store=False)

    assert np.allclose(ours().coords, theirs().coords, atol=1e-12)
    return ours, theirs


def main():
    cases = [
        ("StretchMove proposal (32,768 walkers x 64d)", stretch_case),
        ("autocorr.function_1d (262,144 samples)", autocorrelation_case),
        ("EnsembleSampler, vectorized log-prob (4,096 x 32d x 40)", sampler_case),
    ]
    print(f"Machine: {machine()}")
    print()
    print("| case | mojo-emcee | emcee 3.1.6 | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, setup in cases:
        ours, theirs = setup()
        ours()
        ours_time = timeit(ours)
        theirs_time = timeit(theirs)
        ratio = theirs_time / ours_time
        result = (
            f"{ratio:.2f}x faster"
            if ratio >= 1.0
            else f"{1.0 / ratio:.2f}x slower"
        )
        print(
            f"| {name} | {ours_time * 1e3:.2f} ms | "
            f"{theirs_time * 1e3:.2f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
