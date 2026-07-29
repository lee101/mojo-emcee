# mojo-emcee

`mojo-emcee` is a Mojo-accelerated port of the affine-invariant ensemble
sampler in [emcee](https://emcee.readthedocs.io/). It keeps the familiar
Python API and callback model while moving batched stretch proposals,
Metropolis acceptance updates, and FFT autocorrelation calculations into a
compiled Mojo shared library.

Use it as an emcee-shaped module for the covered subset:

```python
import mojo_emcee as emcee
```

## Installation

The repository pins the tested Mojo nightly and all Python dependencies:

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-emcee.so`.
Run Python commands inside the environment with `pixi run python`, or enter
it with `pixi shell`.

## Usage

This complete example samples a three-dimensional standard normal:

```python
import numpy as np
import mojo_emcee as emcee

rng = np.random.default_rng(7)
nwalkers, ndim = 32, 3
initial = 1.0e-2 * rng.normal(size=(nwalkers, ndim))

def log_probability(points):
    return -0.5 * np.sum(points * points, axis=1)

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, vectorize=True
)
sampler.run_mcmc(initial, 500)

samples = sampler.get_chain(discard=100, flat=True)
print(samples.shape)
print(sampler.acceptance_fraction.mean())
```

Scalar log-probability callbacks, `args`, `kwargs`, `pool.map`, named
parameters, blobs, weighted custom move schedules, state resumption,
`thin_by`, and `store=False` are also supported.

## Covered API

- `EnsembleSampler`, including `sample`, `run_mcmc`, `compute_log_prob`,
  `reset`, `random_state`, `acceptance_fraction`, and stored-chain accessors
- `State`
- `moves.Move`, `moves.RedBlueMove`, and `moves.StretchMove`
- `backends.Backend`, the in-memory backend
- `autocorr.function_1d`, `autocorr.integrated_time`, and `AutocorrError`
- Current and legacy chain, log-probability, blob, and autocorrelation
  accessors

The test suite compares the implementation directly with emcee 3.1.6 using
identical initial positions and random-number states. Seeded stretch chains,
acceptance decisions, likelihood values, blobs, storage views, and
autocorrelation estimates agree numerically.

Not covered are emcee's `WalkMove`, differential-evolution, snooker, KDE,
Gaussian and general Metropolis-Hastings moves; `HDFBackend`; the progress-bar
UI; and automatic tuning beyond a custom move's own `tune` method. A supplied
custom move can still participate in a weighted move schedule when it
implements emcee's `propose(model, state)` protocol.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz.
Times are the best of three runs and include the Python wrapper and random
number generation. The shared library is warmed before timing.

| case | mojo-emcee | emcee 3.1.6 | result |
| --- | ---: | ---: | ---: |
| StretchMove proposal (32,768 walkers x 64d) | 11.51 ms | 85.15 ms | 7.40x faster |
| `autocorr.function_1d` (262,144 samples) | 50.45 ms | 85.85 ms | 1.70x faster |
| EnsembleSampler, vectorized log-prob (4,096 x 32d x 40) | 131.57 ms | 441.71 ms | 3.36x faster |

These results describe this machine and pinned environment, not a general
performance guarantee. Real sampling speed is often dominated by the user's
log-probability function; expensive Python likelihoods reduce the relative
benefit of accelerating the transition itself.

No GPU path is provided.

## How it works

Python owns every allocation. Coordinates are C-contiguous row-major
`float64` arrays with shape `(nwalkers, ndim)`; log probabilities and stretch
factors are one-dimensional `float64` arrays, while partner and walker
indices are `int64`. Their addresses cross `ctypes` as 64-bit integers and
are reconstructed as mutable `UnsafePointer` values inside non-parametric
`abi("C")` exports. Before a call, the Python boundary checks exact dtype,
shape, contiguity, alignment, writability, index bounds, and non-null data
pointers. The exports also reject null addresses and invalid dimensions.
Calls are synchronous, so the Python references holding every NumPy buffer
remain alive until Mojo returns.

The red-blue update preserves emcee's random draw order. Python invokes the
user likelihood on each proposed half-ensemble, while Mojo constructs the
coordinate proposals and applies accepted coordinates and log probabilities
in place. NumPy computes the small logarithmic stretch-factor vector because
the pinned nightly's fast Mojo logarithm is not accurate enough for
acceptance decisions near the threshold.

Autocorrelation uses a zero-padded iterative radix-2 FFT written in Mojo.
Buffer preparation, the power spectrum, and normalization use hardware-width
SIMD with scalar tails. FFT stages at 131,072 points and above are divided
across CPU workers; smaller transforms remain serial to avoid scheduling
overhead. Its normalization and Sokal window selection match emcee's
implementation; only the FFT and power-spectrum work crosses the FFI boundary.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

`pixi run bench` holds a machine-wide lock so concurrent jobs do not distort
the measurements.

MIT licensed.
