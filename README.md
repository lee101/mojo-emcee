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
| StretchMove proposal (32,768 walkers x 64d) | 6.38 ms | 56.74 ms | 8.90x faster |
| `autocorr.function_1d` (262,144 samples) | 29.04 ms | 69.20 ms | 2.38x faster |
| EnsembleSampler, vectorized log-prob (4,096 x 32d x 40) | 128.13 ms | 371.00 ms | 2.90x faster |

These results describe this machine and pinned environment, not a general
performance guarantee. Real sampling speed is often dominated by the user's
log-probability function; expensive Python likelihoods reduce the relative
benefit of accelerating the transition itself.

### SIMD stretch-proposal pass

The stretch row loop is the one hot loop that takes a vector. Each walker row
is contiguous, visited exactly once, and written to a `proposal` buffer that
never aliases `sample` or `complement`, so `ndim` is safe to widen with
`simd_width_of[DType.float64]()` (4 lanes here) plus a scalar tail; nothing
else in the update has contiguous independent iterations to widen.

Before and after, `pixi run bench`, StretchMove case, same machine and pinned
environment:

| build | mojo-emcee | emcee 3.1.6 | result |
| --- | ---: | ---: | ---: |
| scalar row loop | 7.59 ms | 57.92 ms | 7.63x faster |
| SIMD row loop, run 1 | 6.38 ms | 56.74 ms | 8.90x faster |
| SIMD row loop, run 2 | 7.19 ms | 66.73 ms | 9.27x faster |

The host was busy enough between runs that absolute times drifted ~13%; the
paired comparison below runs both kernels in one process, alternating order,
and reports the median of five. It confirms the kernel gain rather than the
host noise:

| callers | scalar | SIMD | speedup |
| --- | ---: | ---: | ---: |
| 128 x 17 | 5.65 us | 4.85 us | 1.16x |
| 256 x 32 | 12.35 us | 8.50 us | 1.45x |
| 1,024 x 32 | 49.10 us | 33.09 us | 1.48x |
| 4,096 x 32 | 224.21 us | 167.07 us | 1.34x |
| 16,384 x 32 | 891.23 us | 664.00 us | 1.34x |
| 32,768 x 64 | 4,843.71 us | 3,979.30 us | 1.22x |

At 24 walkers x 5 dims (2.5 us) the two builds tie within noise, which is the
expected floor: there is one 4-lane iteration plus a scalar tail per row.

Two fusions were tried and reverted because `pixi run bench` did not justify
them. Moving the stretch-factor `log` into the row loop would have saved a
pass over `scales`, but the result then comes from the Mojo `log` (relative
error ~3e-9), which breaks the accepted `atol=1e-8` factor parity. Folding the
imaginary-buffer clear into the autocorrelation power-spectrum pass, and the
accepted-row copy into a vector loop, measured at or under 1% and inside the
noise floor. Normalisation cannot join the spectrum pass at all, because the
`work_real[0]` denominator it divides by is only produced afterwards by the
inverse transform. The FFT butterflies stay as they were: packing re/im into
one vector halves their loads and stores, but the dialect has no lane-sweep
shuffle for the odd/even deinterleave, and the substitute shuffles cost more
than the 4 multiplies and 2 adds they replace.

No GPU path is provided because these kernels do not have the arithmetic
intensity to justify one. An FFT butterfly moves about 64 bytes for roughly
10--12 floating-point operations (about 0.2 flop/byte), far below the
2 flop/byte cutoff where device transfer and launch costs become plausible.

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

Autocorrelation uses a zero-padded iterative radix-2 FFT written in Mojo. A
forward decimation-in-frequency transform feeds an inverse
decimation-in-time transform, avoiding both bit-reversal passes. FFT
butterflies, buffer preparation, the power spectrum, and normalization use
hardware-width SIMD with scalar tails. The returned array aliases the real
FFT workspace instead of requiring a separate output allocation. Independent
per-walker transforms in `integrated_time` use CPU workers only when the total
work reaches 262,144 samples; smaller jobs stay serial. Normalization and
Sokal window selection match emcee's implementation.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

`pixi run bench` holds a machine-wide lock so concurrent jobs do not distort
the measurements.

MIT licensed.
