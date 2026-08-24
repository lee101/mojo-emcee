import numpy as np
import pytest
from concurrent.futures import ThreadPoolExecutor

import emcee
import mojo_emcee as memcee
from mojo_emcee import _lib
from mojo_emcee import autocorr as memcee_autocorr


def gaussian_log_prob(point):
    return -0.5 * np.dot(point, point)


def vectorized_gaussian(points):
    return -0.5 * np.sum(points * points, axis=1)


def initial_positions(nwalkers=24, ndim=5, seed=8):
    return np.random.RandomState(seed).normal(size=(nwalkers, ndim))


def paired_samplers(log_prob=gaussian_log_prob, **kwargs):
    ours = memcee.EnsembleSampler(24, 5, log_prob, **kwargs)
    theirs = emcee.EnsembleSampler(24, 5, log_prob, **kwargs)
    random_state = np.random.RandomState(42).get_state()
    ours.random_state = random_state
    theirs.random_state = random_state
    return ours, theirs


def test_state_sequence_without_blobs_matches_upstream():
    coords = initial_positions()
    ours = memcee.State(coords, log_prob=np.arange(24.0))
    theirs = emcee.State(coords, log_prob=np.arange(24.0))
    assert len(ours) == len(theirs) == 3
    assert all(
        a is b or np.array_equal(a, b)
        for a, b in zip(tuple(ours), tuple(theirs))
    )
    assert ours[-1] is None


def test_state_sequence_with_blobs_matches_upstream():
    coords = initial_positions()
    blobs = np.arange(24)
    ours = memcee.State(coords, blobs=blobs)
    theirs = emcee.State(coords, blobs=blobs)
    assert len(ours) == len(theirs) == 4
    assert np.array_equal(ours[3], theirs[3])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_walkers_independent_matches_upstream(seed):
    coords = initial_positions(seed=seed)
    assert memcee.walkers_independent(coords) == emcee.walkers_independent(coords)


def test_walkers_independent_rejects_degenerate_ensemble():
    coords = np.ones((24, 5))
    assert not memcee.walkers_independent(coords)
    assert not emcee.walkers_independent(coords)


def test_stretch_proposal_matches_upstream_for_identical_rng():
    sample = initial_positions(128, 17, 10)
    complement = [initial_positions(96, 17, 11)]
    ours_rng = np.random.RandomState(123)
    theirs_rng = np.random.RandomState(123)
    ours = memcee.moves.StretchMove(a=2.3)
    theirs = emcee.moves.StretchMove(a=2.3)
    q1, f1 = ours.get_proposal(sample, complement, ours_rng)
    q2, f2 = theirs.get_proposal(sample, complement, theirs_rng)
    assert np.allclose(q1, q2, rtol=0, atol=2e-15)
    assert np.allclose(f1, f2, rtol=0, atol=1e-8)
    assert np.array_equal(ours_rng.get_state()[1], theirs_rng.get_state()[1])


@pytest.mark.parametrize("ns, ndim", [(1, 1), (3, 5), (9, 7)])
def test_stretch_proposal_handles_scalar_and_tail_lengths(ns, ndim):
    sample = initial_positions(ns, ndim, 10)
    complement = [initial_positions(11, ndim, 11)]
    ours_rng = np.random.RandomState(123)
    theirs_rng = np.random.RandomState(123)
    q1, f1 = memcee.moves.StretchMove(a=2.3).get_proposal(
        sample, complement, ours_rng
    )
    q2, f2 = emcee.moves.StretchMove(a=2.3).get_proposal(
        sample, complement, theirs_rng
    )
    assert np.allclose(q1, q2, rtol=0, atol=2e-15)
    assert np.allclose(f1, f2, rtol=0, atol=1e-8)


def test_ffi_rejects_invalid_partner_before_call():
    sample = np.ones((2, 3), dtype=np.float64)
    complement = np.ones((2, 3), dtype=np.float64)
    scales = np.ones(2, dtype=np.float64)
    proposal = np.empty_like(sample)
    factors = np.empty(2, dtype=np.float64)
    with pytest.raises(IndexError, match="partner"):
        _lib.stretch_proposal(
            sample,
            complement,
            scales,
            np.array([0, 2], dtype=np.int64),
            proposal,
            factors,
        )


def test_ffi_rejects_wrong_dtype_and_noncontiguous_output():
    sample = np.ones((2, 3), dtype=np.float32)
    complement = np.ones((2, 3), dtype=np.float64)
    scales = np.ones(2, dtype=np.float64)
    partners = np.zeros(2, dtype=np.int64)
    factors = np.empty(2, dtype=np.float64)
    with pytest.raises(TypeError, match="sample"):
        _lib.stretch_proposal(
            sample,
            complement,
            scales,
            partners,
            np.empty((2, 3), dtype=np.float64),
            factors,
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        _lib.stretch_proposal(
            sample.astype(np.float64),
            complement,
            scales,
            partners,
            np.empty((3, 2), dtype=np.float64).T,
            factors,
        )


def test_dtype_conversions_do_not_silently_discard_information():
    with pytest.raises(TypeError, match="complex"):
        _lib.f64(np.array([1 + 2j]))
    with pytest.raises(OverflowError, match="int64"):
        _lib.i64(np.array([2**63], dtype=np.uint64))


def test_invalid_random_state_is_not_silently_ignored():
    sampler = memcee.EnsembleSampler(24, 5, gaussian_log_prob)
    with pytest.raises((TypeError, ValueError)):
        sampler.random_state = ("invalid",)


def test_stretch_move_has_upstream_red_blue_hierarchy():
    assert isinstance(
        memcee.moves.StretchMove(), memcee.moves.RedBlueMove
    )


def test_seeded_sampler_chain_matches_upstream():
    ours, theirs = paired_samplers()
    positions = initial_positions()
    ours.run_mcmc(positions, 40)
    theirs.run_mcmc(positions, 40)
    assert np.allclose(ours.get_chain(), theirs.get_chain(), atol=1e-12)
    assert np.allclose(ours.get_log_prob(), theirs.get_log_prob(), atol=1e-12)
    assert np.array_equal(
        ours.acceptance_fraction, theirs.acceptance_fraction
    )


def test_vectorized_sampler_matches_upstream():
    ours, theirs = paired_samplers(vectorized_gaussian, vectorize=True)
    positions = initial_positions()
    ours.run_mcmc(positions, 25)
    theirs.run_mcmc(positions, 25)
    assert np.allclose(ours.get_chain(), theirs.get_chain(), atol=1e-12)
    assert np.allclose(ours.get_log_prob(), theirs.get_log_prob(), atol=1e-12)


def test_args_and_kwargs_match_upstream():
    def log_prob(point, mean, scale=1.0):
        return -0.5 * np.sum(((point - mean) / scale) ** 2)

    kwargs = {"args": [np.arange(5.0)], "kwargs": {"scale": 2.0}}
    ours, theirs = paired_samplers(log_prob, **kwargs)
    positions = initial_positions()
    assert np.allclose(
        ours.compute_log_prob(positions)[0],
        theirs.compute_log_prob(positions)[0],
    )


def test_pool_map_is_used():
    class Pool:
        calls = 0

        def map(self, function, values):
            self.calls += 1
            return list(map(function, values))

    pool = Pool()
    sampler = memcee.EnsembleSampler(24, 5, gaussian_log_prob, pool=pool)
    sampler.compute_log_prob(initial_positions())
    assert pool.calls == 1


def test_named_parameter_list_matches_upstream():
    names = ["a", "b", "c", "d", "e"]

    def log_prob(point):
        return -sum(value * value for value in point.values())

    ours = memcee.EnsembleSampler(24, 5, log_prob, parameter_names=names)
    theirs = emcee.EnsembleSampler(24, 5, log_prob, parameter_names=names)
    positions = initial_positions()
    assert np.allclose(
        ours.compute_log_prob(positions)[0],
        theirs.compute_log_prob(positions)[0],
    )


def test_named_parameter_groups_match_upstream():
    names = {"location": [0, 1], "shape": [2, 3, 4]}

    def log_prob(point):
        return -sum(np.sum(value * value) for value in point.values())

    ours = memcee.EnsembleSampler(24, 5, log_prob, parameter_names=names)
    theirs = emcee.EnsembleSampler(24, 5, log_prob, parameter_names=names)
    positions = initial_positions()
    assert np.allclose(
        ours.compute_log_prob(positions)[0],
        theirs.compute_log_prob(positions)[0],
    )


def test_blobs_match_upstream():
    def log_prob(point):
        return gaussian_log_prob(point), np.sum(point)

    ours, theirs = paired_samplers(log_prob)
    positions = initial_positions()
    ours.run_mcmc(positions, 12)
    theirs.run_mcmc(positions, 12)
    assert np.allclose(ours.get_chain(), theirs.get_chain(), atol=1e-12)
    assert np.allclose(ours.get_blobs(), theirs.get_blobs(), atol=1e-12)
    assert np.array_equal(ours.flatblobs, ours.get_blobs(flat=True))
    assert np.array_equal(
        ours.flatlnprobability, ours.get_log_prob(flat=True)
    )


def test_weighted_custom_move_and_tuning_are_used():
    class TrackingMove(memcee.moves.Move):
        def __init__(self):
            self.proposals = 0
            self.tunes = 0

        def propose(self, model, state):
            self.proposals += 1
            return state, np.zeros(len(state.coords), dtype=bool)

        def tune(self, state, accepted):
            self.tunes += 1

    unused = TrackingMove()
    selected = TrackingMove()
    sampler = memcee.EnsembleSampler(
        24,
        5,
        gaussian_log_prob,
        moves=[(unused, 0.0), (selected, 1.0)],
    )
    sampler.run_mcmc(initial_positions(), 4, tune=True)
    assert unused.proposals == unused.tunes == 0
    assert selected.proposals == selected.tunes == 4


def test_resume_with_none_matches_single_run():
    positions = initial_positions()
    split = memcee.EnsembleSampler(24, 5, gaussian_log_prob)
    whole = memcee.EnsembleSampler(24, 5, gaussian_log_prob)
    random_state = np.random.RandomState(20).get_state()
    split.random_state = random_state
    whole.random_state = random_state
    split.run_mcmc(positions, 10)
    split.run_mcmc(None, 15)
    whole.run_mcmc(positions, 25)
    assert np.allclose(split.get_chain(), whole.get_chain(), atol=1e-12)


def test_reset_clears_backend():
    sampler = memcee.EnsembleSampler(24, 5, gaussian_log_prob)
    sampler.run_mcmc(initial_positions(), 5)
    sampler.reset()
    assert sampler.iteration == 0
    assert sampler.backend.chain.shape == (0, 24, 5)
    assert np.all(sampler.backend.accepted == 0)


def test_thin_by_shape_and_values_match_upstream():
    ours, theirs = paired_samplers()
    positions = initial_positions()
    ours.run_mcmc(positions, 8, thin_by=3)
    theirs.run_mcmc(positions, 8, thin_by=3)
    assert ours.get_chain().shape == (8, 24, 5)
    assert np.allclose(ours.get_chain(), theirs.get_chain(), atol=1e-12)


def test_store_false_does_not_grow_backend():
    sampler = memcee.EnsembleSampler(24, 5, gaussian_log_prob)
    state = sampler.run_mcmc(initial_positions(), 5, store=False)
    assert state.coords.shape == (24, 5)
    assert sampler.iteration == 0


def test_invalid_initial_shape_raises():
    sampler = memcee.EnsembleSampler(24, 5, gaussian_log_prob)
    with pytest.raises(ValueError, match="incompatible input dimensions"):
        sampler.run_mcmc(np.zeros((23, 5)), 1)


def test_low_walker_count_matches_upstream_error():
    positions = initial_positions(6, 5)
    ours = memcee.EnsembleSampler(6, 5, gaussian_log_prob)
    theirs = emcee.EnsembleSampler(6, 5, gaussian_log_prob)
    with pytest.raises(RuntimeError, match="fewer walkers"):
        ours.run_mcmc(positions, 1, skip_initial_state_check=True)
    with pytest.raises(RuntimeError, match="fewer walkers"):
        theirs.run_mcmc(positions, 1, skip_initial_state_check=True)


@pytest.mark.parametrize("n", [2, 3, 7, 16, 31, 100, 1000])
def test_autocorrelation_function_matches_upstream(n):
    values = np.random.default_rng(n).normal(size=n)
    assert np.allclose(
        memcee.autocorr.function_1d(values),
        emcee.autocorr.function_1d(values),
        rtol=2e-13,
        atol=2e-13,
    )


def test_autocorrelation_result_aliases_fft_workspace():
    result = memcee.autocorr.function_1d(np.arange(17.0))
    assert result.base is not None
    assert result.base.shape == (64,)


@pytest.mark.parametrize("n", [32768, 65535])
def test_autocorrelation_large_simd_and_tail_lengths_match_upstream(n):
    values = np.random.default_rng(n).normal(size=n)
    assert np.allclose(
        memcee.autocorr.function_1d(values),
        emcee.autocorr.function_1d(values),
        rtol=2e-13,
        atol=2e-13,
    )


def test_parallel_autocorrelation_initializes_worker_thread_runtime():
    values = np.random.default_rng(4).normal(size=65535)
    with ThreadPoolExecutor(max_workers=1) as executor:
        ours = executor.submit(memcee.autocorr.function_1d, values).result()
    assert np.allclose(
        ours,
        emcee.autocorr.function_1d(values),
        rtol=2e-13,
        atol=2e-13,
    )


def test_integrated_time_1d_matches_upstream():
    rng = np.random.default_rng(0)
    values = np.cumsum(rng.normal(size=4096))
    ours = memcee.autocorr.integrated_time(values, quiet=True)
    theirs = emcee.autocorr.integrated_time(values, quiet=True)
    assert np.allclose(ours, theirs, rtol=1e-12, atol=1e-12)


def test_integrated_time_walker_chain_matches_upstream():
    rng = np.random.default_rng(1)
    values = np.cumsum(rng.normal(size=(2048, 8, 3)), axis=0)
    ours = memcee.autocorr.integrated_time(values, quiet=True)
    theirs = emcee.autocorr.integrated_time(values, quiet=True)
    assert np.allclose(ours, theirs, rtol=2e-12, atol=2e-12)


def test_integrated_time_parallel_threshold_matches_upstream(monkeypatch):
    monkeypatch.setattr(memcee_autocorr, "_PARALLEL_WORK_THRESHOLD", 1)
    values = np.cumsum(
        np.random.default_rng(11).normal(size=(512, 4, 2)), axis=0
    )
    ours = memcee.autocorr.integrated_time(values, quiet=True)
    theirs = emcee.autocorr.integrated_time(values, quiet=True)
    assert np.allclose(ours, theirs, rtol=2e-12, atol=2e-12)


def test_integrated_time_parameter_matrix_matches_upstream():
    rng = np.random.default_rng(2)
    values = np.cumsum(rng.normal(size=(2048, 4)), axis=0)
    ours = memcee.autocorr.integrated_time(
        values, quiet=True, has_walkers=False
    )
    theirs = emcee.autocorr.integrated_time(
        values, quiet=True, has_walkers=False
    )
    assert np.allclose(ours, theirs, rtol=2e-12, atol=2e-12)


def test_autocorr_error_exposes_matching_estimate():
    values = np.cumsum(np.random.default_rng(3).normal(size=32))
    with pytest.raises(memcee.autocorr.AutocorrError) as ours:
        memcee.autocorr.integrated_time(values)
    with pytest.raises(emcee.autocorr.AutocorrError) as theirs:
        emcee.autocorr.integrated_time(values)
    assert np.allclose(ours.value.tau, theirs.value.tau)


def test_backend_views_match_upstream():
    ours, theirs = paired_samplers()
    positions = initial_positions()
    ours.run_mcmc(positions, 20)
    theirs.run_mcmc(positions, 20)
    for kwargs in (
        {"discard": 3},
        {"thin": 3},
        {"discard": 4, "thin": 2},
        {"discard": 2, "thin": 2, "flat": True},
    ):
        assert np.allclose(
            ours.get_chain(**kwargs),
            theirs.get_chain(**kwargs),
            atol=1e-12,
        )
        assert np.allclose(
            ours.get_log_prob(**kwargs),
            theirs.get_log_prob(**kwargs),
            atol=1e-12,
        )


def test_legacy_chain_accessors_have_upstream_layout():
    sampler = memcee.EnsembleSampler(24, 5, gaussian_log_prob)
    sampler.run_mcmc(initial_positions(), 5)
    assert np.array_equal(sampler.chain, np.swapaxes(sampler.get_chain(), 0, 1))
    assert np.array_equal(sampler.flatchain, sampler.get_chain(flat=True))
    assert np.array_equal(
        sampler.lnprobability, np.swapaxes(sampler.get_log_prob(), 0, 1)
    )


def test_last_sample_matches_backend_tail():
    sampler = memcee.EnsembleSampler(24, 5, gaussian_log_prob)
    returned = sampler.run_mcmc(initial_positions(), 10)
    last = sampler.get_last_sample()
    assert np.array_equal(last.coords, sampler.get_chain()[-1])
    assert np.array_equal(last.log_prob, sampler.get_log_prob()[-1])
    assert np.array_equal(last.coords, returned.coords)


def test_sampler_autocorrelation_accessor_matches_upstream():
    ours, theirs = paired_samplers()
    positions = initial_positions()
    ours.run_mcmc(positions, 300)
    theirs.run_mcmc(positions, 300)
    assert np.allclose(
        ours.get_autocorr_time(quiet=True),
        theirs.get_autocorr_time(quiet=True),
        rtol=2e-12,
        atol=2e-12,
    )
    with pytest.raises(memcee.autocorr.AutocorrError):
        _ = ours.acor


def test_gaussian_sampling_recovers_target_moments():
    nwalkers, ndim = 40, 2
    positions = np.random.RandomState(10).normal(size=(nwalkers, ndim))
    sampler = memcee.EnsembleSampler(nwalkers, ndim, gaussian_log_prob)
    sampler.random_state = np.random.RandomState(11).get_state()
    sampler.run_mcmc(positions, 600)
    samples = sampler.get_chain(discard=100, flat=True)
    assert np.all(np.abs(samples.mean(axis=0)) < 0.08)
    assert np.all(np.abs(samples.var(axis=0) - 1.0) < 0.12)
