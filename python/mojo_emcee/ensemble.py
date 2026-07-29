from __future__ import annotations

import warnings
from collections import namedtuple
from itertools import count

import numpy as np

from .backends import Backend
from .moves import StretchMove
from .state import State

Model = namedtuple("Model", ("log_prob_fn", "compute_log_prob_fn", "map_fn", "random"))


class _FunctionWrapper:
    def __init__(self, function, args, kwargs):
        self.function = function
        self.args = args or []
        self.kwargs = kwargs or {}

    def __call__(self, value):
        return self.function(value, *self.args, **self.kwargs)


class EnsembleSampler:
    def __init__(
        self,
        nwalkers,
        ndim,
        log_prob_fn,
        pool=None,
        moves=None,
        args=None,
        kwargs=None,
        backend=None,
        vectorize=False,
        blobs_dtype=None,
        parameter_names=None,
        a=None,
        postargs=None,
        threads=None,
        live_dangerously=None,
        runtime_sortingfn=None,
    ):
        if postargs is not None and args is None:
            args = postargs
        if a is not None:
            warnings.warn(
                "The 'a' argument is deprecated, use 'moves' instead",
                DeprecationWarning,
                stacklevel=2,
            )
        if moves is None:
            move_kwargs = {}
            if a is not None:
                move_kwargs["a"] = a
            if live_dangerously is not None:
                move_kwargs["live_dangerously"] = live_dangerously
            self._moves = [StretchMove(**move_kwargs)]
            self._weights = np.array([1.0])
        elif isinstance(moves, (list, tuple)):
            try:
                self._moves, weights = zip(*moves)
                self._moves = list(self._moves)
                self._weights = np.asarray(weights, dtype=float)
            except (TypeError, ValueError):
                self._moves = list(moves)
                self._weights = np.ones(len(self._moves))
        else:
            self._moves = [moves]
            self._weights = np.array([1.0])
        self._weights = np.atleast_1d(self._weights).astype(float)
        self._weights /= self._weights.sum()

        self.pool = pool
        self.vectorize = vectorize
        self.blobs_dtype = blobs_dtype
        self.ndim = int(ndim)
        self.nwalkers = int(nwalkers)
        self.backend = Backend() if backend is None else backend
        if not self.backend.initialized:
            self._previous_state = None
            self.reset()
            random_state = np.random.get_state()
        else:
            if self.backend.shape != (self.nwalkers, self.ndim):
                raise ValueError("backend shape is incompatible with the sampler")
            random_state = self.backend.random_state or np.random.get_state()
            self._previous_state = (
                self.get_last_sample() if self.backend.iteration else None
            )

        self._random = np.random.mtrand.RandomState()
        self._random.set_state(random_state)
        self.log_prob_fn = _FunctionWrapper(log_prob_fn, args, kwargs)

        self.params_are_named = parameter_names is not None
        if self.params_are_named:
            if vectorize:
                raise ValueError(
                    "named parameters with vectorization unsupported for now"
                )
            if isinstance(parameter_names, list):
                if len(parameter_names) != self.ndim:
                    raise ValueError(
                        "name all parameters or set parameter_names to None"
                    )
                parameter_names = {
                    name: index for index, name in enumerate(parameter_names)
                }
            values = []
            for value in parameter_names.values():
                values.extend(value if isinstance(value, list) else [value])
            if set(values) != set(range(self.ndim)):
                raise ValueError("not all parameter indices appear")
            self.parameter_names = parameter_names

    @property
    def random_state(self):
        return self._random.get_state()

    @random_state.setter
    def random_state(self, state):
        if state is not None:
            self._random.set_state(state)

    @property
    def iteration(self):
        return self.backend.iteration

    @property
    def acceptance_fraction(self):
        return self.backend.accepted / float(self.backend.iteration)

    def reset(self):
        self.backend.reset(self.nwalkers, self.ndim)

    def compute_log_prob(self, coords):
        positions = np.asarray(coords)
        if np.any(np.isinf(positions)):
            raise ValueError("At least one parameter value was infinite")
        if np.any(np.isnan(positions)):
            raise ValueError("At least one parameter value was NaN")
        values = positions
        if self.params_are_named:
            values = [
                {key: point[index] for key, index in self.parameter_names.items()}
                for point in positions
            ]
        if self.vectorize:
            results = self.log_prob_fn(values)
        else:
            mapper = map if self.pool is None else self.pool.map
            results = list(mapper(self.log_prob_fn, values))
        if (
            self.vectorize
            and isinstance(results, np.ndarray)
            and results.ndim == 1
            and results.dtype.kind in "fiu"
        ):
            log_prob = results
            blobs = None
        else:
            try:
                blobs = [result[1:] for result in results if len(result) > 1]
                if not blobs:
                    raise IndexError
                log_prob = np.array([_scalar(result[0]) for result in results])
            except (IndexError, TypeError):
                log_prob = np.array([_scalar(result) for result in results])
                blobs = None
            else:
                dtype = self.blobs_dtype
                if dtype is None:
                    dtype = np.atleast_1d(blobs[0]).dtype
                    if dtype.kind in "US":
                        dtype = np.dtype("object")
                blobs = np.array(blobs, dtype=dtype)
                shape = blobs.shape[1:]
                if shape:
                    axes = np.arange(len(shape))[np.array(shape) == 1] + 1
                    if len(axes):
                        blobs = np.squeeze(blobs, tuple(axes))
        if np.any(np.isnan(log_prob)):
            raise ValueError("Probability function returned NaN")
        return log_prob, blobs

    def sample(
        self,
        initial_state,
        log_prob0=None,
        rstate0=None,
        blobs0=None,
        iterations=1,
        tune=False,
        skip_initial_state_check=False,
        thin_by=1,
        thin=None,
        store=True,
        progress=False,
        progress_kwargs=None,
    ):
        if iterations is None and store:
            raise ValueError("'store' must be False when 'iterations' is None")
        state = State(initial_state, copy=True)
        if np.shape(state.coords) != (self.nwalkers, self.ndim):
            raise ValueError(f"incompatible input dimensions {np.shape(state.coords)}")
        if not skip_initial_state_check and not walkers_independent(state.coords):
            raise ValueError(
                "Initial state has a large condition number. Make sure that "
                "your walkers are linearly independent for the best performance"
            )
        if rstate0 is not None:
            state.random_state = rstate0
        self.random_state = state.random_state
        if log_prob0 is not None:
            state.log_prob = log_prob0
        if blobs0 is not None:
            state.blobs = blobs0
        if state.log_prob is None:
            state.log_prob, state.blobs = self.compute_log_prob(state.coords)
        state.log_prob = np.asarray(state.log_prob, dtype=np.float64)
        if state.log_prob.shape != (self.nwalkers,):
            raise ValueError("incompatible input dimensions")
        if np.any(np.isnan(state.log_prob)):
            raise ValueError("The initial log_prob was NaN")

        if thin is not None:
            thin = int(thin)
            if thin <= 0:
                raise ValueError("Invalid thinning argument")
            yield_step = 1
            checkpoint_step = thin
            if store:
                self.backend.grow(iterations // thin, state.blobs)
        else:
            thin_by = int(thin_by)
            if thin_by <= 0:
                raise ValueError("Invalid thinning argument")
            yield_step = checkpoint_step = thin_by
            if store:
                self.backend.grow(iterations, state.blobs)

        mapper = map if self.pool is None else self.pool.map
        model = Model(self.log_prob_fn, self.compute_log_prob, mapper, self._random)
        proposal_number = 0
        iterator = count() if iterations is None else range(iterations)
        for _ in iterator:
            for _ in range(yield_step):
                move = self._random.choice(self._moves, p=self._weights)
                state, accepted = move.propose(model, state)
                state.random_state = self.random_state
                if tune:
                    move.tune(state, accepted)
                if store and (proposal_number + 1) % checkpoint_step == 0:
                    self.backend.save_step(state, accepted)
                proposal_number += 1
            yield state

    def run_mcmc(self, initial_state, nsteps, **kwargs):
        if initial_state is None:
            if self._previous_state is None:
                raise ValueError(
                    "Cannot have `initial_state=None` if run_mcmc has never been called."
                )
            initial_state = self._previous_state
        result = None
        for result in self.sample(initial_state, iterations=nsteps, **kwargs):
            pass
        self._previous_state = result
        return result

    def get_chain(self, **kwargs):
        return self.backend.get_chain(**kwargs)

    def get_log_prob(self, **kwargs):
        return self.backend.get_log_prob(**kwargs)

    def get_blobs(self, **kwargs):
        return self.backend.get_blobs(**kwargs)

    def get_last_sample(self, **kwargs):
        return self.backend.get_last_sample()

    def get_autocorr_time(self, **kwargs):
        return self.backend.get_autocorr_time(**kwargs)

    def get_value(self, name, **kwargs):
        return self.backend.get_value(name, **kwargs)

    @property
    def chain(self):
        return np.swapaxes(self.get_chain(), 0, 1)

    @property
    def flatchain(self):
        return self.get_chain(flat=True)

    @property
    def lnprobability(self):
        return np.swapaxes(self.get_log_prob(), 0, 1)

    @property
    def flatlnprobability(self):
        return self.get_log_prob(flat=True)

    @property
    def blobs(self):
        return self.get_blobs()

    @property
    def flatblobs(self):
        return self.get_blobs(flat=True)

    @property
    def acor(self):
        return self.get_autocorr_time()


def _scalar(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(np.asarray(value).item())


def walkers_independent(coords):
    coords = np.asarray(coords)
    if not np.all(np.isfinite(coords)):
        return False
    centered = coords - np.mean(coords, axis=0)[None, :]
    column_max = np.amax(np.abs(centered), axis=0)
    if np.any(column_max == 0):
        return False
    centered /= column_max
    column_sum = np.sqrt(np.sum(centered**2, axis=0))
    centered /= column_sum
    return np.linalg.cond(centered.astype(float)) <= 1e8


__all__ = ["EnsembleSampler", "walkers_independent"]
