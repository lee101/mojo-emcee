from __future__ import annotations

import numpy as np

from .. import autocorr
from ..state import State


class Backend:
    def __init__(self, dtype=None):
        self.initialized = False
        self.dtype = np.float64 if dtype is None else dtype

    def reset(self, nwalkers, ndim):
        self.nwalkers = int(nwalkers)
        self.ndim = int(ndim)
        self.iteration = 0
        self.accepted = np.zeros(self.nwalkers, dtype=self.dtype)
        self.chain = np.empty((0, self.nwalkers, self.ndim), dtype=self.dtype)
        self.log_prob = np.empty((0, self.nwalkers), dtype=self.dtype)
        self.blobs = None
        self.random_state = None
        self.initialized = True

    @property
    def shape(self):
        return self.nwalkers, self.ndim

    def has_blobs(self):
        return self.blobs is not None

    def get_value(self, name, flat=False, thin=1, discard=0):
        if self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with 'store == True' before accessing the results"
            )
        if name == "blobs" and not self.has_blobs():
            return None
        value = getattr(self, name)[
            discard + thin - 1 : self.iteration : thin
        ]
        if flat:
            shape = list(value.shape[1:])
            shape[0] = np.prod(value.shape[:2])
            return value.reshape(shape)
        return value

    def get_chain(self, **kwargs):
        return self.get_value("chain", **kwargs)

    def get_log_prob(self, **kwargs):
        return self.get_value("log_prob", **kwargs)

    def get_blobs(self, **kwargs):
        return self.get_value("blobs", **kwargs)

    def get_last_sample(self):
        if not self.initialized or self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with 'store == True' before accessing the results"
            )
        last = self.iteration - 1
        blobs = self.get_blobs(discard=last)
        return State(
            self.get_chain(discard=last)[0],
            log_prob=self.get_log_prob(discard=last)[0],
            blobs=None if blobs is None else blobs[0],
            random_state=self.random_state,
        )

    def get_autocorr_time(self, discard=0, thin=1, **kwargs):
        values = self.get_chain(discard=discard, thin=thin)
        return thin * autocorr.integrated_time(values, **kwargs)

    def _check_blobs(self, blobs):
        has_blobs = self.has_blobs()
        if has_blobs and blobs is None:
            raise ValueError("inconsistent use of blobs")
        if self.iteration > 0 and blobs is not None and not has_blobs:
            raise ValueError("inconsistent use of blobs")

    def grow(self, ngrow, blobs):
        self._check_blobs(blobs)
        extra = ngrow - (len(self.chain) - self.iteration)
        if extra <= 0:
            return
        coords = np.empty(
            (extra, self.nwalkers, self.ndim), dtype=self.dtype
        )
        self.chain = np.concatenate((self.chain, coords), axis=0)
        log_prob = np.empty((extra, self.nwalkers), dtype=self.dtype)
        self.log_prob = np.concatenate((self.log_prob, log_prob), axis=0)
        if blobs is not None:
            dtype = np.dtype((blobs.dtype, blobs.shape[1:]))
            values = np.empty((extra, self.nwalkers), dtype=dtype)
            if self.blobs is None:
                self.blobs = values
            else:
                self.blobs = np.concatenate((self.blobs, values), axis=0)

    def save_step(self, state, accepted):
        if state.coords.shape != self.shape:
            raise ValueError(f"invalid coordinate dimensions; expected {self.shape}")
        if state.log_prob.shape != (self.nwalkers,):
            raise ValueError(
                f"invalid log probability size; expected {self.nwalkers}"
            )
        self._check_blobs(state.blobs)
        if accepted.shape != (self.nwalkers,):
            raise ValueError(f"invalid acceptance size; expected {self.nwalkers}")
        self.chain[self.iteration] = state.coords
        self.log_prob[self.iteration] = state.log_prob
        if state.blobs is not None:
            self.blobs[self.iteration] = state.blobs
        self.accepted += accepted
        self.random_state = state.random_state
        self.iteration += 1

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass


__all__ = ["Backend"]
