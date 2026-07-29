from __future__ import annotations

import numpy as np

from .._lib import accept_proposals, f64, i64
from .move import Move


class RedBlueMove(Move):
    def __init__(
        self, nsplits=2, randomize_split=True, live_dangerously=False
    ):
        self.nsplits = int(nsplits)
        self.randomize_split = randomize_split
        self.live_dangerously = live_dangerously

    def setup(self, coords):
        pass

    def get_proposal(self, sample, complement, random):
        raise NotImplementedError("The proposal must be implemented by subclasses")

    def propose(self, model, state):
        nwalkers, ndim = state.coords.shape
        if nwalkers < 2 * ndim and not self.live_dangerously:
            raise RuntimeError(
                "It is unadvisable to use a red-blue move with fewer walkers "
                "than twice the number of dimensions."
            )
        if self.nsplits < 2:
            raise ValueError("nsplits must be at least 2")
        self.setup(state.coords)

        state.coords = f64(state.coords)
        state.log_prob = f64(state.log_prob)
        accepted_i64 = np.zeros(nwalkers, dtype=np.int64)
        all_inds = np.arange(nwalkers)
        split_ids = all_inds % self.nsplits
        if self.randomize_split:
            model.random.shuffle(split_ids)

        for split in range(self.nsplits):
            selected = split_ids == split
            sets = [state.coords[split_ids == j] for j in range(self.nsplits)]
            proposal, factors = self.get_proposal(
                sets[split], sets[:split] + sets[split + 1 :], model.random
            )
            new_log_prob, new_blobs = model.compute_log_prob_fn(proposal)
            new_log_prob = f64(new_log_prob)
            indices = i64(all_inds[selected])
            log_uniform = f64(np.log(model.random.rand(len(proposal))))
            proposal = f64(proposal)
            factors = f64(factors)
            accept_proposals(
                state.coords,
                state.log_prob,
                proposal,
                new_log_prob,
                factors,
                log_uniform,
                indices,
                accepted_i64,
            )
            if new_blobs is not None:
                if state.blobs is None:
                    raise ValueError(
                        "If you start sampling with a given log_prob, you also "
                        "need to provide the current list of blobs at that position."
                    )
                local_accepted = accepted_i64[indices].astype(bool)
                state.blobs[indices[local_accepted]] = new_blobs[local_accepted]

        return state, accepted_i64.astype(bool)


__all__ = ["RedBlueMove"]
