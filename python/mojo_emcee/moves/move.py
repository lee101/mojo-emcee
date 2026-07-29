from __future__ import annotations

import numpy as np


class Move:
    def tune(self, state, accepted):
        pass

    def update(self, old_state, new_state, accepted, subset=None):
        if subset is None:
            subset = np.ones(len(old_state.coords), dtype=bool)
        m1 = subset & accepted
        m2 = accepted[subset]
        old_state.coords[m1] = new_state.coords[m2]
        old_state.log_prob[m1] = new_state.log_prob[m2]
        if new_state.blobs is not None:
            if old_state.blobs is None:
                raise ValueError(
                    "If you start sampling with a given log_prob, "
                    "you also need to provide the current list of blobs at that position."
                )
            old_state.blobs[m1] = new_state.blobs[m2]
        return old_state
