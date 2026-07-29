from __future__ import annotations

import numpy as np

from .._lib import f64, i64, stretch_proposal
from .red_blue import RedBlueMove


class StretchMove(RedBlueMove):
    def __init__(self, a=2.0, **kwargs):
        self.a = a
        super().__init__(**kwargs)

    def get_proposal(self, s, c, random):
        sample = f64(s)
        complement = f64(c[0] if len(c) == 1 else np.concatenate(c, axis=0))
        ns, ndim = sample.shape
        nc = len(complement)
        scales = f64(((self.a - 1.0) * random.rand(ns) + 1.0) ** 2.0 / self.a)
        partners = i64(random.randint(nc, size=ns))
        proposal = np.empty_like(sample)
        factors = np.empty(ns, dtype=np.float64)
        stretch_proposal(
            sample, complement, scales, partners, proposal, factors
        )
        factors[:] = (ndim - 1.0) * np.log(scales)
        return proposal, factors
