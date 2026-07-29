from __future__ import annotations

from copy import deepcopy

import numpy as np


class State:
    __slots__ = "coords", "log_prob", "blobs", "random_state"

    def __init__(
        self, coords, log_prob=None, blobs=None, random_state=None, copy=False
    ):
        dc = deepcopy if copy else lambda value: value
        if hasattr(coords, "coords"):
            self.coords = dc(coords.coords)
            self.log_prob = dc(coords.log_prob)
            self.blobs = dc(coords.blobs)
            self.random_state = dc(coords.random_state)
            return
        self.coords = dc(np.atleast_2d(coords))
        self.log_prob = dc(log_prob)
        self.blobs = dc(blobs)
        self.random_state = dc(random_state)

    def __len__(self):
        return 3 if self.blobs is None else 4

    def __repr__(self):
        return "State({0}, log_prob={1}, blobs={2}, random_state={3})".format(
            self.coords, self.log_prob, self.blobs, self.random_state
        )

    def __iter__(self):
        if self.blobs is None:
            return iter((self.coords, self.log_prob, self.random_state))
        return iter((self.coords, self.log_prob, self.random_state, self.blobs))

    def __getitem__(self, index):
        if index < 0:
            return self[len(self) + index]
        if index == 0:
            return self.coords
        if index == 1:
            return self.log_prob
        if index == 2:
            return self.random_state
        if index == 3 and self.blobs is not None:
            return self.blobs
        raise IndexError(f"Invalid index '{index}'")
