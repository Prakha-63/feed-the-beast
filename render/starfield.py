"""
Lightweight procedural 360-degree starfield: fixed points on a sphere of
radius STARFIELD_RADIUS, seeded deterministically.  Purely a visual
environment (procedural, not a real sky catalogue - see README).
"""
import numpy as np
import taichi as ti

import config as cfg


class Starfield:
    def __init__(self, n, rng):
        self.n = n
        self.pos = ti.Vector.field(3, ti.f32, n)
        self.color = ti.Vector.field(3, ti.f32, n)
        self.radius = ti.field(ti.f32, n)
        v = rng.standard_normal((n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        self.pos.from_numpy((v * cfg.STARFIELD_RADIUS).astype(np.float32))
        mag = rng.random(n) ** 2.5                     # few bright, many faint
        tint = rng.random(n)
        col = np.stack([0.85 + 0.15 * tint, 0.85 + 0.1 * tint, 0.9 + 0.1 * (1 - tint)], axis=1)
        self.color.from_numpy((col * (0.25 + 0.75 * mag)[:, None]).astype(np.float32))
        self.radius.from_numpy((0.45 + 0.9 * mag).astype(np.float32))
