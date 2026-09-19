"""
Scene assembly: owns all visual modules and draws one frame from a
SystemModel.  Physics never reads anything from here.

Scene (priority order): neutron star (lit sphere + accretion hot spots),
accretion disk (ring-driven particles), pulsar beams (wire-frame cones),
dipole field lines + reference rings, particle outflow, procedural starfield;
HUD on top.

Everything is submitted in 3 GGUI draw calls (see render/composite.py).
"""
import numpy as np

import config as cfg
from render.disk_particles import DiskParticles
from render.fieldlines import FieldLines
from render.beams import Beams
from render.wind import Wind
from render.starfield import Starfield
from render.camera import OrbitCamera
from render.hud import Hud
from render.composite import Composite


class Renderer:
    def __init__(self, window, seed):
        self.window = window
        self.canvas = window.get_canvas()
        self.scene = window.get_scene()
        self.camera = OrbitCamera()
        self.hud = Hud()
        self.field = FieldLines(cfg.N_FIELD_LINES, cfg.FIELD_LINE_SEGMENTS)
        self.reseed(seed)
        self.canvas.set_background_color((0.005, 0.005, 0.012))

    def reseed(self, seed):
        """(Re)build every procedural visual element from a deterministic seed."""
        rng = np.random.default_rng(seed)
        self.disk = DiskParticles(cfg.N_DISK_PARTICLES, rng)
        self.beams = Beams(cfg.N_BEAM_PARTICLES, rng)
        self.wind = Wind(cfg.N_WIND_PARTICLES, rng)
        self.stars = Starfield(cfg.N_STARS, rng)
        self.composite = Composite(
            [self.stars, self.disk, self.beams, self.wind],
            [(self.field.verts, self.field.colors, self.field.nv),
             (self.beams.wire_verts, self.beams.wire_colors, self.beams.wire_nv)],
            [(self.field.ring_verts, self.field.ring_colors, self.field.ring_nv),
             (self.beams.core_verts, self.beams.core_colors, self.beams.n_core_v)])

    def update(self, model):
        basis = model.magnetosphere.basis
        self.disk.update(model)
        self.field.update(model)
        self.beams.update(model, basis)
        self.wind.update(model, basis)
        self.composite.update(model, basis)

    def draw(self, sim, real_dt):
        model = sim.model
        if sim.run.show_window:
            self.camera.handle_input(self.window, real_dt)
        self.camera.ease(real_dt)
        cam_pos = self.camera.apply(self.scene)
        act = model.activity
        self.scene.ambient_light((0.10, 0.10, 0.14))
        self.scene.point_light(pos=(0.0, 0.0, 0.0), color=(0.9 * act + 0.3, 0.8 * act + 0.3, 1.0))
        self.scene.point_light(pos=cam_pos, color=(0.6, 0.6, 0.66))
        c = self.composite
        self.scene.particles(c.pos, radius=0.1, per_vertex_color=c.color, per_vertex_radius=c.radius)
        self.scene.lines(c.thin_verts, width=1.2, per_vertex_color=c.thin_colors)
        self.scene.lines(c.thick_verts, width=3.0, per_vertex_color=c.thick_colors)
        self.canvas.scene(self.scene)
        self.hud.draw(self.window, self.canvas, sim, self.camera.name)
