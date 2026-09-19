"""
Draw-call batching for GGUI.

Every scene.particles()/scene.lines() call costs a few milliseconds of fixed
overhead (field -> vertex-buffer copies), so all particle systems are gathered
into ONE particle buffer and all line systems into two line buffers (thin and
thick) by a single Taichi kernel each frame.  The small "bodies" group
(neutron star, two polar hot spots) is generated directly from kernel
arguments - no per-frame numpy uploads.
"""
import numpy as np
import taichi as ti


N_BODIES = 3
AXIS_NV = 4          # spin-axis line: 2 segments (north/south)


@ti.data_oriented
class Composite:
    def __init__(self, particle_sources, thin_sources, thick_sources):
        # lists of fields (ti.static can unroll over fields, not over arbitrary objects)
        self.p_pos = [s.pos for s in particle_sources]
        self.p_col = [s.color for s in particle_sources]
        self.p_rad = [s.radius for s in particle_sources]
        self.thin_v = [v for v, _, _ in thin_sources]
        self.thin_c = [c for _, c, _ in thin_sources]
        self.thick_v = [v for v, _, _ in thick_sources]
        self.thick_c = [c for _, c, _ in thick_sources]
        self.p_off = np.cumsum([0] + [s.pos.shape[0] for s in particle_sources])
        self.n_p = int(self.p_off[-1]) + N_BODIES
        self.thin_off = np.cumsum([0] + [c for _, _, c in thin_sources])
        self.n_thin = int(self.thin_off[-1])
        self.thick_off = np.cumsum([0] + [c for _, _, c in thick_sources])
        self.n_thick = int(self.thick_off[-1]) + AXIS_NV
        self.pos = ti.Vector.field(3, ti.f32, self.n_p)
        self.color = ti.Vector.field(3, ti.f32, self.n_p)
        self.radius = ti.field(ti.f32, self.n_p)
        self.thin_verts = ti.Vector.field(3, ti.f32, self.n_thin)
        self.thin_colors = ti.Vector.field(3, ti.f32, self.n_thin)
        self.thick_verts = ti.Vector.field(3, ti.f32, max(1, self.n_thick))
        self.thick_colors = ti.Vector.field(3, ti.f32, max(1, self.n_thick))

    @ti.kernel
    def _gather(self, m: ti.types.vector(3, ti.f32), hot: ti.f32, star_col: ti.types.vector(3, ti.f32)):
        # --- particle systems --------------------------------------------------
        for s in ti.static(range(len(self.p_pos))):
            sp = ti.static(self.p_pos[s])
            sc = ti.static(self.p_col[s])
            sr = ti.static(self.p_rad[s])
            off = ti.static(int(self.p_off[s]))
            for i in range(sp.shape[0]):
                self.pos[off + i] = sp[i]
                self.color[off + i] = sc[i]
                self.radius[off + i] = sr[i]
        # --- bodies --------------------------------------------------------------
        b = ti.static(int(self.p_off[-1]))
        hidden = ti.Vector([0.0, 0.0, -1.0e5])
        self.pos[b] = ti.Vector([0.0, 0.0, 0.0])
        self.color[b] = star_col
        self.radius[b] = 1.0
        hs = ti.Vector([1.0, 0.6 * hot, 0.2 * hot]) * ti.min(1.0, hot)
        for k in ti.static(range(2)):
            sgn = 1.0 if k == 0 else -1.0
            self.pos[b + 1 + k] = sgn * m * 0.92 if hot >= 0.4 else hidden
            self.color[b + 1 + k] = hs
            self.radius[b + 1 + k] = 0.28 * ti.min(1.0, hot)
        # --- thin lines ------------------------------------------------------------
        for s in ti.static(range(len(self.thin_v))):
            src_v = ti.static(self.thin_v[s])
            src_c = ti.static(self.thin_c[s])
            off = ti.static(int(self.thin_off[s]))
            for i in range(src_v.shape[0]):
                self.thin_verts[off + i] = src_v[i]
                self.thin_colors[off + i] = src_c[i]
        # --- spin axis (white, dim): makes the tilt of the magnetic axis / beams obvious
        ab = ti.static(int(self.thick_off[-1]))
        self.thick_verts[ab] = ti.Vector([0.0, 0.0, 1.0])
        self.thick_verts[ab + 1] = ti.Vector([0.0, 0.0, 2.6])
        self.thick_verts[ab + 2] = ti.Vector([0.0, 0.0, -1.0])
        self.thick_verts[ab + 3] = ti.Vector([0.0, 0.0, -2.6])
        for k in ti.static(range(4)):
            self.thick_colors[ab + k] = ti.Vector([0.7, 0.7, 0.75]) * (1.0 if k % 2 == 0 else 0.2)
        # --- thick lines -----------------------------------------------------------
        for s in ti.static(range(len(self.thick_v))):
            src_v = ti.static(self.thick_v[s])
            src_c = ti.static(self.thick_c[s])
            off = ti.static(int(self.thick_off[s]))
            for i in range(src_v.shape[0]):
                self.thick_verts[off + i] = src_v[i]
                self.thick_colors[off + i] = src_c[i]

    def update(self, model, basis):
        _, _, m = basis
        hot = 0.35 + model.beam_orange
        # star colour: hotter/brighter with activity (surface heated by accretion)
        act = model.activity
        star_col = np.array([0.85 + 0.15 * act, 0.9 + 0.1 * act, 1.0], dtype=np.float32)
        self._gather(m.astype(np.float32), float(hot), star_col)
