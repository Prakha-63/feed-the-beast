"""
Application core: central run configuration, simulation state ownership and
the main loop with RESET / PAUSE / RESUME / RESTART hooks.

    RunConfig   immutable per-run settings (seed, backend, resolution, ...)
    Simulation  owns the SystemModel (physical state), the Renderer
                (visual state) and the clocks; exposes the control hooks
                used by the keyboard, the HUD buttons and tests.

Determinism: every random visual element is seeded from RunConfig.seed
(numpy Generator for initial layouts, Taichi random_seed for in-kernel
jitter).  The physics contains no randomness at all.
"""
import time
from dataclasses import dataclass

import config as cfg


@dataclass(frozen=True)
class RunConfig:
    seed: int = cfg.DEFAULT_SEED
    force_cpu: bool = False
    resolution: tuple = cfg.WINDOW_RES
    control: float = 0.5                 # initial accretion control 0..1
    warp_index: int = cfg.DEFAULT_WARP_INDEX
    sim_seconds_per_real_second: float = None   # explicit rate overrides warp_index
    max_frames: int = 0                  # >0: exit after N frames (smoke tests)
    demo: bool = False                   # scripted feed/starve sequence (demo.py)
    demo_duration: float = 60.0          # seconds of real time for the whole sequence
    show_window: bool = True
    vsync: bool = True


def init_taichi(run: RunConfig):
    """Initialise Taichi: Vulkan when available, CPU fallback. Returns backend name."""
    import taichi as ti
    if not run.force_cpu:
        try:
            ti.init(arch=ti.vulkan, random_seed=run.seed)
            return "VULKAN"
        except Exception as exc:          # noqa: BLE001 - any backend failure -> CPU
            print(f"[sim] Vulkan unavailable ({exc}); falling back to CPU")
    ti.init(arch=ti.cpu, random_seed=run.seed)
    return "CPU"


class Simulation:
    """Owns physical state + visual state and drives the frame loop."""

    def __init__(self, run: RunConfig):
        self.run = run
        self.backend = init_taichi(run)
        import taichi as ti
        from physics.model import SystemModel
        from render.renderer import Renderer

        self._ti = ti
        self.model = SystemModel()
        self._apply_initial_state()
        self.window = ti.ui.Window("Feed the Beast - " + cfg.REF_SYSTEM_NAME, run.resolution,
                                   vsync=run.vsync, show_window=run.show_window)
        self.renderer = Renderer(self.window, run.seed)
        self.frame = 0
        self.fps = 0.0
        self.running = True
        self.demo = None
        if run.demo:
            from demo import DemoScript
            self.demo = DemoScript(run.demo_duration)

    # ------------------------------------------------------------ hooks
    def _apply_initial_state(self):
        self.model.set_control(self.run.control)
        if self.run.sim_seconds_per_real_second is not None:
            self.model.warp = self.run.sim_seconds_per_real_second
        else:
            self.model.warp_index = max(0, min(len(cfg.TIME_WARPS) - 1, self.run.warp_index))

    def reset(self):
        """RESET: restore the documented initial physical state
        (SystemModel.INITIAL_STATE_DOC) and clear the visual buffers, keeping
        the user's current control, time-warp, pause flag and camera."""
        control, rate, paused = self.model.control, self.model.warp, self.model.paused
        self.model.reset()
        self.model.set_control(control)
        self.model.warp = rate
        self.model.paused = paused
        self.renderer.reseed(self.run.seed)
        self.frame = 0

    def restart(self):
        """RESTART: reproduce the initial simulation of this run - initial
        physical state, the run's initial control and time warp, visuals
        rebuilt from the run seed, unpaused.  Same seed => same simulation."""
        self.model.reset()
        self._apply_initial_state()
        self.model.paused = False
        self.renderer.reseed(self.run.seed)
        self.frame = 0

    def pause(self):
        self.model.paused = True

    def resume(self):
        self.model.paused = False

    def toggle_pause(self):
        self.model.paused = not self.model.paused

    # ------------------------------------------------------------ input
    def _handle_events(self):
        ti = self._ti
        for ev in self.window.get_events(ti.ui.PRESS):
            k = ev.key
            if k == ti.ui.ESCAPE:
                self.running = False
            elif k == ti.ui.SPACE:
                self.toggle_pause()
            elif k == "r":
                self.reset()
            elif k == "t":
                self.restart()
            elif k == "[":
                self.model.change_warp(-1)
            elif k == "]":
                self.model.change_warp(+1)
            elif k in "123456":
                self.renderer.camera.set_preset(k)
            elif k == "0":
                self.renderer.camera.recentre()
            elif k == ti.ui.LEFT:
                self.model.set_control(self.model.control - 0.05)
            elif k == ti.ui.RIGHT:
                self.model.set_control(self.model.control + 0.05)
            elif k == "x":
                self.model.set_control(0.0)          # STARVE
            elif k == "f":
                self.model.set_control(1.0)          # FEED

    # ------------------------------------------------------------ loop
    def step_frame(self, real_dt):
        """One frame: physics -> visual buffers -> draw."""
        self.model.step(real_dt)          # secular + kinematic clocks
        self.renderer.update(self.model)  # physical state -> visual buffers
        self.renderer.draw(self, real_dt)
        self.frame += 1

    def loop(self):
        last = time.perf_counter()
        fps_n, fps_t = 0, last
        while self.running and self.window.running:
            now = time.perf_counter()
            real_dt = min(now - last, 0.1)     # clamp hitches so physics never jumps
            last = now
            if self.run.show_window:          # input polling needs a visible window
                self._handle_events()
            if self.demo is not None:
                self.demo.update(self, real_dt)   # drives only the accretion control
                if self.demo.done:
                    self.running = False
            self.step_frame(real_dt)
            fps_n += 1
            if now - fps_t >= 0.5:
                self.fps = fps_n / (now - fps_t)
                fps_n, fps_t = 0, now
            if self.run.show_window:
                self.window.show()
            else:                              # headless: GGUI cannot show(); render to buffer
                self.window.get_image_buffer_as_numpy()
            if self.run.max_frames and self.frame >= self.run.max_frames:
                self.running = False
        self.window.destroy()
        return 0
