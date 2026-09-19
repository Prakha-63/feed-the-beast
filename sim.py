"""
FEED THE BEAST - The Life of a Neutron Star
Interactive accretion / spin-evolution laboratory built on PSR J1023+0038.

    python sim.py                 run (Vulkan GPU if available, else CPU)
    python sim.py --cpu           force the CPU backend
    python sim.py --check         run the scientific validation suite (no window)
    python sim.py --provenance    print what is observed / computed / simplified
    python sim.py --seed 7        deterministic seed for procedural visuals
    python sim.py --warp 4        initial time-warp preset index (see config.TIME_WARPS)
    python sim.py --rate 3600     simulation seconds per real second (any positive float)
    python sim.py --frames 60     exit after N frames (smoke test); add
                                  --headless to run without showing a window

Keys: LEFT/RIGHT feed -/+5 %, X starve (0 %), F feed (100 %),
      SPACE pause/resume, R reset, T restart, [ ] time warp, ESC quit.
Camera: 1-6 presets, 0 recentre, left-drag orbit, right-drag pan,
        middle-drag / Shift+left-drag / UP,DOWN zoom.
"""
import argparse
import sys

import config as cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="run scientific validation checks and exit")
    p.add_argument("--provenance", action="store_true", help="print the OBSERVED / COMPUTED / SIMPLIFIED table and exit")
    p.add_argument("--cpu", action="store_true", help="force CPU backend")
    p.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED, help="seed for procedural visuals")
    p.add_argument("--warp", type=int, default=cfg.DEFAULT_WARP_INDEX, help="initial time-warp preset index")
    p.add_argument("--rate", type=float, default=None,
                   help="simulation seconds per real second (overrides --warp), e.g. 3600")
    p.add_argument("--control", type=float, default=0.5, help="initial accretion control 0..1")
    p.add_argument("--frames", type=int, default=0, help="exit after N frames (0 = run until closed)")
    p.add_argument("--headless", action="store_true", help="do not show the window (with --frames)")
    args = p.parse_args(argv)

    if args.provenance:
        from physics.provenance import format_table
        print(format_table(markdown=False))
        return 0
    if args.check:
        from checks import run_checks
        return 0 if run_checks(force_cpu=args.cpu) else 1

    from simulation import RunConfig, Simulation
    run = RunConfig(seed=args.seed, force_cpu=args.cpu, control=args.control, warp_index=args.warp,
                    sim_seconds_per_real_second=args.rate, max_frames=args.frames, show_window=not args.headless)
    return Simulation(run).loop()


if __name__ == "__main__":
    sys.exit(main())
