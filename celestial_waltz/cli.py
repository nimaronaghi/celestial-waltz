"""Command-line entry point for reproducible simulations and validation."""

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

from . import __version__, BarnesHutSimulation


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "benchmark":
        from benchmark import main as benchmark_main
        return benchmark_main(argv[1:])
    parser = argparse.ArgumentParser(description="Celestial Waltz: reproducible N-body experiments")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    simulate = commands.add_parser("simulate", help="Run a seeded headless simulation")
    simulate.add_argument("--particles", type=int, default=100)
    simulate.add_argument("--steps", type=int, default=100)
    simulate.add_argument("--dt", type=float, default=0.01)
    simulate.add_argument("--eps", type=float, default=0.05)
    simulate.add_argument("--theta", type=float, default=0.5)
    simulate.add_argument("--seed", type=int, default=0)
    simulate.add_argument("--mode", choices=("direct", "bh"), default="bh")
    simulate.add_argument("--integrator", choices=("euler", "leapfrog", "rk4"), default="leapfrog")
    simulate.add_argument("--initial", choices=("spiral", "plummer", "kuzmin", "two_galaxies"), default="spiral")
    simulate.add_argument("--output", type=Path, default=Path("results/simulation.json"))
    validate = commands.add_parser("validate", help="Reproduce the analytic two-body validation report")
    validate.add_argument("--output", type=Path, default=Path("results/two-body"))
    validate.add_argument("--orbits", type=int, default=3)
    validate.add_argument("--plot", action="store_true", help="Requires the plot extra")
    figures = commands.add_parser("figures", help="Export publication figures from saved measurements")
    figures.add_argument("--validation", type=Path, help="Directory with a saved validation report")
    figures.add_argument("--benchmark", type=Path, help="Directory with saved benchmark data")
    figures.add_argument("--output", type=Path, help="Optional export root; defaults to the source directories")
    commands.add_parser("benchmark", help="Measure force accuracy and runtime; use benchmark --help")
    args = parser.parse_args(argv)
    try:
        if args.command == "figures":
            if args.validation is None and args.benchmark is None:
                parser.error("figures requires --validation and/or --benchmark")
            if args.validation is not None:
                from .validation import render_saved_report
                destination = args.output / "validation" if args.output else args.validation
                render_saved_report(args.validation, output=destination)
                print(f"Exported validation PDF, SVG, PNG and caption to {destination}")
            if args.benchmark is not None:
                from benchmark import plot_benchmark
                destination = args.output / "benchmark" if args.output else args.benchmark
                plot_benchmark(args.benchmark, destination)
                print(f"Exported benchmark PDF, SVG, PNG and caption to {destination}")
            return 0
        if args.command == "validate":
            from .validation import generate_report
            report = generate_report(args.output, orbits=args.orbits, plot=args.plot)
            print(f"Validation {'passed' if report['passed'] else 'failed'}: {args.output / 'report.json'}")
            return 0 if report["passed"] else 1
        config = {key: getattr(args, key) for key in
                  ("dt", "eps", "theta", "seed", "mode", "integrator", "initial")}
        config["num_particles"] = args.particles
        sim = BarnesHutSimulation(**config)
        diagnostics = sim.run(args.steps, verbose=False)
        payload = {"package_version": __version__, "python": platform.python_version(),
                   "config": dict(config, steps=args.steps), "diagnostics": diagnostics,
                   "particles": [asdict(particle) for particle in sim.particles]}
        encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(f"Saved {len(sim.particles)} particles and diagnostics to {args.output}")
        return 0
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
