"""Reproducible CPU force benchmarks against direct summation.

Run ``python benchmark.py --counts 32 64 128 --repeats 5``. Both solvers
receive identical, seeded positions with total mass fixed at one. Timings
cover one acceleration evaluation (including Barnes-Hut tree construction),
not initialization, integration, error calculation, plotting, or file I/O.
This synthetic cube is a force-solver fixture, not an equilibrium galaxy.

Raw samples and environment metadata are saved without optional dependencies.
Use ``--plot`` to also save a timing plot; only that option needs matplotlib.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

from nbody import BarnesHutSimulation, Particle
import nbody
from celestial_waltz import __version__


DEFAULT_COUNTS = (32, 64, 128)
SAMPLE_FIELDS = (
    "n_particles", "mode", "repeat", "seconds", "relative_rms_force_error"
)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def make_particles(n_particles, seed=0, total_mass=1.0):
    """Return reproducible uniform-cube particles using an isolated RNG.

    Positions are uniform on [-1, 1]^3, velocities are zero, and every
    particle has mass ``total_mass / n_particles``. The same seed gives
    shared position prefixes across resolutions, with masses rescaled.
    """
    _positive_integer(n_particles, "n_particles")
    if not math.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("total_mass must be finite and positive")
    rng = random.Random(seed)
    return [
        Particle(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1),
                 0.0, 0.0, 0.0, total_mass / n_particles)
        for _ in range(n_particles)
    ]


def relative_rms_force_error(approximate, reference):
    """Return RMS vector error divided by RMS direct acceleration.

    Particle-vector norms are used, rather than a mean of per-particle
    relative errors, so nearly force-free particles do not dominate. If
    both fields are zero the error is zero. If only the reference is zero,
    the ratio is undefined and returned as None (JSON null / empty CSV).
    """
    approximate, reference = list(approximate), list(reference)
    if len(approximate) != len(reference) or not reference:
        raise ValueError("force fields must have the same nonzero length")
    if any(len(v) != 3 or not all(math.isfinite(x) for x in v)
           for field in (approximate, reference) for v in field):
        raise ValueError("force fields must contain finite 3D vectors")
    difference_norm = math.sqrt(math.fsum(
        (a - r) ** 2
        for av, rv in zip(approximate, reference) for a, r in zip(av, rv)
    ))
    reference_norm = math.sqrt(math.fsum(x * x for v in reference for x in v))
    if reference_norm == 0:
        return 0.0 if difference_norm == 0 else None
    # The common sqrt(number of particles) in both RMS values cancels.
    return difference_norm / reference_norm


def _fixture_digest(particles):
    values = [[p.x, p.y, p.z, p.vx, p.vy, p.vz, p.mass] for p in particles]
    payload = json.dumps(values, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_benchmark(particle_counts=None, *, repeats=5, seed=0, theta=0.5,
                  eps=0.05, total_mass=1.0):
    """Measure repeated direct/Barnes-Hut accelerations on unchanged fixtures.

    Return a dict with ``metadata``, raw ``samples`` and per-solver
    ``summary`` records. Each solver gets one untimed warmup per particle
    count. Timed solver order alternates each repeat to reduce order bias.
    No universal speedup is assumed, especially for small particle counts.
    """
    counts = list(DEFAULT_COUNTS if particle_counts is None else particle_counts)
    if not counts:
        raise ValueError("particle_counts must not be empty")
    for count in counts:
        _positive_integer(count, "particle count")
    if len(set(counts)) != len(counts):
        raise ValueError("particle counts must be unique")
    _positive_integer(repeats, "repeats")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not math.isfinite(theta) or theta < 0:
        raise ValueError("theta must be finite and nonnegative")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not math.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("total_mass must be finite and positive")

    samples, summary, fixtures = [], [], []
    for count in counts:
        particles = make_particles(count, seed=seed, total_mass=total_mass)
        fixtures.append({"n_particles": count, "sha256": _fixture_digest(particles)})
        solvers = {
            mode: BarnesHutSimulation(particles=particles, mode=mode,
                                      theta=theta, eps=eps)
            for mode in ("direct", "bh")
        }
        reference = solvers["direct"].accelerations()
        solvers["bh"].accelerations()
        for repeat in range(1, repeats + 1):
            modes = ("direct", "bh") if repeat % 2 else ("bh", "direct")
            for mode in modes:
                start = time.perf_counter()
                result = solvers[mode].accelerations()
                elapsed = time.perf_counter() - start
                samples.append({
                    "n_particles": count,
                    "mode": mode,
                    "repeat": repeat,
                    "seconds": elapsed,
                    "relative_rms_force_error": relative_rms_force_error(result, reference),
                })
        for mode in ("direct", "bh"):
            group = [row for row in samples
                     if row["n_particles"] == count and row["mode"] == mode]
            times = [row["seconds"] for row in group]
            summary.append({
                "n_particles": count, "mode": mode,
                "mean_seconds": statistics.mean(times),
                "median_seconds": statistics.median(times),
                "stdev_seconds": statistics.stdev(times) if repeats > 1 else None,
                "min_seconds": min(times), "max_seconds": max(times),
                "relative_rms_force_error": group[0]["relative_rms_force_error"],
            })
    metadata = {
        "schema_version": 1,
        "package_version": __version__,
        "solver_source_sha256": hashlib.sha256(Path(nbody.__file__).read_bytes()).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "particle_counts": counts, "repeats": repeats, "seed": seed,
        "theta": theta, "eps": eps, "G": 1.0, "total_mass": total_mass,
        "fixture": "uniform cube [-1, 1]^3; zero velocity; equal particle masses",
        "fixture_hash_encoding": "SHA256 of compact JSON [x,y,z,vx,vy,vz,mass] rows",
        "fixtures": fixtures,
        "timing_scope": "CPU acceleration evaluation; includes Barnes-Hut tree construction; "
                        "excludes initialization, integration, error calculation, plotting and I/O",
        "clock": "time.perf_counter",
        "clock_resolution_seconds": time.get_clock_info("perf_counter").resolution,
        "warmup_evaluations_per_solver": 1,
        "solver_order": "direct,bh on odd repeats; bh,direct on even repeats",
        "force_error": "RMS norm of acceleration difference / RMS norm of direct acceleration",
        "zero_reference_policy": "0 if both fields are zero; null/empty CSV if only reference is zero",
        "environment": {
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
            "logical_cpu_count": os.cpu_count(),
        },
    }
    return {"metadata": metadata, "samples": samples, "summary": summary}


def write_results(result, output_dir):
    """Write every timing sample plus JSON metadata and summary statistics."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = output / "raw_timings.csv", output / "metadata.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(result["samples"])
    document = dict(result["metadata"], summary=result["summary"], samples_file=csv_path.name)
    json_path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return csv_path, json_path


def benchmark(particle_counts=None):
    """Compatibility helper returning counts and mean force-evaluation times."""
    result = run_benchmark(particle_counts)
    counts = result["metadata"]["particle_counts"]
    direct = [r["mean_seconds"] for r in result["summary"] if r["mode"] == "direct"]
    bh = [r["mean_seconds"] for r in result["summary"] if r["mode"] == "bh"]
    return counts, direct, bh


def plot_results(counts, direct_times, bh_times, output_path="benchmark.png"):
    """Export a publication-style mean-runtime plot for legacy callers.

    Only supplied mean timings are shown: this signature contains neither
    replicate measurements nor force errors. Use :func:`plot_benchmark`
    for the two-panel figure backed by the complete benchmark record.
    ``output_path`` supplies the basename for PNG, PDF and SVG exports.
    """
    import matplotlib.pyplot as plt
    from celestial_waltz.plotting import (
        COLORS, export_figure, figure_size, label_panel, publication_style,
    )

    counts, direct_times, bh_times = list(counts), list(direct_times), list(bh_times)
    if not counts or len(counts) != len(direct_times) or len(counts) != len(bh_times):
        raise ValueError("counts and timing series must have the same nonzero length")
    if any(not math.isfinite(t) or t < 0 for t in direct_times + bh_times):
        raise ValueError("timing values must be finite and nonnegative")
    output_stem = Path(output_path).with_suffix("")
    with publication_style():
        fig, ax = plt.subplots(figsize=figure_size(width="double", height_mm=85),
                               layout="constrained")
        ax.plot(counts, [t * 1000 for t in direct_times], "o-",
                color=COLORS["direct"], label="Direct summation", markerfacecolor="white")
        ax.plot(counts, [t * 1000 for t in bh_times], "s--",
                color=COLORS["bh"], label="Barnes-Hut", markerfacecolor="white")
        if min(direct_times + bh_times) > 0:
            ax.set_yscale("log")
        ax.set(xlabel="Particle number, $N$", ylabel="Mean force-evaluation time (ms)")
        label_panel(ax, "a", title="Force evaluation time")
        ax.legend(loc="best")
        caption = (
            "Mean CPU force-evaluation times supplied to the legacy plot_results interface. "
            "No replicate variability or force accuracy is inferred from these mean values. "
            "Connecting lines are visual guides, not fitted scaling laws."
        )
        try:
            return export_figure(
                fig, output_stem, caption=caption, data_sources=[],
                metadata={"figure_type": "benchmark_runtime_legacy",
                          "runtime_statistic": "supplied mean", "time_unit": "ms",
                          "particle_counts": counts},
            )
        finally:
            plt.close(fig)


def plot_benchmark(result, output_dir, *, data_sources=None, show_replicates=False):
    """Export runtime and force accuracy from recorded benchmark samples.

    ``result`` is the mapping returned by :func:`run_benchmark` or a path
    to a directory containing ``raw_timings.csv`` and ``metadata.json``.
    A saved directory can therefore be replotted without rerunning timings.
    The timing panel shows median and 25th--75th percentiles; the accuracy
    panel shows median normalized RMS error as a percentage on a linear axis.
    PNG (600 dpi), PDF, SVG, caption and provenance exports share the basename
    ``benchmark``. Optional plotting packages are imported only on demand.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    import numpy as np
    from celestial_waltz.plotting import (
        COLORS, export_figure, figure_size, label_panel, publication_style,
    )

    if isinstance(result, (str, os.PathLike)):
        source = Path(result)
        metadata_path, samples_path = source / "metadata.json", source / "raw_timings.csv"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with samples_path.open(newline="", encoding="utf-8") as stream:
            samples = [
                {"n_particles": int(row["n_particles"]), "mode": row["mode"],
                 "repeat": int(row["repeat"]), "seconds": float(row["seconds"]),
                 "relative_rms_force_error": (
                     float(row["relative_rms_force_error"])
                     if row["relative_rms_force_error"] else None)}
                for row in csv.DictReader(stream)
            ]
        if data_sources is None:
            data_sources = [samples_path, metadata_path]
    else:
        metadata, samples = result["metadata"], result["samples"]
        if data_sources is None:
            # In-memory inputs must not be attributed to unrelated files that
            # happen to exist in the destination. The CLI supplies its paths.
            data_sources = []

    counts = sorted(metadata["particle_counts"])
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("the benchmark must contain positive particle counts")
    groups = {}
    undefined_errors = 0
    for mode in ("direct", "bh"):
        for count in counts:
            rows = [row for row in samples
                    if row["mode"] == mode and row["n_particles"] == count]
            if not rows:
                raise ValueError(f"no recorded {mode} samples for N={count}")
            if (len(rows) != metadata["repeats"] or
                    {row["repeat"] for row in rows} != set(range(1, metadata["repeats"] + 1))):
                raise ValueError(f"incomplete or duplicated {mode} repeats for N={count}")
            times = np.asarray([row["seconds"] for row in rows], dtype=float) * 1000
            if not np.isfinite(times).all() or (times < 0).any():
                raise ValueError("recorded times must be finite and nonnegative")
            errors = [row["relative_rms_force_error"] for row in rows]
            undefined_errors += sum(error is None for error in errors)
            finite_errors = [float(error) * 100 for error in errors if error is not None]
            if any(not math.isfinite(error) or error < 0 for error in finite_errors):
                raise ValueError("recorded force errors must be finite and nonnegative")
            groups[mode, count] = {
                "times": times, "quartiles": np.percentile(times, [25, 50, 75]),
                "error_percent": float(np.median(finite_errors)) if finite_errors else np.nan,
            }

    with publication_style():
        fig, axes = plt.subplots(1, 2, figsize=figure_size(width="double", height_mm=85),
                                 layout="constrained")
        styles = {"direct": ("o", "-", "Direct summation"),
                  "bh": ("s", "--", "Barnes-Hut")}
        for mode, (marker, linestyle, label) in styles.items():
            quartiles = np.asarray([groups[mode, count]["quartiles"] for count in counts]).T
            q25, median, q75 = quartiles
            axes[0].errorbar(
                counts, median, yerr=[median - q25, q75 - median],
                marker=marker, linestyle=linestyle, color=COLORS[mode],
                markerfacecolor="white", capsize=2.5, label=label, zorder=3,
            )
            if show_replicates:
                for count in counts:
                    values = groups[mode, count]["times"]
                    axes[0].scatter(np.full(values.size, count), values, s=7,
                                    color=COLORS[mode], alpha=0.35, zorder=2)
            axes[1].plot(
                counts, [groups[mode, count]["error_percent"] for count in counts],
                marker=marker, linestyle=linestyle, color=COLORS[mode],
                markerfacecolor="white", label=label, zorder=3,
            )
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xticks(counts)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlabel("Particle number, $N$")
            ax.margins(x=0.12)
        if all((group["times"] > 0).all() for group in groups.values()):
            axes[0].set_yscale("log")
        axes[0].set_ylabel("Force-evaluation time (ms)")
        axes[1].set_ylabel("Normalized RMS force error (%)")
        # Keep the direct-summation zero visible; no log transform or epsilon floor.
        errors = [group["error_percent"] for group in groups.values()
                  if math.isfinite(group["error_percent"])]
        error_scale = max(errors, default=0.0)
        if error_scale == 0:
            axes[1].set_ylim(-0.05, 0.5)
        else:
            axes[1].set_ylim(-0.07 * error_scale, 1.15 * error_scale)
        label_panel(axes[0], "a", title="Runtime: median and IQR")
        label_panel(axes[1], "b", title="Force accuracy")
        axes[0].legend(loc="upper left")

        caption = (
            "Direct summation and Barnes-Hut force evaluation on identical seeded, "
            "fixed-total-mass particle fixtures. (a) Median elapsed wall-clock time with error bars "
            "spanning the 25th to 75th percentiles (linear-interpolated sample quantiles). "
            "Timing spread describes execution variability, not physical uncertainty. "
            "(b) Median normalized RMS acceleration error relative to direct summation, "
            "expressed as a percentage: 100 times the RMS norm of the acceleration "
            "difference divided by the RMS norm of the direct acceleration. "
            "The direct reference is zero on the linear accuracy axis. "
            f"Particle counts: {', '.join(str(n) for n in counts)}; "
            f"seed={metadata['seed']}; theta={metadata['theta']}; "
            f"softening eps={metadata['eps']}; G={metadata['G']}; "
            f"total mass={metadata['total_mass']}; {metadata['repeats']} repeats per solver "
            f"and count; {metadata['warmup_evaluations_per_solver']} untimed warmup per solver. "
            f"Fixture: {metadata['fixture']}. Scope: {metadata['timing_scope']}. "
            "Connecting lines are visual guides, not fitted scaling laws."
        )
        if show_replicates:
            caption += " Faint points in panel (a) show individual timing replicates."
        if undefined_errors:
            caption += (f" {undefined_errors} undefined normalized-error samples "
                        "(zero reference norm with nonzero discrepancy) are omitted.")
        plot_metadata = {
            "figure_type": "benchmark_runtime_and_accuracy",
            "renderer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime_statistic": "median with 25th--75th percentile interval",
            "quantile_method": "linear interpolation", "time_unit": "ms",
            "accuracy_statistic": "median normalized RMS force error", "accuracy_unit": "percent",
            "show_replicates": show_replicates,
            "experiment": {key: metadata[key] for key in (
                "particle_counts", "repeats", "seed", "theta", "eps", "G", "total_mass",
                "fixture", "timing_scope", "warmup_evaluations_per_solver")},
            "benchmark_environment": metadata.get("environment", {}),
            "benchmark_created_at_utc": metadata.get("created_at_utc"),
            "solver_source_sha256": metadata.get("solver_source_sha256"),
            "plotted_values": [
                {"n_particles": count, "mode": mode,
                 "q25_milliseconds": float(group["quartiles"][0]),
                 "median_milliseconds": float(group["quartiles"][1]),
                 "q75_milliseconds": float(group["quartiles"][2]),
                 "median_error_percent": (group["error_percent"]
                                          if math.isfinite(group["error_percent"]) else None)}
                for (mode, count), group in groups.items()
            ],
        }
        try:
            return export_figure(fig, Path(output_dir) / "benchmark", caption=caption,
                                 data_sources=data_sources, metadata=plot_metadata)
        finally:
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=list(DEFAULT_COUNTS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--plot", action="store_true", help="also save a plot (requires matplotlib)")
    args = parser.parse_args(argv)
    try:
        result = run_benchmark(args.counts, repeats=args.repeats, seed=args.seed, theta=args.theta)
    except ValueError as exc:
        parser.error(str(exc))
    csv_path, json_path = write_results(result, args.output)
    for row in result["summary"]:
        error = row["relative_rms_force_error"]
        error_text = "undefined" if error is None else f"{error:.3e}"
        print(f"N={row['n_particles']:5d} {row['mode']:6s} "
              f"median={row['median_seconds']:.6f}s relative RMS force error={error_text}")
    print(f"Saved {csv_path} and {json_path}")
    if args.plot:
        try:
            plot_benchmark(result, args.output, data_sources=[csv_path, json_path])
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            print("Plot skipped: install the plotting extra to use --plot. Raw results were saved.",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
