"""Compare numerical equal-mass circular orbits with their analytic solution."""

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import nbody

from . import (
    __version__,
    BarnesHutSimulation,
    Particle,
    compute_total_angular_momentum,
    compute_total_energy,
    compute_total_momentum,
)


def circular_binary():
    """Return M=1, separation=1, G=1: angular frequency 1, period 2*pi."""
    return [
        Particle(-0.5, 0.0, 0.0, 0.0, -0.5, 0.0, mass=0.5),
        Particle(0.5, 0.0, 0.0, 0.0, 0.5, 0.0, mass=0.5),
    ]


def orbit_experiment(integrator="leapfrog", steps_per_orbit=256, orbits=3):
    """Measure position error and conserved quantities at every completed step."""
    if not isinstance(steps_per_orbit, int) or steps_per_orbit < 16:
        raise ValueError("steps_per_orbit must be an integer >= 16")
    if not isinstance(orbits, int) or orbits < 1:
        raise ValueError("orbits must be a positive integer")
    dt = 2.0 * math.pi / steps_per_orbit
    sim = BarnesHutSimulation(
        particles=circular_binary(), mode="direct", integrator=integrator,
        dt=dt, eps=0.0,
    )
    initial_energy = compute_total_energy(sim.particles, eps=0.0)
    initial_angular = compute_total_angular_momentum(sim.particles)
    rows = []
    for step in range(steps_per_orbit * orbits + 1):
        if step:
            sim.step()
        time = step * dt
        p = sim.particles[0]
        expected_x = -0.5 * math.cos(time)
        expected_y = -0.5 * math.sin(time)
        momentum = compute_total_momentum(sim.particles)
        angular = compute_total_angular_momentum(sim.particles)
        rows.append({
            "step": step, "time": time, "x": p.x, "y": p.y, "z": p.z,
            "analytic_x": expected_x, "analytic_y": expected_y,
            "position_error": math.dist((p.x, p.y, p.z), (expected_x, expected_y, 0.0)),
            "relative_energy_error": abs(compute_total_energy(sim.particles, eps=0.0) - initial_energy) / abs(initial_energy),
            "momentum_error": math.sqrt(sum(value * value for value in momentum)),
            "angular_momentum_error": math.dist(angular, initial_angular),
        })
    summary = {
        "integrator": integrator, "steps_per_orbit": steps_per_orbit,
        "orbits": orbits, "dt": dt,
        "final_position_error": rows[-1]["position_error"],
        "max_position_error": max(row["position_error"] for row in rows),
        "max_relative_energy_error": max(row["relative_energy_error"] for row in rows),
        "max_momentum_error": max(row["momentum_error"] for row in rows),
        "max_angular_momentum_error": max(row["angular_momentum_error"] for row in rows),
    }
    return summary, rows


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_report(output, *, orbits=3, plot=False):
    """Write raw trajectories, convergence data and explicit validation checks.

    This verifies one smooth, unsoftened, circular two-body problem. It does not
    establish accuracy of arbitrary many-body configurations or tree gravity.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    trajectories = {}
    orders = {}
    checks = {}
    for integrator, minimum_order in (("leapfrog", 1.8), ("rk4", 3.6)):
        results = []
        for steps in (64, 128, 256):
            summary, rows = orbit_experiment(integrator, steps, orbits)
            summaries.append(summary)
            results.append(summary)
        trajectories[integrator] = rows
        _write_csv(output / f"{integrator}_trajectory.csv", rows)
        errors = [result["final_position_error"] for result in results]
        orders[integrator] = [math.log2(a / b) if a > 0 and b > 0 else None
                              for a, b in zip(errors, errors[1:])]
        checks[f"{integrator}_convergence"] = all(
            order is not None and order >= minimum_order for order in orders[integrator])
        checks[f"{integrator}_energy"] = results[-1]["max_relative_energy_error"] < 1e-4
        checks[f"{integrator}_momentum"] = results[-1]["max_momentum_error"] < 1e-12
        checks[f"{integrator}_angular_momentum"] = results[-1]["max_angular_momentum_error"] < 1e-6
    _write_csv(output / "convergence.csv", summaries)
    report = {
        "package_version": __version__, "python": platform.python_version(),
        "solver_source_sha256": hashlib.sha256(Path(nbody.__file__).read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "problem": {"G": 1.0, "total_mass": 1.0, "separation": 1.0,
                    "softening": 0.0, "period": 2 * math.pi, "orbits": orbits,
                    "solver": "direct", "steps_per_orbit": [64, 128, 256]},
        "acceptance_criteria": {"minimum_order": {"leapfrog": 1.8, "rk4": 3.6},
                                "max_relative_energy_error": 1e-4,
                                "max_momentum_error": 1e-12,
                                "max_angular_momentum_error": 1e-6,
                                "invariants_evaluated_at_steps_per_orbit": 256},
        "observed_orders": orders, "checks": checks,
        "passed": all(checks.values()), "experiments": summaries,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if plot:
        plot_report(output, summaries, trajectories)
    return report


def plot_report(output, summaries, trajectories, report=None, *, data_sources=None):
    """Export a four-panel verification figure with a standalone caption."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator
    from .plotting import (
        COLORS, export_figure, figure_size, label_panel,
        positive_or_nan, publication_style,
    )

    output = Path(output)
    if report is None:
        report_file = output / "report.json"
        report = json.loads(report_file.read_text(encoding="utf-8")) if report_file.exists() else {}
    problem = report.get("problem", {})
    period = problem.get("period", 2 * math.pi)
    separation = problem.get("separation", 1.0)
    orbits = summaries[0]["orbits"]
    steps = len(trajectories["leapfrog"]) - 1
    steps_per_orbit = round(steps / orbits)
    if data_sources is None:
        data_sources = [output / name for name in
                        ("report.json", "convergence.csv", "leapfrog_trajectory.csv", "rk4_trajectory.csv")
                        if (output / name).is_file()]

    caption = (
        f"Circular-binary verification of the direct Newtonian solver. "
        f"Two equal masses have total mass M={problem.get('total_mass', 1):g}, "
        f"initial separation d0={separation:g}, G={problem.get('G', 1):g}, and "
        f"softening epsilon={problem.get('softening', 0):g}, in code units. "
        f"(a) The first particle's orbit at {steps_per_orbit} steps per period, "
        f"compared with the analytic circular solution. "
        f"(b) Position residual norm, normalized by d0, across {orbits:g} periods "
        f"at the same resolution. (c) Final position residual after {orbits:g} periods "
        f"versus normalized timestep h=Delta t/P. Reference slopes h^2 and h^4 "
        f"are normalized to 0.4 times the finest measured error for the corresponding "
        f"method; they are guides, not fits. "
        f"(d) Absolute relative energy error at {steps_per_orbit} steps per period. "
        f"Exact zeros are omitted from logarithmic axes; positive values are not clipped. "
        f"These are deterministic numerical errors, not statistical confidence intervals. "
        f"This experiment verifies a smooth two-body problem and does not establish "
        f"accuracy for arbitrary many-body systems or galaxy initial conditions."
    )
    styles = {
        "leapfrog": {"color": COLORS["leapfrog"], "linestyle": "-", "marker": "o", "label": "Leapfrog"},
        "rk4": {"color": COLORS["rk4"], "linestyle": "--", "marker": "s", "label": "RK4"},
    }
    with publication_style():
        fig = Figure(figsize=figure_size(height_mm=132), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 2)
        orbit, residual, convergence, energy = axes.flat
        try:
            reference = trajectories["leapfrog"]
            orbit.plot([row["analytic_x"] / separation for row in reference],
                       [row["analytic_y"] / separation for row in reference],
                       color=COLORS["reference"], lw=1.3, label="Analytic")
            for integrator, style in styles.items():
                rows = trajectories[integrator]
                time = [row["time"] / period for row in rows]
                # Sparse open markers keep overlapping orbits distinguishable in print.
                orbit.plot([row["x"] / separation for row in rows],
                           [row["y"] / separation for row in rows],
                           **style, markevery=max(1, steps_per_orbit // 10),
                           mfc="white", ms=2.5, lw=0.8, alpha=0.9)
                residual.semilogy(time,
                                 positive_or_nan([row["position_error"] / separation for row in rows]),
                                 **style, markevery=max(1, steps // 12), mfc="white", ms=3)
                energy.semilogy(time,
                               positive_or_nan([row["relative_energy_error"] for row in rows]),
                               **style, markevery=max(1, steps // 12), mfc="white", ms=3)
                selected = sorted((row for row in summaries if row["integrator"] == integrator),
                                  key=lambda row: row["dt"])
                h = [row["dt"] / period for row in selected]
                errors = [row["final_position_error"] / separation for row in selected]
                convergence.loglog(h, positive_or_nan(errors), **style, mfc="white")
                order = 2 if integrator == "leapfrog" else 4
                if errors[0] > 0:
                    guide = [0.4 * errors[0] * (value / h[0]) ** order for value in h]
                    convergence.loglog(h, guide, color="0.45",
                                       linestyle=":" if order == 2 else "-.",
                                       lw=0.8, label=rf"$h^{order}$ guide")
            orbit.set(xlabel=r"$x/d_0$", ylabel=r"$y/d_0$", aspect="equal",
                      xlim=(-0.6, 0.6), ylim=(-0.6, 0.6),
                      xticks=(-0.5, 0, 0.5), yticks=(-0.5, 0, 0.5))
            orbit.xaxis.set_minor_locator(AutoMinorLocator(2))
            orbit.yaxis.set_minor_locator(AutoMinorLocator(2))
            residual.set(xlabel=r"$t/P$", ylabel=r"$\|\delta\mathbf{r}\|/d_0$", xlim=(0, orbits))
            convergence.set(xlabel=r"$h=\Delta t/P$", ylabel=r"$\|\delta\mathbf{r}(t_{\mathrm{f}})\|/d_0$")
            energy.set(xlabel=r"$t/P$", ylabel=r"$|E(t)-E_0|/|E_0|$", xlim=(0, orbits))
            for axis in (residual, energy):
                axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, min_n_ticks=3))
                axis.xaxis.set_minor_locator(AutoMinorLocator(2))
                # Zero-only data have no positive values to define logarithmic limits.
                if not any(math.isfinite(value) and value > 0 for line in axis.lines for value in line.get_ydata()):
                    axis.set_ylim(1e-16, 1.0)
                    axis.text(0.5, 0.5, "No positive errors to display",
                              ha="center", va="center", transform=axis.transAxes)
            orbit.legend(loc="center")
            residual.legend(loc="lower right")
            convergence.legend(loc="center left", bbox_to_anchor=(0.02, 0.57),
                               ncols=2, columnspacing=0.8,
                               handlelength=1.8, fontsize=7)
            energy.legend(loc="lower right")
            for axis, letter, title in zip(
                    axes.flat, "abcd",
                    ("Orbit", "Position residual", "Timestep convergence", "Energy conservation")):
                label_panel(axis, letter, title)
            return export_figure(
                fig, output / "validation", caption=caption, data_sources=data_sources,
                metadata={
                    "title": "Circular-binary numerical verification",
                    "renderer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "normalized_time": "t/P", "normalized_length": "r/d0",
                    "log_zero_policy": "Nonpositive/undefined values omitted; no positive-value floor",
                    "convergence_reference_orders": [2, 4], "reference_guides_are_fits": False,
                    "problem": problem,
                },
            )
        finally:
            fig.clear()


def render_saved_report(source, output=None):
    """Re-render recorded measurements without rerunning a solver or changing data."""
    source = Path(source)
    output = source if output is None else Path(output)
    report = json.loads((source / "report.json").read_text(encoding="utf-8"))

    def read_rows(path, text_fields=()):
        with path.open(newline="", encoding="utf-8") as stream:
            return [{key: value if key in text_fields else float(value)
                     for key, value in row.items()} for row in csv.DictReader(stream)]

    summaries = read_rows(source / "convergence.csv", text_fields=("integrator",))
    trajectories = {name: read_rows(source / f"{name}_trajectory.csv") for name in ("leapfrog", "rk4")}
    paths = [source / name for name in
             ("report.json", "convergence.csv", "leapfrog_trajectory.csv", "rk4_trajectory.csv")]
    return plot_report(output, summaries, trajectories, report=report, data_sources=paths)
