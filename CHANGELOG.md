# Changelog

## 0.2.0 — Numerical verification and reproducible experiments

- Correct direct-force accumulation, mass-weighted tree centers, dynamic tree
  bounds, and self-interaction handling for close or coincident particles.
- Build trees in a local coordinate frame and restore state after failed CPU steps.
- Preserve physical close-pair separations independently of tree partition
  coordinates, including systems with very distant outliers.
- Replace the Plummer toy velocity prescription with the isotropic continuum
  distribution, and center/deboost collision components before placing them.
- Verify softened force–potential consistency, eccentric unequal-mass Kepler
  motion, invariants, symmetry, time reversal, and initial-state distributions.
- Make GUI timestep/softening displays match actual code units and cull invisible
  particles instead of pinning them to image boundaries.
- Centralize acceleration dispatch, remove per-step Python thread pools, and
  support copied particle fixtures and explicit random seeds.
- Use leapfrog by default; retain symplectic Euler and RK4.
- Add robust energy, momentum and angular momentum diagnostics.
- Correct tensor force direction and tree handling; add optional parity tests.
- Add an installable package and simulation, validation and benchmark commands.
- Add analytic two-body reports, repeated force benchmarks, raw results and metadata.
- Replace smoke-only checks with numerical regression tests and configure CI.
- Document model limitations and illustrative galaxy initial conditions.
- Add a shared scientific figure style with fixed physical dimensions, vector
  PDF/SVG and 600 dpi PNG exports, standalone captions, and provenance manifests.
- Plot timing medians and interquartile ranges alongside force accuracy;
  preserve zero and undefined measurements without artificial log-axis floors.
- Add a `figures` command to redraw saved data without rerunning experiments,
  export regression tests, and research-style methods and figure documentation.
