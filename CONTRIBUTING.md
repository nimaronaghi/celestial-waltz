# Contributing

Use Python 3.10 or newer and a virtual environment. Install with
`python -m pip install -e .`; add `.[plot]` for figures or `.[gpu]` for tensor tests.

Before proposing a change:

1. Run `python -m unittest discover -s tests -v`.
2. Run `python -m celestial_waltz validate --output results/two-body`.
3. For numerical changes, state the assumptions and demonstrate an analytic
   result, convergence, or agreement with an independent reference.
4. For optimizations, preserve the input fixture, report accuracy and repeated
   raw timings, and distinguish setup, transfer, and computation costs.

Prefer physical invariants, known solutions, boundary cases and reproducibility
over tests that only check execution. Do not assert wall-clock speed thresholds.
Document optional dependencies and hardware needed for checks you could not run.

Issues should include versions, full configuration and seed, a reproduction
command, expected behavior, and observed result. Pull requests should explain
the problem, design choice, and validation. Keep changes focused; exclude large
datasets and virtual environments from Git.
