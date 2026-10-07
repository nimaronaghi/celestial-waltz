"""Public API for dimensionless Newtonian gravity experiments (G = 1)."""

from nbody import (
    BarnesHutSimulation,
    Particle,
    compute_total_angular_momentum,
    compute_total_energy,
    compute_total_momentum,
)

__version__ = "0.2.0"
__all__ = [
    "BarnesHutSimulation",
    "Particle",
    "compute_total_angular_momentum",
    "compute_total_energy",
    "compute_total_momentum",
]
