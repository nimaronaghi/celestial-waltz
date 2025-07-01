import os
import sys
import unittest
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa: E402
from nbody import BarnesHutSimulation, compute_total_energy  # noqa: E402


class TestSimulation(unittest.TestCase):
    def test_energy_conservation(self):
        random.seed(0)
        sim = BarnesHutSimulation(num_particles=10, dt=0.01, integrator="leapfrog")
        e0 = compute_total_energy(sim.particles, eps=sim.eps)
        sim.run(5)
        e1 = compute_total_energy(sim.particles, eps=sim.eps)
        self.assertAlmostEqual(e0, e1, delta=10.0)


if __name__ == "__main__":
    unittest.main()
