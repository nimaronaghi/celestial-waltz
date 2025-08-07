import os
import sys
import unittest
import random
import math

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

    def test_initial_conditions(self):
        """Verify that custom initial condition generators produce the expected number of particles."""
        sim_plummer = BarnesHutSimulation(num_particles=20, initial="plummer")
        self.assertEqual(len(sim_plummer.particles), 20)
        sim_kuzmin = BarnesHutSimulation(num_particles=15, initial="kuzmin")
        self.assertEqual(len(sim_kuzmin.particles), 15)
        sim_collision = BarnesHutSimulation(num_particles=12, initial="two_galaxies")
        # two galaxies yield twice the number of particles passed in
        self.assertEqual(len(sim_collision.particles), 24)

    def test_rk4_runs(self):
        """Ensure that the RK4 integrator executes without errors for a few steps."""
        sim = BarnesHutSimulation(num_particles=5, integrator="rk4", initial="spiral")
        # run a few steps and check that positions remain finite
        sim.step()
        sim.step()
        for p in sim.particles:
            self.assertTrue(math.isfinite(p.x) and math.isfinite(p.y) and math.isfinite(p.z))


if __name__ == "__main__":
    unittest.main()
