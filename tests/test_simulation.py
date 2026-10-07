import contextlib
import io
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nbody import (BarnesHutSimulation, OctreeNode, Particle,
                   compute_total_angular_momentum, compute_total_energy,
                   compute_total_momentum)


def circular_binary():
    speed = math.sqrt(0.5)
    return [Particle(-0.5, 0, 0, 0, -speed, 0), Particle(0.5, 0, 0, 0, speed, 0)]


class ForceTests(unittest.TestCase):
    def assertVectorsClose(self, actual, expected, tolerance=1e-12):
        self.assertEqual(len(actual), len(expected))
        for first, second in zip(actual, expected):
            self.assertLessEqual(math.dist(first, second), tolerance)

    def test_analytic_unequal_mass_two_body(self):
        particles = [Particle(-1, 0, 0, 0, 0, 0, mass=2),
                     Particle(1, 0, 0, 0, 0, 0, mass=3)]
        for mode in ("direct", "bh"):
            for theta in (0, 0.5, 100):
                with self.subTest(mode=mode, theta=theta):
                    sim = BarnesHutSimulation(particles=particles, mode=mode, theta=theta, eps=0)
                    self.assertVectorsClose(sim.accelerations(), [(0.75, 0, 0), (-0.5, 0, 0)])

    def test_direct_pairwise_momentum_balance(self):
        rng = random.Random(123)
        particles = [Particle(*(rng.uniform(-1, 1) for _ in range(3)), 0, 0, 0,
                              mass=rng.uniform(0.2, 2)) for _ in range(20)]
        sim = BarnesHutSimulation(particles=particles, mode="direct")
        accelerations = sim.accelerations()
        for axis in range(3):
            self.assertAlmostEqual(math.fsum(p.mass * a[axis] for p, a in zip(particles, accelerations)),
                                   0, delta=1e-12)

    def test_weighted_center_of_mass_and_idempotent_finalize(self):
        tree = OctreeNode((0, 0, 0), 4)
        tree.insert(Particle(1, 2, 3, 0, 0, 0, mass=2))
        tree.finalize()
        self.assertEqual(tree.com, [1, 2, 3])
        tree.insert(Particle(-1, 0, 1, 0, 0, 0, mass=3))
        tree.finalize()
        self.assertVectorsClose([tree.com], [(-0.2, 0.8, 1.8)])
        tree.finalize()
        self.assertVectorsClose([tree.com], [(-0.2, 0.8, 1.8)])
        self.assertEqual(tree.mass, 5)

    def test_exact_tree_matches_direct_outside_original_bounds(self):
        rng = random.Random(42)
        particles = [Particle(*(1000 + rng.uniform(-20, 20) for _ in range(3)), 0, 0, 0,
                              mass=rng.uniform(0.1, 3)) for _ in range(32)]
        direct = BarnesHutSimulation(particles=particles, mode="direct", eps=0.1)
        tree = BarnesHutSimulation(particles=particles, mode="bh", theta=0, eps=0.1)
        self.assertVectorsClose(tree.accelerations(), direct.accelerations())
        root = tree._build_tree()
        for particle in tree.particles:
            for value, center in zip((particle.x, particle.y, particle.z), root.center):
                self.assertLessEqual(abs(value - center), root.half_size)

    def test_single_particle_has_no_self_force(self):
        for mode in ("direct", "bh"):
            sim = BarnesHutSimulation(particles=[Particle(100, -200, 300, 0, 0, 0, mass=5)],
                                     mode=mode, theta=1000, eps=0)
            self.assertEqual(sim.accelerations(), [(0.0, 0.0, 0.0)])

    def test_tree_force_is_stable_under_large_translation(self):
        positions = [(-1, 0, 0), (1, 0, 0), (0, 2, 0), (0.25, 0, 0.5),
                     (-0.5, 0.5, 0), (2, -1, 0.25), (-2, -0.5, -1)]
        particles = [Particle(*position, 0, 0, 0, mass=index + 1)
                     for index, position in enumerate(positions)]
        reference = BarnesHutSimulation(particles=particles, theta=0.65, eps=0.05).accelerations()
        for shift in ((500, -500, 500), (1e6, -1e6, 1e6)):
            with self.subTest(shift=shift):
                translated = [Particle(p.x + shift[0], p.y + shift[1], p.z + shift[2],
                                       p.vx, p.vy, p.vz, p.mass) for p in particles]
                sim = BarnesHutSimulation(particles=translated, theta=0.65, eps=0.05)
                original_list = sim.particles
                original_particles = list(sim.particles)
                original_states = [vars(p).copy() for p in sim.particles]
                self.assertVectorsClose(sim.accelerations(), reference)
                self.assertIs(sim.particles, original_list)
                for particle, original, state in zip(sim.particles, original_particles, original_states):
                    self.assertIs(particle, original)
                    self.assertEqual(vars(particle), state)

    def test_tighter_opening_improves_seeded_force_accuracy(self):
        rng = random.Random(17)
        particles = [Particle(*(rng.uniform(-1, 1) for _ in range(3)), 0, 0, 0,
                              mass=1 / 64) for _ in range(64)]
        reference = BarnesHutSimulation(particles=particles, mode="direct").accelerations()
        normalization = sum(sum(value * value for value in acceleration) for acceleration in reference)
        errors = []
        for theta in (1.0, 0.5, 0.2):
            trial = BarnesHutSimulation(particles=particles, theta=theta).accelerations()
            errors.append(math.sqrt(sum(math.dist(a, b) ** 2 for a, b in zip(reference, trial)) / normalization))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertLess(errors[1], 0.01)
        self.assertLess(errors[2], 0.0001)

    def test_coincident_bucket_preserves_other_interactions(self):
        particles = [Particle(1, 2, 3, 0, 0, 0, mass=2),
                     Particle(1, 2, 3, 0, 0, 0, mass=3),
                     Particle(1 + 1e-10, 2, 3, 0, 0, 0, mass=4),
                     Particle(-1, 2, 3, 0, 0, 0, mass=5)]
        direct = BarnesHutSimulation(particles=particles, mode="direct", eps=0.05)
        tree = BarnesHutSimulation(particles=particles, mode="bh", theta=0, eps=0.05)
        self.assertVectorsClose(tree.accelerations(), direct.accelerations())
        coincident = BarnesHutSimulation(particles=particles[:2], eps=0.05)
        self.assertEqual(coincident.accelerations(), [(0, 0, 0), (0, 0, 0)])

    def test_unsoftened_coincident_particles_raise(self):
        particles = [Particle(0, 0, 0, 0, 0, 0), Particle(0, 0, 0, 0, 0, 0)]
        for mode in ("bh", "direct"):
            sim = BarnesHutSimulation(particles=particles, mode=mode, eps=0)
            with self.assertRaisesRegex(ValueError, "Coincident"):
                sim.accelerations()
        with self.assertRaisesRegex(ValueError, "Coincident"):
            compute_total_energy(particles, eps=0)


class IntegrationTests(unittest.TestCase):
    def test_singular_intermediate_stage_restores_state_and_identity(self):
        for integrator, speed in (("leapfrog", 0.875), ("rk4", 2.0)):
            with self.subTest(integrator=integrator):
                sim = BarnesHutSimulation(
                    particles=[Particle(-1, 0, 0, speed, 0, 0),
                               Particle(1, 0, 0, -speed, 0, 0)],
                    mode="direct", integrator=integrator, eps=0, dt=1,
                )
                original_list = sim.particles
                original_particles = list(sim.particles)
                original_states = [vars(p).copy() for p in sim.particles]
                with self.assertRaisesRegex(ValueError, "Coincident"):
                    sim.step()
                self.assertIs(sim.particles, original_list)
                for particle, original_particle, state in zip(sim.particles, original_particles, original_states):
                    self.assertIs(particle, original_particle)
                    self.assertEqual(vars(particle), state)
                self.assertEqual(sim.time, 0)

    def test_nonfinite_final_state_restores_all_integrators(self):
        for integrator in ("euler", "leapfrog", "rk4"):
            with self.subTest(integrator=integrator):
                sim = BarnesHutSimulation(particles=[Particle(1e308, 0, 0, 1e308, 0, 0)],
                                         mode="direct", integrator=integrator, dt=1)
                original = vars(sim.particles[0]).copy()
                with self.assertRaises((ValueError, ArithmeticError)):
                    sim.step()
                self.assertEqual(vars(sim.particles[0]), original)
                self.assertEqual(sim.time, 0)

    def orbital_error(self, integrator, dt):
        sim = BarnesHutSimulation(particles=circular_binary(), mode="direct", eps=0,
                                 integrator=integrator, dt=dt)
        for _ in range(round(1 / dt)):
            sim.step()
        expected = (-0.5 * math.cos(math.sqrt(2)), -0.5 * math.sin(math.sqrt(2)), 0)
        return math.dist((sim.particles[0].x, sim.particles[0].y, sim.particles[0].z), expected)

    def test_integrator_convergence_orders(self):
        for integrator, minimum_ratio, maximum_ratio in (("euler", 1.7, 2.3),
                                                          ("leapfrog", 3.7, 4.3),
                                                          ("rk4", 14, 18)):
            with self.subTest(integrator=integrator):
                coarse = self.orbital_error(integrator, 0.05)
                fine = self.orbital_error(integrator, 0.025)
                self.assertGreater(coarse / fine, minimum_ratio)
                self.assertLess(coarse / fine, maximum_ratio)

    def test_leapfrog_long_orbit_conservation(self):
        period = 2 * math.pi / math.sqrt(2)
        sim = BarnesHutSimulation(particles=circular_binary(), mode="direct", eps=0,
                                 integrator="leapfrog", dt=period / 256)
        energy = compute_total_energy(sim.particles, eps=0)
        angular = compute_total_angular_momentum(sim.particles)
        maximum_relative_energy_error = 0
        for _ in range(256 * 10):
            sim.step()
            maximum_relative_energy_error = max(maximum_relative_energy_error,
                abs((compute_total_energy(sim.particles, eps=0) - energy) / energy))
        self.assertLess(maximum_relative_energy_error, 2e-6)
        self.assertLess(math.dist(compute_total_momentum(sim.particles), (0, 0, 0)), 1e-12)
        self.assertLess(math.dist(compute_total_angular_momentum(sim.particles), angular), 1e-12)

    def test_all_integrators_match_exact_tree(self):
        for integrator in ("euler", "leapfrog", "rk4"):
            with self.subTest(integrator=integrator):
                direct = BarnesHutSimulation(particles=circular_binary(), mode="direct", integrator=integrator, eps=0)
                tree = BarnesHutSimulation(particles=circular_binary(), mode="bh", theta=0, integrator=integrator, eps=0)
                for _ in range(20):
                    direct.step()
                    tree.step()
                for first, second in zip(direct.particles, tree.particles):
                    for name in ("x", "y", "z", "vx", "vy", "vz"):
                        self.assertAlmostEqual(getattr(first, name), getattr(second, name), delta=1e-12)


class ApiTests(unittest.TestCase):
    def test_iterable_diagnostics(self):
        particles = [Particle(1, 2, 3, 4, 5, 6, mass=2), Particle(-1, 1, 0, 3, 2, 1, mass=3)]
        for diagnostic in (compute_total_momentum, compute_total_angular_momentum, compute_total_energy):
            with self.subTest(diagnostic=diagnostic.__name__):
                self.assertEqual(diagnostic(iter(particles)), diagnostic(particles))
        self.assertEqual(compute_total_momentum(iter(particles)), (17, 16, 15))
        self.assertEqual(compute_total_angular_momentum(iter(particles)), (-3, 15, -21))

    def test_seed_is_reproducible_without_global_rng_changes(self):
        before = random.getstate()
        for initial in ("spiral", "plummer", "kuzmin", "two_galaxies"):
            with self.subTest(initial=initial):
                first = BarnesHutSimulation(num_particles=10, initial=initial, seed=23)
                second = BarnesHutSimulation(num_particles=10, initial=initial, seed=23)
                third = BarnesHutSimulation(num_particles=10, initial=initial, seed=24)
                self.assertEqual(first.particles, second.particles)
                self.assertNotEqual(first.particles, third.particles)
                self.assertEqual(len(first.particles), 20 if initial == "two_galaxies" else 10)
        self.assertEqual(before, random.getstate())

    def test_supplied_particles_are_copied(self):
        particles = circular_binary()
        sim = BarnesHutSimulation(particles=iter(particles))
        self.assertIsNot(sim.particles[0], particles[0])
        sim.step()
        self.assertEqual(particles, circular_binary())

    def test_run_reports_zero_components_and_can_be_silent(self):
        sim = BarnesHutSimulation(particles=[Particle(0, 0, 0, 1, 0, 0)], dt=0.1)
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            result = sim.run(3, verbose=False)
        self.assertEqual(capture.getvalue(), "")
        self.assertEqual(result["steps"], 3)
        self.assertAlmostEqual(result["simulated_time"], 0.3)
        self.assertEqual(result["momentum_error"], 0)
        self.assertEqual(result["angular_momentum_error"], 0)
        self.assertEqual(result["relative_energy_error"], 0)
        self.assertEqual(result["final_momentum"], (1, 0, 0))
        self.assertGreaterEqual(result["elapsed_seconds"], 0)
        self.assertAlmostEqual(sim.time, 0.3)

    def test_zero_energy_empty_simulation(self):
        sim = BarnesHutSimulation(particles=[])
        self.assertEqual(sim.accelerations(), [])
        result = sim.run(1, verbose=False)
        self.assertIsNone(result["relative_energy_error"])
        self.assertEqual(result["energy_change"], 0)

    def test_parameter_validation(self):
        cases = ({"dt": 0}, {"dt": float("nan")}, {"theta": -1}, {"eps": -1},
                 {"num_particles": -1}, {"num_particles": 2.5}, {"num_particles": True},
                 {"integrator": "invalid"}, {"mode": "invalid"}, {"initial": "invalid"})
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BarnesHutSimulation(**kwargs)
        for particle in (Particle(0, 0, 0, 0, 0, 0, mass=0), Particle(float("inf"), 0, 0, 0, 0, 0)):
            with self.assertRaises(ValueError):
                BarnesHutSimulation(particles=[particle])
            for diagnostic in (compute_total_momentum, compute_total_angular_momentum, compute_total_energy):
                with self.assertRaises(ValueError):
                    diagnostic([particle])
        sim = BarnesHutSimulation(particles=[])
        for iterations in (-1, 0.5, True):
            with self.assertRaises(ValueError):
                sim.run(iterations, verbose=False)

    def test_recorder_receives_completed_steps(self):
        class Recorder:
            def __init__(self):
                self.frames = []

            def add_frame(self, particles):
                self.frames.append([(p.x, p.y, p.z) for p in particles])

        for integrator in ("euler", "leapfrog", "rk4"):
            recorder = Recorder()
            sim = BarnesHutSimulation(particles=[Particle(0, 0, 0, 1, 0, 0)], dt=0.1,
                                     integrator=integrator, recorder=recorder)
            sim.run(2, verbose=False)
            self.assertEqual(recorder.frames, [[(0.1, 0, 0)], [(0.2, 0, 0)]])


if __name__ == "__main__":
    unittest.main()
