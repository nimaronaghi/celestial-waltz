"""Independent mechanics checks beyond the equal-mass circular example.

The elliptic oracle solves Kepler's equation, rather than integrating the
equations of motion. Orbital-plane coordinates follow the formulae documented
by NASA/JPL: https://ssd.jpl.nasa.gov/planets/approx_pos.html (in radians here).
No planetary element tables or approximate ephemerides are used.
"""

import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nbody import (BarnesHutSimulation, Particle, compute_total_angular_momentum,
                   compute_total_energy, compute_total_momentum)


def position(particle):
    return particle.x, particle.y, particle.z


def velocity(particle):
    return particle.vx, particle.vy, particle.vz


def rotate(vector):
    """Rodrigues rotation about (1, 2, 3), avoiding octree symmetry axes."""
    axis = tuple(value / math.sqrt(14) for value in (1, 2, 3))
    cosine, sine = math.cos(0.73), math.sin(0.73)
    projection = sum(a * v for a, v in zip(axis, vector))
    cross = (axis[1] * vector[2] - axis[2] * vector[1],
             axis[2] * vector[0] - axis[0] * vector[2],
             axis[0] * vector[1] - axis[1] * vector[0])
    return tuple(v * cosine + c * sine + a * projection * (1 - cosine)
                 for v, c, a in zip(vector, cross, axis))


def eccentric_binary(time, semimajor=1.4, eccentricity=0.6, masses=(0.7, 1.3)):
    """Exact isolated binary state, G=1, pericenter at time zero, zero COM."""
    total_mass = sum(masses)
    mean_motion = math.sqrt(total_mass / semimajor ** 3)
    mean_anomaly = math.remainder(mean_motion * time, 2 * math.pi)
    eccentric_anomaly = mean_anomaly
    for _ in range(50):
        residual = eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly) - mean_anomaly
        correction = residual / (1 - eccentricity * math.cos(eccentric_anomaly))
        eccentric_anomaly -= correction
        if abs(correction) < 2e-15:
            break
    else:
        raise AssertionError("Independent Kepler oracle failed to converge")
    q = math.sqrt(1 - eccentricity ** 2)
    relative_position = (semimajor * (math.cos(eccentric_anomaly) - eccentricity),
                         semimajor * q * math.sin(eccentric_anomaly), 0)
    scale = semimajor * mean_motion / (1 - eccentricity * math.cos(eccentric_anomaly))
    relative_velocity = (-scale * math.sin(eccentric_anomaly),
                         scale * q * math.cos(eccentric_anomaly), 0)
    return [Particle(*(fraction * x for x in relative_position),
                     *(fraction * v for v in relative_velocity), mass=mass)
            for fraction, mass in ((-masses[1] / total_mass, masses[0]),
                                   (masses[0] / total_mass, masses[1]))]


def interacting_fixture():
    return [Particle(-0.8, 0.25, 0.4, 0.1, -0.2, 0.3, mass=0.7),
            Particle(0.6, -0.5, 0.2, -0.1, 0.3, -0.2, mass=1.3),
            Particle(0.2, 0.8, -0.6, 0.2, 0.1, -0.1, mass=0.4),
            Particle(-0.2, -0.4, -0.9, -0.2, -0.1, 0.2, mass=2.1)]


class TestPhysicalForces(unittest.TestCase):
    def assertVectorsClose(self, actual, expected, tolerance=2e-12):
        self.assertEqual(len(actual), len(expected))
        for result, reference in zip(actual, expected):
            self.assertLessEqual(math.dist(result, reference),
                                 tolerance * max(1, math.hypot(*reference)))

    def test_acceleration_is_negative_mass_normalized_potential_gradient(self):
        # A fourth-order central difference checks every force component against
        # the independent energy diagnostic, with and without softening.
        h = 1e-4
        for eps in (0, 0.3):
            for mode in ("direct", "bh"):
                with self.subTest(eps=eps, mode=mode):
                    sim = BarnesHutSimulation(particles=interacting_fixture(), mode=mode, theta=0, eps=eps)
                    actual = sim.accelerations()
                    for index, particle in enumerate(sim.particles):
                        for axis, name in enumerate(("x", "y", "z")):
                            coordinate = getattr(particle, name)
                            energies = []
                            for offset in (2 * h, h, -h, -2 * h):
                                setattr(particle, name, coordinate + offset)
                                energies.append(compute_total_energy(sim.particles, eps=eps))
                            setattr(particle, name, coordinate)
                            derivative = (-energies[0] + 8 * energies[1] - 8 * energies[2] + energies[3]) / (12 * h)
                            self.assertAlmostEqual(actual[index][axis], -derivative / particle.mass, delta=2e-10)

    def test_pair_softening_matches_closed_form(self):
        separation, eps = 0.4, 0.3
        masses = (0.7, 1.3)
        particles = [Particle(0, 0, 0, 0, 0, 0, mass=masses[0]),
                     Particle(separation, 0, 0, 0, 0, 0, mass=masses[1])]
        radius = math.hypot(separation, eps)
        expected = [(masses[1] * separation / radius ** 3, 0, 0),
                    (-masses[0] * separation / radius ** 3, 0, 0)]
        for mode in ("direct", "bh"):
            sim = BarnesHutSimulation(particles=particles, mode=mode, eps=eps, theta=0)
            self.assertVectorsClose(sim.accelerations(), expected)
        self.assertAlmostEqual(compute_total_energy(particles, eps=eps),
                               -masses[0] * masses[1] / radius, delta=1e-14)

    def test_multiscale_tree_preserves_close_physical_separations(self):
        # Root-relative float64 coordinates collapse the first two positions.
        # Tree geometry may bucket them, but physical pair forces must not.
        particles = [Particle(0, 0, 0, 0, 0, 0, mass=0.7),
                     Particle(1e-7, 0, 0, 0, 0, 0, mass=1.3),
                     Particle(1e10, 0, 0, 0, 0, 0, mass=2)]
        for eps in (0, 2e-7, 0.05):
            reference = BarnesHutSimulation(particles=particles, mode="direct", eps=eps).accelerations()
            for theta in (0, 0.5):
                with self.subTest(eps=eps, theta=theta):
                    tree = BarnesHutSimulation(particles=particles, mode="bh", theta=theta, eps=eps)
                    self.assertVectorsClose(tree.accelerations(), reference, tolerance=3e-14)

    def test_force_rotation_and_particle_permutation(self):
        particles = interacting_fixture()
        rotated = [Particle(*rotate(position(p)), *rotate(velocity(p)), mass=p.mass) for p in particles]
        order = [2, 0, 3, 1]
        for mode in ("direct", "bh"):
            with self.subTest(mode=mode):
                original = BarnesHutSimulation(particles=particles, mode=mode, theta=0, eps=0.2).accelerations()
                rotated_acc = BarnesHutSimulation(particles=rotated, mode=mode, theta=0, eps=0.2).accelerations()
                self.assertVectorsClose(rotated_acc, [rotate(a) for a in original])
                permuted = BarnesHutSimulation(particles=[particles[i] for i in order], mode=mode, theta=0, eps=0.2)
                self.assertVectorsClose(permuted.accelerations(), [original[i] for i in order])

    def test_deep_tree_membership_excludes_self_despite_rounded_cell_bounds(self):
        # At depth 51, a descendant falls outside its nominal floating-point
        # bounds. Geometric containment alone would permit a self monopole.
        particles = [Particle(x, 0, 0, 0, 0, 0) for x in (0, 0.03125, 1e14)]
        reference = BarnesHutSimulation(particles=particles, mode="direct", eps=1e-4).accelerations()
        for theta in (0, 0.5, 100):
            with self.subTest(theta=theta):
                tree = BarnesHutSimulation(particles=particles, mode="bh", theta=theta, eps=1e-4)
                self.assertVectorsClose(tree.accelerations(), reference, tolerance=3e-14)

    def test_internal_force_and_torque_vanish(self):
        particles = interacting_fixture()
        for mode in ("direct", "bh"):
            accelerations = BarnesHutSimulation(particles=particles, mode=mode, theta=0, eps=0.2).accelerations()
            forces = [tuple(p.mass * a for a in acceleration) for p, acceleration in zip(particles, accelerations)]
            self.assertLess(math.hypot(*(math.fsum(f[axis] for f in forces) for axis in range(3))), 2e-14)
            torques = [(p.y * fz - p.z * fy, p.z * fx - p.x * fz, p.x * fy - p.y * fx)
                       for p, (fx, fy, fz) in zip(particles, forces)]
            self.assertLess(math.hypot(*(math.fsum(t[axis] for t in torques) for axis in range(3))), 2e-14)

    def test_tree_tightening_controls_error_in_clustered_unequal_mass_fixture(self):
        rng = random.Random(904)
        particles = [Particle((i % 2) * 3 + rng.gauss(0, 0.1), rng.gauss(0, 0.2),
                              rng.gauss(0, 0.3), 0, 0, 0, mass=10 ** rng.uniform(-2, 1))
                     for i in range(48)]
        reference = BarnesHutSimulation(particles=particles, mode="direct", eps=0.1).accelerations()
        norm_squared = math.fsum(math.hypot(*acceleration) ** 2 for acceleration in reference)
        errors = []
        for theta in (0.8, 0.2, 0):
            actual = BarnesHutSimulation(particles=particles, theta=theta, eps=0.1).accelerations()
            errors.append(math.sqrt(math.fsum(math.dist(a, b) ** 2 for a, b in zip(actual, reference)) / norm_squared))
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[1], 5e-4)
        self.assertLess(errors[2], 2e-14)


class TestOrbitalMechanics(unittest.TestCase):
    def test_eccentric_unequal_mass_kepler_convergence_and_invariants(self):
        semimajor, eccentricity, masses = 1.4, 0.6, (0.7, 1.3)
        period = 2 * math.pi * math.sqrt(semimajor ** 3 / sum(masses))
        expected_energy = -masses[0] * masses[1] / (2 * semimajor)
        reduced_mass = math.prod(masses) / sum(masses)
        expected_angular = reduced_mass * math.sqrt(sum(masses) * semimajor * (1 - eccentricity ** 2))
        for integrator, order_range, position_bound, energy_bound in (
                ("leapfrog", (1.9, 2.1), 0.004, 0.0003),
                ("rk4", (3.9, 4.5), 2e-7, 2e-8)):
            errors = []
            for steps_per_orbit in (512, 1024):
                with self.subTest(integrator=integrator, steps_per_orbit=steps_per_orbit):
                    sim = BarnesHutSimulation(particles=eccentric_binary(0), mode="direct", eps=0,
                                             integrator=integrator, dt=period / steps_per_orbit)
                    self.assertAlmostEqual(compute_total_energy(sim.particles, eps=0), expected_energy, delta=2e-15)
                    self.assertAlmostEqual(compute_total_angular_momentum(sim.particles)[2], expected_angular, delta=2e-15)
                    maximum_position, maximum_energy = 0.0, 0.0
                    for step in range(1, 5 * steps_per_orbit // 4 + 1):
                        sim.step()
                        exact = eccentric_binary(step * sim.dt)
                        error = max(math.dist(position(p), position(q)) / semimajor
                                    for p, q in zip(sim.particles, exact))
                        maximum_position = max(maximum_position, error)
                        maximum_energy = max(maximum_energy, abs(compute_total_energy(sim.particles, eps=0)
                                                                 - expected_energy) / abs(expected_energy))
                    errors.append(error)
                    if steps_per_orbit == 1024:
                        self.assertLess(maximum_position, position_bound)
                        self.assertLess(maximum_energy, energy_bound)
                        self.assertLess(math.hypot(*compute_total_momentum(sim.particles)), 2e-13)
                        center = tuple(math.fsum(p.mass * getattr(p, axis) for p in sim.particles) / sum(masses)
                                       for axis in ("x", "y", "z"))
                        self.assertLess(math.hypot(*center), 2e-13)
            order = math.log2(errors[0] / errors[1])
            self.assertGreater(order, order_range[0])
            self.assertLess(order, order_range[1])

    def test_softened_binary_uses_softened_circular_frequency(self):
        masses, separation, eps = (0.7, 1.3), 1.2, 0.3
        total_mass = sum(masses)
        omega = math.sqrt(total_mass / (separation ** 2 + eps ** 2) ** 1.5)
        particles = [Particle(fraction * separation, 0, 0, 0, fraction * separation * omega, 0, mass=mass)
                     for fraction, mass in ((-masses[1] / total_mass, masses[0]),
                                            (masses[0] / total_mass, masses[1]))]
        sim = BarnesHutSimulation(particles=particles, mode="direct", integrator="rk4",
                                 eps=eps, dt=2 * math.pi / omega / 512)
        for _ in range(512):
            sim.step()
        for actual, initial in zip(sim.particles, particles):
            self.assertLess(math.dist(position(actual), position(initial)) / separation, 2e-8)

    def test_galilean_covariance_of_all_integrators(self):
        shift, boost, steps, dt = (4, -8, 2), (3, -2, 0.75), 40, 0.002
        original = interacting_fixture()
        moved = [Particle(*(x + offset for x, offset in zip(position(p), shift)),
                          *(v + speed for v, speed in zip(velocity(p), boost)), mass=p.mass) for p in original]
        for mode in ("direct", "bh"):
            for integrator in ("euler", "leapfrog", "rk4"):
                with self.subTest(mode=mode, integrator=integrator):
                    reference = BarnesHutSimulation(particles=original, mode=mode, theta=0, eps=0.2,
                                                   integrator=integrator, dt=dt)
                    transformed = BarnesHutSimulation(particles=moved, mode=mode, theta=0, eps=0.2,
                                                     integrator=integrator, dt=dt)
                    for _ in range(steps):
                        reference.step()
                        transformed.step()
                    for first, second in zip(reference.particles, transformed.particles):
                        expected_position = tuple(x + offset + speed * steps * dt
                                                  for x, offset, speed in zip(position(first), shift, boost))
                        expected_velocity = tuple(v + speed for v, speed in zip(velocity(first), boost))
                        self.assertLess(math.dist(position(second), expected_position), 2e-12)
                        self.assertLess(math.dist(velocity(second), expected_velocity), 2e-12)

    def test_leapfrog_time_reversibility(self):
        initial = interacting_fixture()
        sim = BarnesHutSimulation(particles=initial, mode="direct", integrator="leapfrog", dt=0.002, eps=0.3)
        for _ in range(100):
            sim.step()
        for p in sim.particles:
            p.vx, p.vy, p.vz = -p.vx, -p.vy, -p.vz
        for _ in range(100):
            sim.step()
        for final, original in zip(sim.particles, initial):
            self.assertLess(math.dist(position(final), position(original)), 2e-12)
            self.assertLess(math.dist(velocity(final), tuple(-v for v in velocity(original))), 2e-12)


if __name__ == "__main__":
    unittest.main()
