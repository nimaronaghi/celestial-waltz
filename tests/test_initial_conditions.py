"""Distribution and center-of-mass checks against analytic initial-state physics."""

import math
import random
import unittest

from nbody import (generate_plummer_sphere, generate_two_galaxies, compute_total_energy,
                   generate_spiral_galaxy, generate_kuzmin_disk)


class InitialConditionTests(unittest.TestCase):
    def test_plummer_matches_radial_cdf_and_isotropic_velocity_moments(self):
        count, scale = 12000, 1.7
        particles = generate_plummer_sphere(count, scale, random.Random(924))
        radii = [math.sqrt(p.x*p.x + p.y*p.y + p.z*p.z) for p in particles]
        for radius in (scale, 2*scale, 4*scale):
            expected = (radius / math.hypot(radius, scale)) ** 3
            actual = sum(r < radius for r in radii) / count
            self.assertAlmostEqual(actual, expected, delta=0.015)
        q_squared, radial_cosines = [], []
        normalized_components = [[], [], []]
        for p, radius in zip(particles, radii):
            velocity = (p.vx, p.vy, p.vz)
            speed_squared = math.fsum(v*v for v in velocity)
            escape_squared = 2*count / math.hypot(radius, scale)
            self.assertLess(speed_squared, escape_squared)
            q_squared.append(speed_squared / escape_squared)
            radial_cosines.append((p.x*p.vx+p.y*p.vy+p.z*p.vz) /
                                  (radius*math.sqrt(speed_squared)))
            for axis in range(3):
                normalized_components[axis].append(velocity[axis]/math.sqrt(escape_squared))
        # From f(E) proportional to (-E)^(7/2) and the 4*pi*v^2 measure:
        # <q^2>=1/4, <q^4>=5/56. Isotropy gives <cos^2>=1/3.
        self.assertAlmostEqual(math.fsum(q_squared)/count, 1/4, delta=0.007)
        self.assertAlmostEqual(math.fsum(q*q for q in q_squared)/count, 5/56, delta=0.004)
        self.assertAlmostEqual(math.fsum(radial_cosines)/count, 0, delta=0.02)
        self.assertAlmostEqual(math.fsum(c*c for c in radial_cosines)/count, 1/3, delta=0.015)
        for values in normalized_components:
            self.assertAlmostEqual(math.fsum(values)/count, 0, delta=0.012)
            self.assertAlmostEqual(math.fsum(v*v for v in values)/count, 1/12, delta=0.005)

    def test_plummer_sample_has_expected_virial_scale(self):
        count, scale = 1024, 1.3
        particles = generate_plummer_sphere(count, scale, random.Random(73))
        kinetic = math.fsum(0.5*p.mass*(p.vx*p.vx+p.vy*p.vy+p.vz*p.vz) for p in particles)
        energy = compute_total_energy(particles, eps=0)
        potential = energy - kinetic
        # Finite random samples need not have exactly 2K/|U|=1.
        expected_kinetic = 3*math.pi*count**2/(64*scale)
        self.assertAlmostEqual(kinetic/expected_kinetic, 1, delta=0.1)
        self.assertAlmostEqual(2*kinetic/abs(potential), 1, delta=0.1)
        self.assertLess(energy, 0)

    def test_plummer_obeys_newtonian_length_velocity_scaling(self):
        first = generate_plummer_sphere(32, 1, random.Random(23))
        scaled = generate_plummer_sphere(32, 4, random.Random(23))
        for p, q in zip(first, scaled):
            for name in ('x', 'y', 'z'):
                self.assertAlmostEqual(getattr(q, name), 4*getattr(p, name))
            for name in ('vx', 'vy', 'vz'):
                self.assertAlmostEqual(getattr(q, name), 0.5*getattr(p, name))

    def test_two_galaxies_obey_requested_com_separation_and_velocity(self):
        count, separation, approach_speed = 127, 4.7, 1.3
        particles = generate_two_galaxies(count, separation, approach_speed, random.Random(42))
        self.assertEqual(len(particles), 2*count)
        for group, sign in ((particles[:count], -1), (particles[count:], 1)):
            mass = math.fsum(p.mass for p in group)
            means = [math.fsum(p.mass*getattr(p, name) for p in group)/mass
                     for name in ('x', 'y', 'z', 'vx', 'vy', 'vz')]
            expected = [sign*separation/2, 0, 0, -sign*approach_speed/2, 0, 0]
            for actual, target in zip(means, expected):
                self.assertAlmostEqual(actual, target, delta=1e-12)

    def test_plummer_rejects_nonphysical_sizes(self):
        for count, scale in [(-1, 1), (1.5, 1), (True, 1), (1, 0), (1, -1), (1, math.inf)]:
            with self.subTest(count=count, scale=scale), self.assertRaises(ValueError):
                generate_plummer_sphere(count, scale)
        self.assertEqual(generate_plummer_sphere(0), [])

    def test_other_initializers_reject_nonphysical_parameters(self):
        for generator in (generate_spiral_galaxy, generate_kuzmin_disk, generate_two_galaxies):
            for count in (-1, 1.5, True):
                with self.subTest(generator=generator.__name__, count=count), self.assertRaises(ValueError):
                    generator(count)
        for generator, kwargs in (
                (generate_spiral_galaxy, {'radius': 0}),
                (generate_kuzmin_disk, {'radius': -1}),
                (generate_kuzmin_disk, {'thickness': -1}),
                (generate_two_galaxies, {'separation': math.inf}),
                (generate_two_galaxies, {'separation': -1}),
                (generate_two_galaxies, {'relative_velocity': math.nan})):
            with self.subTest(generator=generator.__name__, kwargs=kwargs), self.assertRaises(ValueError):
                generator(2, **kwargs)


if __name__ == '__main__':
    unittest.main()
