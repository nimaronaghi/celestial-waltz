"""Optional PyTorch tests run on CPU; CUDA hardware is not required."""

import math
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from gpu_sim import _bh_force, _build_tree, generate_spiral_galaxy, run, step_bh, step_direct
    from nbody import BarnesHutSimulation, Particle


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class TestTensorGravity(unittest.TestCase):
    def state(self, positions, masses, dtype=None):
        dtype = dtype or torch.float64
        pos = torch.tensor(positions, dtype=dtype, device="cpu")
        return pos, torch.zeros_like(pos), torch.tensor(masses, dtype=dtype, device="cpu")

    def test_two_body_attraction_and_momentum(self):
        for step in (step_direct, step_bh):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(step=step.__name__, dtype=dtype):
                    pos, vel, mass = self.state([[-1, 0, 0], [1, 0, 0]], [2, 3], dtype)
                    start = pos.clone()
                    # a_0 = 3 / 2**2; a_1 = -2 / 2**2 in units G=1.
                    expected_velocity = pos.new_tensor([[0.075, 0, 0], [-0.05, 0, 0]])
                    step(pos, vel, mass, dt=0.1, eps=0)
                    torch.testing.assert_close(vel, expected_velocity)
                    torch.testing.assert_close(pos, start + expected_velocity * 0.1)
                    torch.testing.assert_close((mass[:, None] * vel).sum(0), pos.new_zeros(3))
                    self.assertEqual(pos.dtype, dtype)
                    self.assertEqual(vel.dtype, dtype)

    def test_softened_unequal_pair_with_nondefault_gravity(self):
        # Separation 5 and eps=sqrt(11) give softened radius 6. This checks
        # each vector component, source mass, G, and the velocity-first drift.
        for step in (step_direct, step_bh):
            with self.subTest(step=step.__name__):
                pos, vel, mass = self.state([[0, 0, 0], [3, 4, 0]], [2, 7])
                start_pos = pos.clone()
                vel[:] = pos.new_tensor([[0.2, -0.3, 0.4], [-0.5, 0.6, -0.7]])
                start_vel = vel.clone()
                acceleration = pos.new_tensor([
                    [2.5 * 7 * 3 / 216, 2.5 * 7 * 4 / 216, 0],
                    [-2.5 * 2 * 3 / 216, -2.5 * 2 * 4 / 216, 0],
                ])
                step(pos, vel, mass, dt=0.125, G=2.5, eps=math.sqrt(11))
                torch.testing.assert_close(vel, start_vel + 0.125 * acceleration,
                                           rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(pos, start_pos + 0.125 * vel,
                                           rtol=1e-12, atol=1e-12)

    def test_acceleration_matches_softened_pair_potential_gradient(self):
        # Independent finite differences of U=-G*m1*m2/sqrt(r^2+eps^2).
        target, source = [-1.0, 0.5, -0.25], [2.0, -1.5, 0.75]
        G, eps, m1, m2, h = 1.7, 0.4, 2.0, 5.0, 1e-5

        def potential(point):
            return -G * m1 * m2 / math.sqrt(
                sum((a - b) ** 2 for a, b in zip(point, source)) + eps ** 2
            )

        expected = []
        for axis in range(3):
            plus, minus = target.copy(), target.copy()
            plus[axis] += h
            minus[axis] -= h
            expected.append(-(potential(plus) - potential(minus)) / (2 * h * m1))
        for step in (step_direct, step_bh):
            with self.subTest(step=step.__name__):
                pos, vel, mass = self.state([target, source], [m1, m2])
                step(pos, vel, mass, dt=1, G=G, eps=eps)
                torch.testing.assert_close(vel[0], pos.new_tensor(expected), rtol=1e-8, atol=1e-10)

    def test_softened_monopole_uses_mass_weighted_center(self):
        pos, _, mass = self.state([[-1, 0, 0], [1, 0, 0], [10, 2, 0]], [2, 6, 9])
        node = _build_tree(pos, mass, torch.tensor([0, 1]), pos.new_zeros(3), 1.0)
        torch.testing.assert_close(node.com, pos.new_tensor([0.5, 0, 0]))
        # The accepted source node has mass 8 at x=0.5, independent of target mass.
        expected = pos.new_tensor([-9.5, -2, 0]) * (3 * 8 / (9.5 ** 2 + 2 ** 2 + 0.75 ** 2) ** 1.5)
        actual = _bh_force(node, 2, pos, theta=1, G=3, eps=0.75)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_single_particle_has_no_self_force_without_softening(self):
        for step in (step_direct, step_bh):
            with self.subTest(step=step.__name__):
                pos, vel, mass = self.state([[3, -4, 5]], [10])
                vel[0] = pos.new_tensor([1, -2, 3])
                step(pos, vel, mass, dt=0.5, eps=0)
                torch.testing.assert_close(vel, pos.new_tensor([[1, -2, 3]]))
                torch.testing.assert_close(pos, pos.new_tensor([[3.5, -5, 6.5]]))

    def test_softened_coincidence_is_finite_and_has_zero_pair_force(self):
        for step in (step_direct, step_bh):
            with self.subTest(step=step.__name__):
                pos, vel, mass = self.state([[3, 4, 5], [3, 4, 5]], [2, 3])
                start = pos.clone()
                step(pos, vel, mass, dt=0.1, eps=0.2)
                torch.testing.assert_close(vel, torch.zeros_like(vel))
                torch.testing.assert_close(pos, start)

    def test_unsoftened_coincidence_raises_before_mutating(self):
        for step in (step_direct, step_bh):
            with self.subTest(step=step.__name__):
                pos, vel, mass = self.state([[0, 0, 0], [0, 0, 0]], [2, 3])
                start = pos.clone()
                with self.assertRaisesRegex(ValueError, "coincident"):
                    step(pos, vel, mass, dt=0.1, eps=0)
                torch.testing.assert_close(pos, start)
                torch.testing.assert_close(vel, torch.zeros_like(vel))

    def test_tree_bucket_retains_individual_close_particle_forces(self):
        positions = [[0, 0, 0], [0, 0, 0], [1e-6, 0, 0], [3, 1, 0]]
        masses = [2, 3, 5, 7]
        pos, vel, mass = self.state(positions, masses)
        direct_pos, direct_vel = pos.clone(), vel.clone()
        step_direct(direct_pos, direct_vel, mass, dt=0.01, eps=0.05)
        step_bh(pos, vel, mass, dt=0.01, theta=0, eps=0.05)
        torch.testing.assert_close(pos, direct_pos, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(vel, direct_vel, rtol=1e-12, atol=1e-12)

    def test_distant_outlier_does_not_erase_resolved_close_pair(self):
        # Root recentering alone rounds the first two x values to the same
        # float64 number. Tree partitioning must not replace physical distances.
        for eps in (0, 0.05):
            with self.subTest(eps=eps):
                pos, vel, mass = self.state([[0, 0, 0], [1e-7, 0, 0], [1e10, 0, 0]], [2, 3, 5])
                dt = 1e-10
                close_factor = 1e-7 / (1e-14 + eps ** 2) ** 1.5
                far_factor = 1e10 / (1e20 + eps ** 2) ** 1.5
                expected = pos.new_tensor([
                    [(3 * close_factor + 5 * far_factor) * dt, 0, 0],
                    [(-2 * close_factor + 5 * far_factor) * dt, 0, 0],
                    [-5 * far_factor * dt, 0, 0],
                ])
                step_bh(pos, vel, mass, dt=dt, theta=0, eps=eps)
                torch.testing.assert_close(vel, expected, rtol=2e-14, atol=0)

    def test_exact_forces_preserve_linear_and_angular_momentum(self):
        for step in (step_direct, step_bh):
            with self.subTest(step=step.__name__):
                pos, vel, mass = self.state([[-1, 0.5, 0], [1, -0.5, 0.25], [0, 2, -1]], [2, 3, 5])
                vel[:] = pos.new_tensor([[0.3, -0.2, 0.1], [-0.4, 0.1, 0.2], [0.1, 0.3, -0.2]])
                momentum = (mass[:, None] * vel).sum(0)
                angular = (mass[:, None] * torch.linalg.cross(pos, vel)).sum(0)
                kwargs = {"theta": 0} if step is step_bh else {}
                step(pos, vel, mass, dt=0.05, G=1.7, eps=0.4, **kwargs)
                torch.testing.assert_close((mass[:, None] * vel).sum(0), momentum, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close((mass[:, None] * torch.linalg.cross(pos, vel)).sum(0),
                                           angular, rtol=1e-12, atol=1e-12)

    def test_softened_circular_orbit_has_first_order_timestep_convergence(self):
        # Relative radius R=2, masses 2 and 3, G=0.8; softened two-body motion
        # admits a circular solution with omega^2=G*M/(R^2+eps^2)^(3/2).
        omega = math.sqrt(0.8 * 5 / (4 + 0.5 ** 2) ** 1.5)
        expected = [-1.2 * math.cos(omega), -1.2 * math.sin(omega), 0]
        for step in (step_direct, step_bh):
            errors = []
            for count in (64, 128):
                pos, vel, mass = self.state([[-1.2, 0, 0], [0.8, 0, 0]], [2, 3])
                vel[:, 1] = pos[:, 0] * omega
                kwargs = {"theta": 0} if step is step_bh else {}
                for _ in range(count):
                    step(pos, vel, mass, dt=1 / count, G=0.8, eps=0.5, **kwargs)
                errors.append(math.dist(pos[0].tolist(), expected))
            with self.subTest(step=step.__name__):
                self.assertGreater(errors[0] / errors[1], 1.9)
                self.assertLess(errors[0] / errors[1], 2.1)

    def test_target_containing_nodes_are_opened_even_at_large_theta(self):
        pos, vel, mass = self.state([[-5, 0, 0], [5, 0, 0]], [2, 9])
        step_bh(pos, vel, mass, dt=1, theta=1e6, eps=0)
        torch.testing.assert_close(vel, pos.new_tensor([[0.09, 0, 0], [-0.02, 0, 0]]))

    def test_opening_criterion_uses_full_width_and_unsoftened_distance(self):
        # The source node has width 2 and target distance 4: it must open for
        # theta=0.4, even when a large softening length is used in the force.
        pos, _, mass = self.state([[1, 0, 0], [3, 0, 0], [-2, 0, 0]], [1, 1, 7])
        node = _build_tree(pos, mass, torch.tensor([0, 1]), pos.new_tensor([2, 0, 0]), 1.0)
        for eps in (0, 10):
            with self.subTest(eps=eps):
                expected = 3 / (9 + eps ** 2) ** 1.5 + 5 / (25 + eps ** 2) ** 1.5
                actual = _bh_force(node, 2, pos, theta=0.4, eps=eps)
                torch.testing.assert_close(actual, pos.new_tensor([expected, 0, 0]), rtol=1e-12, atol=1e-14)

    def test_tree_approximation_is_translation_invariant(self):
        positions = [
            [-1, 0, 0], [1, 0, 0], [0, 2, 0], [0.25, 0, 0.5],
            [-0.5, 0.5, 0], [2, -1, 0.25], [-2, -0.5, -1],
        ]
        pos, vel, mass = self.state(positions, [1, 2, 3, 4, 5, 6, 7])
        shift = pos.new_tensor([500, -800, 200])
        shifted_pos, shifted_vel = pos + shift, vel.clone()
        step_bh(pos, vel, mass, dt=0.01, theta=0.65)
        step_bh(shifted_pos, shifted_vel, mass, dt=0.01, theta=0.65)
        torch.testing.assert_close(shifted_vel, vel, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(shifted_pos - shift, pos, rtol=1e-11, atol=1e-12)

    def test_tensor_steps_match_python_core_symplectic_euler(self):
        positions = [[-2, 1, 0], [0, 0, 0.5], [3, -1, 2], [0.25, 2, -1]]
        velocities = [[0.1, 0, -0.2], [0, 0.2, 0], [-0.1, 0, 0.1], [0, -0.1, 0]]
        masses = [2, 1, 4, 3]
        dt = 0.02
        for step in (step_direct, step_bh):
            for eps in (0, 0.125):
                with self.subTest(step=step.__name__, eps=eps):
                    pos, _, mass = self.state(positions, masses)
                    vel = pos.new_tensor(velocities)
                    particles = [Particle(*p, *v, mass=m) for p, v, m in zip(positions, velocities, masses)]
                    sim = BarnesHutSimulation(particles=particles, dt=dt, mode="direct", integrator="euler", eps=eps)
                    sim.step()
                    kwargs = {"theta": 0} if step is step_bh else {}
                    step(pos, vel, mass, dt=dt, eps=eps, **kwargs)
                    expected_pos = pos.new_tensor([[p.x, p.y, p.z] for p in sim.particles])
                    expected_vel = vel.new_tensor([[p.vx, p.vy, p.vz] for p in sim.particles])
                    torch.testing.assert_close(pos, expected_pos, rtol=1e-12, atol=1e-12)
                    torch.testing.assert_close(vel, expected_vel, rtol=1e-12, atol=1e-12)

    def test_seeded_generation_is_repeatable_and_preserves_global_rng(self):
        before = torch.random.get_rng_state().clone()
        first = generate_spiral_galaxy(8, device="cpu", seed=42, dtype=torch.float64)
        second = generate_spiral_galaxy(8, device="cpu", seed=42, dtype=torch.float64)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(a.dtype, torch.float64)

    def test_seeded_run_is_repeatable(self):
        for mode in ("direct", "bh"):
            with self.subTest(mode=mode):
                first = run(8, 2, 0.001, device="cpu", mode=mode, seed=7, dtype=torch.float64)
                second = run(8, 2, 0.001, device="cpu", mode=mode, seed=7, dtype=torch.float64)
                for a, b in zip(first, second):
                    self.assertTrue(torch.equal(a, b))

    def test_invalid_parameters_and_states(self):
        for step in (step_direct, step_bh):
            for kwargs in ({"dt": 0}, {"dt": float("nan")}, {"eps": -1}, {"G": 0}):
                with self.subTest(step=step.__name__, kwargs=kwargs):
                    pos, vel, mass = self.state([[0, 0, 0], [1, 0, 0]], [1, 2])
                    arguments = {"dt": 0.1, **kwargs}
                    with self.assertRaises(ValueError):
                        step(pos, vel, mass, **arguments)
            pos, vel, mass = self.state([[0, 0, 0], [1, 0, 0]], [1, -2])
            with self.assertRaisesRegex(ValueError, "mass"):
                step(pos, vel, mass, dt=0.1)
            with self.assertRaisesRegex(ValueError, "dtype"):
                step(pos, vel.float(), mass, dt=0.1)
        for kwargs in ({"mode": "invalid"}, {"iterations": -1}, {"n_particles": 0}, {"theta": -1}):
            with self.subTest(kwargs=kwargs):
                arguments = {"n_particles": 2, "iterations": 0, "dt": 0.1, "device": "cpu", **kwargs}
                with self.assertRaises(ValueError):
                    run(**arguments)


if __name__ == "__main__":
    unittest.main()
