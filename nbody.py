"""Newtonian N-body experiments in dimensionless units with G = 1.

The force solvers use Plummer softening. The Plummer initializer samples its
isotropic continuum distribution function; the disk/spiral generators are toys.
"""

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import time


@dataclass
class Particle:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    mass: float = 1.0


class OctreeNode:
    """An octree with direct-interaction buckets for coincident/nearby particles."""

    MIN_SIZE = 1e-5

    def __init__(self, center: Tuple[float, float, float], half_size: float,
                 geometry_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.center = list(center)
        self.geometry_origin = geometry_origin
        self.half_size = half_size
        self.particle: Optional[Particle] = None
        self.particles: List[Particle] = []
        self._particle_ids = set()
        self.children: List[Optional["OctreeNode"]] = [None] * 8
        self.mass = 0.0
        self.com = [0.0, 0.0, 0.0]
        self._reference_position = None
        self._weighted_offset = [0.0, 0.0, 0.0]
        self._com_offset = [0.0, 0.0, 0.0]

    def _is_leaf(self) -> bool:
        return all(child is None for child in self.children)

    def _subdivide(self):
        quarter = self.half_size / 2.0
        for i in range(8):
            offset = (
                quarter if i & 1 else -quarter,
                quarter if i & 2 else -quarter,
                quarter if i & 4 else -quarter,
            )
            new_center = (
                self.center[0] + offset[0],
                self.center[1] + offset[1],
                self.center[2] + offset[2],
            )
            self.children[i] = OctreeNode(new_center, quarter, self.geometry_origin)

    def _geometry_position(self, p: Particle):
        return tuple(coordinate - origin for coordinate, origin in
                     zip((p.x, p.y, p.z), self.geometry_origin))

    def _child_index(self, p: Particle) -> int:
        idx = 0
        x, y, z = self._geometry_position(p)
        if x > self.center[0]:
            idx |= 1
        if y > self.center[1]:
            idx |= 2
        if z > self.center[2]:
            idx |= 4
        return idx

    def insert(self, p: Particle):
        self._particle_ids.add(id(p))
        self.mass += p.mass
        if self._reference_position is None:
            self._reference_position = (p.x, p.y, p.z)
        for axis, coordinate in enumerate((p.x, p.y, p.z)):
            self._weighted_offset[axis] += p.mass * (coordinate - self._reference_position[axis])
        if self._is_leaf():
            # Partition coordinates can coincide after recentering a multiscale
            # system. Keep the original particles in a bucket: force evaluation
            # must still resolve their original physical separation.
            geometry_position = self._geometry_position(p)
            coincident = self.particles and all(
                self._geometry_position(q) == geometry_position for q in self.particles
            )
            if not self.particles or self.half_size <= self.MIN_SIZE or coincident:
                self.particles.append(p)
                self.particle = p if len(self.particles) == 1 else None
                return
            self._subdivide()
            previous = self.particles
            self.particles = []
            self.particle = None
            for existing in previous:
                self.children[self._child_index(existing)].insert(existing)
        self.children[self._child_index(p)].insert(p)

    def finalize(self):
        if self.mass > 0:
            self._com_offset = [coordinate / self.mass for coordinate in self._weighted_offset]
            self.com = [reference + offset for reference, offset in
                        zip(self._reference_position, self._com_offset)]
        if not self._is_leaf():
            for child in self.children:
                if child:
                    child.finalize()

    def compute_force_on(
        self, p: Particle, theta: float = 0.5, G: float = 1.0, eps: float = 0.05
    ) -> Tuple[float, float, float]:
        """Return acceleration, excluding self even for large opening angles.

        Opening uses full node width divided by the unsoftened COM distance. A
        node containing the target is always opened. theta=0 visits every source.
        """
        if self.mass == 0:
            return 0.0, 0.0, 0.0
        if self._is_leaf():
            contributions = [
                _pair_acceleration(p, q, G, eps) for q in self.particles if q is not p
            ]
            return tuple(math.fsum(a[axis] for a in contributions) for axis in range(3))
        # Do not reconstruct a large absolute COM before subtracting the target.
        # Referencing an actual source retains close separations in small nodes.
        dx, dy, dz = tuple((reference - coordinate) + offset
                          for reference, coordinate, offset in
                          zip(self._reference_position, (p.x, p.y, p.z), self._com_offset))
        distance_squared = dx * dx + dy * dy + dz * dz
        # Membership is exact even if floating-point child bounds round away
        # from a descendant in a deeply subdivided, widely separated system.
        contains_target = id(p) in self._particle_ids
        if (not contains_target and distance_squared > 0
                and 2.0 * self.half_size / math.sqrt(distance_squared) < theta):
            factor = G * self.mass / ((distance_squared + eps * eps) ** 1.5)
            return dx * factor, dy * factor, dz * factor
        fx = fy = fz = 0.0
        for child in self.children:
            if child:
                cfx, cfy, cfz = child.compute_force_on(p, theta, G, eps)
                fx += cfx
                fy += cfy
                fz += cfz
        return fx, fy, fz


def _pair_acceleration(target, source, G, eps):
    dx, dy, dz = source.x - target.x, source.y - target.y, source.z - target.z
    radius_squared = dx * dx + dy * dy + dz * dz + eps * eps
    if radius_squared == 0:
        raise ValueError("Coincident particles require positive softening (eps).")
    factor = G * source.mass / radius_squared ** 1.5
    return dx * factor, dy * factor, dz * factor


def _validate_generator_count(num):
    if isinstance(num, bool) or not isinstance(num, int) or num < 0:
        raise ValueError("particle count must be a nonnegative integer.")


def generate_spiral_galaxy(num: int, radius: float = 1.0, rng=None) -> List[Particle]:
    """Generate a decorative spiral with approximate point-mass orbital speeds."""
    _validate_generator_count(num)
    _finite_number(radius, "radius", positive=True)
    rng = random if rng is None else rng
    particles: List[Particle] = []
    central_mass = num
    G = 1.0
    for _ in range(num):
        r = math.sqrt(rng.random()) * radius
        angle = r * 4.0 + rng.uniform(-0.2, 0.2)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        z = rng.gauss(0, 0.05)
        v_mag = math.sqrt(G * central_mass / (r + 0.01))
        vx = -v_mag * math.sin(angle)
        vy = v_mag * math.cos(angle)
        vz = rng.gauss(0, 0.01)
        particles.append(Particle(x, y, z, vx, vy, vz))
    return particles

# Additional initial condition generators inspired by contemporary N‑body projects.
def generate_plummer_sphere(num: int, scale: float = 1.0, rng=None) -> List[Particle]:
    """Sample the isotropic Plummer phase-space distribution in G=1 units.

    Each particle has mass one, so the total mass is ``num``. Positions follow
    M(<r)/M = r^3/(r^2 + scale^2)^(3/2). At fixed r, q=v/v_escape has density
    proportional to q^2(1-q^2)^(7/2): equivalently q^2 ~ Beta(3/2, 9/2).
    Velocity directions are isotropic and independent of position directions.

    This samples equilibrium in the smooth, unsoftened Plummer potential.
    Finite-N sampling and softened pair forces do not give an exact equilibrium.
    Samples are not recentered or rescaled: their finite-sample COM can drift.
    See docs/physics-verification.md for derivation, tests and limitations.
    """
    _validate_generator_count(num)
    _finite_number(scale, "scale", positive=True)
    rng = random if rng is None else rng
    particles: List[Particle] = []

    def direction():
        phi = rng.uniform(0, 2 * math.pi)
        cosine = rng.uniform(-1, 1)
        sine = math.sqrt(max(0.0, 1 - cosine * cosine))
        return sine * math.cos(phi), sine * math.sin(phi), cosine

    for _ in range(num):
        fraction = rng.random()
        r = (scale / math.sqrt(math.expm1(-2 * math.log(fraction) / 3))
             if fraction else 0.0)
        position = tuple(r * component for component in direction())
        escape_squared = 2 * num / math.hypot(r, scale)
        speed = math.sqrt(escape_squared * rng.betavariate(1.5, 4.5))
        velocity = tuple(speed * component for component in direction())
        particles.append(Particle(*position, *velocity))
    return particles


def generate_kuzmin_disk(num: int, radius: float = 1.0, thickness: float = 0.05, rng=None) -> List[Particle]:
    """
    Generate an illustrative exponential-radius disk (legacy Kuzmin API name).

    This sampler does not implement the analytic Kuzmin surface-density profile
    or an equilibrium velocity distribution. The name is retained for API compatibility.

    Parameters
    ----------
    num : int
        Number of particles.
    radius : float, optional
        Characteristic radius of the disk.
    thickness : float, optional
        Standard deviation of the vertical distribution.

    Returns
    -------
    List[Particle]
        Particles representing a non-equilibrium toy rotating disk.
    """
    _validate_generator_count(num)
    _finite_number(radius, "radius", positive=True)
    _finite_number(thickness, "thickness", nonnegative=True)
    rng = random if rng is None else rng
    particles: List[Particle] = []
    G = 1.0
    total_mass = num
    for _ in range(num):
        # sample radius with exponential distribution
        r = -radius * math.log(1.0 - rng.random())
        theta = rng.uniform(0, 2 * math.pi)
        x_pos = r * math.cos(theta)
        y_pos = r * math.sin(theta)
        z_pos = rng.gauss(0.0, thickness)
        # compute circular velocity
        v_mag = math.sqrt(G * total_mass / (r + 0.1))
        vx = -v_mag * math.sin(theta)
        vy = v_mag * math.cos(theta)
        vz = rng.gauss(0.0, thickness * 0.1)
        particles.append(Particle(x_pos, y_pos, z_pos, vx, vy, vz))
    return particles


def generate_two_galaxies(num_per_galaxy: int, separation: float = 5.0, relative_velocity: float = 0.0, rng=None) -> List[Particle]:
    """
    Create a simple two‑galaxy collision scenario. Two spiral galaxies are placed apart and given a relative velocity.

    Parameters
    ----------
    num_per_galaxy : int
        Number of particles per galaxy.
    separation : float, optional
        Initial separation between the centers of the galaxies.
    relative_velocity : float, optional
        Signed approach velocity along x; negative values make the centers recede.

    Returns
    -------
    List[Particle]
        Combined list of particles belonging to two galaxies.
    """
    _validate_generator_count(num_per_galaxy)
    _finite_number(separation, "separation", nonnegative=True)
    _finite_number(relative_velocity, "relative_velocity")

    def recenter(galaxy):
        if not galaxy:
            return
        mass = math.fsum(p.mass for p in galaxy)
        for component in ("x", "y", "z", "vx", "vy", "vz"):
            mean = math.fsum(p.mass * getattr(p, component) for p in galaxy) / mass
            for p in galaxy:
                setattr(p, component, getattr(p, component) - mean)

    # Remove each realization's COM position and bulk velocity before placing it.
    galaxy1 = generate_spiral_galaxy(num_per_galaxy, radius=1.0, rng=rng)
    recenter(galaxy1)
    for p in galaxy1:
        p.x -= separation / 2.0
        # apply half the relative velocity in +x direction
        p.vx += 0.5 * relative_velocity
    # second galaxy centered at (+separation/2, 0, 0)
    galaxy2 = generate_spiral_galaxy(num_per_galaxy, radius=1.0, rng=rng)
    recenter(galaxy2)
    for p in galaxy2:
        p.x += separation / 2.0
        # apply half the relative velocity in -x direction
        p.vx -= 0.5 * relative_velocity
    return galaxy1 + galaxy2


def _finite_number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive.")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative.")


def _validate_particles(particles):
    for index, particle in enumerate(particles):
        if not isinstance(particle, Particle):
            raise ValueError(f"Particle {index} must be a Particle instance.")
        for name in ("x", "y", "z", "vx", "vy", "vz", "mass"):
            _finite_number(getattr(particle, name), f"particle {index} {name}", positive=name == "mass")


def compute_total_momentum(particles: Iterable[Particle]):
    """Return linear momentum; accepts one-shot iterables."""
    particles = list(particles)
    _validate_particles(particles)
    return tuple(math.fsum(p.mass * getattr(p, component) for p in particles)
                 for component in ("vx", "vy", "vz"))


def compute_total_angular_momentum(particles: Iterable[Particle]):
    """Return angular momentum about the origin; accepts one-shot iterables."""
    particles = list(particles)
    _validate_particles(particles)
    return (
        math.fsum(p.mass * (p.y * p.vz - p.z * p.vy) for p in particles),
        math.fsum(p.mass * (p.z * p.vx - p.x * p.vz) for p in particles),
        math.fsum(p.mass * (p.x * p.vy - p.y * p.vx) for p in particles),
    )


def compute_total_energy(particles: Iterable[Particle], G: float = 1.0, eps: float = 0.05):
    """Kinetic plus pair potential energy, using the force solver's softening."""
    _finite_number(G, "G", nonnegative=True)
    _finite_number(eps, "eps", nonnegative=True)
    particles = list(particles)
    _validate_particles(particles)
    kinetic = math.fsum(0.5 * p.mass * (p.vx ** 2 + p.vy ** 2 + p.vz ** 2) for p in particles)

    def potential_terms():
        for i, first in enumerate(particles):
            for j in range(i + 1, len(particles)):
                second = particles[j]
                radius_squared = ((first.x - second.x) ** 2 + (first.y - second.y) ** 2
                                  + (first.z - second.z) ** 2 + eps ** 2)
                if radius_squared == 0:
                    raise ValueError("Coincident particles require positive softening (eps).")
                yield -G * first.mass * second.mass / math.sqrt(radius_squared)

    return kinetic + math.fsum(potential_terms())


class BarnesHutSimulation:
    """A reproducible Newtonian simulation in dimensionless units (G = 1).

    ``particles`` overrides generated initial conditions and is defensively copied.
    ``seed`` uses a local RNG; seed=None preserves the legacy global-RNG behavior.
    The Plummer option samples a continuum equilibrium distribution; spiral and
    disk options are illustrative non-equilibrium states.
    ``euler`` is velocity-first symplectic Euler; leapfrog is kick-drift-kick.
    Barnes-Hut approximations do not guarantee exact pairwise momentum conservation.
    """

    def __init__(self, num_particles: int = 100, dt: float = 0.01, theta: float = 0.5,
                 eps: float = 0.05, mode: str = "bh", integrator: str = "leapfrog",
                 initial: str = "spiral", recorder=None, seed=None, particles=None):
        if isinstance(num_particles, bool) or not isinstance(num_particles, int) or num_particles < 0:
            raise ValueError("num_particles must be a nonnegative integer.")
        _finite_number(dt, "dt", positive=True)
        _finite_number(theta, "theta", nonnegative=True)
        _finite_number(eps, "eps", nonnegative=True)
        if mode not in ("bh", "direct"):
            raise ValueError("mode must be 'bh' or 'direct'.")
        if not isinstance(integrator, str) or integrator.lower() not in ("euler", "leapfrog", "rk4"):
            raise ValueError("integrator must be 'euler', 'leapfrog', or 'rk4'.")
        if initial not in ("spiral", "plummer", "kuzmin", "two_galaxies"):
            raise ValueError("initial must be 'spiral', 'plummer', 'kuzmin', or 'two_galaxies'.")
        self.dt = dt
        self.theta = theta
        self.eps = eps
        self.mode = mode
        self.integrator = integrator.lower()
        self.seed = seed
        self.time = 0.0
        self.recorder = recorder
        if particles is None:
            rng = random if seed is None else random.Random(seed)
            if initial == "plummer":
                self.particles = generate_plummer_sphere(num_particles, rng=rng)
            elif initial == "kuzmin":
                self.particles = generate_kuzmin_disk(num_particles, rng=rng)
            elif initial == "two_galaxies":
                self.particles = generate_two_galaxies(num_particles, separation=4.0, relative_velocity=1.0, rng=rng)
            else:
                self.particles = generate_spiral_galaxy(num_particles, rng=rng)
        else:
            supplied = list(particles)
            _validate_particles(supplied)
            self.particles = [Particle(p.x, p.y, p.z, p.vx, p.vy, p.vz, p.mass) for p in supplied]
        _validate_particles(self.particles)

    @staticmethod
    def _bounding_center(particles):
        if not particles:
            return 0.0, 0.0, 0.0
        lower = [min(getattr(p, axis) for p in particles) for axis in ("x", "y", "z")]
        upper = [max(getattr(p, axis) for p in particles) for axis in ("x", "y", "z")]
        return tuple(low / 2.0 + high / 2.0 for low, high in zip(lower, upper))

    def _build_tree(self, particles=None, geometry_origin=(0.0, 0.0, 0.0)) -> OctreeNode:
        particles = self.particles if particles is None else particles
        if not particles:
            return OctreeNode((0.0, 0.0, 0.0), 1.0, geometry_origin)
        geometry = [tuple(coordinate - origin for coordinate, origin in
                          zip((p.x, p.y, p.z), geometry_origin)) for p in particles]
        center = tuple(min(position[axis] for position in geometry) / 2.0
                       + max(position[axis] for position in geometry) / 2.0 for axis in range(3))
        half_size = max(abs(position[axis] - center[axis])
                        for position in geometry for axis in range(3))
        root = OctreeNode(center, max(OctreeNode.MIN_SIZE, half_size * (1.0 + 1e-12)), geometry_origin)
        for particle in particles:
            root.insert(particle)
        root.finalize()
        return root

    def _direct_forces(self) -> List[Tuple[float, float, float]]:
        accelerations = [[0.0, 0.0, 0.0] for _ in self.particles]
        for i, first in enumerate(self.particles):
            for j in range(i + 1, len(self.particles)):
                second = self.particles[j]
                dx, dy, dz = second.x - first.x, second.y - first.y, second.z - first.z
                radius_squared = dx * dx + dy * dy + dz * dz + self.eps * self.eps
                if radius_squared == 0:
                    raise ValueError("Coincident particles require positive softening (eps).")
                inverse_radius_cubed = radius_squared ** -1.5
                for axis, displacement in enumerate((dx, dy, dz)):
                    pair = displacement * inverse_radius_cubed
                    accelerations[i][axis] += pair * second.mass
                    accelerations[j][axis] -= pair * first.mass
        return [tuple(acceleration) for acceleration in accelerations]

    def accelerations(self) -> List[Tuple[float, float, float]]:
        """Evaluate the configured force solver at the current positions.

        theta=0 opens the entire tree and provides an exact direct-summation
        comparison. With eps=0, coincident distinct particles are singular.
        Tree partitioning uses root-midpoint-relative coordinates. Interactions
        retain original pair displacements and source-relative centers of mass,
        so recentering cannot erase close pairs in widely separated clusters.
        """
        _validate_particles(self.particles)
        if self.mode == "direct":
            return self._direct_forces()
        center = self._bounding_center(self.particles)
        tree = self._build_tree(geometry_origin=center)
        return [tree.compute_force_on(p, self.theta, eps=self.eps) for p in self.particles]

    def step(self):
        """Advance one step, restoring the state if numerical evaluation fails.

        Successful numerical steps are committed before calling the recorder.
        Recorder failures therefore do not roll back the physical state or time.
        """
        _validate_particles(self.particles)
        previous = [(p.x, p.y, p.z, p.vx, p.vy, p.vz) for p in self.particles]
        try:
            if self.integrator == "rk4":
                self._step_rk4()
            else:
                accelerations = self.accelerations()
                kick = self.dt * (0.5 if self.integrator == "leapfrog" else 1.0)
                for particle, (ax, ay, az) in zip(self.particles, accelerations):
                    particle.vx += ax * kick
                    particle.vy += ay * kick
                    particle.vz += az * kick
                    particle.x += particle.vx * self.dt
                    particle.y += particle.vy * self.dt
                    particle.z += particle.vz * self.dt
                if self.integrator == "leapfrog":
                    for particle, (ax, ay, az) in zip(self.particles, self.accelerations()):
                        particle.vx += ax * kick
                        particle.vy += ay * kick
                        particle.vz += az * kick
            _validate_particles(self.particles)
            next_time = self.time + self.dt
            _finite_number(next_time, "simulation time", nonnegative=True)
        except (ValueError, ArithmeticError):
            for particle, state in zip(self.particles, previous):
                particle.x, particle.y, particle.z, particle.vx, particle.vy, particle.vz = state
            raise
        self.time = next_time
        if self.recorder is not None:
            self.recorder.add_frame(self.particles)

    def _step_rk4(self):
        original_positions = [(p.x, p.y, p.z) for p in self.particles]
        original_velocities = [(p.vx, p.vy, p.vz) for p in self.particles]
        position_derivatives = [original_velocities]
        velocity_derivatives = [self.accelerations()]
        for fraction in (0.5, 0.5, 1.0):
            velocities = []
            for i, particle in enumerate(self.particles):
                position = tuple(original_positions[i][axis] + fraction * self.dt * position_derivatives[-1][i][axis]
                                 for axis in range(3))
                velocity = tuple(original_velocities[i][axis] + fraction * self.dt * velocity_derivatives[-1][i][axis]
                                 for axis in range(3))
                particle.x, particle.y, particle.z = position
                particle.vx, particle.vy, particle.vz = velocity
                velocities.append(velocity)
            position_derivatives.append(velocities)
            velocity_derivatives.append(self.accelerations())
        weights = (1.0, 2.0, 2.0, 1.0)
        for i, particle in enumerate(self.particles):
            position = tuple(original_positions[i][axis] + self.dt / 6.0 *
                             sum(weight * stage[i][axis] for weight, stage in zip(weights, position_derivatives))
                             for axis in range(3))
            velocity = tuple(original_velocities[i][axis] + self.dt / 6.0 *
                             sum(weight * stage[i][axis] for weight, stage in zip(weights, velocity_derivatives))
                             for axis in range(3))
            particle.x, particle.y, particle.z = position
            particle.vx, particle.vy, particle.vz = velocity

    def run(self, iterations: int, verbose: bool = True):
        """Run steps and return diagnostics; verbose=False makes the API silent.

        Energy diagnostics use exact pair summation, O(N^2), outside the timed
        integration loop. Relative energy error is None if initial energy is zero.
        Momentum errors are absolute vector norms and remain defined at zero.
        """
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
            raise ValueError("iterations must be a nonnegative integer.")
        _validate_particles(self.particles)
        initial_energy = compute_total_energy(self.particles, eps=self.eps)
        initial_momentum = compute_total_momentum(self.particles)
        initial_angular_momentum = compute_total_angular_momentum(self.particles)
        start = time.perf_counter()
        for _ in range(iterations):
            self.step()
        elapsed = time.perf_counter() - start
        final_energy = compute_total_energy(self.particles, eps=self.eps)
        final_momentum = compute_total_momentum(self.particles)
        final_angular_momentum = compute_total_angular_momentum(self.particles)
        change = final_energy - initial_energy
        diagnostics = {
            "steps": iterations,
            "simulated_time": iterations * self.dt,
            "elapsed_seconds": elapsed,
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_change": change,
            "relative_energy_error": abs(change / initial_energy) if initial_energy != 0 else None,
            "initial_momentum": initial_momentum,
            "final_momentum": final_momentum,
            "momentum_error": math.dist(initial_momentum, final_momentum),
            "initial_angular_momentum": initial_angular_momentum,
            "final_angular_momentum": final_angular_momentum,
            "angular_momentum_error": math.dist(initial_angular_momentum, final_angular_momentum),
        }
        if verbose:
            print("Momentum:", initial_momentum, "->", final_momentum)
            print("Momentum error:", diagnostics["momentum_error"])
            print("Energy drift:", change)
            print("Relative energy error:", diagnostics["relative_energy_error"])
            print("Angular momentum error:", diagnostics["angular_momentum_error"])
            print("Simulation time: %.2f s" % elapsed)
        return diagnostics
