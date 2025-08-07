import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import time
from concurrent.futures import ThreadPoolExecutor


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
    MIN_SIZE = 1e-5

    def __init__(self, center: Tuple[float, float, float], half_size: float):
        self.center = list(center)
        self.half_size = half_size
        self.particle: Optional[Particle] = None
        self.children: List[Optional["OctreeNode"]] = [None] * 8
        self.mass = 0.0
        self.com = [0.0, 0.0, 0.0]

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
            self.children[i] = OctreeNode(new_center, quarter)

    def _child_index(self, p: Particle) -> int:
        idx = 0
        if p.x > self.center[0]:
            idx |= 1
        if p.y > self.center[1]:
            idx |= 2
        if p.z > self.center[2]:
            idx |= 4
        return idx

    def insert(self, p: Particle):
        if self._is_leaf():
            if self.particle is None:
                self.particle = p
                self.mass = p.mass
                self.com = [p.x, p.y, p.z]
                return
            elif self.half_size < self.MIN_SIZE:
                # Avoid infinite subdivision when particles overlap closely
                self.mass += p.mass
                self.com[0] += p.mass * p.x
                self.com[1] += p.mass * p.y
                self.com[2] += p.mass * p.z
                return
            else:
                self._subdivide()
                existing = self.particle
                self.particle = None
                self.children[self._child_index(existing)].insert(existing)
        self.children[self._child_index(p)].insert(p)
        # Update center of mass and mass
        self.mass += p.mass
        self.com[0] += p.mass * p.x
        self.com[1] += p.mass * p.y
        self.com[2] += p.mass * p.z

    def finalize(self):
        if self.mass > 0:
            self.com[0] /= self.mass
            self.com[1] /= self.mass
            self.com[2] /= self.mass
        if not self._is_leaf():
            for child in self.children:
                if child:
                    child.finalize()

    def compute_force_on(
        self, p: Particle, theta: float = 0.5, G: float = 1.0, eps: float = 0.05
    ) -> Tuple[float, float, float]:
        if self.mass == 0 or (self.particle is p and self._is_leaf()):
            return 0.0, 0.0, 0.0
        dx = self.com[0] - p.x
        dy = self.com[1] - p.y
        dz = self.com[2] - p.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz + eps * eps)
        if self._is_leaf() or self.half_size / dist < theta:
            factor = G * self.mass / (dist ** 3)
            return dx * factor, dy * factor, dz * factor
        fx = fy = fz = 0.0
        for child in self.children:
            if child:
                cfx, cfy, cfz = child.compute_force_on(p, theta, G, eps)
                fx += cfx
                fy += cfy
                fz += cfz
        return fx, fy, fz


def generate_spiral_galaxy(num: int, radius: float = 1.0) -> List[Particle]:
    particles: List[Particle] = []
    central_mass = num
    G = 1.0
    for _ in range(num):
        r = math.sqrt(random.random()) * radius
        angle = r * 4.0 + random.uniform(-0.2, 0.2)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        z = random.gauss(0, 0.05)
        v_mag = math.sqrt(G * central_mass / (r + 0.01))
        vx = -v_mag * math.sin(angle)
        vy = v_mag * math.cos(angle)
        vz = random.gauss(0, 0.01)
        particles.append(Particle(x, y, z, vx, vy, vz))
    return particles

# Additional initial condition generators inspired by contemporary N‑body projects.
def generate_plummer_sphere(num: int, scale: float = 1.0) -> List[Particle]:
    """
    Generate a spherical distribution of particles following the Plummer model.

    The Plummer profile is commonly used in astrophysical simulations to model star
    clusters. Particles are drawn from a density profile with finite mass and
    characteristic scale length.

    Parameters
    ----------
    num : int
        Number of particles to generate.
    scale : float, optional
        Scale radius of the Plummer sphere.

    Returns
    -------
    List[Particle]
        List of Particles with positions and velocities drawn from the Plummer model.
    """
    particles: List[Particle] = []
    G = 1.0
    M = num  # total mass proportional to number of particles
    for _ in range(num):
        # sample radius from Plummer distribution
        x = random.random()
        r = scale / math.sqrt(x ** (-2.0 / 3.0) - 1.0)
        # isotropic angles
        phi = random.uniform(0, 2 * math.pi)
        costheta = random.uniform(-1.0, 1.0)
        sintheta = math.sqrt(1 - costheta * costheta)
        x_pos = r * sintheta * math.cos(phi)
        y_pos = r * sintheta * math.sin(phi)
        z_pos = r * costheta
        # simple circular velocity estimate
        v_mag = math.sqrt(G * M / (r + 1e-3))
        # random direction perpendicular to position vector for approximate circular orbit
        # choose arbitrary vector not parallel to position
        if abs(x_pos) < abs(y_pos):
            ax = 1.0
            ay = 0.0
            az = 0.0
        else:
            ax = 0.0
            ay = 1.0
            az = 0.0
        # cross product to get perpendicular vector
        vx_dir = (y_pos * az - z_pos * ay, z_pos * ax - x_pos * az, x_pos * ay - y_pos * ax)
        norm = math.sqrt(vx_dir[0] ** 2 + vx_dir[1] ** 2 + vx_dir[2] ** 2) + 1e-12
        vx_dir = (vx_dir[0] / norm, vx_dir[1] / norm, vx_dir[2] / norm)
        vx = v_mag * vx_dir[0]
        vy = v_mag * vx_dir[1]
        vz = v_mag * vx_dir[2]
        particles.append(Particle(x_pos, y_pos, z_pos, vx, vy, vz))
    return particles


def generate_kuzmin_disk(num: int, radius: float = 1.0, thickness: float = 0.05) -> List[Particle]:
    """
    Generate a Kuzmin disk galaxy with exponential radial distribution and small vertical thickness.

    The Kuzmin disk is a simple model for a thin rotating galaxy. Particles are distributed
    in a disk with exponential decay in radius and small Gaussian scatter in the vertical direction.

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
        Particles representing a rotating Kuzmin disk.
    """
    particles: List[Particle] = []
    G = 1.0
    total_mass = num
    for _ in range(num):
        # sample radius with exponential distribution
        r = -radius * math.log(1.0 - random.random())
        theta = random.uniform(0, 2 * math.pi)
        x_pos = r * math.cos(theta)
        y_pos = r * math.sin(theta)
        z_pos = random.gauss(0.0, thickness)
        # compute circular velocity
        v_mag = math.sqrt(G * total_mass / (r + 0.1))
        vx = -v_mag * math.sin(theta)
        vy = v_mag * math.cos(theta)
        vz = random.gauss(0.0, thickness * 0.1)
        particles.append(Particle(x_pos, y_pos, z_pos, vx, vy, vz))
    return particles


def generate_two_galaxies(num_per_galaxy: int, separation: float = 5.0, relative_velocity: float = 0.0) -> List[Particle]:
    """
    Create a simple two‑galaxy collision scenario. Two spiral galaxies are placed apart and given a relative velocity.

    Parameters
    ----------
    num_per_galaxy : int
        Number of particles per galaxy.
    separation : float, optional
        Initial separation between the centers of the galaxies.
    relative_velocity : float, optional
        Magnitude of the initial relative velocity between the galaxies.

    Returns
    -------
    List[Particle]
        Combined list of particles belonging to two galaxies.
    """
    # first galaxy centered at (-separation/2, 0, 0)
    galaxy1 = generate_spiral_galaxy(num_per_galaxy, radius=1.0)
    for p in galaxy1:
        p.x -= separation / 2.0
        # apply half the relative velocity in +x direction
        p.vx += 0.5 * relative_velocity
    # second galaxy centered at (+separation/2, 0, 0)
    galaxy2 = generate_spiral_galaxy(num_per_galaxy, radius=1.0)
    for p in galaxy2:
        p.x += separation / 2.0
        # apply half the relative velocity in -x direction
        p.vx -= 0.5 * relative_velocity
    return galaxy1 + galaxy2


def compute_total_momentum(particles: Iterable[Particle]):
    px = sum(p.mass * p.vx for p in particles)
    py = sum(p.mass * p.vy for p in particles)
    pz = sum(p.mass * p.vz for p in particles)
    return px, py, pz


def compute_total_energy(particles: Iterable[Particle], G: float = 1.0, eps: float = 0.05):
    KE = sum(0.5 * p.mass * (p.vx ** 2 + p.vy ** 2 + p.vz ** 2) for p in particles)
    PE = 0.0
    plist = list(particles)
    for i in range(len(plist)):
        for j in range(i + 1, len(plist)):
            dx = plist[i].x - plist[j].x
            dy = plist[i].y - plist[j].y
            dz = plist[i].z - plist[j].z
            r = math.sqrt(dx * dx + dy * dy + dz * dz + eps * eps)
            PE -= G * plist[i].mass * plist[j].mass / r
    return KE + PE


class BarnesHutSimulation:
    def __init__(self, num_particles: int = 100, dt: float = 0.01, theta: float = 0.5,
                 eps: float = 0.05, mode: str = "bh", integrator: str = "euler",
                 initial: str = "spiral", recorder=None):
        """
        Parameters
        ----------
        num_particles : int
            Number of particles to simulate (per galaxy for two_galaxies).
        dt : float
            Time step.
        theta : float
            Opening angle parameter for Barnes‑Hut approximation.
        eps : float
            Softening length.
        mode : str, optional
            'bh' for Barnes‑Hut or 'direct' for direct summation.
        integrator : str, optional
            Choice of integrator: 'euler', 'leapfrog', or 'rk4'.
        initial : str, optional
            Initial condition type: 'spiral', 'plummer', 'kuzmin', or 'two_galaxies'.
            When using 'two_galaxies' the `num_particles` refers to the number of particles
            per galaxy, resulting in 2*num_particles total.
        recorder : SimulationRecorder, optional
            If provided, records particle positions at each step.
        """
        self.dt = dt
        self.theta = theta
        self.eps = eps
        self.mode = mode
        self.integrator = integrator.lower()
        # generate initial particle set
        if initial == "plummer":
            self.particles = generate_plummer_sphere(num_particles)
        elif initial == "kuzmin":
            self.particles = generate_kuzmin_disk(num_particles)
        elif initial == "two_galaxies":
            # default separation and relative velocity for collisions
            self.particles = generate_two_galaxies(num_particles, separation=4.0, relative_velocity=1.0)
        else:
            # default spiral galaxy
            self.particles = generate_spiral_galaxy(num_particles)
        self.recorder = recorder

    def _build_tree(self) -> OctreeNode:
        root = OctreeNode((0.0, 0.0, 0.0), 2.0)
        for p in self.particles:
            root.insert(p)
        root.finalize()
        return root

    def _direct_forces(self) -> List[Tuple[float, float, float]]:
        n = len(self.particles)
        forces = [(0.0, 0.0, 0.0)] * n
        G = 1.0
        for i in range(n):
            fx, fy, fz = 0.0, 0.0, 0.0
            for j in range(i + 1, n):
                pi = self.particles[i]
                pj = self.particles[j]
                dx = pj.x - pi.x
                dy = pj.y - pi.y
                dz = pj.z - pi.z
                r = math.sqrt(dx * dx + dy * dy + dz * dz + self.eps * self.eps)
                f = G / (r ** 3)
                fx_ij = dx * f
                fy_ij = dy * f
                fz_ij = dz * f
                fx += fx_ij * pj.mass
                fy += fy_ij * pj.mass
                fz += fz_ij * pj.mass
                forces[j] = (
                    forces[j][0] - fx_ij * pi.mass,
                    forces[j][1] - fy_ij * pi.mass,
                    forces[j][2] - fz_ij * pi.mass,
                )
            forces[i] = (fx, fy, fz)
        return forces

    def step(self):
        """
        Advance the simulation by one time step using the selected integrator.
        """
        if self.integrator == "rk4":
            self._step_rk4()
            if self.recorder is not None:
                self.recorder.add_frame(self.particles)
            return
        # compute forces once at the beginning of the step
        if self.mode == "direct":
            forces = self._direct_forces()
        else:
            tree = self._build_tree()
            with ThreadPoolExecutor() as ex:
                forces = list(ex.map(lambda p: tree.compute_force_on(p, self.theta, eps=self.eps), self.particles))
        if self.integrator == "leapfrog":
            # half kick
            for (p, (ax, ay, az)) in zip(self.particles, forces):
                p.vx += ax * self.dt * 0.5
                p.vy += ay * self.dt * 0.5
                p.vz += az * self.dt * 0.5
                p.x += p.vx * self.dt
                p.y += p.vy * self.dt
                p.z += p.vz * self.dt
            # recompute forces mid‑step
            if self.mode == "direct":
                forces = self._direct_forces()
            else:
                tree = self._build_tree()
                with ThreadPoolExecutor() as ex:
                    forces = list(ex.map(lambda p: tree.compute_force_on(p, self.theta, eps=self.eps), self.particles))
            # second half kick
            for (p, (ax, ay, az)) in zip(self.particles, forces):
                p.vx += ax * self.dt * 0.5
                p.vy += ay * self.dt * 0.5
                p.vz += az * self.dt * 0.5
        else:
            # default Euler integrator
            for (p, (ax, ay, az)) in zip(self.particles, forces):
                p.vx += ax * self.dt
                p.vy += ay * self.dt
                p.vz += az * self.dt
                p.x += p.vx * self.dt
                p.y += p.vy * self.dt
                p.z += p.vz * self.dt
        if self.recorder is not None:
            self.recorder.add_frame(self.particles)

    def _step_rk4(self):
        """
        Perform a single Runge–Kutta 4th order integration step.

        This integrator provides higher accuracy than Euler or leapfrog at the cost of additional force evaluations.
        """
        n = len(self.particles)
        # store original positions and velocities
        orig_pos = [(p.x, p.y, p.z) for p in self.particles]
        orig_vel = [(p.vx, p.vy, p.vz) for p in self.particles]

        # helper to compute accelerations for a list of Particles
        def compute_accelerations():
            if self.mode == "direct":
                forces = self._direct_forces()
            else:
                tree = self._build_tree()
                with ThreadPoolExecutor() as ex:
                    forces = list(ex.map(lambda p: tree.compute_force_on(p, self.theta, eps=self.eps), self.particles))
            acc = []
            for i, (ax, ay, az) in enumerate(forces):
                m = self.particles[i].mass if hasattr(self.particles[i], "mass") else 1.0
                acc.append((ax, ay, az))
            return acc

        # k1: acceleration at current positions
        a1 = compute_accelerations()
        # store k1 velocities and positions increments
        k1_v = [(ax * self.dt, ay * self.dt, az * self.dt) for (ax, ay, az) in a1]
        k1_x = [(vx * self.dt, vy * self.dt, vz * self.dt) for (vx, vy, vz) in orig_vel]

        # update positions and velocities for k2
        for i, p in enumerate(self.particles):
            p.x = orig_pos[i][0] + 0.5 * k1_x[i][0]
            p.y = orig_pos[i][1] + 0.5 * k1_x[i][1]
            p.z = orig_pos[i][2] + 0.5 * k1_x[i][2]
            p.vx = orig_vel[i][0] + 0.5 * k1_v[i][0]
            p.vy = orig_vel[i][1] + 0.5 * k1_v[i][1]
            p.vz = orig_vel[i][2] + 0.5 * k1_v[i][2]
        a2 = compute_accelerations()
        k2_v = [(ax * self.dt, ay * self.dt, az * self.dt) for (ax, ay, az) in a2]
        k2_x = [( (orig_vel[i][0] + 0.5 * k1_v[i][0]) * self.dt,
                  (orig_vel[i][1] + 0.5 * k1_v[i][1]) * self.dt,
                  (orig_vel[i][2] + 0.5 * k1_v[i][2]) * self.dt ) for i in range(n)]

        # update positions and velocities for k3
        for i, p in enumerate(self.particles):
            p.x = orig_pos[i][0] + 0.5 * k2_x[i][0]
            p.y = orig_pos[i][1] + 0.5 * k2_x[i][1]
            p.z = orig_pos[i][2] + 0.5 * k2_x[i][2]
            p.vx = orig_vel[i][0] + 0.5 * k2_v[i][0]
            p.vy = orig_vel[i][1] + 0.5 * k2_v[i][1]
            p.vz = orig_vel[i][2] + 0.5 * k2_v[i][2]
        a3 = compute_accelerations()
        k3_v = [(ax * self.dt, ay * self.dt, az * self.dt) for (ax, ay, az) in a3]
        k3_x = [( (orig_vel[i][0] + 0.5 * k2_v[i][0]) * self.dt,
                  (orig_vel[i][1] + 0.5 * k2_v[i][1]) * self.dt,
                  (orig_vel[i][2] + 0.5 * k2_v[i][2]) * self.dt ) for i in range(n)]

        # update positions and velocities for k4
        for i, p in enumerate(self.particles):
            p.x = orig_pos[i][0] + k3_x[i][0]
            p.y = orig_pos[i][1] + k3_x[i][1]
            p.z = orig_pos[i][2] + k3_x[i][2]
            p.vx = orig_vel[i][0] + k3_v[i][0]
            p.vy = orig_vel[i][1] + k3_v[i][1]
            p.vz = orig_vel[i][2] + k3_v[i][2]
        a4 = compute_accelerations()
        k4_v = [(ax * self.dt, ay * self.dt, az * self.dt) for (ax, ay, az) in a4]
        k4_x = [( (orig_vel[i][0] + k3_v[i][0]) * self.dt,
                  (orig_vel[i][1] + k3_v[i][1]) * self.dt,
                  (orig_vel[i][2] + k3_v[i][2]) * self.dt ) for i in range(n)]

        # final update of positions and velocities using RK4 combination
        for i, p in enumerate(self.particles):
            dx = (k1_x[i][0] + 2*k2_x[i][0] + 2*k3_x[i][0] + k4_x[i][0]) / 6.0
            dy = (k1_x[i][1] + 2*k2_x[i][1] + 2*k3_x[i][1] + k4_x[i][1]) / 6.0
            dz = (k1_x[i][2] + 2*k2_x[i][2] + 2*k3_x[i][2] + k4_x[i][2]) / 6.0
            dvx = (k1_v[i][0] + 2*k2_v[i][0] + 2*k3_v[i][0] + k4_v[i][0]) / 6.0
            dvy = (k1_v[i][1] + 2*k2_v[i][1] + 2*k3_v[i][1] + k4_v[i][1]) / 6.0
            dvz = (k1_v[i][2] + 2*k2_v[i][2] + 2*k3_v[i][2] + k4_v[i][2]) / 6.0
            p.x = orig_pos[i][0] + dx
            p.y = orig_pos[i][1] + dy
            p.z = orig_pos[i][2] + dz
            p.vx = orig_vel[i][0] + dvx
            p.vy = orig_vel[i][1] + dvy
            p.vz = orig_vel[i][2] + dvz

    def run(self, iterations: int):
        start_mom = compute_total_momentum(self.particles)
        start_energy = compute_total_energy(self.particles, eps=self.eps)
        t0 = time.time()
        for _ in range(iterations):
            self.step()
        elapsed = time.time() - t0
        end_mom = compute_total_momentum(self.particles)
        end_energy = compute_total_energy(self.particles, eps=self.eps)
        print("Momentum:", start_mom, "->", end_mom)
        if sum(abs(m) for m in start_mom) > 0:
            delta = [abs(e - s) / abs(s) for s, e in zip(start_mom, end_mom)]
            print("Relative change:", delta)
        print("Energy drift:", end_energy - start_energy)
        print("Simulation time: %.2f s" % elapsed)
