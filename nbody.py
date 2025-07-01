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
                 eps: float = 0.05, mode: str = "bh", integrator: str = "euler", recorder=None):
        self.dt = dt
        self.theta = theta
        self.eps = eps
        self.mode = mode
        self.integrator = integrator
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
        if self.mode == "direct":
            forces = self._direct_forces()
        else:
            tree = self._build_tree()
            with ThreadPoolExecutor() as ex:
                forces = list(ex.map(lambda p: tree.compute_force_on(p, self.theta, eps=self.eps), self.particles))

        if self.integrator == "leapfrog":
            for (p, (ax, ay, az)) in zip(self.particles, forces):
                p.vx += ax * self.dt * 0.5
                p.vy += ay * self.dt * 0.5
                p.vz += az * self.dt * 0.5
                p.x += p.vx * self.dt
                p.y += p.vy * self.dt
                p.z += p.vz * self.dt
            if self.mode == "direct":
                forces = self._direct_forces()
            else:
                tree = self._build_tree()
                with ThreadPoolExecutor() as ex:
                    forces = list(ex.map(lambda p: tree.compute_force_on(p, self.theta, eps=self.eps), self.particles))
            for (p, (ax, ay, az)) in zip(self.particles, forces):
                p.vx += ax * self.dt * 0.5
                p.vy += ay * self.dt * 0.5
                p.vz += az * self.dt * 0.5
        else:  # euler
            for (p, (ax, ay, az)) in zip(self.particles, forces):
                p.vx += ax * self.dt
                p.vy += ay * self.dt
                p.vz += az * self.dt
                p.x += p.vx * self.dt
                p.y += p.vy * self.dt
                p.z += p.vz * self.dt

        if self.recorder is not None:
            self.recorder.add_frame(self.particles)

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
