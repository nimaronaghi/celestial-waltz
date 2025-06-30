import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


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


class BarnesHutSimulation:
    def __init__(self, num_particles: int = 100, dt: float = 0.01, theta: float = 0.5):
        self.dt = dt
        self.theta = theta
        self.particles = generate_spiral_galaxy(num_particles)

    def _build_tree(self) -> OctreeNode:
        root = OctreeNode((0.0, 0.0, 0.0), 2.0)
        for p in self.particles:
            root.insert(p)
        root.finalize()
        return root

    def step(self):
        tree = self._build_tree()
        for p in self.particles:
            ax, ay, az = tree.compute_force_on(p, self.theta)
            p.vx += ax * self.dt
            p.vy += ay * self.dt
            p.vz += az * self.dt
        for p in self.particles:
            p.x += p.vx * self.dt
            p.y += p.vy * self.dt
            p.z += p.vz * self.dt
