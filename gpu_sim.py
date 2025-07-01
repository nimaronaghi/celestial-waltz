import argparse
import torch

def generate_spiral_galaxy(num, radius=1.0, device="cuda"):
    central_mass = num
    G = 1.0
    r = torch.sqrt(torch.rand(num, device=device)) * radius
    angle = r * 4.0 + (torch.rand(num, device=device) * 0.4 - 0.2)
    x = r * torch.cos(angle)
    y = r * torch.sin(angle)
    z = torch.randn(num, device=device) * 0.05
    v_mag = torch.sqrt(G * central_mass / (r + 0.01))
    vx = -v_mag * torch.sin(angle)
    vy = v_mag * torch.cos(angle)
    vz = torch.randn(num, device=device) * 0.01
    pos = torch.stack((x, y, z), dim=1)
    vel = torch.stack((vx, vy, vz), dim=1)
    mass = torch.ones(num, device=device)
    return pos, vel, mass

def step_direct(pos, vel, mass, dt, G=1.0, eps=0.05):
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist_sqr = (diff ** 2).sum(-1) + eps ** 2
    inv_dist3 = dist_sqr.pow(-1.5)
    accel = (diff * inv_dist3.unsqueeze(-1) * mass.view(1, -1, 1)).sum(1) * G
    vel += accel * dt
    pos += vel * dt


class BHNode:
    def __init__(self, indices, center, half_size):
        self.indices = indices
        self.center = center
        self.half_size = half_size
        self.children = []
        self.mass = 0.0
        self.com = torch.zeros(3, device=center.device)


def _build_tree(pos, mass, indices, center, half_size, min_size=1e-5):
    node = BHNode(indices, center, half_size)
    if indices.numel() == 0:
        return node
    if indices.numel() == 1 or half_size < min_size:
        node.mass = mass[indices].sum()
        node.com = (pos[indices] * mass[indices].unsqueeze(1)).sum(0) / node.mass
        return node
    half = half_size / 2.0
    xmask = pos[indices, 0] > center[0]
    ymask = pos[indices, 1] > center[1]
    zmask = pos[indices, 2] > center[2]
    for i in range(8):
        mask = (
            (xmask == bool(i & 1))
            & (ymask == bool(i & 2))
            & (zmask == bool(i & 4))
        )
        sub_idx = indices[mask]
        offset = torch.tensor(
            [half if i & 1 else -half, half if i & 2 else -half, half if i & 4 else -half],
            device=pos.device,
        )
        child_center = center + offset
        child = _build_tree(pos, mass, sub_idx, child_center, half, min_size)
        if child.indices.numel() > 0:
            node.children.append(child)
    if node.children:
        node.mass = sum(c.mass for c in node.children)
        node.com = sum(c.mass * c.com for c in node.children) / node.mass
    return node


def _bh_force(node, i, pos, theta, G=1.0, eps=0.05):
    if node.mass == 0:
        return torch.zeros(3, device=pos.device)
    dx = node.com - pos[i]
    dist = torch.sqrt((dx * dx).sum() + eps * eps)
    if node.indices.numel() == 1 and node.indices[0] == i:
        return torch.zeros(3, device=pos.device)
    if not node.children or node.half_size / dist < theta:
        return dx * (G * node.mass / (dist ** 3))
    acc = torch.zeros(3, device=pos.device)
    for child in node.children:
        acc = acc + _bh_force(child, i, pos, theta, G, eps)
    return acc


def step_bh(pos, vel, mass, dt, theta=0.5, G=1.0, eps=0.05):
    center = torch.tensor([0.0, 0.0, 0.0], device=pos.device)
    indices = torch.arange(pos.size(0), device=pos.device)
    root = _build_tree(pos, mass, indices, center, 2.0)
    forces = torch.stack([_bh_force(root, i, pos, theta, G, eps) for i in range(pos.size(0))])
    vel += forces * dt
    pos += vel * dt


def run(n_particles, iterations, dt, device=None, mode="direct", theta=0.5, eps=0.05):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pos, vel, mass = generate_spiral_galaxy(n_particles, device=device)
    for _ in range(iterations):
        if mode == "bh":
            step_bh(pos, vel, mass, dt, theta=theta, eps=eps)
        else:
            step_direct(pos, vel, mass, dt, eps=eps)
    return pos, vel


def main():
    parser = argparse.ArgumentParser(description="GPU accelerated N-body simulation")
    parser.add_argument("--particles", type=int, default=100000, help="Number of particles")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step in Myr")
    parser.add_argument("--mode", choices=["direct", "bh"], default="direct", help="Force computation mode")
    parser.add_argument("--theta", type=float, default=0.5, help="Barnes-Hut opening angle")
    parser.add_argument("--eps", type=float, default=0.05, help="Softening parameter")
    args = parser.parse_args()
    run(args.particles, args.iterations, args.dt, mode=args.mode, theta=args.theta, eps=args.eps)


if __name__ == "__main__":
    main()
