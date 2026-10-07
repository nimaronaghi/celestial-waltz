"""Optional PyTorch gravity solvers in dimensionless units.

Both backends use velocity-first (symplectic) Euler integration and Plummer
softening: a_i = G sum_j m_j (r_j-r_i) / (|r_j-r_i|^2 + eps^2)^(3/2).
The corresponding pair potential is -G m_i m_j / sqrt(r_ij^2 + eps^2).
Direct summation allocates O(N**2) storage. The recursive Barnes-Hut
backend is a reference implementation: Python traversal and device
synchronization can outweigh any GPU benefit, so benchmark before choosing it.
Its target-dependent approximations do not preserve pairwise momentum exactly;
the symplectic property of Euler applies to the exact conservative force field.
"""

import argparse
import math

import torch


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_scalar(value, name, *, allow_zero=False):
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")


def _validate_state(pos, vel, mass, dt, G, eps):
    """Reject incompatible tensors before either in-place state update."""
    if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] == 0:
        raise ValueError("pos must have shape (N, 3), with N > 0")
    if vel.shape != pos.shape or mass.shape != (pos.shape[0],):
        raise ValueError("vel must match pos and mass must have shape (N,)")
    if pos.dtype not in (torch.float32, torch.float64):
        raise ValueError("state tensors must use torch.float32 or torch.float64")
    for tensor in (vel, mass):
        if tensor.device != pos.device or tensor.dtype != pos.dtype:
            raise ValueError("pos, vel, and mass must share a device and dtype")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in (pos, vel, mass)):
        raise ValueError("state tensors must contain only finite values")
    if not bool((mass > 0).all()):
        raise ValueError("particle masses must be positive")
    _finite_scalar(dt, "dt")
    _finite_scalar(G, "G")
    _finite_scalar(eps, "eps", allow_zero=True)


def generate_spiral_galaxy(num, radius=1.0, device="cuda", seed=None, dtype=None):
    """Return illustrative disk tensors; a seed uses an isolated RNG stream.

    This initial condition is a visualization example, not an equilibrium model.
    All particle masses are one, so total mass and the point-mass speed estimate
    scale with num and sqrt(num), respectively. The speed estimate assumes G=1;
    no central point mass is added to the simulated particles.
    Reproducibility is within the same PyTorch version and device backend.
    """
    _positive_int(num, "num")
    _finite_scalar(radius, "radius")
    dtype = dtype or torch.get_default_dtype()
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    options = {"device": device, "dtype": dtype, "generator": generator}
    central_mass = num
    r = torch.sqrt(torch.rand(num, **options)) * radius
    angle = r * 4.0 + (torch.rand(num, **options) * 0.4 - 0.2)
    x = r * torch.cos(angle)
    y = r * torch.sin(angle)
    z = torch.randn(num, **options) * 0.05
    v_mag = torch.sqrt(central_mass / (r + 0.01))
    vx = -v_mag * torch.sin(angle)
    vy = v_mag * torch.cos(angle)
    vz = torch.randn(num, **options) * 0.01
    pos = torch.stack((x, y, z), dim=1)
    vel = torch.stack((vx, vy, vz), dim=1)
    mass = torch.ones(num, device=device, dtype=dtype)
    return pos, vel, mass


def step_direct(pos, vel, mass, dt, G=1.0, eps=0.05):
    """Advance tensors in place using all distinct particle pairs.

    ``eps=0`` is supported for distinct positions. Coincident unsoftened
    particles are singular and raise ValueError before modifying the state.
    """
    _validate_state(pos, vel, mass, dt, G, eps)
    # Entry [i, j] points from target i toward source j: gravity is attractive.
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    dist_sqr = (diff ** 2).sum(-1) + eps ** 2
    # Exclude self interactions before inversion, including when eps == 0.
    dist_sqr.fill_diagonal_(float("inf"))
    if bool((dist_sqr == 0).any()):
        raise ValueError("coincident particles require positive softening (eps)")
    inv_dist3 = dist_sqr.pow(-1.5)
    accel = (diff * inv_dist3.unsqueeze(-1) * mass.view(1, -1, 1)).sum(1) * G
    vel.add_(accel * dt)
    pos.add_(vel * dt)


class BHNode:
    """Octree node with exact particle membership and bucket-leaf masses."""

    def __init__(self, indices, center, half_size):
        self.indices = indices
        self.center = center
        self.half_size = half_size
        self.children = []
        self.mass = center.new_zeros(())
        self.com = center.new_zeros(3)
        self.com_reference = center.new_zeros(3)
        self.com_offset = center.new_zeros(3)
        self.particle_masses = None


def _build_tree(pos, mass, indices, center, half_size, min_size=1e-5, geometry_pos=None):
    # Partition coordinates may be recentered, but physical source positions
    # remain untouched: recentering a tiny pair near a distant outlier can round
    # its separation to zero even when the original coordinates resolve it.
    geometry_pos = pos if geometry_pos is None else geometry_pos
    node = BHNode(indices, center, half_size)
    if indices.numel() == 0:
        return node
    source_mass = mass[indices]
    node.mass = source_mass.sum()
    node.com_reference = pos[indices[0]]
    node.com_offset = (
        (pos[indices] - node.com_reference) * source_mass.unsqueeze(1)
    ).sum(0) / node.mass
    node.com = node.com_reference + node.com_offset
    if (
        indices.numel() == 1
        or half_size <= min_size
        or bool((pos[indices] == pos[indices[0]]).all())
    ):
        # Close and coincident particles stay individually addressable.
        node.particle_masses = source_mass
        return node
    half = half_size / 2.0
    xmask = geometry_pos[indices, 0] > center[0]
    ymask = geometry_pos[indices, 1] > center[1]
    zmask = geometry_pos[indices, 2] > center[2]
    for i in range(8):
        mask = (
            (xmask == bool(i & 1))
            & (ymask == bool(i & 2))
            & (zmask == bool(i & 4))
        )
        sub_idx = indices[mask]
        if sub_idx.numel() == 0:
            continue
        offset = pos.new_tensor(
            [half if i & 1 else -half, half if i & 2 else -half, half if i & 4 else -half]
        )
        child = _build_tree(pos, mass, sub_idx, center + offset, half, min_size, geometry_pos)
        node.children.append(child)
    return node


def _bh_force(node, i, pos, theta, G=1.0, eps=0.05):
    """Return acceleration; the opening criterion uses unsoftened distance."""
    if node.indices.numel() == 0:
        return pos.new_zeros(3)
    if not node.children:
        # Do not collapse a bucket into one mass: it may contain the target.
        mask = node.indices != i
        diff = pos[node.indices[mask]] - pos[i]
        dist_sqr = (diff ** 2).sum(-1) + eps ** 2
        if bool((dist_sqr == 0).any()):
            raise ValueError("coincident particles require positive softening (eps)")
        factors = G * node.particle_masses[mask] * dist_sqr.pow(-1.5)
        return (diff * factors.unsqueeze(1)).sum(0)
    # Subtract the target before adding the COM offset, preserving resolved
    # source-target differences that an absolute COM coordinate may round away.
    dx = (node.com_reference - pos[i]) + node.com_offset
    distance = torch.linalg.vector_norm(dx)
    contains_target = bool((node.indices == i).any())
    if not contains_target and 2.0 * node.half_size < theta * distance:
        dist_sqr = (dx ** 2).sum() + eps ** 2
        return dx * (G * node.mass * dist_sqr.pow(-1.5))
    acc = pos.new_zeros(3)
    for child in node.children:
        acc = acc + _bh_force(child, i, pos, theta, G, eps)
    return acc


def step_bh(pos, vel, mass, dt, theta=0.5, G=1.0, eps=0.05):
    """Advance tensors in place with a tree enclosing the current positions.

    ``theta=0`` evaluates all sources exactly. Larger values trade accuracy
    for fewer node visits; nodes containing the target are always opened.
    """
    _validate_state(pos, vel, mass, dt, G, eps)
    _finite_scalar(theta, "theta", allow_zero=True)
    if eps == 0 and torch.unique(pos, dim=0).shape[0] != pos.shape[0]:
        raise ValueError("coincident particles require positive softening (eps)")
    lower, upper = pos.amin(dim=0), pos.amax(dim=0)
    center = lower / 2.0 + upper / 2.0
    # Partition the tree in a local frame. Otherwise adding small
    # cell offsets to a large absolute center can round away the bounds padding,
    # changing octants (and approximated groups) after a simple translation.
    local_pos = pos - center
    extent = float(local_pos.abs().amax())
    half_size = max(extent * (1.0 + 4.0 * torch.finfo(pos.dtype).eps), 1e-5)
    indices = torch.arange(pos.size(0), device=pos.device)
    root = _build_tree(pos, mass, indices, pos.new_zeros(3), half_size, geometry_pos=local_pos)
    accel = torch.stack([
        _bh_force(root, i, pos, theta, G, eps) for i in range(pos.size(0))
    ])
    vel.add_(accel * dt)
    pos.add_(vel * dt)


def run(n_particles, iterations, dt, device=None, mode="direct", theta=0.5, eps=0.05,
        seed=None, dtype=None):
    """Run a seeded example, returning final positions and velocities."""
    _positive_int(n_particles, "n_particles")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if mode not in ("direct", "bh"):
        raise ValueError("mode must be 'direct' or 'bh'")
    _finite_scalar(dt, "dt")
    _finite_scalar(theta, "theta", allow_zero=True)
    _finite_scalar(eps, "eps", allow_zero=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pos, vel, mass = generate_spiral_galaxy(n_particles, device=device, seed=seed, dtype=dtype)
    for _ in range(iterations):
        if mode == "bh":
            step_bh(pos, vel, mass, dt, theta=theta, eps=eps)
        else:
            step_direct(pos, vel, mass, dt, eps=eps)
    return pos, vel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=1000, help="Number of particles")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step in dimensionless units")
    parser.add_argument("--mode", choices=["direct", "bh"], default="direct", help="Force computation mode")
    parser.add_argument("--theta", type=float, default=0.5, help="Barnes-Hut opening angle")
    parser.add_argument("--eps", type=float, default=0.05, help="Softening length in dimensionless units")
    parser.add_argument("--seed", type=int, default=None, help="Initial-condition random seed")
    parser.add_argument("--device", default=None, help="PyTorch device, e.g. cpu or cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()
    run(args.particles, args.iterations, args.dt, device=args.device, mode=args.mode,
        theta=args.theta, eps=args.eps, seed=args.seed, dtype=getattr(torch, args.dtype))


if __name__ == "__main__":
    main()
