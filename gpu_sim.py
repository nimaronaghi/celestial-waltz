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

def step(pos, vel, mass, dt, G=1.0, eps=0.05):
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist_sqr = (diff ** 2).sum(-1) + eps ** 2
    inv_dist3 = dist_sqr.pow(-1.5)
    accel = (diff * inv_dist3.unsqueeze(-1) * mass.view(1, -1, 1)).sum(1) * G
    vel += accel * dt
    pos += vel * dt


def run(n_particles, iterations, dt, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pos, vel, mass = generate_spiral_galaxy(n_particles, device=device)
    for _ in range(iterations):
        step(pos, vel, mass, dt)
    return pos, vel


def main():
    parser = argparse.ArgumentParser(description="GPU accelerated N-body simulation")
    parser.add_argument("--particles", type=int, default=100000, help="Number of particles")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step in Myr")
    args = parser.parse_args()
    run(args.particles, args.iterations, args.dt)


if __name__ == "__main__":
    main()
