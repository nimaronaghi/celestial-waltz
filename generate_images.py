"""
generate_images.py
-------------------

Generate illustrative images of the N-body simulation using different
initial conditions and integrators. These are not accuracy-validation figures.
Positions and elapsed simulation time use dimensionless code units (G=1).
The resulting images are stored in the
``images/`` directory. These can be embedded in documentation to showcase the
capabilities of the project without requiring users to run simulations
themselves.

Usage:

    python3 generate_images.py

It produces the following files:

``spiral_galaxy.png``
    A scatter plot of a spiral galaxy after a short evolution.

``plummer_sphere.png``
    A scatter plot of a Plummer sphere.

``kuzmin_disk.png``
    A scatter plot of a toy exponential-radius disk (legacy Kuzmin API name).

``galaxy_collision.png``
    A snapshot from a two‑galaxy collision simulation.

"""
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from nbody import BarnesHutSimulation


ILLUSTRATION_SEED = 42
INITIAL_LABELS = {
    "spiral": "Toy spiral initial state",
    "plummer": "Sampled Plummer sphere",
    "kuzmin": "Toy exponential-radius disk",
    "two_galaxies": "Toy two-galaxy interaction",
}


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def run_and_plot(initial: str, filename: str, integrator: str = "leapfrog", steps: int = 200, perspective_3d: bool = False):
    """Run a simulation with a given initial condition and save a scatter plot."""
    sim = BarnesHutSimulation(num_particles=500, dt=0.01, mode="bh", integrator=integrator,
                             initial=initial, seed=ILLUSTRATION_SEED)
    for _ in range(steps):
        sim.step()
    xs = [p.x for p in sim.particles]
    ys = [p.y for p in sim.particles]
    zs = [p.z for p in sim.particles]
    title = (f"{INITIAL_LABELS[initial]} (illustration)\n"
             f"t = {sim.time:g} code units; seed = {ILLUSTRATION_SEED}")
    plt.figure(figsize=(6, 6))
    if perspective_3d:
        ax = plt.axes(projection="3d")
        ax.scatter(xs, ys, zs, s=1, alpha=0.6)
        ax.set_xlabel("x (code units)")
        ax.set_ylabel("y (code units)")
        ax.set_zlabel("z (code units)")
        ax.set_title(title)
    else:
        plt.scatter(xs, ys, s=1, alpha=0.6)
        plt.xlabel("x (code units)")
        plt.ylabel("y (code units)")
        plt.title(title)
        plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join("images", filename), dpi=200)
    plt.close()
    print(f"Saved {filename}")


def main():
    ensure_dir("images")
    # Toy spiral after 300 steps
    run_and_plot("spiral", "spiral_galaxy.png", integrator="leapfrog", steps=300)
    # Plummer sphere without time evolution (snapshot at t=0)
    run_and_plot("plummer", "plummer_sphere.png", integrator="leapfrog", steps=0)
    # Toy exponential-radius disk after short evolution
    run_and_plot("kuzmin", "kuzmin_disk.png", integrator="leapfrog", steps=200)
    # Two galaxy collision snapshot
    sim = BarnesHutSimulation(num_particles=300, dt=0.01, mode="bh", integrator="leapfrog",
                             initial="two_galaxies", seed=ILLUSTRATION_SEED)
    # run for 200 steps to allow interaction
    for _ in range(200):
        sim.step()
    xs = [p.x for p in sim.particles]
    ys = [p.y for p in sim.particles]
    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, s=1, alpha=0.6)
    plt.xlabel("x (code units)")
    plt.ylabel("y (code units)")
    plt.title(f"Toy two-galaxy interaction (illustration)\n"
              f"t = {sim.time:g} code units; seed = {ILLUSTRATION_SEED}")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join("images", "galaxy_collision.png"), dpi=200)
    plt.close()
    print("Saved galaxy_collision.png")


if __name__ == "__main__":
    main()
