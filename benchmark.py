"""
benchmark.py
---------------

This script compares the performance of the direct (O(N^2)) and Barnes–Hut (O(N log N))
gravity solvers implemented in ``nbody.py``. It runs a small number of simulation
steps for varying particle counts and measures the wall clock time required.

The results are plotted using matplotlib and saved to ``benchmark.png`` in the project
directory. You can run this script from the command line:

    python3 benchmark.py

The output figure contains two curves: one for direct force calculation and one for
Barnes–Hut. The horizontal axis shows the number of particles and the vertical axis
shows the average time per integration step (in seconds).  This allows you to
visually compare the asymptotic scaling of the two algorithms.

"""
import time
import matplotlib.pyplot as plt

from nbody import BarnesHutSimulation


def run_simulation(n_particles: int, mode: str, integrator: str = "euler", steps: int = 3) -> float:
    """Run a short simulation and return the average time per step."""
    sim = BarnesHutSimulation(num_particles=n_particles, dt=0.01, mode=mode, integrator=integrator)
    # warm up
    sim.step()
    t0 = time.time()
    for _ in range(steps):
        sim.step()
    elapsed = time.time() - t0
    return elapsed / steps


def benchmark(particle_counts=None):
    if particle_counts is None:
        particle_counts = [50, 100, 200, 400, 800]
    direct_times = []
    bh_times = []
    for n in particle_counts:
        print(f"Benchmarking {n} particles...")
        t_direct = run_simulation(n, mode="direct", integrator="euler", steps=2)
        t_bh = run_simulation(n, mode="bh", integrator="euler", steps=2)
        direct_times.append(t_direct)
        bh_times.append(t_bh)
    return particle_counts, direct_times, bh_times


def plot_results(counts, direct_times, bh_times):
    plt.figure(figsize=(8, 5))
    plt.plot(counts, direct_times, marker="o", label="Direct (O(N^2))")
    plt.plot(counts, bh_times, marker="s", label="Barnes–Hut (O(N log N))")
    plt.xlabel("Number of particles")
    plt.ylabel("Average time per step (s)")
    plt.title("Performance comparison: Direct vs Barnes–Hut")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("benchmark.png")
    print("Saved benchmark plot as benchmark.png")


if __name__ == "__main__":
    counts, direct, bh = benchmark()
    plot_results(counts, direct, bh)