# celestial-waltz

This repository contains a simple N-body simulation of a spiral galaxy. The
simulation uses the Barnes-Hut algorithm to accelerate gravitational
calculations and provides a small Tkinter-based interface for experimenting
with different parameters.

## Algorithm Overview

Each pair of particles interacts through Newtonian gravity. The force between
two bodies with masses $m_i$ and $m_j$ separated by $\mathbf{r}_{ij}$ is

$$
\mathbf{F_ij} = G   \frac{m_i m_j}{\lVert \mathbf{r_ij} \rVert^3}  \mathbf{r}_{ij}
$$


where $G$ is the gravitational constant and a small softening term $\epsilon$
is added in the implementation to avoid singularities. The Barnes-Hut tree
approximates distant regions by their center of mass. A node of size $s$ at
distance $d$ from the particle is treated as a single body when
$\frac{s}{d} < \theta$.

Time integration is performed either with the simple Euler method or the more
stable leapfrog scheme

$$
\begin{aligned}
\mathbf{v}^{n+1/2} &= \mathbf{v}^n + \tfrac{1}{2}\mathbf{a}^n \Delta t,\\
\mathbf{x}^{n+1} &= \mathbf{x}^n + \mathbf{v}^{n+1/2} \Delta t,\\
\mathbf{v}^{n+1} &= \mathbf{v}^{n+1/2} + \tfrac{1}{2}\,\mathbf{a}^{n+1} \Delta t,
\end{aligned}
$$

which conserves energy much better than Euler for long runs.

## Requirements

The code only uses the Python standard library and therefore does not require
any third-party packages.

Optional features like PNG/GIF conversion use Pillow which can be installed via:

```bash
pip install pillow
```

## How to Run the Simulation

1. Ensure Python 3.8 or newer is installed. Optionally install Pillow for GIF/PNG output and PyTorch for GPU acceleration:
   ```bash
   pip install pillow torch
   ```
2. Launch the Tkinter GUI with:
   ```bash
   python3 galaxy_gui.py
   ```
3. For headless execution use the library directly:
   ```bash
   python3 -c "from nbody import BarnesHutSimulation; \
   sim = BarnesHutSimulation(num_particles=500, eps=0.02, mode='bh', integrator='leapfrog'); \
   sim.run(200)"
   ```
   Adjust parameters as needed (`mode` is `bh` or `direct`; `integrator` is `euler` or `leapfrog`).
4. GPU acceleration can be tried with:
   ```bash
   python3 gpu_sim.py --particles 10000 --iterations 100 --mode bh
   ```

## Running the GUI

To start the simulation run:

```bash
python3 galaxy_gui.py
```

A window will appear with sliders controlling the number of particles, the time
step, and how many iterations to run. Press **Start** to launch the simulation.

Command line simulations can also be run headless using `nbody.py`:

```bash
python3 -c "from nbody import BarnesHutSimulation; sim=BarnesHutSimulation(); sim.run(100)"
```

## Simulation Parameters

The main parameters controlling the simulation are listed below:

- **Number of Particles** – total bodies in the simulation (more particles
  produce a more detailed galaxy but slow down execution).
- **Time Step (dt)** – integration step in **million years**. A smaller value
  yields more accurate dynamics at the cost of speed.
- **Total Iterations** – number of simulation steps performed.
- **Galaxy Type** – currently a simple spiral galaxy generator is provided.
- **Mass Scale** and **Size Scale** – overall scaling factors for initial
  conditions.
- **Visualization Options** – trails, velocity vectors, and coloring by
  velocity.

Additional options include:

- **mode** – choose `bh` (Barnes-Hut) or `direct` pairwise forces.
- **integrator** – `euler` or `leapfrog` for more stable integration.
- **eps** – softening parameter controlling force calculation.

### Diagnostics

The command line runner prints total momentum and energy before and after the
simulation to help verify numerical stability.

## Jupyter Notebook

An example notebook `example.ipynb` is included which demonstrates how to start
the GUI from a notebook cell. It also shows how to record a short simulation
sequence to disk using `SimulationRecorder`.

## Simulation GIF

The example notebook saves a recording to `simulation.bin`. You can convert this
binary file into a GIF using:

```bash
python3 convert_to_gif.py
```

This will create `simulation.gif` in the project directory. The GIF is not
included in the repository to keep the codebase lightweight, so run the above
command locally to generate it.
`convert_to_gif.py` will also save a PNG of the final frame for higher quality
inspection.


## GPU Simulation

For experimentation on Google Colab or any machine with a CUDA capable GPU,
`gpu_sim.py` provides a minimal PyTorch implementation. Install PyTorch in the
notebook and run:

```python
!python gpu_sim.py --particles 100000 --iterations 1000 --dt 0.01
```

The command line flags allow choosing the number of particles and iterations.
