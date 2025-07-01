# celestial-waltz

This repository contains a simple N-body simulation of a spiral galaxy. The
simulation uses the Barnes--Hut algorithm to accelerate gravitational
calculations and provides a small Tkinter based interface for experimenting
with different parameters.

## Requirements

The code only uses the Python standard library and therefore does not require
any third-party packages.

Optional features like PNG/GIF conversion use Pillow which can be installed via:

```bash
pip install pillow
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
