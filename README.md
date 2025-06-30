# celestial-waltz

This repository contains a simple N-body simulation of a spiral galaxy. The
simulation uses the Barnes--Hut algorithm to accelerate gravitational
calculations and provides a small Tkinter based interface for experimenting
with different parameters.

## Requirements

The code only uses the Python standard library and therefore does not require
any third-party packages.

## Running the GUI

To start the simulation run:

```bash
python3 galaxy_gui.py
```

A window will appear with sliders controlling the number of particles, the time
step, and how many iterations to run. Press **Start** to launch the simulation.

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

## Jupyter Notebook

An example notebook `example.ipynb` is included which demonstrates how to start
the GUI from a notebook cell. It also shows how to record a short simulation
sequence to disk using `SimulationRecorder`.

## GPU Simulation

For experimentation on Google Colab or any machine with a CUDA capable GPU,
`gpu_sim.py` provides a minimal PyTorch implementation. Install PyTorch in the
notebook and run:

```python
!python gpu_sim.py --particles 100000 --iterations 1000 --dt 0.01
```

The command line flags allow choosing the number of particles and iterations.
