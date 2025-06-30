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

## Jupyter Notebook

An example notebook `example.ipynb` is included which demonstrates how to start
the GUI from a notebook cell.
