# celestial-waltz

This repository contains a simple N-body simulation of a spiral galaxy. The simulation uses the Barnes–Hut algorithm to accelerate gravitational calculations and provides a small Tkinter-based interface for experimenting with different parameters.

## Algorithm Overview

Each pair of particles interacts through Newtonian gravity. The force between two bodies with masses \( m_i \) and \( m_j \), separated by vector \( \mathbf{r}_{ij} \), is given by:

$$
\mathbf{F}_{ij} = G \frac{m_i m_j}{\lVert \mathbf{r}_{ij} \rVert^3} \, \mathbf{r}_{ij},
$$

where \( G \) is the gravitational constant. A small softening term \( \epsilon \) is added in the implementation to avoid singularities. The Barnes–Hut tree approximates distant regions by representing them as a single body located at their center of mass. A node of size \( s \), at distance \( d \) from a particle, is treated as a single body when:

$$
\frac{s}{d} < \theta.
$$

Time integration can be performed either with the simple Euler method or with the more stable leapfrog scheme:

$$
\begin{aligned}
\mathbf{v}^{n+1/2} &= \mathbf{v}^n + \tfrac{1}{2} \, \mathbf{a}^n \, \Delta t, \\
\mathbf{x}^{n+1} &= \mathbf{x}^n + \mathbf{v}^{n+1/2} \, \Delta t, \\
\mathbf{v}^{n+1} &= \mathbf{v}^{n+1/2} + \tfrac{1}{2} \, \mathbf{a}^{n+1} \, \Delta t,
\end{aligned}
$$

which conserves energy much better than Euler over long runs.

## Requirements

The core code only uses the Python standard library and therefore requires no third-party packages.

Optional features like PNG/GIF conversion use Pillow, which can be installed via:

```bash
pip install pillow
