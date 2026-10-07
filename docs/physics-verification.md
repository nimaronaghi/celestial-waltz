# Physics verification and supported model

Celestial Waltz evolves positive point masses under isolated, nonrelativistic
gravity, optionally with Plummer softening. Verification covers this numerical
model. It does not establish a model of an observed galaxy, resolve physical
stellar collisions, or replace convergence studies for a new scientific problem.

## Force law, potential, and integration

The implemented acceleration and pair potential are

$$
\mathbf a_i=G\sum_{j\ne i}m_j
\frac{\mathbf x_j-\mathbf x_i}
{\left(|\mathbf x_j-\mathbf x_i|^2+\varepsilon^2\right)^{3/2}},
\qquad
U=-G\sum_{i<j}\frac{m_i m_j}
{\sqrt{|\mathbf x_j-\mathbf x_i|^2+\varepsilon^2}}.
$$

The sign is attractive, the source mass determines acceleration, and
$m_i\mathbf a_{ij}=-m_j\mathbf a_{ji}$ for exact pair evaluation. The potential
uses the same softening as the force. Positive softening changes close-range
physics; it is not just a numerical guard against division by zero. With zero
softening, coincident distinct particles are singular and rejected.

The CPU integrators are kick–drift–kick leapfrog, classical RK4, and
velocity-first symplectic Euler. The tensor backend implements symplectic Euler.
Leapfrog is second order for the smooth conservative direct-force problem;
RK4 is fourth order and is not symplectic. Small energy drift alone does not
establish accurate trajectories. Fixed timesteps must resolve the shortest
encounter timescale, which can be much shorter than an orbital period.

Barnes–Hut approximates distant groups by their mass-weighted monopole.
The opening criterion uses full cell width divided by unsoftened distance.
Nodes containing the target are opened and leaf evaluation excludes self-force.
Containment uses descendant particle identity, because rounded geometric bounds
can otherwise omit their own member at very large spatial dynamic range.
At finite opening angle, independently accepted groups need not produce
equal-and-opposite forces. Exact momentum conservation, rotational invariance,
and symplectic behavior are therefore not guaranteed for approximate tree forces.
The exact-tree limit, `theta=0`, is checked against pair summation.

A multiscale regression exposed a precision loss when a close pair near the
origin shared a tree with a distant outlier: recentering their physical
coordinates erased the pair separation. Tree partition coordinates are now
separate from force coordinates. Exact interactions retain original differences;
monopoles use an anchored center-of-mass displacement. This retains separations
already represented in the input, but cannot recover precision absent from the
input floating-point coordinates.

## Plummer phase-space sampling

The `plummer` initializer now samples both positions and velocities from the
isotropic continuum Plummer model. It replaces the previous point-mass circular
speed prescription, which did not represent a bound Plummer equilibrium.
For scale radius $a$ and total mass $M$, the model has

$$
\Phi(r)=-\frac{GM}{\sqrt{r^2+a^2}},\qquad
\frac{M(<r)}M=\frac{r^3}{(r^2+a^2)^{3/2}},\qquad
f(E)\propto(-E)^{7/2}\quad(E<0).
$$

At fixed position, the velocity-space measure supplies a factor $v^2$.
Writing $q=v/v_{\mathrm{esc}}$ with
$v_{\mathrm{esc}}^2=2GM/\sqrt{r^2+a^2}$ gives

$$
p(q)\propto q^2(1-q^2)^{7/2},\quad 0\le q<1,
\qquad q^2\sim\operatorname{Beta}(3/2,9/2).
$$

Positions follow the inverse radial cumulative distribution. Velocity directions
are isotropic and independent of position directions; their speeds follow the
beta distribution above. This implementation is independently derived from the
[isotropic Plummer distribution documented by galpy](https://docs.galpy.org/en/stable/reference/dfplummer.html).
No galpy code or runtime dependency is used.

Tests compare the radial cumulative distribution, bound-speed condition,
$\langle q^2\rangle=1/4$, $\langle q^4\rangle=5/56$, angular isotropy,
and the continuum kinetic-energy scale $3\pi GM^2/(64a)$. A seeded finite
sample is also checked for a virial ratio near unity using the actual
unsoftened pair potential. Statistical tolerances allow sampling fluctuations;
the generator does not artificially rescale velocities to force a virial ratio.

Each generated particle retains mass one, so $M=N$ and characteristic speeds
scale as $\sqrt{N/a}$. Changing particle count changes the physical model unless
the user deliberately constructs fixed-total-mass particles. Raw samples retain
their finite-sample center-of-mass position and bulk velocity. They sample
continuum equilibrium, not an exact finite-particle or softened equilibrium.
Increasing softening changes the potential and requires a separate assessment.

## Other initial conditions and units

Spiral and legacy `kuzmin` disk options remain illustrative states. The latter
samples exponential radii, not a Kuzmin surface-density law. Their velocity
prescriptions assume a point-mass speed scale even though no central point mass
is inserted. They should not be interpreted as equilibrium galactic disks.

The two-galaxy initializer removes each component's sampled center-of-mass
position and velocity before adding the specified separation and opposite
approach velocities. Tests verify the resulting centers and velocities directly.
This fixes the collision geometry; it does not make the component spirals
self-consistent galaxy models.

CPU simulations set $G=1$ in code units. If physical length and mass units are
$L_0$ and $M_0$, the corresponding time unit is

$$
T_0=\sqrt{\frac{L_0^3}{G_{\mathrm{physical}}M_0}},
\qquad V_0=L_0/T_0.
$$

Time therefore has no automatic Myr interpretation. The GUI now passes the
displayed timestep and softening directly to the solver in code units. Visual
projection excludes invisible particles instead of accumulating them at screen
edges; projection and colors do not alter the physical state.

## Reproducing the checks

```bash
python -m unittest discover -s tests -v
celestial-waltz validate --output results/physics-validation --plot
```

Core tests use analytic forces and orbits, conservation laws, coordinate
transformations, and timestep refinement. The Plummer tests require only the
standard library. Optional tensor tests also compare against analytic softened
forces, a finite-difference potential gradient, unequal-mass monopoles, and a
softened circular orbit with first-order timestep convergence. They execute on
CPU; passing these tests does not establish CUDA execution or performance.

The eccentric-orbit test solves Kepler's equation independently of the numerical
integrator, using orbital-plane relations also documented by
[NASA/JPL](https://ssd.jpl.nasa.gov/planets/approx_pos.html). It uses an exact
isolated binary, not JPL's approximate planetary elements. With masses 0.7 and
1.3, relative semimajor axis 1.4, eccentricity 0.6, zero softening, and a duration
of 1.25 periods, the local verification gave:

| Integrator | Maximum position error / semimajor axis, 1024 steps/period | Maximum relative energy error | Final-position order, 512→1024 steps/period |
|---|---:|---:|---:|
| Leapfrog | $3.3447\times10^{-3}$ | $2.7903\times10^{-4}$ | 2.0025 |
| RK4 | $1.2283\times10^{-7}$ | $1.1949\times10^{-8}$ | 4.0969 |

The force–potential test's maximum component discrepancy was approximately
$1.44\times10^{-11}$ using a fourth-order finite difference with displacement
$10^{-4}$. The tests specify tolerances and fixtures; these values are rounded
observations, not universal error bounds. The eccentric test is deliberately
harder than the circular example because its pericenter passage is faster.

The [numerical-methods report](numerical-methods.md) contains saved binary and
force-benchmark results. Those experiments complement these regression checks;
neither verifies arbitrary long-term many-body dynamics. For demanding close
encounters, independently validate the chosen timestep or compare with an
adaptive reference integrator such as [REBOUND IAS15](https://rebound.hanno-rein.de/integrators/ias15/).
REBOUND's [Plummer example](https://rebound.hanno-rein.de/c_examples/selfgravity_plummer/)
also discusses encounter resolution with a fixed timestep.
