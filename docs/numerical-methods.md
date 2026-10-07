# Numerical methods and verification

Celestial Waltz implements Newtonian particle dynamics in Python. The examples
below assess two distinct properties: integration error against an analytic
circular orbit, and the cost and force error of a Barnes–Hut approximation.
They verify specific numerical behavior; they do not establish a new physical
result or validate a model of an observed galaxy.

## Equations and numerical implementation

All quantities use consistent dimensionless units with $G=1$. For
$\mathbf r_{ij}=\mathbf x_j-\mathbf x_i$, the acceleration is

$$
\mathbf a_i = G\sum_{j\ne i}
\frac{m_j\mathbf r_{ij}}{\left(|\mathbf r_{ij}|^2+\varepsilon^2\right)^{3/2}}.
$$

The energy diagnostic uses the matching pair potential,

$$
U=-G\sum_{i<j}\frac{m_i m_j}
{\sqrt{|\mathbf r_{ij}|^2+\varepsilon^2}},\qquad
E=U+\frac12\sum_i m_i|\mathbf v_i|^2.
$$

Softening changes the force law. The circular-orbit experiment therefore uses
$\varepsilon=0$ when comparing with an unsoftened Kepler solution. Distinct
particles at exactly the same position require positive softening; an
unsoftened coincidence raises an error.

Direct summation evaluates each pair once and updates both accelerations.
The Barnes–Hut octree groups distant sources at their mass-weighted center of
mass. A node is accepted when its full width divided by the unsoftened distance
to that center is less than $\theta$. A node containing the target is always
opened; $\theta=0$ opens every internal node. Leaf buckets evaluate their source
particles directly and exclude the target by identity. Bounds follow the
current particle distribution, and tree geometry is evaluated relative to the
root midpoint to reduce sensitivity to large world-coordinate offsets. Original
pair separations and anchored mass-center differences are retained for forces,
so tree recentering cannot erase a resolved close pair near a distant outlier.
Approximate tree forces need not conserve pairwise momentum exactly.

The default integrator is kick–drift–kick leapfrog. Classical RK4 and
velocity-first symplectic Euler are also available. All use the same acceleration
dispatch. If a CPU integration step encounters a numerical error, its position
and velocity changes are rolled back; simulation time and recording advance
only after a finite result is accepted. Recording occurs after that numerical
commit. The standard-library CPU core and optional plotting/tensor dependencies
remain separate.

## Analytic circular-binary experiment

Two particles have masses $m_1=m_2=1/2$, separation $d_0=1$, and opposite
tangential speeds $1/2$. The total mass is $M=1$, angular frequency is
$\omega=\sqrt{GM/d_0^3}=1$, and orbital period is $P=2\pi$. The first particle's
analytic trajectory is

$$
\mathbf x_1(t)=\left[-\frac12\cos t,-\frac12\sin t,0\right].
$$

Both integrators evolve this fixture with direct summation for three periods,
using 64, 128, and 256 steps per period. The experiment records the first
particle's position error, total energy, linear momentum, and angular momentum
after each completed step. Figure axes use $t/P$, the normalized timestep
$h=\Delta t/P$, positions divided by $d_0$, and the trajectory error
$|\mathbf x_1-\mathbf x_{1,\mathrm{analytic}}|/d_0$. Here $d_0=1$, so
normalization leaves the stored position-error values unchanged.

Observed convergence order is calculated from final-time errors at matched
physical times:

$$
p_{\mathrm{obs}}=\log_2\left[\frac{e(h)}{e(h/2)}\right].
$$

The following acceptance criteria are recorded in
[the saved report](../examples/validation/report.json). Both adjacent timestep
pairs must pass the order check; conserved quantities are checked at 256 steps
per period. These are regression thresholds for this fixture and duration.

| Quantity | Acceptance criterion |
|---|---:|
| Leapfrog observed order | At least 1.8 |
| RK4 observed order | At least 3.6 |
| Maximum $\lvert E(t)-E(0)\rvert/\lvert E(0)\rvert$ | Below $10^{-4}$ |
| Maximum $\lVert\mathbf P(t)\rVert$ | Below $10^{-12}$ in code units |
| Maximum $\lVert\mathbf L(t)-\mathbf L(0)\rVert$ | Below $10^{-6}$ in code units |

Initial momentum is zero, so its norm is also its absolute drift. Momentum and
angular momentum are retained in the report even though the principal figure
focuses on trajectory and energy errors.

### Recorded results

All eight saved checks pass. At 256 steps per period:

| Integrator | Final trajectory error / $d_0$ | Maximum relative energy error | Observed orders, 64→128 and 128→256 |
|---|---:|---:|---:|
| Leapfrog | $1.8920\times10^{-3}$ | $9.0637\times10^{-8}$ | 1.9959, 1.9990 |
| RK4 | $1.1133\times10^{-7}$ | $4.6639\times10^{-9}$ | 4.5424, 4.3738 |

These values are rounded from [convergence.csv](../examples/validation/convergence.csv).
Leapfrog's measured trajectory errors follow the expected second-order trend.
RK4 gives smaller errors for this smooth orbit and these timesteps, at the cost
of more force evaluations per step. Its finite-resolution slopes above four
do not establish a higher-order method. The plotted order guides use
$g_p(h)=0.4\,e(h_{\min})(h/h_{\min})^p$, with $p=2$ for leapfrog and $p=4$
for RK4. Each guide is offset below that method's finest-resolution error for
visual separation; these are not fitted models or uncertainty estimates.
This experiment does not compare the integrators at equal computational cost.

![Circular-orbit verification](../examples/validation/validation.png)

The orbit panel alone cannot resolve small phase errors. The accompanying
trajectory-error panel exposes those deviations, while convergence and energy
panels assess different numerical properties. Exactly zero errors cannot be
shown on logarithmic axes: those observations are omitted rather than replaced
with an artificial positive floor, and remain in the saved data.

## Force accuracy and timing experiment

The benchmark draws equal-mass particles uniformly from $[-1,1]^3$ with seed
42 and zero velocities. Total mass remains one as $N$ changes through
32, 64, 128, and 256. Both solvers receive the same unchanged fixture at each
$N$, with $\theta=0.5$ and $\varepsilon=0.05$. Reusing the seed gives shared
position prefixes across particle counts; these are not independent samples
of a physical population.

Each solver receives one untimed warmup followed by five timed acceleration
evaluations. Solver order alternates between repeats. Timings include
Barnes–Hut construction and acceleration evaluation; initialization,
integration, error calculation, plotting, and file I/O are excluded.
The figure shows median wall-clock time and the interquartile range of repeated
timings. That range describes execution variability, not a confidence interval
or a scientific uncertainty on the gravitational model.

For accelerations $\mathbf a_i^{\rm BH}$ and $\mathbf a_i^{\rm direct}$, the
reported force-error ratio is

$$
\eta=\sqrt{\frac{\sum_i|\mathbf a_i^{\rm BH}-\mathbf a_i^{\rm direct}|^2}
{\sum_i|\mathbf a_i^{\rm direct}|^2}}.
$$

The figure displays $100\eta$ as a percentage; the raw CSV and JSON store
$\eta$ as a fraction. Global normalization avoids division by individual
near-zero accelerations. A zero reference and zero approximation give zero
error; a nonzero approximation with a zero reference gives an undefined ratio.
These are force comparisons on fixed positions, not trajectory-error estimates.

### Recorded results

The saved run used CPython 3.12.14 on Windows 11. Full processor identification,
clock details, fixture hashes, package version, and solver-source hash are in
[metadata.json](../examples/force-benchmark/metadata.json); all repeated timings
are in [raw_timings.csv](../examples/force-benchmark/raw_timings.csv).

| $N$ | Direct median (ms) | Barnes–Hut median (ms) | Barnes–Hut force error (%) |
|---:|---:|---:|---:|
| 32 | 0.323 | 2.333 | 0.773 |
| 64 | 1.183 | 7.010 | 0.549 |
| 128 | 4.652 | 20.574 | 0.929 |
| 256 | 18.592 | 57.966 | 0.773 |

Direct summation is faster at all four tested sizes on this machine. The tree's
normalized force error is below 1% for these particular fixtures. Neither
observation establishes an asymptotic scaling law, a crossover particle count,
or an accuracy bound for other distributions. A broader performance study
would vary $N$, clustering, $\theta$, and hardware while comparing cost at a
specified acceptable force error. No GPU speedup follows from this CPU study.

![CPU force accuracy and runtime](../examples/force-benchmark/benchmark.png)

## Reproduction and scope

To redraw the figures from the saved experiment files without changing the
measurements, install the plotting extra and run:

```bash
celestial-waltz figures --validation examples/validation --benchmark examples/force-benchmark
```

To collect new numerical results, use fresh output directories:

```bash
celestial-waltz validate --output results/two-body --plot
celestial-waltz benchmark --counts 32 64 128 256 --repeats 5 --seed 42 --theta 0.5 --output results/force-benchmark --plot
```

The [figure guide](figure-guide.md) describes physical sizing, export formats,
caption conventions, and provenance. Input data, plotting changes, and new
measurements should remain distinguishable when comparing revisions.

The circular binary tests a smooth two-body problem. It does not verify
arbitrary many-body trajectories, close encounters, long-duration dynamics, or
equilibrium galaxy models. The Plummer initializer now samples the isotropic
continuum distribution function, with separate radial and velocity-moment tests;
finite-particle and softened equilibrium require further assessment. The spiral,
legacy `kuzmin`, and collision initializers remain illustrative. The
[physics verification report](physics-verification.md) documents these distinctions,
eccentric-orbit checks, and fixes to initial conditions. Real-galaxy inference and
claims of research novelty are outside the evidence presented here.

## Related methods and reporting guidance

[REBOUND's gravity solvers](https://rebound.hanno-rein.de/gravity/) and
[leapfrog documentation](https://rebound.hanno-rein.de/integrators/leapfrog/)
provide established comparison targets for future independent verification.
The present results use analytic two-body motion and direct summation as their
references. The [JOSS review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html)
provide a useful checklist for installation, documentation, examples, and
objective verification; they do not imply a publication or review endorsement.
