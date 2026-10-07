# Scientific figure guide

Figures in this repository are generated from saved numerical data. Their
purpose is to make the evidence, numerical assumptions, and limitations easy
to inspect. The design is a consistent scientific house style, not a claim of
compliance with every journal or of research novelty.

## Redraw existing results

Install the optional plotting dependencies once, then render the saved examples:

```bash
python -m pip install -e '.[plot]'
celestial-waltz figures --validation examples/validation --benchmark examples/force-benchmark
```

This command reads the existing reports and CSV files and redraws their
figures. It does not rerun integration or timing measurements. To put the
exports in a separate directory:

```bash
celestial-waltz figures --validation examples/validation --benchmark examples/force-benchmark --output results/figures
```

That output contains `validation/` and `benchmark/` subdirectories. The base
filename is `validation` or `benchmark`, respectively:

| Export | Use |
|---|---|
| `.pdf` | Vector figure for a manuscript or report |
| `.svg` | Vector figure with editable text; downstream tools need STIXGeneral for matching typography |
| `.png` | 600 dpi raster preview and document embedding |
| `.caption.txt` | Figure caption describing the plotted experiment |
| `.figure.json` | Input hashes, renderer identity, environment, and physical export dimensions |

Keep captions and provenance with the image files. Raw CSV/JSON measurements
remain the numerical record. A new benchmark run produces new timings;
changing typography or layout should not silently replace those measurements.
The [methods document](numerical-methods.md) gives commands for new experiments
and explains how to interpret their results.

## Physical layout and typography

The shared `publication_style`, `figure_size`, and `export_figure` functions
control sizing and output. This iteration has no CLI width or dpi switches.
New figure types should use those shared functions rather than redefine fonts,
colors, or export behavior independently.

| Property | House default |
|---|---|
| Double-column width | 183 mm |
| Single-column width for future figures | 89 mm |
| Four-panel validation figure | 183 × 132 mm |
| Two-panel force benchmark | 183 × 85 mm |
| Typeface | STIXGeneral serif with matching mathematical notation |
| Base, tick, and legend text | 8 pt at the intended physical size; convergence legend 7 pt |
| Axis labels | 9 pt |
| Panel labels | 9 pt bold |
| Raster export | 600 dpi |
| Vector exports | PDF and SVG |

These are physical sizes, not instructions to stretch an image after export.
Inspect the PDF at its intended placement size. A 600 dpi PNG preserves raster
detail but cannot compensate for unreadable labels or a poor layout. Prefer
the vector PDF when placing line plots in a manuscript. The PDF embeds fonts;
the SVG preserves editable text and depends on STIXGeneral being installed in
the downstream viewer or editor. Font substitution can change SVG spacing,
so inspect the file after opening it on another system.

Panel labels establish reading order. Series use redundant visual distinctions
such as color, marker shape, and line pattern so that color is not the sole
identifier. Axis labels define physical or normalized quantities; legends
identify methods, while captions describe the experiment. Keep long parameter
lists and interpretation outside the plotting area.

## Reading the validation figure

The four panels answer complementary questions:

| Panel | Quantity | Interpretation |
|---|---|---|
| (a) Orbit | $x/d_0$ versus $y/d_0$ at 256 steps per period | Gross geometry compared with the analytic circle |
| (b) Trajectory error | $\lVert\delta\mathbf r\rVert/d_0$ versus $t/P$ | Deviations that are hidden by overlapping orbit curves |
| (c) Convergence | Final $\lVert\delta\mathbf r\rVert/d_0$ versus $h=\Delta t/P$ | Error change as the timestep is reduced |
| (d) Energy | $\lvert E(t)-E(0)\rvert/\lvert E(0)\rvert$ versus $t/P$ | Conservation behavior over the recorded interval |

Here $d_0$ is the initial separation, $P$ is the analytic period,
$\Delta t$ is the integration timestep, and $h=\Delta t/P$ is dimensionless.
The residual is the position-error norm for the first particle, not a fit
residual or an observational error bar. The reference curves use

$$
g_p(h)=0.4\,e(h_{\min})\left(\frac{h}{h_{\min}}\right)^p,
$$

with $p=2$ for leapfrog and $p=4$ for RK4. Here $e(h_{\min})$ is the respective
method's measured error at the finest normalized timestep. The factor 0.4
places each guide below its corresponding data for visual separation. These
curves are not fitted slopes. Pairwise observed orders are calculated from the
saved errors and reported separately.

Exactly zero errors are masked on logarithmic axes; no positive floor is
inserted. This includes the initially exact state and any later exact-zero
samples. The values remain zero in the source tables. Such omissions must be
mentioned in the caption, since an artificial floor could otherwise be mistaken
for numerical precision or a measured error plateau.

**Caption for the recorded validation example.** Circular-binary verification
with direct Newtonian gravity. Two particles of mass $1/2$ have total mass
$M=1$, initial separation $d_0=1$, opposite tangential speeds $1/2$, $G=1$, and
zero softening. The analytic period is $P=2\pi$. Leapfrog and classical RK4 are
integrated for $3P$ at 64, 128, and 256 steps per period. (a) The first particle's
orbit compared with the analytic solution, using 256 steps per period.
(b) Its position-error norm divided by $d_0$ at the same resolution.
(c) Final position error at $3P$ versus $h=\Delta t/P$; the $h^2$ and $h^4$
guides are normalized to 0.4 times the corresponding finest-resolution error,
not fitted to the data. (d) Absolute relative energy error at 256 steps
per period. Exact zeros are omitted from logarithmic displays and retained in
the data. Package version 0.2.0; the saved report records Python 3.12.14 and
identifies the solver by SHA256. This smooth two-body check does not validate
arbitrary galaxy trajectories.

Use the generated caption when rendering another duration or dataset: the
example text above describes only the checked-in three-period experiment.

## Reading the force benchmark

The timing panel uses the median of repeated measurements, with the 25th–75th
percentile interval calculated from the raw samples by linear interpolation.
This is an interquartile range, not a standard error or confidence interval.
The five repeats time the same fixture; they are not five independent draws
from a particle distribution. A narrow interval establishes repeatability
within that run, not portability to another machine.

The accuracy panel displays normalized RMS acceleration error as a percentage,
$100\eta$, where

$$
\eta=\sqrt{\frac{\sum_i|\mathbf a_i^{\rm BH}-\mathbf a_i^{\rm direct}|^2}
{\sum_i|\mathbf a_i^{\rm direct}|^2}}.
$$

CSV and JSON files retain the fractional ratio. Direct summation is the force
reference, so its self-comparison has zero error. These deterministic force
errors do not acquire uncertainty bars from repeated wall-clock measurements.
A timing curve over four small particle counts does not by itself establish
an asymptotic complexity or a larger-$N$ crossover.

**Caption for the recorded benchmark example.** CPU force-evaluation cost and
accuracy for $N=32,64,128,256$ equal-mass particles, sampled uniformly in
$[-1,1]^3$ with seed 42 and zero velocities. Total mass is one, $G=1$,
softening is $\varepsilon=0.05$, and the Barnes–Hut opening parameter is
$\theta=0.5$. Both solvers receive the same unchanged positions and masses at
each $N$. (a) Median acceleration-evaluation time from five repeats after one
untimed warmup; intervals span the linearly interpolated 25th and 75th
percentiles and describe execution variability. Solver order alternates across
repeats. Timing includes Barnes–Hut construction and excludes initialization,
integration, error calculation, plotting, and file I/O. (b) Barnes–Hut RMS
acceleration error relative to direct summation, expressed as a percentage.
Measurements used CPython 3.12.14 on Windows 11; the saved metadata provides
the processor identifier, clock information, fixture hashes, and solver-source
hash. These measurements assess the recorded implementation and machine;
they do not imply a general tree speedup or an error bound for other systems.

## Provenance and editing

The saved experiment reports identify the solver version and source hash. The
figure metadata identifies the input files and rendering code used for that
export. These identities answer different questions: which implementation
produced the numbers, and which renderer turned those numbers into a figure.
A figure redraw must preserve the former rather than substitute the current
solver's identity for an older experiment.

Make plotting changes in code and rerender. If a journal requires external
layout edits, retain the generated originals and record those edits separately.
Do not change data values to improve visual agreement, suppress inconvenient
samples, or relabel runtime variability as physical uncertainty. Preserve raw
values and explain any normalization, omission, or aggregation in the caption.

## Adapting to a journal

The [AAS Graphics Guide](https://journals.aas.org/graphics-guide/) favors vector
PDF/EPS, legible type and strokes, and distinctions beyond color. Its guidance
also covers raster resolution and the placement of panel labels. These are
useful design checks, rather than a certification of this export preset.

[Nature's final-submission guide](https://www.nature.com/nature/for-authors/final-submission)
provides 89 mm and 183 mm widths and calls for editable vector line art. Its
ordinary text-size guidance differs from this repository's 8–9 pt serif style;
Nature's [research-figure specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)
also recommend standard sans-serif fonts and accessible colors. The shared
widths therefore do not make this a Nature-specific preset.

Before submission, consult the chosen journal's current instructions for figure
size, fonts, panel lettering, file format, and captions. Review exported figures
at final size, in grayscale, and with the actual manuscript layout. The
publisher's requirements take precedence over this repository's house style.
