"""Journal-oriented figure styling and traceable vector/raster exports.

Matplotlib is optional and imported only when rendering. The house style is not
a claim of compliance with a particular journal's submission specifications.
"""

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import platform

from . import __version__


COLORS = {
    "direct": "#0072B2", "bh": "#D55E00",
    "leapfrog": "#0072B2", "rk4": "#D55E00", "reference": "#333333",
}
WIDTHS_MM = {"single": 89.0, "double": 183.0}


def figure_size(width="double", height_mm=132):
    """Return the requested physical canvas size in inches, without cropping."""
    width_mm = WIDTHS_MM[width] if isinstance(width, str) else float(width)
    if not all(math.isfinite(value) and value > 0 for value in (width_mm, height_mm)):
        raise ValueError("figure dimensions must be finite and positive")
    return width_mm / 25.4, height_mm / 25.4


@contextmanager
def publication_style():
    """Temporarily apply a print-scale style; do not change global backends."""
    import matplotlib as mpl

    style = {
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "text.usetex": False,
        "font.size": 8.0, "axes.labelsize": 9.0, "axes.titlesize": 9.0,
        "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0, "legend.frameon": False,
        "legend.handlelength": 2.5, "legend.borderaxespad": 0.5,
        "axes.linewidth": 0.65, "lines.linewidth": 1.15,
        "lines.markersize": 4.0, "lines.markeredgewidth": 0.75,
        "axes.grid": False, "axes.axisbelow": True,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "axes.formatter.use_mathtext": True,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.edgecolor": "white",
        "savefig.transparent": False, "savefig.bbox": None,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "celestial-waltz",
        "figure.constrained_layout.w_pad": 0.035,
        "figure.constrained_layout.h_pad": 0.035,
        "figure.constrained_layout.wspace": 0.04,
        "figure.constrained_layout.hspace": 0.07,
    }
    with mpl.rc_context(style):
        yield


def label_panel(ax, letter, title=None):
    """Use consistent panel identifiers, separate from the scientific caption."""
    label = f"({letter})" + (f"  {title}" if title else "")
    return ax.set_title(label, loc="left", fontweight="bold", pad=7)


def positive_or_nan(values):
    """Mask nonpositive/undefined values on a log axis without inventing a floor."""
    return [float(value) if value is not None and math.isfinite(value) and value > 0
            else float("nan") for value in values]


def export_figure(fig, output_stem, *, caption, data_sources=(), metadata=None):
    """Write fixed-size PDF/SVG, 600 dpi PNG, caption and provenance manifest.

    Source measurements are read only. Figure dimensions are deliberately not
    changed by bbox_inches='tight'. PDF embeds TrueType fonts; SVG keeps text
    editable and therefore requires the recorded fonts in a downstream editor.
    """
    import matplotlib

    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    source_records = []
    for item in data_sources:
        path = Path(item)
        source_records.append({
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    paths = {suffix: Path(str(stem) + "." + suffix) for suffix in
             ("pdf", "svg", "png", "caption.txt", "figure.json")}
    title = (metadata or {}).get("title", stem.name.replace("_", " "))
    # Context also controls export-time font handling when called independently.
    with publication_style():
        fig.savefig(paths["pdf"], format="pdf", bbox_inches=None,
                    metadata={"Title": title, "Subject": caption,
                              "Creator": f"Celestial Waltz {__version__}",
                              "CreationDate": None, "ModDate": None})
        fig.savefig(paths["svg"], format="svg", bbox_inches=None,
                    metadata={"Title": title, "Description": caption,
                              "Creator": f"Celestial Waltz {__version__}", "Date": None})
        fig.savefig(paths["png"], format="png", dpi=600, bbox_inches=None,
                    metadata={"Title": title, "Description": caption})
    paths["caption.txt"].write_text(caption.strip() + "\n", encoding="utf-8")
    details = {
        "schema_version": 1, "package_version": __version__,
        "python_version": platform.python_version(),
        "matplotlib_version": matplotlib.__version__,
        "style": "celestial-waltz research house style",
        "font": "STIXGeneral", "math_font": "STIX", "base_font_pt": 8,
        "axes_label_font_pt": 9, "width_mm": float(fig.get_figwidth() * 25.4),
        "height_mm": float(fig.get_figheight() * 25.4), "png_dpi": 600,
        "formats": ["pdf", "svg", "png"], "caption": caption,
        "sources": source_records,
        "style_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "figure_metadata": metadata or {},
    }
    paths["figure.json"].write_text(json.dumps(details, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return paths
