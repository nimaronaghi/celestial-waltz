"""Optional figure-export and scientific-data integrity regressions."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET


HAS_PLOTTING = all(importlib.util.find_spec(name) is not None for name in ("matplotlib", "PIL"))
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@unittest.skipUnless(HAS_PLOTTING, "optional Matplotlib/Pillow dependencies are not installed")
class TestScientificFigures(unittest.TestCase):
    @staticmethod
    def hashes(paths):
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    @staticmethod
    def figure():
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from celestial_waltz.plotting import figure_size

        fig = Figure(figsize=figure_size("single", height_mm=45))
        FigureCanvasAgg(fig)
        fig.subplots().plot([0, 1], [0, 1])
        return fig

    def test_log_values_preserve_small_measurements_and_mask_undefined_values(self):
        from celestial_waltz.plotting import positive_or_nan

        result = positive_or_nan([0, None, -1, float("nan"), float("inf"), 1e-20, 0.5])
        self.assertTrue(all(math.isnan(value) for value in result[:5]))
        self.assertEqual(result[5:], [1e-20, 0.5])

    def test_exports_have_fixed_physical_size_and_source_provenance(self):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from PIL import Image
        from celestial_waltz.plotting import export_figure, publication_style

        before_style = dict(mpl.rcParams)
        before_figures = plt.get_fignums()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "measurement.csv"
            source.write_text("time,error\n0,0\n1,1e-20\n", encoding="utf-8")
            original_hash = self.hashes([source])
            with publication_style():
                fig = self.figure()
                paths = export_figure(fig, root / "figure", caption="An export regression fixture.",
                                      data_sources=[source], metadata={"title": "Fixture"})
            with Image.open(paths["png"]) as raster:
                # Matplotlib's raster canvas uses integer pixel conversion;
                # a fractional physical pixel cannot be represented in PNG.
                self.assertEqual(raster.size, (int(89 / 25.4 * 600), int(45 / 25.4 * 600)))
                for dpi in raster.info["dpi"]:
                    self.assertAlmostEqual(dpi, 600, delta=0.02)
            expected_points = (89 / 25.4 * 72, 45 / 25.4 * 72)
            svg = ET.parse(paths["svg"]).getroot()
            for attribute, expected in zip(("width", "height"), expected_points):
                self.assertTrue(svg.attrib[attribute].endswith("pt"))
                self.assertAlmostEqual(float(svg.attrib[attribute][:-2]), expected, delta=1e-5)
            viewbox = [float(value) for value in svg.attrib["viewBox"].split()]
            self.assertEqual(viewbox[:2], [0, 0])
            for actual, expected in zip(viewbox[2:], expected_points):
                self.assertAlmostEqual(actual, expected, delta=1e-5)
            media_box = re.search(rb"/MediaBox\s*\[\s*([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\s*\]",
                                  paths["pdf"].read_bytes())
            self.assertIsNotNone(media_box, "PDF must declare its physical page size")
            x0, y0, x1, y1 = map(float, media_box.groups())
            self.assertAlmostEqual(x1 - x0, expected_points[0], delta=1e-5)
            self.assertAlmostEqual(y1 - y0, expected_points[1], delta=1e-5)
            manifest = json.loads(paths["figure.json"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["sources"], [{"file": source.name, "sha256": original_hash[source.name]}])
            self.assertEqual(self.hashes([source]), original_hash)
            self.assertEqual(paths["caption.txt"].read_text(encoding="utf-8").strip(),
                             "An export regression fixture.")
            fig.clear()
        self.assertEqual(dict(mpl.rcParams), before_style)
        self.assertEqual(plt.get_fignums(), before_figures)

    def test_export_failure_restores_style_and_does_not_open_figures(self):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from celestial_waltz.plotting import export_figure, publication_style

        with mpl.rc_context({"font.size": 13.25, "axes.linewidth": 1.73, "savefig.dpi": 91}):
            before_style = dict(mpl.rcParams)
            before_figures = plt.get_fignums()
            fig = self.figure()
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(OSError, "simulated export failure"):
                    with publication_style(), patch.object(fig, "savefig", side_effect=OSError("simulated export failure")):
                        export_figure(fig, Path(directory) / "failed", caption="Failure fixture.")
            self.assertEqual(dict(mpl.rcParams), before_style)
            self.assertEqual(plt.get_fignums(), before_figures)
            fig.clear()

    def test_saved_reports_rerender_without_solving_or_changing_measurements(self):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from benchmark import plot_benchmark
        from celestial_waltz.validation import render_saved_report

        before_style = dict(mpl.rcParams)
        before_figures = plt.get_fignums()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation, benchmark = root / "validation", root / "benchmark"
            validation.mkdir()
            benchmark.mkdir()
            sources = []
            for folder, destination, filenames in (
                (EXAMPLES / "validation", validation,
                 ("report.json", "convergence.csv", "leapfrog_trajectory.csv", "rk4_trajectory.csv")),
                (EXAMPLES / "force-benchmark", benchmark, ("metadata.json", "raw_timings.csv")),
            ):
                for filename in filenames:
                    target = destination / filename
                    shutil.copyfile(folder / filename, target)
                    sources.append(target)
            before_hashes = self.hashes(sources)
            with patch("celestial_waltz.validation.orbit_experiment", side_effect=AssertionError("rerender reran simulation")), \
                 patch("benchmark.run_benchmark", side_effect=AssertionError("rerender reran timings")), \
                 patch("matplotlib.use", side_effect=AssertionError("renderer changed the global backend")):
                render_saved_report(validation)
                plot_benchmark(benchmark, benchmark)
            self.assertEqual(self.hashes(sources), before_hashes)
            for destination, stem in ((validation, "validation"), (benchmark, "benchmark")):
                for suffix in ("pdf", "svg", "png", "caption.txt", "figure.json"):
                    self.assertTrue((destination / f"{stem}.{suffix}").is_file())
        self.assertEqual(dict(mpl.rcParams), before_style)
        self.assertEqual(plt.get_fignums(), before_figures)

    def test_benchmark_uses_median_iqr_and_preserves_undefined_accuracy(self):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from benchmark import plot_benchmark

        metadata = json.loads((EXAMPLES / "force-benchmark" / "metadata.json").read_text(encoding="utf-8"))
        metadata.update(particle_counts=[8], repeats=3)
        samples = [
            {"n_particles": 8, "mode": mode, "repeat": index, "seconds": seconds,
             "relative_rms_force_error": 0.0 if mode == "direct" else None}
            for mode, times in (("direct", [1, 2, 100]), ("bh", [2, 4, 200]))
            for index, seconds in enumerate(times, start=1)
        ]
        result = {"metadata": metadata, "samples": samples, "summary": []}
        before_style, before_figures = dict(mpl.rcParams), plt.get_fignums()
        captured = {}

        def inspect_figure(fig, stem, **kwargs):
            runtime, accuracy = fig.axes
            captured["runtime_medians"] = [container.lines[0].get_ydata().tolist()
                                           for container in runtime.containers]
            captured["errors"] = [line.get_ydata().tolist() for line in accuracy.lines]
            captured["accuracy_scale"] = accuracy.get_yscale()
            captured.update(kwargs)
            return {}

        with tempfile.TemporaryDirectory() as directory:
            with patch("celestial_waltz.plotting.export_figure", side_effect=inspect_figure):
                plot_benchmark(result, directory)
            self.assertEqual(captured["runtime_medians"], [[2000], [4000]])
            self.assertEqual(captured["errors"][0], [0.0])
            self.assertTrue(math.isnan(captured["errors"][1][0]))
            self.assertEqual(captured["accuracy_scale"], "linear")
            rows = captured["metadata"]["plotted_values"]
            self.assertEqual([(r["q25_milliseconds"], r["q75_milliseconds"]) for r in rows],
                             [(1500, 51000), (3000, 102000)])
            self.assertIsNone(rows[1]["median_error_percent"])
            self.assertIn("3 undefined", captured["caption"])
            self.assertIn("25th to 75th", captured["caption"])
            with patch("celestial_waltz.plotting.export_figure", side_effect=OSError("simulated export failure")):
                with self.assertRaisesRegex(OSError, "simulated export failure"):
                    plot_benchmark(result, directory)
        self.assertEqual(dict(mpl.rcParams), before_style)
        self.assertEqual(plt.get_fignums(), before_figures)


if __name__ == "__main__":
    unittest.main()
