"""Benchmark integrity checks; elapsed-time performance is never asserted."""

import csv
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import make_particles, relative_rms_force_error, run_benchmark, write_results
from nbody import BarnesHutSimulation


class TestBenchmark(unittest.TestCase):
    def test_seeded_fixture_preserves_global_rng_and_total_mass(self):
        before = random.getstate()
        small = make_particles(8, seed=19)
        large = make_particles(16, seed=19)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(small, make_particles(8, seed=19))
        self.assertNotEqual(small, make_particles(8, seed=20))
        for particles in (small, large):
            self.assertAlmostEqual(math.fsum(p.mass for p in particles), 1.0)
        self.assertEqual([(p.x, p.y, p.z) for p in small],
                         [(p.x, p.y, p.z) for p in large[:8]])

    def test_error_is_global_rms_ratio(self):
        self.assertAlmostEqual(relative_rms_force_error([(3, 4, 0)], [(0, 4, 0)]), 0.75)
        self.assertAlmostEqual(relative_rms_force_error(
            [(2, 0, 0), (1, 0, 0)], [(1, 0, 0), (0, 0, 0)]), math.sqrt(2))
        self.assertEqual(relative_rms_force_error([(0, 0, 0)], [(0, 0, 0)]), 0)
        self.assertIsNone(relative_rms_force_error([(1, 0, 0)], [(0, 0, 0)]))
        for approximate, reference in [([], []), ([(0, 0, 0)], []),
                                       ([(float("nan"), 0, 0)], [(0, 0, 0)])]:
            with self.subTest(approximate=approximate), self.assertRaises(ValueError):
                relative_rms_force_error(approximate, reference)

    def test_each_solver_receives_same_unchanged_particles(self):
        calls, snapshots = [], {}

        class RecordingSimulation(BarnesHutSimulation):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                snapshots[self.mode] = [(p.x, p.y, p.z, p.mass) for p in self.particles]

            def accelerations(self):
                calls.append(self.mode)
                current = [(p.x, p.y, p.z, p.mass) for p in self.particles]
                if current != snapshots[self.mode]:
                    raise AssertionError("benchmark changed the fixture")
                return super().accelerations()

            def step(self):
                raise AssertionError("force-only benchmark must not integrate")

        with patch("benchmark.BarnesHutSimulation", RecordingSimulation):
            result = run_benchmark([8], repeats=2, seed=19, theta=0)
        self.assertEqual(snapshots["direct"], snapshots["bh"])
        self.assertEqual(calls, ["direct", "bh", "direct", "bh", "bh", "direct"])
        self.assertEqual(len(result["samples"]), 4)
        for row in result["samples"]:
            self.assertLess(row["relative_rms_force_error"], 1e-12)

    def test_raw_samples_and_metadata_round_trip(self):
        result = run_benchmark([1, 6], repeats=2, seed=7)
        with tempfile.TemporaryDirectory() as directory:
            csv_path, json_path = write_results(result, directory)
            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 8)
        self.assertEqual({(int(r["n_particles"]), r["mode"], int(r["repeat"])) for r in rows},
                         {(n, mode, repeat) for n in (1, 6)
                          for mode in ("direct", "bh") for repeat in (1, 2)})
        self.assertEqual(metadata["seed"], 7)
        self.assertEqual(metadata["particle_counts"], [1, 6])
        self.assertEqual(metadata["total_mass"], 1.0)
        self.assertIn("tree construction", metadata["timing_scope"])
        self.assertIn("python_version", metadata["environment"])
        self.assertIn("processor", metadata["environment"])
        self.assertIn("package_version", metadata)
        self.assertEqual(len(metadata["solver_source_sha256"]), 64)
        self.assertEqual(len(metadata["summary"]), 4)
        same_fixture = run_benchmark([1, 6], repeats=1, seed=7)
        self.assertEqual(metadata["fixtures"], same_fixture["metadata"]["fixtures"])

    def test_rejects_invalid_configuration(self):
        for kwargs in ({"particle_counts": []}, {"particle_counts": [0]},
                       {"particle_counts": [2, 2]}, {"repeats": 0},
                       {"theta": -0.1}, {"eps": 0}, {"total_mass": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_benchmark(**kwargs)

    def test_cli_and_import_work_without_optional_packages(self):
        # -S disables site packages, including installed NumPy/matplotlib.
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.run(
                [sys.executable, "-S", "-B", str(repository / "benchmark.py"),
                 "--counts", "4", "--repeats", "1", "--seed", "3",
                 "--output", directory], capture_output=True, text=True,
                cwd=repository, check=False)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertTrue((Path(directory) / "raw_timings.csv").is_file())
            self.assertTrue((Path(directory) / "metadata.json").is_file())


if __name__ == "__main__":
    unittest.main()
