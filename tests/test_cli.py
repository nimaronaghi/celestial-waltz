import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from celestial_waltz.cli import main
from celestial_waltz.validation import generate_report, orbit_experiment


class TestExperiments(unittest.TestCase):
    def test_seeded_cli_serializes_repeatable_states(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"run-{index}.json" for index in range(2)]
            for path in paths:
                with contextlib.redirect_stdout(io.StringIO()):
                    code = main(["simulate", "--particles", "8", "--steps", "3",
                                 "--seed", "42", "--mode", "direct", "--output", str(path)])
                self.assertEqual(code, 0)
            first, second = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            self.assertEqual(first["particles"], second["particles"])
            self.assertEqual(first["config"], second["config"])
            self.assertEqual(first["config"]["seed"], 42)
            self.assertEqual(len(first["particles"]), 8)
            self.assertIn("diagnostics", first)

    def test_invalid_simulation_does_not_write_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["simulate", "--dt", "0", "--output", str(output)])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())

    def test_validation_has_raw_data_and_passes_physical_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            report = generate_report(directory, orbits=1)
            self.assertTrue(report["passed"], report["checks"])
            self.assertEqual(len(report["experiments"]), 6)
            for name in ("report.json", "convergence.csv", "leapfrog_trajectory.csv", "rk4_trajectory.csv"):
                self.assertTrue((Path(directory) / name).is_file())
            self.assertAlmostEqual(report["observed_orders"]["leapfrog"][-1], 2.0, delta=0.1)
            self.assertAlmostEqual(report["observed_orders"]["rk4"][-1], 4.0, delta=0.2)

    def test_analytic_start_and_orbit_period(self):
        summary, rows = orbit_experiment("rk4", 256, 1)
        self.assertEqual(rows[0]["position_error"], 0.0)
        self.assertLess(summary["final_position_error"], 1e-6)
        self.assertAlmostEqual(rows[-1]["analytic_x"], -0.5)
        self.assertAlmostEqual(rows[-1]["analytic_y"], 0.0)

    def test_validation_cli_returns_failure_when_checks_fail(self):
        from unittest.mock import patch
        with patch("celestial_waltz.validation.generate_report", return_value={"passed": False}):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["validate"]), 1)


if __name__ == "__main__":
    unittest.main()
