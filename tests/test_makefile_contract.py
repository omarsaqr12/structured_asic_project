"""Dependency and ECO output naming regression tests; does not run EDA tools."""
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MakefileContractTest(unittest.TestCase):
    def test_clean_output_dry_run_includes_all_stages(self):
        command = ["make", "-B", "-n", "all", "DESIGN=arith"]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = result.stdout
        for stage in ("placer.py", "--no-sa", "--initial-placement", "eco_generator.py", "--output-json", "rename.py", "make_def.py", "route.tcl", "sta.tcl"):
            with self.subTest(stage=stage):
                self.assertIn(stage, output)

    def test_eco_file_names_are_not_conflated(self):
        text = (ROOT / "Makefile").read_text()
        self.assertIn("$(DESIGN)_sa_placement_eco.json", text)
        self.assertIn("$(DESIGN)_eco_netlist.json", text)
        self.assertIn("--output-json $(ECO_NETLIST_JSON)", text)
        self.assertIn("$(RENAMED_VERILOG): $(ECO_VERILOG) $(ECO_MAP)", text)


if __name__ == "__main__":
    unittest.main()
