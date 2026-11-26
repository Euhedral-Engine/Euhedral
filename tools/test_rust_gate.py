from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rust_gate  # noqa: E402

# A stand-in for `system-run`. It reads the arm's command after `-- timeout <time>`, writes the rows the plan gives
# that arm to the output path of the arm's derived configuration, and exits with the planned code. The arm is named
# by the executable's name: `java-arm` or `rust-arm`.
FAKE_LAUNCHER = """#!PYTHON
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
argv = sys.argv[1:]
command = argv[argv.index("--") + 3:]
arm = Path(command[0]).name.split("-")[0]
fork = int(command[4])
with (here / "ran.txt").open("a") as log:
    log.write(f"{arm} fork {fork}\\n")
plan = json.loads((here / "plan.json").read_text())[arm]
if fork not in plan.get("skip_forks", []) and plan.get("write", True):
    config = json.loads(Path(command[2]).read_text())
    Path(config["output"]).write_text("".join(json.dumps(row) + "\\n" for row in plan.get("rows", [])))
sys.exit(plan["exit"])
"""


def row(iteration: int = 1, warmup: bool = False, prompt: str = "p1", output: str = "o1") -> dict:
    return {
        "scenario": {"name": "decode", "promptSha256": prompt},
        "iteration": iteration,
        "warmup": warmup,
        "status": "success",
        "work": {"promptTokens": 8, "generatedTokens": 4, "outputSha256": output},
        "timings": None,
        "throughput": None,
    }


class RustGateTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.launcher = self.root / "system-run"
        self.launcher.write_text(FAKE_LAUNCHER.replace("PYTHON", sys.executable))
        self.launcher.chmod(0o755)
        self.library = self.root / "libeuhedral.so"
        self.library.write_bytes(b"")
        self.out = self.root / "out"
        self.plan: dict[str, dict] = {}
        for name in ("java", "rust"):
            self.write_config(name, self.library)
        # The arms are not built here: main gets the stand-in executables without touching the build.
        for target, value in (("java_start_script", self.root / "java-arm"), ("rust_binary", self.root / "rust-arm")):
            patcher = mock.patch.object(rust_gate, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_config(self, arm: str, library: Path) -> None:
        path = self.root / f"{arm}.json"
        path.write_text(json.dumps({"artifact": "model.edrl", "cudaLibrary": str(library), "scenarios": []}))

    def set_arm(self, arm: str, exit_code: int, rows: list[dict] | None = None, **extra) -> None:
        self.plan[arm] = {"exit": exit_code, "rows": rows or [], **extra}

    def run_gate(self, forks: int = 1) -> tuple[int, str, str]:
        (self.root / "plan.json").write_text(json.dumps(self.plan))
        argv = [
            "rust_gate.py",
            "--java-config", str(self.root / "java.json"),
            "--rust-config", str(self.root / "rust.json"),
            "--forks", str(forks),
            "--launcher", str(self.launcher),
            "--no-build",
            "--out-dir", str(self.out),
        ]
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = rust_gate.main()
        return code, stdout.getvalue(), stderr.getvalue()

    def summary(self) -> str:
        return (self.out / "summary.txt").read_text()

    def arms_started(self) -> list[str]:
        ran = self.root / "ran.txt"
        return ran.read_text().splitlines() if ran.exists() else []

    def test_both_arms_exiting_2_fail_the_gate(self) -> None:
        self.set_arm("java", 2)
        self.set_arm("rust", 2)
        code, _, _ = self.run_gate()
        self.assertEqual(code, 2)

    def test_both_arms_exiting_0_without_a_result_file_is_not_a_pass(self) -> None:
        self.set_arm("java", 0, write=False)
        self.set_arm("rust", 0, write=False)
        code, _, _ = self.run_gate()
        self.assertNotEqual(code, 0)
        self.assertIn("no pair was compared", self.summary())

    def test_both_arms_exiting_0_with_an_empty_result_file_is_not_a_pass(self) -> None:
        self.set_arm("java", 0)
        self.set_arm("rust", 0)
        code, _, _ = self.run_gate()
        self.assertNotEqual(code, 0)
        self.assertIn("no pair was compared", self.summary())

    def test_a_fork_with_no_measured_pair_is_not_a_pass_even_when_another_fork_compared(self) -> None:
        self.set_arm("java", 0, rows=[row()], skip_forks=[2])
        self.set_arm("rust", 0, rows=[row()], skip_forks=[2])
        code, _, _ = self.run_gate(forks=2)
        self.assertNotEqual(code, 0)
        self.assertIn("fork 2: no pair was compared", self.summary())

    def test_warmup_rows_alone_are_not_a_measured_pair(self) -> None:
        self.set_arm("java", 0, rows=[row(warmup=True)])
        self.set_arm("rust", 0, rows=[row(warmup=True)])
        code, _, _ = self.run_gate()
        self.assertNotEqual(code, 0)
        self.assertIn("no pair was compared", self.summary())

    def test_one_arm_exiting_1_with_matching_rows_keeps_the_unchanged_result(self) -> None:
        self.set_arm("java", 0, rows=[row()])
        self.set_arm("rust", 1, rows=[row()])
        code, _, _ = self.run_gate()
        self.assertEqual(code, 0)
        self.assertIn("1 pairs", self.summary())

    def test_missing_cuda_library_refuses_before_any_arm_runs(self) -> None:
        for arm in ("java", "rust"):
            with self.subTest(config=arm):
                missing = self.root / "missing" / "libeuhedral.so"
                self.write_config(arm, missing)
                self.set_arm("java", 0, rows=[row()])
                self.set_arm("rust", 0, rows=[row()])
                code, _, stderr = self.run_gate()
                self.assertNotEqual(code, 0)
                self.assertIn(str(missing), stderr)
                self.assertEqual(self.arms_started(), [])
                self.write_config(arm, self.library)


if __name__ == "__main__":
    unittest.main()
