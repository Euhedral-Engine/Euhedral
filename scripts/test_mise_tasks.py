"""Mise task names and the deployment entrypoint are part of the operator contract."""

import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MiseTasksTest(unittest.TestCase):
    def test_common_build_optimization_and_deployment_tasks_exist(self):
        data = tomllib.loads((ROOT / "mise.toml").read_text())
        tasks = data["tasks"]
        expected = {
            "format-check", "format", "test", "native-build", "native-test",
            "native-verify", "cuda-test", "benchmark-validate", "benchmark-smoke",
            "benchmark-baseline", "full-build", "deploy", "test-deploy",
        }
        self.assertLessEqual(expected, tasks.keys())
        self.assertIn("nativeVerify", tasks["full-build"]["run"])
        self.assertIn("build", tasks["full-build"]["run"])
        self.assertIn("scripts/deploy-main.py", tasks["deploy"]["run"])
        workflow = (ROOT / ".github/workflows/native-package.yaml").read_text()
        self.assertIn('      - "mise.toml"', workflow)
        self.assertNotIn('      - ".mise.toml"', workflow)
        self.assertIn("  push:\n    branches: [main]", workflow)
        for action in (
            "actions/checkout@v7.0.1",
            "jdx/mise-action@v5.0.0",
            "gradle/actions/setup-gradle@v6.4.0",
            "actions/cache@v6.1.0",
            "actions/setup-python@v7.0.0",
            "actions/upload-artifact@v7.0.1",
        ):
            self.assertIn(f"uses: {action}", workflow)
        self.assertIn("cache-provider: basic", workflow)
        self.assertIn("native/.zig-cache", workflow)
        self.assertIn("native/.zig-global-cache", workflow)
        self.assertIn("hashFiles('mise.toml', 'native/build.zig', 'native/native-products.json')", workflow)


if __name__ == "__main__":
    unittest.main()
