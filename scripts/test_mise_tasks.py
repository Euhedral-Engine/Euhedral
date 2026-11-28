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
            "native-test", "rust-check", "rust-test", "rust-native", "rust-native-verify", "deploy", "test-deploy",
        }
        self.assertLessEqual(expected, tasks.keys())
        self.assertFalse(any("gradlew" in str(task.get("run")) for task in tasks.values()))
        self.assertNotIn("java", data["tools"])
        self.assertNotIn("gradle", data["tools"])
        self.assertIn("scripts/deploy-main.py", tasks["deploy"]["run"])
        self.assertNotIn("docker-run", tasks)
        self.assertIn('arg "<file>"', tasks["deploy"]["usage"])
        self.assertIn('${usage_file?}', tasks["deploy"]["run"])
        workflow = (ROOT / ".github/workflows/native-package.yaml").read_text()
        self.assertIn('      - "mise.toml"', workflow)
        self.assertNotIn('      - ".mise.toml"', workflow)
        self.assertIn("  push:\n    branches: [main]", workflow)
        for action in (
            "actions/checkout@v7.0.1",
            "jdx/mise-action@v5.0.0",
            "actions/cache@v6.1.0",
            "actions/setup-python@v7.0.0",
            "actions/upload-artifact@v7.0.1",
        ):
            self.assertIn(f"uses: {action}", workflow)
        self.assertIn("native/.zig-cache", workflow)
        self.assertIn("native/.zig-global-cache", workflow)
        self.assertIn("hashFiles('mise.toml', 'native/build.zig', 'native/native-products.json')", workflow)


if __name__ == "__main__":
    unittest.main()
