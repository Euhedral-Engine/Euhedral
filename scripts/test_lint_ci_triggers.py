"""The trigger lint holds ci.yaml to the land rule: a pull request runs only on the label land, a push tests nothing."""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lint_ci_triggers", ROOT / "scripts/lint-ci-triggers.py")
lint_ci_triggers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lint_ci_triggers)

GOOD = """name: ci
on:
  pull_request:
    types: [labeled]
  push:
    branches: ["rust/next*"]
  workflow_dispatch:
concurrency:
  group: ci-${{ contains(github.event.changes.added_labels.*.name, 'land') && 'land' || 'other' }}
jobs:
  changes:
    if: contains(github.event.changes.added_labels.*.name, 'land')
    runs-on: x
  rust:
    needs: changes
    if: contains(github.event.changes.added_labels.*.name, 'land') && needs.changes.outputs.rust == 'true'
    runs-on: x
  land-guard:
    runs-on: x
"""


def broken(old, new):
    assert old in GOOD, old
    return lint_ci_triggers.lint(GOOD.replace(old, new))


class LintCiTriggersTest(unittest.TestCase):
    def test_the_repository_ci_passes(self):
        self.assertEqual(lint_ci_triggers.lint((ROOT / ".gitea/workflows/ci.yaml").read_text()), [])

    def test_the_model_passes(self):
        self.assertEqual(lint_ci_triggers.lint(GOOD), [])

    def test_a_pull_request_on_every_push_fails(self):
        self.assertEqual(len(broken("    types: [labeled]\n", "")), 1)
        self.assertEqual(len(broken("types: [labeled]", "types: [opened, synchronize]")), 1)

    def test_a_path_filter_fails(self):
        self.assertEqual(len(broken("    types: [labeled]\n", "    types: [labeled]\n    paths:\n      - rust/**\n")), 1)

    def test_a_push_to_main_or_any_branch_fails(self):
        self.assertEqual(len(broken('["rust/next*"]', '[main, "rust/next*"]')), 1)
        self.assertEqual(len(broken('    branches: ["rust/next*"]\n', "")), 1)

    def test_no_push_trigger_passes(self):
        self.assertEqual(lint_ci_triggers.lint(GOOD.replace('  push:\n    branches: ["rust/next*"]\n', "")), [])

    def test_a_job_not_gated_on_land_fails(self):
        self.assertEqual(len(broken("    if: contains(github.event.changes.added_labels.*.name, 'land')\n", "")), 1)
        self.assertEqual(len(broken("    if: contains(github.event.changes.added_labels.*.name, 'land') && needs", "    if: needs")), 1)

    def test_a_missing_guard_fails(self):
        self.assertEqual(len(broken("  land-guard:\n    runs-on: x\n", "")), 1)

    def test_a_concurrency_group_that_cannot_tell_a_land_run_fails(self):
        self.assertEqual(len(broken(" && 'land' || 'other' }}", " }}")), 1)


if __name__ == "__main__":
    unittest.main()
