"""The workflow lint stops a request above 32G and any cpu form, and passes the repository's own workflows."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lint_workflows", ROOT / "scripts/lint-workflows.py")
lint_workflows = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lint_workflows)


def problems(text):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "w.yaml"
        path.write_text(text)
        return lint_workflows.lint(path)


class LintWorkflowsTest(unittest.TestCase):
    def test_the_repository_workflows_pass(self):
        self.assertEqual(lint_workflows.main([str(ROOT / ".gitea/workflows")]), 0)

    def test_the_ceiling_is_inclusive(self):
        for value in ("--mem 32G", "--mem=32768M", "--spec mem=shared:32", "--mem 1G --est 60"):
            self.assertEqual(problems(f'      ARBITER_ARGS: "{value}"\n'), [], value)

    def test_more_than_the_ceiling_fails(self):
        for value in ("--mem 33G", "--mem 32.5G", "--mem 1T", "--mem=40960M", "--spec mem=shared:33", "--res mem=shared:64G"):
            self.assertEqual(len(problems(f'      ARBITER_ARGS: "{value}"\n')), 1, value)

    def test_no_cap_and_whole_machine_fail(self):
        for value in ("--mem 0", "--mem none", "--system"):
            self.assertEqual(len(problems(f'      ARBITER_ARGS: "{value}"\n')), 1, value)

    def test_every_cpu_form_fails(self):
        for value in ("--cpu shared:2", "--cpu=2", "--spec cpu=shared:2", "--spec mem=shared:8,cpu=shared:2", "--res cpu=shared:2"):
            self.assertEqual(len(problems(f'      ARBITER_ARGS: "{value}"\n')), 1, value)

    def test_a_value_that_cannot_be_read_fails(self):
        self.assertEqual(len(problems("      ARBITER_ARGS: ${{ vars.ARGS }}\n")), 1)
        self.assertEqual(len(problems('      ARBITER_ARGS: "--mem 8"\n')), 1)

    def test_forms_that_hide_the_value_fail(self):
        for text in (
            '      ARBITER_ARGS: "--mem 64G"  # note\n',
            '      env: {ARBITER_ARGS: "--mem 64G", ARBITER_EST: "60"}\n',
            "      ARBITER_ARGS: >-\n        --mem 64G\n",
            "      ARBITER_ARGS: |\n        --mem 64G\n",
            '      ARBITER_ARGS: "--mem 1G" "--mem 64G"\n',
            "      ARBITER_ARGS:\n",
        ):
            self.assertGreaterEqual(len(problems(text)), 1, text)

    def test_a_comment_after_a_readable_value_is_fine(self):
        self.assertEqual(problems('      ARBITER_ARGS: "--mem 8G"  # note\n'), [])
        self.assertEqual(problems("      ARBITER_ARGS: --mem 8G # note\n"), [])

    def test_system_run_lines_are_read_too(self):
        self.assertEqual(len(problems("          system-run --mem 64G --est 60 -- true\n")), 1)
        self.assertEqual(problems("          system-run --mem 1G --est 300 -- ./x --mem 99G\n"), [])

    def test_comments_are_ignored(self):
        self.assertEqual(problems("# ARBITER_ARGS: --mem 99G\n"), [])

    def test_a_path_with_no_workflows_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(lint_workflows.main([directory]), 2)


if __name__ == "__main__":
    unittest.main()
