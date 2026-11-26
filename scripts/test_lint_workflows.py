"""The workflow lint stops a request above 32G and any cpu form, and passes the repository's own workflows."""

import importlib.util
import re
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

    def test_no_cap_spellings_fail(self):
        for value in ("--mem off", "--mem unlimited", "--mem Unlimited", "--mem=0", "--spec mem=shared:0", "--spec mem=none"):
            self.assertEqual(len(problems(f'      ARBITER_ARGS: "{value}"\n')), 1, value)

    def test_a_mem_in_a_spec_or_res_above_the_ceiling_fails(self):
        for value in ("--spec mem=33G", "--res mem=34G", "--spec=mem=shared:33", "--res mem=shared:99999M", "--spec mem=shared:8,mem=64G"):
            self.assertEqual(len(problems(f'      ARBITER_ARGS: "{value}"\n')), 1, value)

    def test_any_cpu_token_in_a_spec_or_res_fails(self):
        for value in ("--spec cpu:2", "--res cpu", "--spec mem=shared:8,CPU=shared:2", "--res 'mem=shared:8 cpu_shared=2'"):
            self.assertEqual(len(problems(f'      ARBITER_ARGS: "{value}"\n')), 1, value)

    def test_abbreviated_options_fail(self):
        for value in ("--me 64G", "--m 64G", "--mem64G", "--memo 8G", "--sp mem=shared:64G", "--re mem=shared:64G", "--sys", "--cp shared:2"):
            self.assertEqual(len(problems(f'      ARBITER_ARGS: "{value}"\n')), 1, value)

    def test_other_options_pass(self):
        self.assertEqual(problems('      ARBITER_ARGS: "--mem 8G --threads 4 --est 60 --who x"\n'), [])

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

    def test_the_exact_option_names_pass(self):
        value = "--mem 8G --est 60 --spec mem=shared:8 --res mem=shared:4G --gpu shared:4 --disk --headset --threads 4 --who x --background --tool-timeout 5"
        self.assertEqual(problems(f'      ARBITER_ARGS: "{value}"\n'), [])
        self.assertEqual(problems('      ARBITER_ARGS: "--mem=8G --est=60 --threads=4"\n'), [])

    def test_other_options_and_short_options_fail(self):
        for value in ("--vram 8", "--shared", "--exclusive", "--lease-for 1", "--reserved", "-m 64G", "--mem 8G --nope"):
            self.assertEqual(len(problems(f'      ARBITER_ARGS: "{value}"\n')), 1, value)

    def test_system_in_a_spec_or_res_fails(self):
        for value in ("--spec system", "--res system", "--spec mem=shared:8,system", "--res SYSTEM"):
            self.assertEqual(len(problems(f'      ARBITER_ARGS: "{value}"\n')), 1, value)

    def test_a_mem_with_no_amount_fails(self):
        for value in ("--res mem=exclusive", "--spec mem=shared"):
            self.assertEqual(len(problems(f'      ARBITER_ARGS: "{value}"\n')), 1, value)

    def test_size_spellings_above_the_ceiling_fail(self):
        for value in ("--mem 33g", "--mem 33GB", "--mem 33GiB", "--mem=33G", "--mem 34000M", "--mem 2T", "--mem max", "--mem infinity", "--mem inf"):
            self.assertEqual(len(problems(f'      ARBITER_ARGS: "{value}"\n')), 1, value)

    def test_quoted_keys_fail_like_plain_ones(self):
        self.assertEqual(len(problems('      "ARBITER_ARGS": "--mem 64G"\n')), 1)
        self.assertEqual(len(problems("      'ARBITER_ARGS': \"--mem 64G\"\n")), 1)
        self.assertEqual(problems('      "ARBITER_ARGS": "--mem 8G"\n'), [])

    def test_a_workflow_level_key_is_read(self):
        self.assertEqual(len(problems('env:\n  ARBITER_ARGS: "--mem 64G"\n')), 1)

    def test_a_folded_plain_scalar_fails(self):
        self.assertEqual(len(problems("      ARBITER_ARGS: --mem 1G\n        --mem 64G\n")), 1)
        self.assertEqual(problems("      ARBITER_ARGS: --mem 1G\n      ARBITER_EST: 60\n"), [])

    def test_escapes_tags_anchors_and_matrix_values_fail(self):
        for text in (
            '      ARBITER_ARGS: "--mem \\x36\\x34G"\n',
            "      ARBITER_ARGS: !!str --mem 64G\n",
            "      ARBITER_ARGS: &a --mem 64G\n",
            "      ARBITER_ARGS: ${{ matrix.args }}\n",
            '      ARBITER_ARGS: "--mem $MEM"\n',
        ):
            self.assertEqual(len(problems(text)), 1, text)

    def test_gpu_run_lines_are_read(self):
        self.assertEqual(len(problems("        run: gpu-run --mem 64G -- true\n")), 1)
        self.assertEqual(len(problems("        run: gpu-run --vram 8 -- true\n")), 1)

    def test_a_line_that_expands_arbiter_args_is_still_read(self):
        self.assertEqual(len(problems("        run: system-run $ARBITER_ARGS --mem 64G -- true\n")), 1)
        self.assertEqual(problems("        run: system-run $ARBITER_ARGS --mem 8G -- true\n"), [])
        self.assertEqual(problems('        run: system-run "${ARBITER_ARGS}" --est 60 -- true\n'), [])

    def test_a_continued_shell_line_is_read_whole(self):
        self.assertEqual(len(problems("        run: |\n          system-run --est 60 \\\n            --mem 64G -- true\n")), 1)

    def test_two_system_runs_on_one_line_are_both_read(self):
        self.assertEqual(len(problems("        run: system-run --mem 1G -- a && system-run --mem 64G -- b\n")), 1)

    def test_an_abbreviated_sys_and_me_fail(self):
        self.assertEqual(len(problems('      ARBITER_ARGS: "--sys"\n')), 1)
        self.assertEqual(len(problems("        run: system-run --me 64G -- true\n")), 1)

    def test_a_path_with_no_workflows_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(lint_workflows.main([directory]), 2)


def jobs_of(text):
    """The job blocks of a workflow, by name, found by indentation (no YAML parser is installed on the runner)."""
    _, _, body = text.partition("\njobs:\n")
    parts = re.split(r"^  ([a-z][a-z-]*):\n", body, flags=re.MULTILINE)
    return {parts[i]: parts[i + 1].rstrip("\n").split("\n") for i in range(1, len(parts), 2)}


class FailFastTest(unittest.TestCase):
    """One workflow runs for a pull request, and every job of it cancels the run when it fails."""

    ACTION = "uses: ./.gitea/actions/fail-fast"
    AGGREGATES = ("device", "gate-ok")  # cancelled with the run; they only report

    def test_only_ci_runs_for_a_pull_request(self):
        for path in sorted((ROOT / ".gitea/workflows").glob("*.y*ml")):
            triggers = re.search(r"^on:\n((?:[ #].*\n|\n)*)", path.read_text(), flags=re.MULTILINE)
            has_pull_request = triggers is not None and re.search(r"^  pull_request:", triggers.group(1), flags=re.MULTILINE)
            self.assertEqual(bool(has_pull_request), path.name == "ci.yaml", path.name)

    def test_every_job_ends_by_cancelling_the_run_when_it_fails(self):
        jobs = jobs_of((ROOT / ".gitea/workflows/ci.yaml").read_text())
        reusable = [name for name, lines in jobs.items() if any(line.startswith("    uses: ") for line in lines)]
        self.assertTrue(reusable, "the Bazel jobs are reusable workflows")
        for name, lines in jobs.items():
            if name in self.AGGREGATES or name in reusable:
                continue
            last = [line for line in lines if line.strip() and not line.startswith("  #")][-1]
            self.assertTrue(last.strip().endswith(self.ACTION), f"{name} does not end with the fail-fast step")
            if not name.endswith("-failed"):
                self.assertEqual(lines[lines.index(last) - 1], "        if: failure()", f"{name}: the step runs on failure only")
        for name in reusable:
            watcher = jobs.get(f"{name}-failed")
            self.assertIsNotNone(watcher, f"{name} is a reusable workflow: it needs a {name}-failed job")
            self.assertIn(f"    needs: {name}", watcher)


if __name__ == "__main__":
    unittest.main()
