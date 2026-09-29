"""Safety tests for the local production deployment script."""

import importlib.util
from contextlib import nullcontext
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("deploy-main.py")
SPEC = importlib.util.spec_from_file_location("deploy_main", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model.edrl"
        self.model.touch()
        self.tokenizer = self.root / "tokenizer"
        self.tokenizer.mkdir()
        self.driver = self.root / "libcuda.so.1"
        self.driver.touch()
        self.ptx = self.root / "libnvidia-ptxjitcompiler.so.1"
        self.ptx.touch()
        self.events = []

    def command(self, *args):
        self.events.append(args)
        if args[:3] == ("git", "branch", "--show-current"):
            return "main"
        if args[:3] == ("git", "status", "--porcelain"):
            return ""
        if args[:3] == ("git", "rev-parse", "HEAD"):
            return "e39805f194a0d2904f7dcb16ef68d9db666d3123"
        if args[:3] == ("git", "rev-parse", "origin/main"):
            return "e39805f194a0d2904f7dcb16ef68d9db666d3123"
        if args[:3] == ("docker", "ps", "-a"):
            return "euhedral-inference-serve\neuhedral-inference-serve-rollback-old\neuhedral-json-tool-probe\neuhedral-orchestration-probe"
        if args[:3] == ("docker", "inspect", "--format"):
            if ".State.Health.Status" in args[3]:
                return "healthy"
            if ".Image" in args[3]:
                if args[-1] == "euhedral-orchestration-probe":
                    return "sha256:foreign"
                return "sha256:oldimage"
            return "oldcontainerfullid"
        if args[:3] == ("docker", "image", "inspect"):
            if args[-1] == "sha256:oldimage" or args[-1] != "euhedral-inference:e39805f":
                return "sha256:oldimage"
            return "sha256:newimage"
        if args[:3] == ("docker", "image", "ls"):
            return "euhedral-inference:4b3ba7c\neuhedral-inference:e39805f\neuhedral-inference:tool-calling-candidate"
        return ""

    def deploy(self, ready):
        with patch.object(MODULE, "build_context", return_value=nullcontext(self.root)), patch.object(MODULE, "command", side_effect=self.command), patch.object(
            MODULE, "verify_ready", side_effect=ready
        ):
            return MODULE.deploy(self.root, self.model, self.tokenizer, self.driver, self.ptx, 18080)

    def test_cleanup_only_after_new_service_is_ready(self):
        def ready(port):
            self.events.append(("ready", port))
            return "qwen3.8-27b-compact-q3"

        revision = self.deploy(ready)
        self.assertEqual("e39805f194a0d2904f7dcb16ef68d9db666d3123", revision)
        events = self.events
        index = lambda predicate: next(i for i, event in enumerate(events) if predicate(event))
        self.assertLess(index(lambda e: e[:2] == ("docker", "build")), index(lambda e: e[:2] == ("docker", "stop")))
        self.assertLess(index(lambda e: e[:2] == ("docker", "run")), index(lambda e: e[0] == "ready"))
        self.assertLess(index(lambda e: e[0] == "ready"), index(lambda e: e[:2] == ("docker", "rm")))
        self.assertTrue(any(e[:3] == ("docker", "image", "rm") and "euhedral-inference:4b3ba7c" in e for e in events))
        self.assertTrue(any(e[:3] == ("docker", "image", "rm") and "sha256:oldimage" in e for e in events))
        self.assertTrue(any("tool-calling-candidate" in str(e) and e[:3] == ("docker", "image", "rm") for e in events))
        self.assertTrue(any(e[:3] == ("docker", "rm", "euhedral-json-tool-probe") for e in events))
        self.assertFalse(any("euhedral-orchestration-probe" in e and e[:2] == ("docker", "rm") for e in events))
        self.assertLess(index(lambda e: e[:3] == ("docker", "inspect", "--format") and "State.Health.Status" in str(e)), index(lambda e: e[:2] == ("docker", "rm")))
        self.assertFalse(any(e[:3] == ("docker", "inspect", "--format") and "State.Health.Status" in str(e) for e in events[index(lambda e: e[:2] == ("docker", "rm")):]))
        run = next(e for e in events if e[:2] == ("docker", "run"))
        self.assertNotIn("euhedral.qwen.prefillRegions", str(run))
        self.assertIn("EUHEDRAL_INFERENCE_MODEL_ID=qwen3.8-27b-compact-q3", run)

    def test_failure_restores_old_service_without_deleting_its_image(self):
        def not_ready(port):
            self.events.append(("ready", port))
            if not any(e[:2] == ("docker", "start") for e in self.events):
                raise RuntimeError("candidate model unavailable")
            return "restored-model"

        with self.assertRaisesRegex(RuntimeError, "candidate model unavailable"):
            self.deploy(not_ready)
        events = self.events
        self.assertTrue(any(e[:3] == ("docker", "rm", "-f") for e in events))
        self.assertTrue(any(e[:2] == ("docker", "start") for e in events))
        self.assertEqual(2, sum(e[0] == "ready" for e in events))
        self.assertFalse(any(e[:3] == ("docker", "image", "rm") for e in events))
        self.assertFalse(any("rollback-old" in str(e) and e[:2] == ("docker", "rm") for e in events))

    def test_unhealthy_after_chat_but_before_cleanup_still_rolls_back(self):
        def checked_command(*args):
            if args[:3] == ("docker", "inspect", "--format") and "State.Health.Status" in args[3]:
                self.events.append(args)
                if not any(e[:2] == ("docker", "start") for e in self.events):
                    return "unhealthy"
                return "healthy"
            return self.command(*args)

        with patch.object(MODULE, "build_context", return_value=nullcontext(self.root)), patch.object(MODULE, "command", side_effect=checked_command), patch.object(MODULE, "verify_ready", return_value="qwen3.8-27b-compact-q3"):
            with self.assertRaisesRegex(RuntimeError, "became unhealthy before cleanup"):
                MODULE.deploy(self.root, self.model, self.tokenizer, self.driver, self.ptx, 18080)
        self.assertTrue(any(e[:2] == ("docker", "start") for e in self.events))
        self.assertFalse(any(e[:3] == ("docker", "image", "rm") for e in self.events))

    def test_build_context_contains_only_committed_source(self):
        with MODULE.build_context(SCRIPT.parents[1]) as context:
            self.assertTrue((context / "Dockerfile").is_file())
            self.assertFalse((context / "benchmark-results").exists())

    def test_zero_token_chat_response_does_not_qualify(self):
        def response(url, data=None):
            if url.endswith("/health"):
                return {"status": "up", "engine": "ready"}
            if url.endswith("/v1/models"):
                return {"data": [{"id": "qwen3.8-27b-compact-q3"}]}
            return {"choices": [{"message": {"content": ""}}], "usage": {"completion_tokens": 0}}

        with patch.object(MODULE, "command", return_value="healthy"), patch.object(MODULE, "get_json", side_effect=response), patch.object(MODULE.time, "monotonic", side_effect=[0, 1, 301]), patch.object(MODULE.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "did not become ready"):
                MODULE.verify_ready(18080)


if __name__ == "__main__":
    unittest.main()
