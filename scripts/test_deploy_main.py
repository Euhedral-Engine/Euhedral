"""Safety tests for the local production deployment script."""

import importlib.util
import csv
import json
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
        self.driver = self.root / "libcuda.so.1"
        self.driver.touch()
        self.ptx = self.root / "libnvidia-ptxjitcompiler.so.1"
        self.ptx.touch()
        self.env_file = self.root / ".env"
        self.env_file.touch()
        self.settings = MODULE.Settings(self.env_file, self.model, "qwen3.8-27b-nvfp4-compressed", 18080,
                                        self.driver, self.ptx)
        self.events = []

    def command(self, *args):
        self.events.append(args)
        if args[:3] == ("git", "branch", "--show-current"):
            return "main"
        if args[:3] == ("git", "status", "--porcelain"):
            return ""
        if args[:3] == ("docker", "info", "--format"):
            return '{"nvidia": {}, "runc": {}}'
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
            return MODULE.deploy(self.root, self.settings)

    def test_cleanup_only_after_new_service_is_ready(self):
        def ready(port):
            self.events.append(("ready", port))
            return "qwen3.8-27b-nvfp4-compressed"

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
        self.assertIn(str(self.env_file), run[run.index("--env-file") + 1])
        self.assertIn("/dev/nvidia0", run)
        self.assertNotIn("--gpus", run)

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

        with patch.object(MODULE, "build_context", return_value=nullcontext(self.root)), patch.object(MODULE, "command", side_effect=checked_command), patch.object(MODULE, "verify_ready", return_value="qwen3.8-27b-nvfp4-compressed"):
            with self.assertRaisesRegex(RuntimeError, "became unhealthy before cleanup"):
                MODULE.deploy(self.root, self.settings)
        self.assertTrue(any(e[:2] == ("docker", "start") for e in self.events))
        self.assertFalse(any(e[:3] == ("docker", "image", "rm") for e in self.events))

    def test_a_pull_that_changes_the_script_runs_the_new_version(self):
        class Replaced(Exception):
            pass

        with patch.object(MODULE, "SCRIPT_SOURCE", b"an older script"), patch.object(MODULE.os, "execv", side_effect=Replaced) as execv:
            with self.assertRaises(Replaced):
                self.deploy(lambda port: "qwen3.8-27b-nvfp4-compressed")
        self.assertEqual(str(SCRIPT.resolve()), execv.call_args.args[1][1])
        pull = next(i for i, e in enumerate(self.events) if e[:2] == ("git", "pull"))
        self.assertFalse(any(e[:2] in (("docker", "build"), ("docker", "stop")) for e in self.events[pull:]))

    def test_reasoning_text_counts_as_generated(self):
        def response(url, data=None):
            if data is not None:
                self.assertNotIn("reasoning_effort", json.loads(data))
                return {"choices": [{"message": {"content": None, "reasoning_content": "The user"}}],
                        "usage": {"completion_tokens": 16}}
            if url.endswith("/health"):
                return {"status": "up", "engine": "ready"}
            return {"data": [{"id": "qwen"}]}

        with patch.object(MODULE, "command", return_value="healthy"), patch.object(MODULE, "get_json", side_effect=response):
            self.assertEqual("qwen", MODULE.verify_ready(18080))

    def test_without_driver_libraries_the_container_toolkit_provides_the_gpu(self):
        toolkit = MODULE.Settings(self.env_file, self.model, "qwen", 18080)
        self.assertEqual(["--gpus", "all"], MODULE.gpu_arguments(toolkit))

    def test_yaml_deployment_mounts_configuration_and_model_and_sets_health_port(self):
        config = self.root / "config, with spaces.yaml"
        config.write_text(f"model:\n  artifact: {self.model}\n  name: qwen\nserver:\n  port: 18081\n")
        self.settings = MODULE.load_settings(config)
        self.deploy(lambda port: "qwen")
        run = next(event for event in self.events if event[:2] == ("docker", "run"))
        self.assertNotIn("--env-file", run)
        self.assertIn("--gpus", run)
        self.assertIn("EUHEDRAL_SERVER_PORT=18081", run)
        self.assertIn("EUHEDRAL_MODEL_ARTIFACT=/models/model.edrl", run)
        self.assertIn("127.0.0.1:18081:18081", run)
        self.assertEqual(("--config", "/etc/euhedral/custom.yaml"), run[-2:])
        mounts = [next(csv.reader([run[i + 1]])) for i, arg in enumerate(run) if arg == "--mount"]
        self.assertIn(["type=bind", f"src={config}", "dst=/etc/euhedral/custom.yaml", "readonly"], mounts)
        self.assertIn(["type=bind", f"src={self.model}", "dst=/models/model.edrl", "readonly"], mounts)

    def test_missing_gpu_support_fails_before_build_or_cutover(self):
        self.settings = MODULE.Settings(self.env_file, self.model, "qwen", 18080)
        with patch.object(MODULE, "select_gpu_access", side_effect=RuntimeError("missing driver libraries")):
            with self.assertRaisesRegex(RuntimeError, "missing driver libraries"):
                self.deploy(lambda port: "qwen")
        self.assertFalse(any(event[:2] in (("docker", "build"), ("docker", "stop"), ("git", "pull"))
                             for event in self.events))

    def test_build_context_contains_only_committed_source(self):
        with MODULE.build_context(SCRIPT.parents[1]) as context:
            self.assertTrue((context / "Dockerfile").is_file())
            self.assertFalse((context / "benchmark-results").exists())

    def test_zero_token_chat_response_does_not_qualify(self):
        def response(url, data=None):
            if url.endswith("/health"):
                return {"status": "up", "engine": "ready"}
            if url.endswith("/v1/models"):
                return {"data": [{"id": "qwen3.8-27b-nvfp4-compressed"}]}
            return {"choices": [{"message": {"content": ""}}], "usage": {"completion_tokens": 0}}

        with patch.object(MODULE, "command", return_value="healthy"), patch.object(MODULE, "get_json", side_effect=response), patch.object(MODULE.time, "monotonic", side_effect=[0, 1, 301]), patch.object(MODULE.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "did not become ready"):
                MODULE.verify_ready(18080)


class GpuAccessTests(unittest.TestCase):
    def setUp(self):
        self.settings = MODULE.Settings(None, Path("/model.edrl"), "qwen", 1738)

    def test_explicit_driver_paths_are_preserved_without_probing_docker(self):
        settings = MODULE.replace(self.settings, driver=Path("/driver"), ptx_jit=Path("/ptx"))
        with patch.object(MODULE, "command") as command:
            self.assertIs(settings, MODULE.select_gpu_access(settings))
        command.assert_not_called()

    def test_nvidia_runtime_uses_docker_gpu_integration(self):
        with patch.object(MODULE, "command", return_value='{"nvidia": {}, "runc": {}}'), \
                patch.object(MODULE, "driver_libraries") as libraries:
            self.assertIs(self.settings, MODULE.select_gpu_access(self.settings))
        libraries.assert_not_called()

    def test_nvidia_cdi_devices_use_docker_gpu_integration(self):
        with patch.object(MODULE, "command", return_value='{"runc": {}}'), \
                patch.object(MODULE, "nvidia_cdi_available", return_value=True), \
                patch.object(MODULE, "driver_libraries") as libraries:
            self.assertIs(self.settings, MODULE.select_gpu_access(self.settings))
        libraries.assert_not_called()

    def test_missing_integration_automatically_mounts_driver_and_devices(self):
        with patch.object(MODULE, "command", return_value='{"runc": {}}'), \
                patch.object(MODULE, "nvidia_cdi_available", return_value=False), \
                patch.object(MODULE, "driver_libraries", return_value=(Path("/driver"), Path("/ptx"))):
            settings = MODULE.select_gpu_access(self.settings)
        arguments = MODULE.gpu_arguments(settings)
        self.assertNotIn("--gpus", arguments)
        self.assertIn("/dev/nvidia0", arguments)
        self.assertIn("/dev/nvidia-uvm", arguments)
        self.assertIn("LD_LIBRARY_PATH=/opt/euhedral/driver", arguments)
        self.assertEqual((Path("/driver"), Path("/ptx")), (settings.driver, settings.ptx_jit))

    def test_driver_libraries_come_from_the_host_linker_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            driver, ptx = Path(directory) / "libcuda.so.1", Path(directory) / "libnvidia-ptxjitcompiler.so.1"
            driver.touch()
            ptx.touch()
            cache = f"libcuda.so.1 (libc6,x86-64) => {driver}\nlibnvidia-ptxjitcompiler.so.1 (libc6,x86-64) => {ptx}\n"
            with patch.object(MODULE.shutil, "which", return_value="/usr/sbin/ldconfig"), \
                    patch.object(MODULE, "command", return_value=cache):
                self.assertEqual((driver, ptx), MODULE.driver_libraries())
            with patch.object(MODULE.shutil, "which", return_value="/usr/sbin/ldconfig"), \
                    patch.object(MODULE, "command", return_value=""):
                with self.assertRaisesRegex(RuntimeError, "host driver libraries are missing"):
                    MODULE.driver_libraries()

    def test_cdi_requires_a_specification_with_nvidia_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = Path(directory) / "nvidia.yaml"
            for contents, expected in (
                ("kind: nvidia.com/gpu\ndevices: [{name: all}]\n", True),
                ("kind: other.com/gpu\ndevices: [{name: all}]\n", False),
                ("kind: nvidia.com/gpu\ndevices: []\n", False),
                ("invalid: [", False),
            ):
                spec.write_text(contents)
                with patch.object(MODULE.Path, "glob", return_value=[spec]):
                    self.assertEqual(expected, MODULE.nvidia_cdi_available())


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / ".env"

    def load(self, text):
        self.path.write_text(text)
        return MODULE.load_settings(self.path)

    def test_the_example_lists_every_required_setting(self):
        example = MODULE.read_env(SCRIPT.parents[1] / ".env.example")
        self.assertLessEqual(set(MODULE.REQUIRED), example.keys())

    def test_reads_docker_env_file_syntax(self):
        settings = self.load("# comment\n\nEUHEDRAL_ARTIFACT_FILE=/m/a b.edrl\n"
                             "EUHEDRAL_MODEL_NAME=qwen\nEUHEDRAL_HOST_CPUS=2-5\nPORT=18080\n")
        self.assertEqual(Path("/m/a b.edrl"), settings.artifact)
        self.assertEqual(("qwen", 18080, None), (settings.model_id, settings.port, settings.driver))

    def test_port_defaults_to_the_servers(self):
        settings = self.load("EUHEDRAL_ARTIFACT_FILE=/a\nEUHEDRAL_MODEL_NAME=q\n"
                             "EUHEDRAL_HOST_CPUS=0\n")
        self.assertEqual(1738, settings.port)

    def test_refuses_a_missing_file_missing_keys_and_unpaired_driver_paths(self):
        with self.assertRaisesRegex(RuntimeError, "copy .env.example"):
            MODULE.load_settings(self.path)
        with self.assertRaisesRegex(RuntimeError, "does not set EUHEDRAL_MODEL_NAME"):
            self.load("EUHEDRAL_ARTIFACT_FILE=/a\n")
        with self.assertRaisesRegex(RuntimeError, "EUHEDRAL_ARTIFACT_FILE"):
            self.load("EUHEDRAL_MODEL_NAME=q\n")
        with self.assertRaisesRegex(RuntimeError, "both EUHEDRAL_CUDA_DRIVER_FILE and EUHEDRAL_PTX_JIT_FILE"):
            self.load("EUHEDRAL_ARTIFACT_FILE=/a\nEUHEDRAL_MODEL_NAME=q\n"
                      "EUHEDRAL_HOST_CPUS=0\nEUHEDRAL_CUDA_DRIVER_FILE=/d\n")
        with self.assertRaisesRegex(RuntimeError, "expected KEY=VALUE"):
            self.load("just words\n")

    def test_env_accepts_the_model_artifact_variable_and_optional_cpus(self):
        settings = self.load("EUHEDRAL_MODEL_ARTIFACT=artifacts/model.edrl\nEUHEDRAL_MODEL_NAME=q\n")
        self.assertEqual(self.path.parent / "artifacts/model.edrl", settings.artifact)
        self.assertIsNone(settings.config_file)

    def test_server_port_variable_overrides_port_alias(self):
        settings = self.load("EUHEDRAL_ARTIFACT_FILE=/a\nEUHEDRAL_MODEL_NAME=q\n"
                             "PORT=1738\nEUHEDRAL_SERVER_PORT=18080\n")
        self.assertEqual(18080, settings.port)

    def test_reads_yaml_host_paths_model_and_port(self):
        self.path = self.path.with_name("settings.yaml")
        settings = self.load('''
model:
  artifact: "models/a b.edrl" # host path relative to this file
  name: "qwen:custom"
server:
  port: 18081
host:
  cpus: auto
''')
        self.assertEqual(self.path.parent / "models/a b.edrl", settings.artifact)
        self.assertEqual(("qwen:custom", 18081), (settings.model_id, settings.port))
        self.assertIsNone(settings.env_file)
        self.assertEqual(self.path, settings.config_file)

    def test_yaml_defaults_port_and_handles_flow_mappings(self):
        self.path = self.path.with_name("settings.yml")
        settings = self.load('model: {artifact: /a.edrl, name: qwen}\n')
        self.assertEqual(1738, settings.port)

    def test_rejects_bad_yaml_and_missing_model_keys(self):
        self.path = self.path.with_name("settings.yaml")
        for text, message in (
            ("model: [", "invalid YAML"),
            ("- model", "expected a YAML mapping"),
            ("model: scalar", "model must be a mapping"),
            ("model: {artifact: /a}", "model.name"),
            ("model: {name: q}", "model.artifact"),
            ("model: {artifact: 123, name: q}", "model.artifact"),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(RuntimeError, message):
                self.load(text)

    def test_rejects_invalid_ports_in_both_file_formats(self):
        for port in ("0", "65536", "bad", "1.5", "true"):
            with self.subTest(format="env", port=port), self.assertRaisesRegex(RuntimeError, "server port"):
                self.load(f"EUHEDRAL_ARTIFACT_FILE=/a\nEUHEDRAL_MODEL_NAME=q\nPORT={port}\n")
            yaml_path = self.path.with_suffix(".yaml")
            yaml_path.write_text(f"model: {{artifact: /a, name: q}}\nserver: {{port: {port}}}\n")
            with self.subTest(format="yaml", port=port), self.assertRaisesRegex(RuntimeError, "server port"):
                MODULE.load_settings(yaml_path)

    def test_cli_accepts_only_a_file_argument(self):
        self.path.write_text("EUHEDRAL_ARTIFACT_FILE=/a\nEUHEDRAL_MODEL_NAME=q\n")
        with patch.object(MODULE, "deploy") as deploy:
            self.assertEqual(0, MODULE.main([str(self.path)]))
        self.assertEqual(self.path, deploy.call_args.args[1].env_file)
        with patch.object(MODULE, "deploy") as deploy, patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                MODULE.main(["--env-file", str(self.path)])
        deploy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
