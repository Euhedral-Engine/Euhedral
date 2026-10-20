#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["PyYAML==6.0.3"]
# ///
"""Pull, build, and deploy main; clean up only after a healthy cutover.

Pass one host YAML or Docker-format .env file. The script reads the model path, public name, and port,
mounts the model and configuration, and checks inference before removing the previous deployment.
"""

import argparse
from dataclasses import dataclass, replace
import csv
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from contextlib import contextmanager


SERVICE = "euhedral-inference-serve"
IMAGE_REPOSITORY = "euhedral-inference"
REPOSITORY = Path(__file__).resolve().parents[1]
# What this process runs; the pull below may bring a newer version of the script.
SCRIPT_SOURCE = Path(__file__).read_bytes()
REQUIRED = ("EUHEDRAL_MODEL_NAME",)
DEVICES = ("nvidia0", "nvidiactl", "nvidia-uvm", "nvidia-modeset")


@dataclass(frozen=True)
class Settings:
    env_file: Path | None
    artifact: Path
    model_id: str
    port: int
    driver: Path | None = None
    ptx_jit: Path | None = None
    config_file: Path | None = None


def read_env(path):
    """Docker's env-file format: KEY=VALUE lines, values taken literally, # comments on their own lines."""
    values = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise RuntimeError(f"{path}:{number}: expected KEY=VALUE")
        values[key] = value
    return values


def host_path(value, settings_file):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else settings_file.parent / path).resolve()


def read_yaml(path):
    import yaml

    try:
        values = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise RuntimeError(f"{path}: invalid YAML: {error}") from error
    if not isinstance(values, dict):
        raise RuntimeError(f"{path}: expected a YAML mapping of settings")
    for section in ("model", "server"):
        if section in values and not isinstance(values[section], dict):
            raise RuntimeError(f"{path}: {section} must be a mapping")
    model = values.get("model", {})
    missing = [f"model.{key}" for key in ("artifact", "name")
               if not isinstance(model.get(key), str) or not model[key].strip()]
    if missing:
        raise RuntimeError(f"{path} does not set {', '.join(missing)}")
    return model, values.get("server", {})


def load_settings(path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"No settings file at {path}; copy .env.example or config/euhedral.example.yaml and fill it in")
    is_yaml = path.suffix.lower() in (".yaml", ".yml")
    if is_yaml:
        model, server = read_yaml(path)
        artifact, model_id, port = model["artifact"], model["name"], server.get("port", 1738)
        driver = ptx_jit = None
    else:
        values = read_env(path)
        missing = [key for key in REQUIRED if not values.get(key, "").strip()]
        artifact = values.get("EUHEDRAL_ARTIFACT_FILE") or values.get("EUHEDRAL_MODEL_ARTIFACT")
        if not artifact or not artifact.strip():
            missing.insert(0, "EUHEDRAL_ARTIFACT_FILE (or EUHEDRAL_MODEL_ARTIFACT)")
        if missing:
            raise RuntimeError(f"{path} does not set {', '.join(missing)} (see .env.example)")
        model_id = values["EUHEDRAL_MODEL_NAME"]
        port = values.get("EUHEDRAL_SERVER_PORT", values.get("PORT", "1738"))
        driver, ptx_jit = values.get("EUHEDRAL_CUDA_DRIVER_FILE"), values.get("EUHEDRAL_PTX_JIT_FILE")
    if bool(driver) != bool(ptx_jit):
        raise RuntimeError(f"{path} must set both EUHEDRAL_CUDA_DRIVER_FILE and EUHEDRAL_PTX_JIT_FILE, or neither")
    try:
        port = int(str(port))
    except ValueError:
        raise RuntimeError(f"{path}: server port must be a whole number") from None
    if not 1 <= port <= 65535:
        raise RuntimeError(f"{path}: server port must be between 1 and 65535")
    return Settings(
        env_file=None if is_yaml else path,
        artifact=host_path(artifact, path),
        model_id=model_id,
        port=port,
        driver=host_path(driver, path) if driver else None,
        ptx_jit=host_path(ptx_jit, path) if ptx_jit else None,
        config_file=path if is_yaml else None,
    )


def mount_file(source, target):
    output = io.StringIO()
    csv.writer(output, lineterminator="").writerow(
        ["type=bind", f"src={source}", f"dst={target}", "readonly"]
    )
    return ["--mount", output.getvalue()]


def nvidia_cdi_available():
    import yaml

    for directory in (Path("/etc/cdi"), Path("/var/run/cdi")):
        for path in directory.glob("*"):
            if path.suffix.lower() not in (".yaml", ".yml", ".json"):
                continue
            try:
                spec = yaml.safe_load(path.read_text())
            except (OSError, yaml.YAMLError):
                continue
            if isinstance(spec, dict) and spec.get("kind") == "nvidia.com/gpu" and spec.get("devices"):
                return True
    return False


def driver_libraries():
    """Find the installed host driver libraries without relying on a distribution's library directory."""
    executable = shutil.which("ldconfig")
    if executable is None:
        raise RuntimeError("Cannot locate NVIDIA driver libraries: ldconfig is unavailable")
    libraries = {}
    for line in command(executable, "-p").splitlines():
        name, separator, location = line.strip().partition(" => ")
        if separator:
            name = name.split()[0]
            if name in ("libcuda.so.1", "libnvidia-ptxjitcompiler.so.1") and Path(location).is_file():
                libraries.setdefault(name, Path(location).resolve())
    missing = {"libcuda.so.1", "libnvidia-ptxjitcompiler.so.1"} - libraries.keys()
    if missing:
        raise RuntimeError("Docker has no NVIDIA GPU integration and host driver libraries are missing: "
                           + ", ".join(sorted(missing)))
    return libraries["libcuda.so.1"], libraries["libnvidia-ptxjitcompiler.so.1"]


def select_gpu_access(settings):
    """Keep explicit driver settings; otherwise use Docker integration or discover the host driver."""
    if settings.driver is not None:
        return settings
    runtimes = json.loads(command("docker", "info", "--format", "{{json .Runtimes}}"))
    if "nvidia" in runtimes or nvidia_cdi_available():
        return settings
    driver, ptx_jit = driver_libraries()
    print("Docker has no NVIDIA runtime or CDI devices; mounting host GPU devices and driver libraries", flush=True)
    return replace(settings, driver=driver, ptx_jit=ptx_jit)


def gpu_arguments(settings):
    """The GPU through Docker integration, or the host devices and driver libraries."""
    if settings.driver is None:
        return ["--gpus", "all"]
    arguments = []
    for device in DEVICES:
        arguments += ["--device", f"/dev/{device}"]
    arguments += mount_file(settings.driver, "/opt/euhedral/driver/libcuda.so.1")
    arguments += mount_file(settings.ptx_jit, "/opt/euhedral/driver/libnvidia-ptxjitcompiler.so.1")
    arguments += ["--env", "LD_LIBRARY_PATH=/opt/euhedral/driver"]
    return arguments


def command(*args):
    print("+", " ".join(str(arg) for arg in args), flush=True)
    result = subprocess.run(args, cwd=REPOSITORY, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, args, output=result.stdout)
    if args[:2] in (("git", "pull"), ("docker", "build")):
        print(result.stdout[-3000:], flush=True)
    return result.stdout.strip()


def get_json(url, data=None):
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=30 if data is not None else 5) as response:
        return json.load(response)


def verify_ready(port):
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 300
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            health = command("docker", "inspect", "--format", "{{.State.Health.Status}}", SERVICE)
            status = get_json(base + "/health")
            models = get_json(base + "/v1/models").get("data", [])
            if health == "healthy" and status.get("status") == "up" and status.get("engine") == "ready" and models:
                model = models[0]["id"]
                # Whatever the image's reasoning default, generated text lands in the answer or in the reasoning.
                response = get_json(
                    base + "/v1/chat/completions",
                    json.dumps({"model": model, "messages": [{"role": "user", "content": "Say READY"}], "max_tokens": 16,
                                "temperature": 0}).encode(),
                )
                choices = response.get("choices") or []
                usage = response.get("usage") or {}
                if usage.get("completion_tokens", 0) > 0 and any(
                    isinstance(choice.get("message", {}).get(field), str) and choice["message"][field].strip()
                    for choice in choices for field in ("content", "reasoning_content")
                ):
                    return model
                last_error = "chat smoke response had no generated text"
            else:
                last_error = f"Docker={health}, HTTP={status}, models={len(models)}"
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            last_error = str(error)
        time.sleep(3)
    raise RuntimeError(f"{SERVICE} did not become ready: {last_error}")


@contextmanager
def build_context(repository):
    """Build exactly the pulled commit, never the untracked benchmark archive."""
    with tempfile.TemporaryDirectory(prefix="euhedral-deploy-", dir=os.environ.get("TMPDIR", str(repository / "benchmark-results"))) as temporary:
        archive = Path(temporary) / "source.tar"
        with archive.open("wb") as output:
            subprocess.run(("git", "archive", "--format=tar", "HEAD"), cwd=repository, stdout=output, check=True)
        context = Path(temporary) / "source"
        context.mkdir()
        with tarfile.open(archive) as source:
            source.extractall(context, filter="data")
        archive.unlink()
        yield context


def deploy(repository, settings):
    model, port = settings.artifact, settings.port
    if command("git", "branch", "--show-current") != "main":
        raise RuntimeError("Deploy only from main; switch to main before invoking this script")
    if command("git", "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked working-tree changes must be committed or moved before deployment")
    settings = select_gpu_access(settings)
    for path in (model, settings.driver, settings.ptx_jit):
        if path is not None and not path.is_file():
            raise RuntimeError(f"Missing required host file: {path}")
    if settings.driver is not None:
        for device in DEVICES:
            if not Path("/dev", device).exists():
                raise RuntimeError(f"Missing NVIDIA device: /dev/{device}")
    if not 1 <= port <= 65535:
        raise RuntimeError("Invalid HTTP port")

    command("git", "pull", "--ff-only", "origin", "main")
    if Path(__file__).read_bytes() != SCRIPT_SOURCE:
        # Deploying main with the script of an older main can pass the wrong settings to the new image.
        print("The pull changed the deploy script; running the new version", flush=True)
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    revision = command("git", "rev-parse", "HEAD")
    if revision != command("git", "rev-parse", "origin/main"):
        raise RuntimeError("Local main does not match origin/main")
    image = f"{IMAGE_REPOSITORY}:{revision[:7]}"
    names = command("docker", "ps", "-a", "--format", "{{.Names}}").splitlines()
    previous = SERVICE in names
    previous_name = None
    previous_image_id = None
    if previous:
        previous_id = command("docker", "inspect", "--format", "{{.Id}}", SERVICE)
        previous_image_id = command("docker", "inspect", "--format", "{{.Image}}", SERVICE)
        previous_name = f"{SERVICE}-previous-{previous_id[:12]}"
        if previous_name in names:
            raise RuntimeError(f"Rollback container already exists: {previous_name}")
    print(f"Building {image} from {revision}; the current service stays up until the build succeeds", flush=True)
    with build_context(repository) as source:
        command("docker", "build", "--label", f"org.opencontainers.image.revision={revision}", "-t", image, str(source))
    new_image_id = command("docker", "image", "inspect", "--format", "{{.Id}}", image)
    old_tags = [
        tag for tag in command("docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}").splitlines()
        if tag.startswith(IMAGE_REPOSITORY + ":") and tag != image
    ]
    old_image_ids = {command("docker", "image", "inspect", "--format", "{{.Id}}", tag) for tag in old_tags}
    if previous_image_id:
        old_image_ids.add(previous_image_id)
    old_stopped = False
    old_renamed = False
    try:
        if previous:
            command("docker", "stop", SERVICE)
            old_stopped = True
            command("docker", "rename", SERVICE, previous_name)
            old_renamed = True
        command(
            "docker", "run", "-d", "--name", SERVICE, "--restart", "unless-stopped", "--stop-timeout", "45",
            *(["--env-file", str(settings.env_file)] if settings.env_file else []),
            *(mount_file(settings.config_file, "/etc/euhedral/custom.yaml") if settings.config_file else []),
            "-e", "EUHEDRAL_MODEL_ARTIFACT=/models/model.edrl",
            "-e", f"EUHEDRAL_SERVER_PORT={port}", "-p", f"127.0.0.1:{port}:{port}",
            *gpu_arguments(settings),
            *mount_file(model, "/models/model.edrl"),
            image,
            *(["--config", "/etc/euhedral/custom.yaml"] if settings.config_file else []),
        )
        model_id = verify_ready(port)
        if model_id != settings.model_id:
            raise RuntimeError(f"Unexpected public model ID {model_id!r}, expected {settings.model_id!r}")
        if command("docker", "image", "inspect", "--format", "{{.Id}}", image) != new_image_id:
            raise RuntimeError("Deployment image changed before cleanup")
        if command("docker", "inspect", "--format", "{{.State.Health.Status}}", SERVICE) != "healthy":
            raise RuntimeError("Deployed container became unhealthy before cleanup")
    except BaseException as error:
        if old_stopped:
            print(f"Deployment failed ({error}); restoring the previous service", file=sys.stderr, flush=True)
            try:
                if old_renamed:
                    try:
                        command("docker", "rm", "-f", SERVICE)
                    except subprocess.CalledProcessError:
                        pass  # Candidate never reached container creation.
                    command("docker", "rename", previous_name, SERVICE)
                command("docker", "start", SERVICE)
                verify_ready(port)
                print("Previous service restored and verified", file=sys.stderr, flush=True)
            except BaseException as rollback_error:
                raise RuntimeError(f"Deployment failed: {error}; ROLLBACK FAILED: {rollback_error}") from error
        raise

    print(f"Healthy candidate {revision} serving {model_id}; cleaning old Euhedral containers/images", flush=True)
    failures = []
    for name in command("docker", "ps", "-a", "--format", "{{.Names}}").splitlines():
        stale_deployment = name.startswith(SERVICE + "-")
        experimental_probe = name.startswith("euhedral-") and name.endswith("-probe")
        owned_probe = experimental_probe and command("docker", "inspect", "--format", "{{.Image}}", name) in old_image_ids
        if stale_deployment or owned_probe:
            try:
                command("docker", "rm", name)  # Never force-stop an unexpected running container.
            except subprocess.CalledProcessError as error:
                failures.append(f"container {name}: {error.output[-300:]}")
    for tag in old_tags:
        try:
            command("docker", "image", "rm", tag)
        except subprocess.CalledProcessError as error:
            failures.append(f"image {tag}: {error.output[-300:]}")
    if previous_image_id and previous_image_id != new_image_id:
        try:
            command("docker", "image", "inspect", "--format", "{{.Id}}", previous_image_id)
        except subprocess.CalledProcessError:
            pass  # Deleting its last tag already removed the old image.
        else:
            try:
                command("docker", "image", "rm", previous_image_id)
            except subprocess.CalledProcessError as error:
                failures.append(f"previous image {previous_image_id}: {error.output[-300:]}")

    if failures:
        raise RuntimeError("Deployed service is healthy, but cleanup was incomplete: " + "; ".join(failures))
    print(f"Deployed {revision} ({model_id}) and removed old Euhedral deployments and experiments", flush=True)
    return revision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", type=Path, help="host YAML or Docker-format .env file")
    arguments = parser.parse_args(argv)
    try:
        deploy(REPOSITORY, load_settings(arguments.file))
    except (RuntimeError, subprocess.CalledProcessError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            print((error.output or "")[-4000:], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
