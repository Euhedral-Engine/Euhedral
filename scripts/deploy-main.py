#!/usr/bin/env python3
"""Pull, build, and deploy main; clean up only after a healthy cutover.

Run on the deployment host from any directory. Every setting comes from the repository's .env (or --env-file):
copy .env.example and fill it in. The file is passed to the container as its environment, and its host paths are
mounted into it.
"""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
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
REQUIRED = ("EUHEDRAL_ARTIFACT_FILE", "EUHEDRAL_CHECKPOINT_DIR", "EUHEDRAL_INFERENCE_MODEL_ID", "EUHEDRAL_INFERENCE_WORKER_CPUS")
# The image fixes these and mounts the host files there.
CONTAINER_PATHS = ("EUHEDRAL_INFERENCE_ARTIFACT_PATH", "EUHEDRAL_INFERENCE_TOKENIZER_DIRECTORY", "EUHEDRAL_INFERENCE_CUDA_LIBRARY_PATH")
DEVICES = ("nvidia0", "nvidiactl", "nvidia-uvm", "nvidia-modeset")


@dataclass(frozen=True)
class Settings:
    env_file: Path
    artifact: Path
    checkpoint: Path
    model_id: str
    port: int
    driver: Path | None = None
    ptx_jit: Path | None = None


def read_env(path):
    """Docker's env-file format: KEY=VALUE lines, values taken literally, # comments on their own lines."""
    values = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if not separator or not key:
            raise RuntimeError(f"{path}:{number}: expected KEY=VALUE, got {line!r}")
        values[key] = value
    return values


def load_settings(path):
    if not path.is_file():
        raise RuntimeError(f"No settings file at {path}; copy .env.example to .env and fill it in")
    values = read_env(path)
    missing = [key for key in REQUIRED if not values.get(key)]
    if missing:
        raise RuntimeError(f"{path} does not set {', '.join(missing)} (see .env.example)")
    fixed = [key for key in CONTAINER_PATHS if key in values]
    if fixed:
        raise RuntimeError(f"{path} sets {', '.join(fixed)}, which the image fixes; set the host paths in "
                           "EUHEDRAL_ARTIFACT_FILE and EUHEDRAL_CHECKPOINT_DIR instead")
    driver, ptx_jit = values.get("EUHEDRAL_CUDA_DRIVER_FILE"), values.get("EUHEDRAL_PTX_JIT_FILE")
    if bool(driver) != bool(ptx_jit):
        raise RuntimeError(f"{path} must set both EUHEDRAL_CUDA_DRIVER_FILE and EUHEDRAL_PTX_JIT_FILE, or neither")
    try:
        port = int(values.get("PORT", "1738"))
    except ValueError:
        raise RuntimeError(f"{path}: PORT must be a number") from None
    return Settings(
        env_file=path.resolve(),
        artifact=Path(values["EUHEDRAL_ARTIFACT_FILE"]),
        checkpoint=Path(values["EUHEDRAL_CHECKPOINT_DIR"]),
        model_id=values["EUHEDRAL_INFERENCE_MODEL_ID"],
        port=port,
        driver=Path(driver) if driver else None,
        ptx_jit=Path(ptx_jit) if ptx_jit else None,
    )


def gpu_arguments(settings):
    """The GPU through the NVIDIA Container Toolkit, or, with the driver libraries set, the device nodes and those
    libraries mounted beside the native library."""
    if settings.driver is None:
        return ["--gpus", "all"]
    arguments = []
    for device in DEVICES:
        arguments += ["--device", f"/dev/{device}"]
    arguments += ["--mount", f"type=bind,src={settings.driver},dst=/opt/euhedral/lib/libcuda.so.1,readonly",
                  "--mount", f"type=bind,src={settings.ptx_jit},dst=/opt/euhedral/lib/libnvidia-ptxjitcompiler.so.1,readonly"]
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
    model, tokenizer, port = settings.artifact, settings.checkpoint, settings.port
    if command("git", "branch", "--show-current") != "main":
        raise RuntimeError("Deploy only from main; switch to main before invoking this script")
    if command("git", "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked working-tree changes must be committed or moved before deployment")
    for path in (model, settings.driver, settings.ptx_jit):
        if path is not None and not path.is_file():
            raise RuntimeError(f"Missing required host file: {path}")
    if not tokenizer.is_dir():
        raise RuntimeError(f"Missing tokenizer directory: {tokenizer}")
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
            "--env-file", str(settings.env_file), "-e", f"PORT={port}", "-p", f"127.0.0.1:{port}:{port}",
            *gpu_arguments(settings),
            "--mount", f"type=bind,src={model},dst=/models/model.edrl,readonly",
            "--mount", f"type=bind,src={tokenizer},dst=/tokenizer,readonly",
            image,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-file", type=Path, default=REPOSITORY / ".env", help="settings file (default: .env)")
    arguments = parser.parse_args()
    try:
        deploy(REPOSITORY, load_settings(arguments.env_file))
    except (RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            print(error.output[-4000:], file=sys.stderr)
        sys.exit(1)
