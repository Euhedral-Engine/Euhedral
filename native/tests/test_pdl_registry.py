"""The programmatic-launch registry under a pthread harness (pdl_registry_harness.c): threads registering shared
keys behind a barrier lose none and double none, and a full registry accepts exactly its limit. The harness is
built against the registry and, as a negative control, against a copy of the former 64-entry registry, which
it must reject. Needs a C compiler and pthreads, no device."""
import pathlib
import shutil
import subprocess
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
NATIVE = HERE.parent
COMPILER = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


@unittest.skipIf(COMPILER is None, "no C compiler")
class PdlRegistryTest(unittest.TestCase):
    def run_harness(self, source):
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "harness"
            subprocess.run([COMPILER, "-O2", "-std=gnu11", "-pthread", "-I", str(NATIVE / "src" / "host"),
                            f'-DREGISTRY_SOURCE="{source}"', str(HERE / "pdl_registry_harness.c"), "-o", str(binary)],
                           check=True)
            return subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)

    def test_the_registry_holds_every_registration_under_racing_threads(self):
        result = self.run_harness(NATIVE / "src" / "host" / "pdl_registry.c")
        self.assertEqual(0, result.returncode, result.stderr)

    def test_the_former_registry_fails_the_same_harness(self):
        result = self.run_harness(HERE / "fixtures" / "pdl_registry_old.c")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("every shared key is found", result.stderr)


if __name__ == "__main__":
    unittest.main()
