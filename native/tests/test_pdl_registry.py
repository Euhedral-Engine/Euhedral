"""The programmatic-launch registry holds every registration, from any thread, and a full one refuses loudly.
Compiles the registry alone with the system C compiler; needs no device."""
import ctypes as C
import pathlib
import shutil
import subprocess
import tempfile
import threading
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
NATIVE = ROOT / "native"
COMPILER = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


@unittest.skipIf(COMPILER is None, "no C compiler")
class PdlRegistryTest(unittest.TestCase):
    def setUp(self):
        # Each test gets its own copy of the library, so the registry state is its own.
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        library = pathlib.Path(self.directory.name) / "pdl_registry.so"
        subprocess.run([COMPILER, "-shared", "-fPIC", "-O2", "-std=c11", "-I", str(NATIVE / "include"),
                        str(NATIVE / "src" / "host" / "pdl_registry.c"), "-o", str(library)], check=True)
        self.lib = C.CDLL(str(library))
        self.lib.euhedral_pdl_registry_add.argtypes = [C.c_void_p]
        self.lib.euhedral_pdl_registry_contains.argtypes = [C.c_void_p]

    def test_more_than_sixty_four_functions_are_all_found(self):
        handles = [0x10000 + 0x40 * index for index in range(300)]
        threads = [threading.Thread(target=lambda part=handles[i::4]: [self.lib.euhedral_pdl_registry_add(h) for h in part])
                   for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        missing = [h for h in handles if not self.lib.euhedral_pdl_registry_contains(h)]
        self.assertEqual([], missing)
        self.assertFalse(self.lib.euhedral_pdl_registry_contains(0x10020))
        self.assertFalse(self.lib.euhedral_pdl_registry_contains(None))

    def test_a_repeated_registration_takes_no_second_slot_and_a_full_registry_refuses(self):
        self.assertEqual(1, self.lib.euhedral_pdl_registry_add(0x900000))
        self.assertEqual(1, self.lib.euhedral_pdl_registry_add(0x900000))
        refused = [h for h in range(0x1000000, 0x1000000 + 0x10 * 2000, 0x10) if not self.lib.euhedral_pdl_registry_add(h)]
        self.assertTrue(refused, "a registry of finite size must refuse, not drop silently")
        self.assertTrue(self.lib.euhedral_pdl_registry_contains(0x900000))
        for h in range(0x1000000, 0x1000000 + 0x10 * 2000, 0x10):
            self.assertEqual(h not in refused, bool(self.lib.euhedral_pdl_registry_contains(h)))


if __name__ == "__main__":
    unittest.main()
