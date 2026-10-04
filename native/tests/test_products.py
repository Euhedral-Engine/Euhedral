"""Validate the installed CUDA products without running a foreign binary."""

import json
import pathlib
import struct
import sys
import unittest
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "native" / "native-products.json"
OUTPUT = ROOT / "build" / "native"


def elf_needed(data):
    """Resolve DT_NEEDED names through the ELF64 program-header load segments."""
    self_header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    program_offset, program_size, program_count = self_header[5], self_header[9], self_header[10]
    segments = [struct.unpack_from("<IIQQQQQQ", data, program_offset + index * program_size)
                for index in range(program_count)]

    def offset(address):
        for kind, _, file_offset, virtual, _, file_size, _, _ in segments:
            if kind == 1 and virtual <= address < virtual + file_size:
                return file_offset + address - virtual
        raise ValueError(f"unmapped ELF address {address:x}")

    dynamic = next(segment for segment in segments if segment[0] == 2)
    entries = [struct.unpack_from("<QQ", data, dynamic[2] + index)
               for index in range(0, dynamic[5], 16)]
    table = offset(next(value for tag, value in entries if tag == 5))
    needed = []
    for tag, value in entries:
        if tag == 1:
            start = table + value
            needed.append(data[start:data.index(b"\0", start)].decode("ascii"))
    return set(needed)


def pe_imports(data):
    """Read DLL names from PE32+ import descriptors, not arbitrary binary strings."""
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    coff = pe + 4
    section_count, optional_size = struct.unpack_from("<H", data, coff + 2)[0], struct.unpack_from("<H", data, coff + 16)[0]
    optional = coff + 20
    if struct.unpack_from("<H", data, optional)[0] != 0x20B:
        raise ValueError("expected PE32+ binary")
    import_rva = struct.unpack_from("<I", data, optional + 112 + 8)[0]
    sections = [struct.unpack_from("<8sIIIIIIHHI", data, optional + optional_size + 40 * index)
                for index in range(section_count)]

    def offset(rva):
        for section in sections:
            _, virtual_size, virtual, raw_size, raw_offset, *_ = section
            if virtual <= rva < virtual + min(virtual_size, raw_size):
                return raw_offset + rva - virtual
        raise ValueError(f"unmapped PE RVA {rva:x}")

    imports = set()
    descriptor = offset(import_rva)
    while True:
        fields = struct.unpack_from("<IIIII", data, descriptor)
        if not any(fields):
            break
        start = offset(fields[3])
        imports.add(data[start:data.index(b"\0", start)].decode("ascii"))
        descriptor += 20
    return imports


class NativeProductsTest(unittest.TestCase):
    def test_boot_jar_does_not_embed_unloadable_native_products(self):
        application = ROOT / "api" / "build" / "libs" / "euhedral-inference-api.jar"
        self.assertTrue(application.is_file(), application)
        with zipfile.ZipFile(application) as packaged:
            self.assertFalse(any(name.startswith("BOOT-INF/native/") for name in packaged.namelist()))

    def test_distribution_contains_only_native_products(self):
        archive = ROOT / "build" / "distributions" / "euhedral-cuda-native.zip"
        self.assertTrue(archive.is_file(), archive)
        with zipfile.ZipFile(archive) as packaged:
            members = {name for name in packaged.namelist() if not name.endswith("/")}
        products = json.loads(MANIFEST.read_text(encoding="utf-8"))["products"]
        sources = {path.relative_to(ROOT / "native" / "src").as_posix()
                   for path in (ROOT / "native" / "src").rglob("*")
                   if path.is_file() and path.suffix in {".cu", ".cuh"}}
        expected = {
            f"{product['id']}/lib/{product['filename']}" for product in products
        } | {
            f"{product['id']}/lib/{product['llguidanceFilename']}" for product in products
        } | {
            f"{product['id']}/share/euhedral_cuda/{source}"
            for product in products for source in sources
        }
        self.assertEqual(members, expected)

    def test_installed_products_match_manifest_and_binary_targets(self):
        products = json.loads(MANIFEST.read_text(encoding="utf-8"))["products"]
        self.assertEqual({item["id"] for item in products}, {"linux-x64", "windows-x64"})
        sources = {path.relative_to(ROOT / "native" / "src").as_posix()
                   for path in (ROOT / "native" / "src").rglob("*")
                   if path.is_file() and path.suffix in {".cu", ".cuh"}}
        self.assertTrue(sources)
        for product in products:
            prefix = OUTPUT / product["id"]
            library = prefix / "lib" / product["filename"]
            with self.subTest(product=product["id"]):
                self.assertTrue(library.is_file(), library)
                binary = library.read_bytes()
                if product["id"] == "linux-x64":
                    self.assertEqual(binary[:4], b"\x7fELF")
                    self.assertEqual(struct.unpack_from("<H", binary, 18)[0], 62)
                    self.assertTrue({"libcuda.so.1", "libcudart.so.13", "libnvrtc.so.13"} <= elf_needed(binary))
                    self.assertFalse(any(name.endswith(".dll") for name in elf_needed(binary)))
                else:
                    self.assertEqual(binary[:2], b"MZ")
                    pe = struct.unpack_from("<I", binary, 0x3C)[0]
                    self.assertEqual(binary[pe : pe + 4], b"PE\x00\x00")
                    self.assertEqual(struct.unpack_from("<H", binary, pe + 4)[0], 0x8664)
                    imports = pe_imports(binary)
                    self.assertTrue({"cudart64_13.dll", "nvrtc64_130_0.dll", "nvcuda.dll"} <= imports)
                    self.assertFalse(any(name.startswith("libcuda.so") for name in imports))
                installed = {path.relative_to(prefix / "share" / "euhedral_cuda").as_posix()
                             for path in (prefix / "share" / "euhedral_cuda").rglob("*") if path.is_file()}
                self.assertEqual(installed, sources)


if __name__ == "__main__":
    unittest.main()
