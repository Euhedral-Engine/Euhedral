import json
import tempfile
import unittest
from pathlib import Path

from euhedral_artifacts import tokenizer_section

# The bytes TokenizerSectionTest (core) checks for the same files.
ENCODED = ("000000040000000e746f6b656e697a65722e6a736f6e0000000c7b226d6f64656c223a7b7d7d"
           "00000015746f6b656e697a65725f636f6e6669672e6a736f6e000000027b7d00000013636861745f74656d706c6174652e"
           "6a696e6a610000000a7b7b2022c3a922207d7d000000137370656369616c5f746f6b656e732e6a736f6e000000147b2265"
           "6f735f746f6b656e5f6964223a5b325d7d00000000000000998fa57935000000014544524c544f4b4e")


class TokenizerSectionTest(unittest.TestCase):
    def test_encodes_the_bytes_the_java_reader_reads(self):
        files = ('{"model":{}}', "{}", '{{ "é" }}', '{"eos_token_id":[2]}')
        self.assertEqual(ENCODED, tokenizer_section.encode(files).hex())

    def test_packages_a_checkpoint_and_finds_the_content_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "tokenizer.json").write_bytes(b'{"a":1}\r\n')
            (directory / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ x }}"}))
            (directory / "generation_config.json").write_text(
                json.dumps({"top_k": 20, "eos_token_id": [248046, 248044], "bos_token_id": 248044}))
            files = tokenizer_section.from_checkpoint(directory)
            self.assertEqual('{"a":1}\r\n', files[0], "the files are packaged byte for byte")
            self.assertEqual("{{ x }}", files[2])
            self.assertEqual('{"bos_token_id":248044,"eos_token_id":[248046,248044]}', files[3])
            artifact = directory / "a.edrl"
            artifact.write_bytes(bytes(range(200)))
            self.assertIsNone(tokenizer_section.read(artifact))
            with self.assertRaisesRegex(tokenizer_section.SectionError, "package-tokenizer"):
                tokenizer_section.require(artifact)
            section = tokenizer_section.encode(files)
            with artifact.open("r+b") as handle:
                tokenizer_section.write(handle, 200, section)
            self.assertEqual(200, tokenizer_section.content_end(artifact))
            self.assertEqual(section, tokenizer_section.read(artifact))
            self.assertEqual(bytes(range(200)), artifact.read_bytes()[:200])


    def test_a_damaged_footer_names_the_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "a.edrl"
            artifact.write_bytes(bytes(64) + tokenizer_section.FOOTER.pack(
                tokenizer_section.MAX_BODY_BYTES + 1, 0, tokenizer_section.VERSION, tokenizer_section.MAGIC))
            with self.assertRaisesRegex(tokenizer_section.SectionError, f"^{artifact}: .*out of range"):
                tokenizer_section.content_end(artifact)

if __name__ == "__main__":
    unittest.main()
