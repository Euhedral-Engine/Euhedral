"""The tokenizer an artifact carries after its content, found from a footer at the end of the file (the Java
`TokenizerSection`, byte for byte). Big-endian:

    body:   int32 entry count (4), then per entry: int32 name length, UTF-8 name, int32 data length, UTF-8 data,
            in the order tokenizer.json, tokenizer_config.json, chat_template.jinja, special_tokens.json
    footer: int64 body length, int32 CRC-32 of the body, int32 version (1), b"EDRLTOKN"

special_tokens.json holds the bos_token_id, eos_token_id and pad_token_id that generation_config.json sets, as JSON
without spaces."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import BinaryIO

ENTRIES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens.json")
SPECIAL_TOKEN_KEYS = ("bos_token_id", "eos_token_id", "pad_token_id")
MAGIC = b"EDRLTOKN"
VERSION = 1
FOOTER = struct.Struct(">qIi8s")


class SectionError(ValueError):
    pass


def from_checkpoint(directory: Path) -> tuple[str, str, str, str]:
    """The four entries' texts from a checkpoint directory: chat_template.jinja, or else tokenizer_config.json's
    chat_template string."""
    def text(name: str) -> str:
        path = directory / name
        if not path.is_file():
            raise SectionError(f"checkpoint has no {name} in {directory}")
        return path.read_bytes().decode("utf-8")

    tokenizer = text("tokenizer.json")
    config = text("tokenizer_config.json")
    jinja = directory / "chat_template.jinja"
    if jinja.is_file():
        template = jinja.read_bytes().decode("utf-8")
    else:
        template = json.loads(config).get("chat_template")
        if not isinstance(template, str):
            raise SectionError(f"checkpoint has no chat template in {directory}")
    generation = json.loads(text("generation_config.json"))
    special = {key: generation[key] for key in SPECIAL_TOKEN_KEYS if key in generation}
    return tokenizer, config, template, json.dumps(special, separators=(",", ":"), ensure_ascii=False)


def encode(files: tuple[str, str, str, str]) -> bytes:
    """The section (body and footer) holding the four entries' texts."""
    body = bytearray(struct.pack(">i", len(ENTRIES)))
    for name, data in zip(ENTRIES, files, strict=True):
        for value in (name.encode("utf-8"), data.encode("utf-8")):
            body += struct.pack(">i", len(value)) + value
    return bytes(body) + FOOTER.pack(len(body), zlib.crc32(body), VERSION, MAGIC)


def _footer(handle: BinaryIO) -> tuple[int, int, int] | None:
    """(section start, body length, CRC-32), or None without a section."""
    size = handle.seek(0, 2)
    if size < FOOTER.size:
        return None
    handle.seek(size - FOOTER.size)
    length, crc, version, magic = FOOTER.unpack(handle.read(FOOTER.size))
    if magic != MAGIC:
        return None
    if version != VERSION:
        raise SectionError(f"unsupported tokenizer section version {version}")
    if length < 0 or length > size - FOOTER.size:
        raise SectionError(f"the tokenizer section's length is out of range: {length}")
    return size - FOOTER.size - length, length, crc


def content_end(path: Path) -> int:
    """Where the artifact's content ends: the start of its tokenizer section, or the end of the file."""
    with path.open("rb") as handle:
        footer = _footer(handle)
        return footer[0] if footer else handle.seek(0, 2)


def read(path: Path) -> bytes | None:
    """The artifact's tokenizer section (body and footer) as it is stored, or None when it carries none."""
    with path.open("rb") as handle:
        footer = _footer(handle)
        if footer is None:
            return None
        start, length, crc = footer
        handle.seek(start)
        section = handle.read(length + FOOTER.size)
    if zlib.crc32(section[:length]) != crc:
        raise SectionError(f"{path}: the tokenizer section fails its CRC-32")
    return section


def require(path: Path) -> bytes:
    """As `read`, failing with the command that adds a section to an artifact without one."""
    section = read(path)
    if section is None:
        raise SectionError(f"{path} carries no tokenizer; add the checkpoint's with: ./gradlew :quantization:run "
                           f"--args=\"package-tokenizer --artifact {path} --tokenizer CHECKPOINT_DIR\"")
    return section


def write(handle: BinaryIO, content_end: int, section: bytes) -> None:
    """Writes `section` after the first `content_end` bytes of the open file, dropping whatever followed them."""
    handle.truncate(content_end)
    handle.seek(content_end)
    handle.write(section)
