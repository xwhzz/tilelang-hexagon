"""SA8650 ELF32/Hexagon load-segment CRC metadata.

Format verified against the board's calculator and calculator64 DSP libraries:
section type 0xd3574956, payload SWIV + IEEE CRC32 + two zero words.
The CRC covers file bytes of each PT_LOAD, in program-header order.
This adds integrity data; it does not disable the DSP loader's verification.
"""

from pathlib import Path
import hashlib
import struct
import zlib

_CRC_SECTION = 0xD3574956


def _layout(data):
    if data[:7] != b"\x7fELF\x01\x01\x01" or len(data) < 52:
        raise ValueError("CRC metadata requires little-endian ELF32")
    header = list(struct.unpack_from("<16sHHIIIIIHHHHHH", data))
    if header[1] != 3 or header[2] != 164:
        raise ValueError("CRC metadata requires a Hexagon shared library")
    if header[9] != 32 or header[11] != 40 or not 0 < header[12] < 65534:
        raise ValueError("Unsupported ELF table layout")
    if header[5] + header[10] * 32 > len(data) or header[6] + header[12] * 40 > len(data):
        raise ValueError("ELF table exceeds file")
    return header


def _crc(data, header):
    result = 0
    for i in range(header[10]):
        segment = struct.unpack_from("<8I", data, header[5] + i * 32)
        if segment[0] == 1:
            offset, size = segment[1], segment[4]
            if offset + size > len(data):
                raise ValueError("ELF load segment exceeds file")
            result = zlib.crc32(data[offset : offset + size], result)
    return result


def verify(data):
    header = _layout(data)
    sections = [struct.unpack_from("<10I", data, header[6] + i * 40) for i in range(header[12])]
    matches = [section for section in sections if section[1] == _CRC_SECTION]
    if len(matches) != 1:
        raise ValueError("Expected exactly one CRC section")
    section = matches[0]
    offset, size = section[4:6]
    if size != 16 or offset + size > len(data):
        raise ValueError("Invalid CRC metadata size")
    magic, expected, reserved0, reserved1 = struct.unpack_from("<4I", data, offset)
    actual = _crc(data, header)
    if magic != 0x56495753 or reserved0 or reserved1 or expected != actual:
        raise ValueError(f"ELF CRC mismatch: stored={expected:08x}, actual={actual:08x}")
    return {"crc32": f"{actual:08x}", "section_type": "0xd3574956", "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def add_load_crc(path):
    path = Path(path)
    original = path.read_bytes()
    header = _layout(original)
    old_sections = original[header[6] : header[6] + header[12] * 40]
    for i in range(header[12]):
        if struct.unpack_from("<I", old_sections, i * 40 + 4)[0] == _CRC_SECTION:
            return verify(original)
    data = bytearray(original)
    data.extend(bytes((-len(data)) % 4))
    header[6] = len(data)
    header[12] += 1
    crc_offset = len(data) + len(old_sections) + 40
    data.extend(old_sections)
    data.extend(struct.pack("<10I", 0, _CRC_SECTION, 0, 0, crc_offset, 16, 0, 0, 0, 0))
    data.extend(bytes(16))
    struct.pack_into("<16sHHIIIIIHHHHHH", data, 0, *header)
    checksum = _crc(data, header)
    struct.pack_into("<4I", data, crc_offset, 0x56495753, checksum, 0, 0)
    evidence = verify(data)
    path.write_bytes(data)
    return evidence
