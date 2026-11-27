//! The shared libraries a native binary imports, read from its headers: the `DT_NEEDED` names of an ELF64 file and the
//! import directory of a PE32+ file. `cargo xtask native-verify` uses it to prove that a product asks for the runtime
//! of its own platform. Every read is bounds-checked: a truncated or foreign file is refused with the offset named.

use std::collections::BTreeSet;

use crate::native::Platform;

const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const ELF_CLASS_64: u8 = 2;
const ELF_LITTLE_ENDIAN: u8 = 1;
const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const DT_NEEDED: u64 = 1;
const DT_STRTAB: u64 = 5;
const DYNAMIC_ENTRY: usize = 16;
const PE_OPTIONAL_MAGIC_64: u16 = 0x20b;
const PE_IMPORT_DIRECTORY_AT: usize = 120;
const PE_SECTION_HEADER: usize = 40;
const PE_IMPORT_DESCRIPTOR: usize = 20;

/// The library names `data` imports, for the binary format of `platform`.
pub(crate) fn binary_imports(data: &[u8], platform: Platform) -> Result<BTreeSet<String>, String> {
    match platform {
        Platform::Linux => elf_imports(data),
        Platform::Windows => pe_imports(data),
    }
}

/// Little-endian reads that refuse to run past the end.
struct Bytes<'a>(&'a [u8]);

impl Bytes<'_> {
    fn take<const N: usize>(&self, at: usize) -> Result<[u8; N], String> {
        at.checked_add(N)
            .and_then(|end| self.0.get(at..end))
            .and_then(|slice| slice.try_into().ok())
            .ok_or_else(|| format!("truncated binary: {N} bytes at offset {at} of {}", self.0.len()))
    }

    fn u16(&self, at: usize) -> Result<u16, String> {
        self.take(at).map(u16::from_le_bytes)
    }

    fn u32(&self, at: usize) -> Result<u32, String> {
        self.take(at).map(u32::from_le_bytes)
    }

    fn u64(&self, at: usize) -> Result<u64, String> {
        self.take(at).map(u64::from_le_bytes)
    }

    /// An offset stored as 64 bits, as an index into this file.
    fn index(&self, at: usize) -> Result<usize, String> {
        let value = self.u64(at)?;
        usize::try_from(value).map_err(|_| format!("offset {value} at {at} does not fit the address space"))
    }

    /// The NUL-terminated ASCII name at `at`.
    fn name(&self, at: usize) -> Result<String, String> {
        let tail = self
            .0
            .get(at..)
            .ok_or_else(|| format!("import name offset {at} is outside the {}-byte file", self.0.len()))?;
        let length = tail.iter().position(|&byte| byte == 0).unwrap_or(tail.len());
        Ok(String::from_utf8_lossy(&tail[..length]).into_owned())
    }
}

/// A `PT_LOAD` segment: where its bytes sit in the file and at which address they are mapped.
struct Segment {
    file_offset: usize,
    address: u64,
    size: u64,
}

fn elf_imports(data: &[u8]) -> Result<BTreeSet<String>, String> {
    let bytes = Bytes(data);
    let identity: [u8; 6] = bytes.take(0)?;
    if identity[..4] != ELF_MAGIC || identity[4] != ELF_CLASS_64 || identity[5] != ELF_LITTLE_ENDIAN {
        return Err("not a little-endian ELF64 file".to_string());
    }
    let table = bytes.index(32)?;
    let entry_size = usize::from(bytes.u16(54)?);
    let count = usize::from(bytes.u16(56)?);
    let mut segments = Vec::new();
    let mut dynamic = None;
    for index in 0..count {
        let entry = index
            .checked_mul(entry_size)
            .and_then(|offset| offset.checked_add(table))
            .ok_or("ELF program header table overflows")?;
        match bytes.u32(entry)? {
            PT_LOAD => segments.push(Segment {
                file_offset: bytes.index(entry + 8)?,
                address: bytes.u64(entry + 16)?,
                size: bytes.u64(entry + 32)?,
            }),
            PT_DYNAMIC => dynamic = Some((bytes.index(entry + 8)?, bytes.index(entry + 32)?)),
            _ => {}
        }
    }
    let (dynamic_at, dynamic_length) = dynamic
        .filter(|&(_, length)| length > 0)
        .ok_or("missing ELF dynamic section")?;
    let entries = (0..dynamic_length)
        .step_by(DYNAMIC_ENTRY)
        .map(|position| Ok((bytes.u64(dynamic_at + position)?, bytes.u64(dynamic_at + position + 8)?)))
        .collect::<Result<Vec<_>, String>>()?;
    let table_address = entries
        .iter()
        .find_map(|&(tag, value)| (tag == DT_STRTAB).then_some(value))
        .ok_or("missing ELF dynamic string table")?;
    let strings = segments
        .iter()
        .find(|segment| table_address >= segment.address && table_address - segment.address < segment.size)
        .and_then(|segment| {
            usize::try_from(table_address - segment.address)
                .ok()?
                .checked_add(segment.file_offset)
        })
        .ok_or_else(|| format!("unmapped ELF import address {table_address}"))?;
    entries
        .iter()
        .filter(|&&(tag, _)| tag == DT_NEEDED)
        .map(|&(_, name)| {
            let name = usize::try_from(name).map_err(|_| format!("import name offset {name} is too large"))?;
            bytes.name(strings.checked_add(name).ok_or("ELF import name offset overflows")?)
        })
        .collect()
}

fn pe_imports(data: &[u8]) -> Result<BTreeSet<String>, String> {
    let bytes = Bytes(data);
    if bytes.take::<2>(0)? != *b"MZ" {
        return Err("not a PE file: missing MZ signature".to_string());
    }
    let pe = usize::try_from(bytes.u32(0x3c)?).map_err(|_| "PE header offset is too large".to_string())?;
    if bytes.take::<4>(pe)? != *b"PE\0\0" {
        return Err(format!("not a PE file: missing PE signature at offset {pe}"));
    }
    let coff = pe + 4;
    let optional = coff + 20;
    let magic = bytes.u16(optional)?;
    if magic != PE_OPTIONAL_MAGIC_64 {
        return Err(format!(
            "expected a PE32+ image (0x20b), found optional header magic {magic:#x}"
        ));
    }
    let import_rva = bytes.u32(optional + PE_IMPORT_DIRECTORY_AT)?;
    let sections = optional + usize::from(bytes.u16(coff + 16)?);
    let section_count = usize::from(bytes.u16(coff + 2)?);
    let file_offset = |rva: u32| -> Result<usize, String> {
        for index in 0..section_count {
            let section = sections + index * PE_SECTION_HEADER;
            let address = bytes.u32(section + 12)?;
            let size = bytes.u32(section + 8)?.min(bytes.u32(section + 16)?);
            if rva >= address && rva - address < size {
                let raw = usize::try_from(bytes.u32(section + 20)?).map_err(|_| "PE section offset is too large")?;
                return Ok(raw + usize::try_from(rva - address).map_err(|_| "PE RVA is too large")?);
            }
        }
        Err(format!("unmapped PE import RVA {rva}"))
    };
    let mut imports = BTreeSet::new();
    let mut descriptor = file_offset(import_rva)?;
    // The directory ends with an all-zero descriptor.
    loop {
        let mut empty = true;
        for word in 0..5 {
            empty &= bytes.u32(descriptor + word * 4)? == 0;
        }
        if empty {
            break;
        }
        imports.insert(bytes.name(file_offset(bytes.u32(descriptor + 12)?)?)?);
        descriptor += PE_IMPORT_DESCRIPTOR;
    }
    Ok(imports)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::iter::once;

    use super::*;

    const BASE: u64 = 0x40_0000;

    /// A minimal ELF64 shared object: one `PT_LOAD` over the whole file, a `PT_DYNAMIC` with the needed names.
    pub(crate) fn elf(needed: &[&str]) -> Vec<u8> {
        let strings: Vec<u8> = once(0u8)
            .chain(needed.iter().flat_map(|name| name.bytes().chain([0])))
            .collect();
        let headers = 64 + 2 * 56;
        let dynamic_at = headers + strings.len();
        let mut offsets = Vec::new();
        let mut at = 1u64;
        for name in needed {
            offsets.push(at);
            at += name.len() as u64 + 1;
        }
        let dynamic_length = (needed.len() + 2) * DYNAMIC_ENTRY;
        let total = dynamic_at + dynamic_length;
        let mut file = vec![0u8; total];
        file[..4].copy_from_slice(&ELF_MAGIC);
        file[4] = ELF_CLASS_64;
        file[5] = ELF_LITTLE_ENDIAN;
        file[32..40].copy_from_slice(&64u64.to_le_bytes());
        file[54..56].copy_from_slice(&56u16.to_le_bytes());
        file[56..58].copy_from_slice(&2u16.to_le_bytes());
        let load = 64;
        file[load..load + 4].copy_from_slice(&PT_LOAD.to_le_bytes());
        file[load + 16..load + 24].copy_from_slice(&BASE.to_le_bytes());
        file[load + 32..load + 40].copy_from_slice(&(total as u64).to_le_bytes());
        let dynamic = load + 56;
        file[dynamic..dynamic + 4].copy_from_slice(&PT_DYNAMIC.to_le_bytes());
        file[dynamic + 8..dynamic + 16].copy_from_slice(&(dynamic_at as u64).to_le_bytes());
        file[dynamic + 32..dynamic + 40].copy_from_slice(&(dynamic_length as u64).to_le_bytes());
        file[headers..dynamic_at].copy_from_slice(&strings);
        let mut entries = vec![(DT_STRTAB, BASE + headers as u64)];
        entries.extend(offsets.into_iter().map(|offset| (DT_NEEDED, offset)));
        for (index, (tag, value)) in entries.into_iter().enumerate() {
            let entry = dynamic_at + index * DYNAMIC_ENTRY;
            file[entry..entry + 8].copy_from_slice(&tag.to_le_bytes());
            file[entry + 8..entry + 16].copy_from_slice(&value.to_le_bytes());
        }
        file
    }

    /// A minimal PE32+ image: one section mapped at RVA 0x1000 holding an import directory with the given names.
    pub(crate) fn pe(imports: &[&str]) -> Vec<u8> {
        let (pe_at, optional_size, raw) = (0x40usize, 240usize, 0x200usize);
        let directory = (imports.len() + 1) * PE_IMPORT_DESCRIPTOR;
        let mut names = Vec::new();
        let mut name_rvas = Vec::new();
        for name in imports {
            name_rvas.push(0x1000 + directory + names.len());
            names.extend(name.bytes().chain([0]));
        }
        let mut file = vec![0u8; raw + directory + names.len()];
        file[..2].copy_from_slice(b"MZ");
        file[0x3c..0x40].copy_from_slice(&(pe_at as u32).to_le_bytes());
        file[pe_at..pe_at + 4].copy_from_slice(b"PE\0\0");
        let coff = pe_at + 4;
        file[coff + 2..coff + 4].copy_from_slice(&1u16.to_le_bytes());
        file[coff + 16..coff + 18].copy_from_slice(&(optional_size as u16).to_le_bytes());
        let optional = coff + 20;
        file[optional..optional + 2].copy_from_slice(&PE_OPTIONAL_MAGIC_64.to_le_bytes());
        file[optional + PE_IMPORT_DIRECTORY_AT..optional + PE_IMPORT_DIRECTORY_AT + 4]
            .copy_from_slice(&0x1000u32.to_le_bytes());
        let section = optional + optional_size;
        let size = (directory + names.len()) as u32;
        file[section + 8..section + 12].copy_from_slice(&size.to_le_bytes());
        file[section + 12..section + 16].copy_from_slice(&0x1000u32.to_le_bytes());
        file[section + 16..section + 20].copy_from_slice(&size.to_le_bytes());
        file[section + 20..section + 24].copy_from_slice(&(raw as u32).to_le_bytes());
        for (index, rva) in name_rvas.into_iter().enumerate() {
            let descriptor = raw + index * PE_IMPORT_DESCRIPTOR;
            file[descriptor + 12..descriptor + 16].copy_from_slice(&(rva as u32).to_le_bytes());
        }
        file[raw + directory..].copy_from_slice(&names);
        file
    }

    fn set(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    #[test]
    fn an_elf_reports_its_needed_libraries() {
        let data = elf(&["libcudart.so.13", "libcuda.so.1"]);
        assert_eq!(
            binary_imports(&data, Platform::Linux),
            Ok(set(&["libcuda.so.1", "libcudart.so.13"]))
        );
    }

    #[test]
    fn a_pe_reports_its_imported_dlls() {
        let data = pe(&["cudart64_13.dll", "nvcuda.dll"]);
        assert_eq!(
            binary_imports(&data, Platform::Windows),
            Ok(set(&["cudart64_13.dll", "nvcuda.dll"]))
        );
    }

    #[test]
    fn a_pe_read_as_elf_and_an_elf_read_as_pe_are_refused() {
        assert!(binary_imports(&pe(&["nvcuda.dll"]), Platform::Linux).is_err());
        assert!(binary_imports(&elf(&["libcuda.so.1"]), Platform::Windows).is_err());
    }

    #[test]
    fn a_truncated_elf_is_refused_with_the_offset() {
        let data = elf(&["libcuda.so.1"]);
        let error = binary_imports(&data[..data.len() - 20], Platform::Linux).expect_err("truncated");
        assert!(error.contains("truncated binary"), "{error}");
    }

    #[test]
    fn a_truncated_pe_is_refused() {
        let data = pe(&["nvcuda.dll"]);
        assert!(binary_imports(&data[..0x150], Platform::Windows).is_err());
    }

    #[test]
    fn an_elf_without_a_dynamic_section_is_refused() {
        let mut data = elf(&["libcuda.so.1"]);
        data[64 + 56..64 + 56 + 4].copy_from_slice(&0u32.to_le_bytes());
        let error = binary_imports(&data, Platform::Linux).expect_err("no dynamic section");
        assert!(error.contains("dynamic section"), "{error}");
    }
}
