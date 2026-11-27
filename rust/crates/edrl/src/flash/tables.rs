use std::collections::HashSet;
use std::str;

use crate::error::EdrlError;
use crate::flash::config::MAX_EXPERTS;
use crate::flash::header::{FlashHeader, OBJECT_ALIGNMENT};
use crate::flash::ngram::NgramLayout;
use crate::format::{TensorDataType, WeightFormat, WeightLayout, accepts_byte_size};
use crate::reader::{Reader, checked_end};

const MAX_OBJECTS: u32 = 1 << 20;
const MAX_PROJECTIONS: u32 = 64;
pub(crate) const MAX_NAME_BYTES: usize = 4096;
pub(crate) const MAX_RANK: usize = 16;

/// The logical component an object belongs to. The ordinal is the `group` field of the tables, so the order is part
/// of the format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComponentGroup {
    /// Always-used text weights that belong to no other group (the n-gram projections of the first layer).
    FixedText = 0,
    /// Routed experts of the text layers, held in expert banks.
    RoutedExpert = 1,
    /// The shared expert of every layer and its gate.
    SharedExpert = 2,
    /// The per-layer routing matrices.
    Router = 3,
    /// Gated DeltaNet projections and convolution kernels.
    Gdn = 4,
    /// Sparse-attention projections and the indexer's projection.
    Qsa = 5,
    /// Hyper-connection mixers, inject weights and norms.
    HyperConnection = 6,
    TokenEmbedding = 7,
    OutputHead = 8,
    /// Norm scales and per-head control vectors: small, read by every layer.
    NormSmallState = 9,
    /// The n-gram embedding tables.
    Ngram = 10,
    /// The multi-token-prediction layer, its routed experts included.
    Mtp = 11,
    Vision = 12,
    Other = 13,
}

impl ComponentGroup {
    /// The ordinal the tables store.
    pub fn ordinal(self) -> i32 {
        self as i32
    }

    pub(crate) fn from_ordinal(ordinal: i32) -> Result<ComponentGroup, EdrlError> {
        Ok(match ordinal {
            0 => ComponentGroup::FixedText,
            1 => ComponentGroup::RoutedExpert,
            2 => ComponentGroup::SharedExpert,
            3 => ComponentGroup::Router,
            4 => ComponentGroup::Gdn,
            5 => ComponentGroup::Qsa,
            6 => ComponentGroup::HyperConnection,
            7 => ComponentGroup::TokenEmbedding,
            8 => ComponentGroup::OutputHead,
            9 => ComponentGroup::NormSmallState,
            10 => ComponentGroup::Ngram,
            11 => ComponentGroup::Mtp,
            12 => ComponentGroup::Vision,
            13 => ComponentGroup::Other,
            _ => {
                return Err(EdrlError::invalid(format!(
                    "component group value is outside the supported range: {ordinal}"
                )));
            }
        })
    }
}

/// A fixed object: a tensor in one contiguous file range, with its component group and the CRC-32 (IEEE) of its
/// payload. `FlashTables::parse` has checked a parsed one against its shape and layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: TensorDataType,
    pub format: WeightFormat,
    pub layout: WeightLayout,
    pub group: ComponentGroup,
    /// Where the payload starts in the file.
    pub offset: u64,
    /// The payload's size.
    pub bytes: u64,
    pub crc32: u32,
}

/// One projection of an expert record: a complete NVFP4 tensor at `record_offset` inside the record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertProjection {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: TensorDataType,
    pub format: WeightFormat,
    pub layout: WeightLayout,
    /// Where the projection starts inside the record.
    pub record_offset: u64,
    pub bytes: u64,
}

/// One expert's record: a contiguous file range holding every projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertRecord {
    pub offset: u64,
    pub bytes: u64,
    pub crc32: u32,
}

/// The routed experts of one layer: one record per expert, each a contiguous file range holding every projection, so
/// a cache miss is one copy. Records need not be equal in size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertBank {
    name: String,
    group: ComponentGroup,
    layer: i32,
    projections: Vec<ExpertProjection>,
    records: Vec<ExpertRecord>,
}

impl ExpertBank {
    /// A bank of `records`, one per expert in expert order. Refuses a bank with no projection or no expert, and a
    /// record smaller than the projections it must hold.
    pub fn new(
        name: impl Into<String>,
        group: ComponentGroup,
        layer: i32,
        projections: Vec<ExpertProjection>,
        records: Vec<ExpertRecord>,
    ) -> Result<ExpertBank, EdrlError> {
        let name = name.into();
        if projections.is_empty() || records.is_empty() {
            return Err(EdrlError::invalid(format!(
                "expert bank {name} is empty: {} experts, {} projections",
                records.len(),
                projections.len()
            )));
        }
        let mut record_end: u64 = 0;
        for projection in &projections {
            let end = checked_end(projection.record_offset, projection.bytes, "expert projection")?;
            record_end = record_end.max(end);
        }
        for (expert, record) in records.iter().enumerate() {
            if record.bytes < record_end {
                return Err(EdrlError::invalid(format!(
                    "expert {expert} of {name} has a record of {} bytes, smaller than its projections' {record_end}",
                    record.bytes
                )));
            }
        }
        Ok(ExpertBank {
            name,
            group,
            layer,
            projections,
            records,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn group(&self) -> ComponentGroup {
        self.group
    }

    /// The layer within its stack (the text layers, or the MTP layers).
    pub fn layer(&self) -> i32 {
        self.layer
    }

    pub fn projections(&self) -> &[ExpertProjection] {
        &self.projections
    }

    /// The records, in expert order.
    pub fn records(&self) -> &[ExpertRecord] {
        &self.records
    }

    pub fn expert_count(&self) -> usize {
        self.records.len()
    }

    /// The bytes of the largest record: the slot size that holds any expert of the bank.
    pub fn max_record_bytes(&self) -> u64 {
        self.records.iter().map(|record| record.bytes).max().unwrap_or(0)
    }
}

/// The fixed objects and expert banks of an artifact, in file order, names unique across both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashTables {
    tensors: Vec<FlashTensor>,
    banks: Vec<ExpertBank>,
}

/// An object whose payload lies in the data section.
#[derive(Clone, Copy)]
enum Placed {
    Tensor(usize),
    Record(usize, usize),
}

impl FlashTables {
    /// Parses and validates the tables region `bytes` (exactly [`FlashHeader::tables_range`]).
    ///
    /// Each fixed object must have the payload size its shape, format and layout give; every object (a fixed object
    /// or an expert record) must start on a 256-byte boundary inside the data section, end inside the content, and
    /// share no byte with another; names are unique across tensors and banks.
    pub fn parse(bytes: &[u8], header: &FlashHeader) -> Result<FlashTables, EdrlError> {
        if bytes.len() as u64 != header.tables_size() {
            return Err(EdrlError::Truncated { what: "tables" });
        }
        let mut reader = Reader::new(bytes);
        let mut names = HashSet::new();
        let tensor_count = count(&mut reader, "tensor count", MAX_OBJECTS)?;
        let mut tensors = Vec::with_capacity(bounded(tensor_count, bytes.len(), 1));
        for _ in 0..tensor_count {
            let tensor = read_tensor(&mut reader)?;
            if !names.insert(tensor.name.clone()) {
                return Err(EdrlError::DuplicateName { name: tensor.name });
            }
            tensors.push(tensor);
        }
        let bank_count = count(&mut reader, "expert bank count", MAX_OBJECTS)?;
        let mut banks = Vec::with_capacity(bounded(bank_count, bytes.len(), 1));
        for _ in 0..bank_count {
            let bank = read_bank(&mut reader)?;
            if !names.insert(bank.name.clone()) {
                return Err(EdrlError::DuplicateName { name: bank.name });
            }
            banks.push(bank);
        }
        if reader.remaining() != 0 {
            return Err(EdrlError::invalid(format!(
                "tables contain {} trailing bytes",
                reader.remaining()
            )));
        }
        let tables = FlashTables { tensors, banks };
        tables.check_ranges(header)?;
        Ok(tables)
    }

    /// The fixed objects, in file order.
    pub fn tensors(&self) -> &[FlashTensor] {
        &self.tensors
    }

    /// The expert banks, in file order.
    pub fn banks(&self) -> &[ExpertBank] {
        &self.banks
    }

    pub fn tensor(&self, name: &str) -> Option<&FlashTensor> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    pub fn bank(&self, name: &str) -> Option<&ExpertBank> {
        self.banks.iter().find(|bank| bank.name == name)
    }

    /// The expert records over all banks.
    pub fn record_count(&self) -> usize {
        self.banks.iter().map(ExpertBank::expert_count).sum()
    }

    fn name_of(&self, placed: Placed) -> String {
        match placed {
            Placed::Tensor(index) => self.tensors[index].name.clone(),
            Placed::Record(bank, expert) => format!("{}#{expert}", self.banks[bank].name),
        }
    }

    fn check_ranges(&self, header: &FlashHeader) -> Result<(), EdrlError> {
        let mut objects: Vec<(u64, u64, Placed)> = Vec::with_capacity(self.tensors.len() + self.record_count());
        for (index, tensor) in self.tensors.iter().enumerate() {
            objects.push((tensor.offset, tensor.bytes, Placed::Tensor(index)));
        }
        for (bank_index, bank) in self.banks.iter().enumerate() {
            for (expert, record) in bank.records.iter().enumerate() {
                objects.push((record.offset, record.bytes, Placed::Record(bank_index, expert)));
            }
        }
        for (offset, bytes, placed) in &objects {
            let name = self.name_of(*placed);
            if !offset.is_multiple_of(OBJECT_ALIGNMENT) {
                return Err(EdrlError::invalid(format!(
                    "{name} at {offset} is not 256-byte aligned"
                )));
            }
            let end = checked_end(*offset, *bytes, "object")
                .map_err(|_| EdrlError::invalid(format!("{name} at {offset} overflows")))?;
            if end > header.file_size() {
                return Err(EdrlError::invalid(format!(
                    "{name} at {offset} extends beyond the file"
                )));
            }
        }
        objects.sort_by_key(|(offset, _, _)| *offset);
        let mut previous: Option<(Placed, u64)> = None;
        for (offset, bytes, placed) in objects {
            match previous {
                None if offset < header.data_offset() => {
                    return Err(EdrlError::invalid(format!(
                        "{} at {offset} starts before the data section",
                        self.name_of(placed)
                    )));
                }
                Some((first, end)) if offset < end => {
                    return Err(EdrlError::Overlap {
                        first: self.name_of(first),
                        second: self.name_of(placed),
                    });
                }
                _ => {}
            }
            // `bytes` was added to `offset` without overflow above.
            previous = Some((placed, offset + bytes));
        }
        Ok(())
    }
}

/// The capacity to reserve for `count` items of at least `min_bytes` each in a region of `region` bytes: what the
/// bytes can hold, so a hostile count allocates nothing large.
fn bounded(count: u32, region: usize, min_bytes: usize) -> usize {
    (count as usize).min(region / min_bytes)
}

fn count(reader: &mut Reader<'_>, what: &'static str, max: u32) -> Result<u32, EdrlError> {
    let value = reader.i32(what)?;
    u32::try_from(value)
        .ok()
        .filter(|value| *value <= max)
        .ok_or_else(|| EdrlError::invalid(format!("{what} is out of range: {value} (0 to {max})")))
}

fn name(reader: &mut Reader<'_>, what: &'static str) -> Result<String, EdrlError> {
    let length = reader.i32(what)?;
    let length = usize::try_from(length)
        .ok()
        .filter(|length| (1..=MAX_NAME_BYTES).contains(length))
        .ok_or_else(|| {
            EdrlError::invalid(format!(
                "{what} length is out of range: {length} (1 to {MAX_NAME_BYTES})"
            ))
        })?;
    str::from_utf8(reader.take(length, what)?)
        .map(str::to_owned)
        .map_err(|_| EdrlError::invalid(format!("{what} is not valid UTF-8")))
}

fn shape(reader: &mut Reader<'_>, owner: &str) -> Result<Vec<u64>, EdrlError> {
    let rank = reader.i32("tensor rank")?;
    let rank = usize::try_from(rank)
        .ok()
        .filter(|rank| *rank <= MAX_RANK)
        .ok_or_else(|| EdrlError::invalid(format!("rank of {owner} is out of range: {rank} (0 to {MAX_RANK})")))?;
    let mut shape = Vec::with_capacity(rank);
    for _ in 0..rank {
        let dimension = reader.i64("tensor shape")?;
        match u64::try_from(dimension) {
            Ok(dimension) if dimension > 0 => shape.push(dimension),
            _ => {
                return Err(EdrlError::invalid(format!(
                    "{owner} has an empty dimension: {dimension}"
                )));
            }
        }
    }
    Ok(shape)
}

fn size(reader: &mut Reader<'_>, owner: &str, what: &'static str) -> Result<u64, EdrlError> {
    let value = reader.i64(what)?;
    u64::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| EdrlError::invalid(format!("{owner} has no payload: {what} {value}")))
}

/// The storage a descriptor names: type, format and layout.
fn storage(reader: &mut Reader<'_>) -> Result<(TensorDataType, WeightFormat, WeightLayout), EdrlError> {
    Ok((
        TensorDataType::from_ordinal(reader.i32("tensor data type")?)?,
        WeightFormat::from_ordinal(reader.i32("tensor weight format")?)?,
        WeightLayout::from_ordinal(reader.i32("tensor weight layout")?)?,
    ))
}

/// Whether `bytes` is the size the layout gives for `shape`, as the Java `checkFixedObject`.
fn check_object(
    name: &str,
    shape: &[u64],
    dtype: TensorDataType,
    format: WeightFormat,
    layout: WeightLayout,
    bytes: u64,
) -> Result<(), EdrlError> {
    use WeightFormat as F;
    use WeightLayout as L;
    let unsupported = |reason: String| EdrlError::UnsupportedTensor {
        name: name.to_owned(),
        reason,
    };
    match (format, layout) {
        (_, L::RowInterleavedNvfp4V1) => {
            let &[rows, k] = shape else {
                return Err(unsupported("an interleaved NVFP4 table is rank 2".into()));
            };
            if format != F::Nvfp4 {
                return Err(unsupported("interleaved rows hold NVFP4".into()));
            }
            let layout = NgramLayout::new(rows, k).map_err(|refusal| unsupported(refusal.to_string()))?;
            if layout.byte_size() != bytes {
                return Err(EdrlError::SizeMismatch {
                    name: name.to_owned(),
                    bytes,
                });
            }
            return Ok(());
        }
        (_, L::RowSplitP2e2V1 | L::RowSplitK128Sd4V1) => {
            return Err(unsupported(
                "a compressed layout that version 3 artifacts do not carry".into(),
            ));
        }
        (F::Nvfp4, layout) if layout != L::RowSplitK128V1 => {
            return Err(unsupported("NVFP4 needs the row-split layout".into()));
        }
        (format, layout) if format != F::Nvfp4 && layout != L::ContiguousLeV1 => {
            return Err(unsupported(format!("{format:?} needs the contiguous layout")));
        }
        _ => {}
    }
    match accepts_byte_size(shape, dtype, format, layout, bytes) {
        Ok(true) => Ok(()),
        Ok(false) => Err(EdrlError::SizeMismatch {
            name: name.to_owned(),
            bytes,
        }),
        Err(reason) => Err(unsupported(reason)),
    }
}

fn read_tensor(reader: &mut Reader<'_>) -> Result<FlashTensor, EdrlError> {
    let name = name(reader, "tensor name")?;
    let shape = shape(reader, &name)?;
    let (dtype, format, layout) = storage(reader)?;
    let group = ComponentGroup::from_ordinal(reader.i32("tensor group")?)?;
    let offset = reader.i64("tensor offset")?;
    let offset =
        u64::try_from(offset).map_err(|_| EdrlError::invalid(format!("{name} has a negative offset: {offset}")))?;
    let bytes = size(reader, &name, "tensor size")?;
    let crc32 = reader.u32("tensor CRC-32")?;
    check_object(&name, &shape, dtype, format, layout, bytes)?;
    Ok(FlashTensor {
        name,
        shape,
        dtype,
        format,
        layout,
        group,
        offset,
        bytes,
        crc32,
    })
}

fn read_bank(reader: &mut Reader<'_>) -> Result<ExpertBank, EdrlError> {
    let bank = name(reader, "expert bank name")?;
    let group = ComponentGroup::from_ordinal(reader.i32("expert bank group")?)?;
    let layer = reader.i32("expert bank layer")?;
    let experts = count(reader, "expert count", MAX_EXPERTS)?;
    let projection_count = count(reader, "projection count", MAX_PROJECTIONS)?;
    if experts == 0 || projection_count == 0 {
        return Err(EdrlError::invalid(format!(
            "expert bank {bank} is empty: {experts} experts, {projection_count} projections"
        )));
    }
    let mut projections = Vec::with_capacity(projection_count as usize);
    let mut projection_names = HashSet::new();
    for _ in 0..projection_count {
        let projection = name(reader, "projection name")?;
        if !projection_names.insert(projection.clone()) {
            return Err(EdrlError::invalid(format!("{bank} repeats projection {projection}")));
        }
        let owner = format!("{bank}/{projection}");
        let shape = shape(reader, &owner)?;
        let (dtype, format, layout) = storage(reader)?;
        let record_offset = reader.i64("projection offset")?;
        let bytes = size(reader, &owner, "projection size")?;
        check_object(&owner, &shape, dtype, format, layout, bytes)?;
        if layout == WeightLayout::RowInterleavedNvfp4V1 {
            return Err(EdrlError::UnsupportedTensor {
                name: owner,
                reason: "expert projections use the row-split layout".into(),
            });
        }
        let record_offset = u64::try_from(record_offset)
            .ok()
            .filter(|offset| offset.is_multiple_of(OBJECT_ALIGNMENT))
            .ok_or_else(|| EdrlError::invalid(format!("{owner} is not 256-byte aligned in its record")))?;
        projections.push(ExpertProjection {
            name: projection,
            shape,
            dtype,
            format,
            layout,
            record_offset,
            bytes,
        });
    }
    // A hostile expert count allocates only what the remaining bytes can hold.
    let mut records = Vec::with_capacity((experts as usize).min(reader.remaining() / 20));
    for _ in 0..experts {
        let offset = reader.i64("expert offset")?;
        let bytes = reader.i64("expert size")?;
        let crc32 = reader.u32("expert CRC-32")?;
        match (u64::try_from(offset), u64::try_from(bytes)) {
            (Ok(offset), Ok(bytes)) => records.push(ExpertRecord { offset, bytes, crc32 }),
            _ => {
                return Err(EdrlError::invalid(format!(
                    "expert record of {bank} has a negative offset {offset} or size {bytes}"
                )));
            }
        }
    }
    ExpertBank::new(bank, group, layer, projections, records)
}

/// Encodes the tables region: `int32 tensorCount`, per tensor `name, rank, shape, dtype, format, layout, group,
/// offset, size, CRC-32`, then `int32 bankCount` and per bank `name, group, layer, expertCount, projectionCount`, its
/// projections (`name, rank, shape, dtype, format, layout, offsetInRecord, size`) and one `(offset, size, CRC-32)`
/// per expert, all big-endian with `int32` string lengths.
///
/// Only what the format cannot represent is refused (a name or rank over its limit, a count over `i32`). Whether the
/// ranges and sizes make sense is [`FlashTables::parse`]'s check, so a test can encode a damaged table.
pub fn encode_flash_tables(tensors: &[FlashTensor], banks: &[ExpertBank]) -> Result<Vec<u8>, EdrlError> {
    let mut out = Vec::new();
    put_count(&mut out, tensors.len(), "tensor count")?;
    for tensor in tensors {
        put_name(&mut out, &tensor.name)?;
        put_shape(&mut out, &tensor.name, &tensor.shape)?;
        put_storage(&mut out, tensor.dtype, tensor.format, tensor.layout);
        out.extend_from_slice(&tensor.group.ordinal().to_be_bytes());
        out.extend_from_slice(&put_i64(tensor.offset, &tensor.name)?.to_be_bytes());
        out.extend_from_slice(&put_i64(tensor.bytes, &tensor.name)?.to_be_bytes());
        out.extend_from_slice(&tensor.crc32.to_be_bytes());
    }
    put_count(&mut out, banks.len(), "expert bank count")?;
    for bank in banks {
        put_name(&mut out, &bank.name)?;
        out.extend_from_slice(&bank.group.ordinal().to_be_bytes());
        out.extend_from_slice(&bank.layer.to_be_bytes());
        put_count(&mut out, bank.records.len(), "expert count")?;
        put_count(&mut out, bank.projections.len(), "projection count")?;
        for projection in &bank.projections {
            put_name(&mut out, &projection.name)?;
            put_shape(&mut out, &projection.name, &projection.shape)?;
            put_storage(&mut out, projection.dtype, projection.format, projection.layout);
            out.extend_from_slice(&put_i64(projection.record_offset, &projection.name)?.to_be_bytes());
            out.extend_from_slice(&put_i64(projection.bytes, &projection.name)?.to_be_bytes());
        }
        for record in &bank.records {
            out.extend_from_slice(&put_i64(record.offset, &bank.name)?.to_be_bytes());
            out.extend_from_slice(&put_i64(record.bytes, &bank.name)?.to_be_bytes());
            out.extend_from_slice(&record.crc32.to_be_bytes());
        }
    }
    Ok(out)
}

fn put_count(out: &mut Vec<u8>, count: usize, what: &str) -> Result<(), EdrlError> {
    let count = i32::try_from(count).map_err(|_| EdrlError::invalid(format!("{what} is out of range: {count}")))?;
    out.extend_from_slice(&count.to_be_bytes());
    Ok(())
}

fn put_i64(value: u64, owner: &str) -> Result<i64, EdrlError> {
    i64::try_from(value)
        .map_err(|_| EdrlError::invalid(format!("{owner} has a value over the format's range: {value}")))
}

fn put_name(out: &mut Vec<u8>, name: &str) -> Result<(), EdrlError> {
    if name.is_empty() || name.len() > MAX_NAME_BYTES {
        return Err(EdrlError::invalid(format!(
            "name length is out of range: {} bytes",
            name.len()
        )));
    }
    out.extend_from_slice(&(name.len() as i32).to_be_bytes());
    out.extend_from_slice(name.as_bytes());
    Ok(())
}

fn put_shape(out: &mut Vec<u8>, owner: &str, shape: &[u64]) -> Result<(), EdrlError> {
    if shape.len() > MAX_RANK {
        return Err(EdrlError::invalid(format!(
            "rank of {owner} is out of range: {}",
            shape.len()
        )));
    }
    out.extend_from_slice(&(shape.len() as i32).to_be_bytes());
    for dimension in shape {
        out.extend_from_slice(&put_i64(*dimension, owner)?.to_be_bytes());
    }
    Ok(())
}

fn put_storage(out: &mut Vec<u8>, dtype: TensorDataType, format: WeightFormat, layout: WeightLayout) {
    out.extend_from_slice(&dtype.ordinal().to_be_bytes());
    out.extend_from_slice(&format.ordinal().to_be_bytes());
    out.extend_from_slice(&layout.ordinal().to_be_bytes());
}
