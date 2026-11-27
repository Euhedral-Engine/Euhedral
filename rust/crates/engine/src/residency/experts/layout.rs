//! The records of every expert bank under one flat key, checked against the artifact.

use std::io::ErrorKind;

use euhedral_inference_edrl::{ExpertBank, ExpertRecord};

use crate::residency::offload::HOST_ALIGNMENT;

/// Which expert: its bank (a layer's routed experts) and its place in the bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExpertId {
    pub bank: u32,
    pub expert: u32,
}

/// Why a record could not be had.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ExpertError {
    #[error("no expert bank was given")]
    NoBanks,
    #[error("the banks hold {experts} experts: a flat key is at most {limit}")]
    TooManyExperts { experts: u64, limit: u64 },
    #[error("expert {expert} of {bank}: record [{offset}, +{bytes}) lies outside the artifact of {file_bytes} bytes")]
    OutsideFile {
        bank: String,
        expert: usize,
        offset: u64,
        bytes: u64,
        file_bytes: u64,
    },
    #[error("there is no bank {bank}: the layout has {banks}")]
    NoSuchBank { bank: u32, banks: usize },
    #[error("there is no expert {expert} in bank {bank}: it has {experts}")]
    NoSuchExpert { bank: u32, expert: u32, experts: usize },
    #[error("a slot of {slot_bytes} bytes cannot hold a record of {needed} bytes (expert {expert} of {bank})")]
    SlotTooSmall {
        bank: String,
        expert: u32,
        slot_bytes: usize,
        needed: usize,
    },
    #[error("slot {slot} is out of range: the arena has {slots}")]
    SlotOutOfRange { slot: usize, slots: usize },
    #[error("slot {slot} is leased and cannot be leased for {wanted}")]
    SlotBusy { slot: usize, wanted: &'static str },
    #[error("a slot of {bytes} bytes is not a whole number of {block}-byte pages")]
    SlotNotPaged { bytes: usize, block: u64 },
    #[error("an arena of {slots} slots of {bytes} bytes does not fit in memory")]
    ArenaSize { slots: usize, bytes: usize },
    #[error("preload with {readers} readers: it needs at least 1")]
    NoReaders { readers: usize },
    #[error("read_split is {split}: it must be at least 1")]
    BadSplit { split: usize },
    #[error("a tier of {slots} slots cannot serve {shards} shards: each shard needs one")]
    TierTooSmall { slots: usize, shards: usize },
    #[error("the settle of slot {slot} ({what}) does not match the slot's state")]
    BadSettle { slot: usize, what: &'static str },
    #[error("the read of expert {expert} of {bank} was cancelled or lost, so the fill stopped")]
    Unfinished { bank: String, expert: u32 },
    #[error("reading expert {expert} of {bank} failed ({kind:?}): {message}")]
    Read {
        bank: String,
        expert: u32,
        kind: ErrorKind,
        message: String,
    },
    #[error("reading expert {expert} of {bank} moved {got} bytes of the {wanted} asked at offset {offset}")]
    Short {
        bank: String,
        expert: u32,
        offset: u64,
        wanted: usize,
        got: usize,
    },
    #[error(
        "expert {expert} of {bank} is damaged: its record has CRC-32 {found:#010x}, the artifact says {expected:#010x}"
    )]
    Damaged {
        bank: String,
        expert: u32,
        expected: u32,
        found: u32,
    },
}

/// The records of the banks, in bank order and expert order, numbered by one flat key, all inside the artifact.
/// A value of this type is valid.
#[derive(Debug)]
pub struct ExpertLayout {
    names: Vec<String>,
    first_key: Vec<usize>,
    records: Vec<ExpertRecord>,
    max_record_bytes: u64,
    file_bytes: u64,
}

impl ExpertLayout {
    /// The layout of `banks` over an artifact of `file_bytes`. Refuses no banks, too many experts for a key, and a
    /// record that does not lie in the file, naming the bank and expert.
    pub fn new(banks: &[ExpertBank], file_bytes: u64) -> Result<ExpertLayout, ExpertError> {
        if banks.is_empty() {
            return Err(ExpertError::NoBanks);
        }
        let limit = u64::from(u32::MAX);
        let experts: u64 = banks.iter().map(|bank| bank.expert_count() as u64).sum();
        if experts > limit {
            return Err(ExpertError::TooManyExperts { experts, limit });
        }
        let mut layout = ExpertLayout {
            names: Vec::with_capacity(banks.len()),
            first_key: Vec::with_capacity(banks.len() + 1),
            records: Vec::with_capacity(experts as usize),
            max_record_bytes: 0,
            file_bytes,
        };
        for bank in banks {
            layout.names.push(bank.name().to_owned());
            layout.first_key.push(layout.records.len());
            for (expert, record) in bank.records().iter().enumerate() {
                let end = record.offset.checked_add(record.bytes);
                if end.is_none_or(|end| end > file_bytes) {
                    return Err(ExpertError::OutsideFile {
                        bank: bank.name().to_owned(),
                        expert,
                        offset: record.offset,
                        bytes: record.bytes,
                        file_bytes,
                    });
                }
                layout.max_record_bytes = layout.max_record_bytes.max(record.bytes);
                layout.records.push(*record);
            }
        }
        layout.first_key.push(layout.records.len());
        Ok(layout)
    }

    /// How many records the layout holds: the keys run from 0 to this.
    pub fn keys(&self) -> usize {
        self.records.len()
    }

    pub fn banks(&self) -> usize {
        self.names.len()
    }

    /// The artifact's length the records were checked against.
    pub fn file_bytes(&self) -> u64 {
        self.file_bytes
    }

    /// The bytes of the largest record.
    pub fn max_record_bytes(&self) -> u64 {
        self.max_record_bytes
    }

    /// The bytes of a slot that holds any record: the largest, to a whole page.
    pub fn slot_bytes(&self) -> usize {
        self.max_record_bytes.next_multiple_of(HOST_ALIGNMENT) as usize
    }

    /// The name of bank `bank`, or an empty name for a bank the layout lacks.
    pub fn bank_name(&self, bank: u32) -> &str {
        self.names.get(bank as usize).map_or("", String::as_str)
    }

    /// The flat key of `id`, or why there is none.
    pub fn key(&self, id: ExpertId) -> Result<usize, ExpertError> {
        let bank = id.bank as usize;
        let Some(&first) = self.first_key.get(bank).filter(|_| bank < self.names.len()) else {
            return Err(ExpertError::NoSuchBank {
                bank: id.bank,
                banks: self.names.len(),
            });
        };
        let experts = self.first_key[bank + 1] - first;
        if id.expert as usize >= experts {
            return Err(ExpertError::NoSuchExpert {
                bank: id.bank,
                expert: id.expert,
                experts,
            });
        }
        Ok(first + id.expert as usize)
    }

    /// The bank and expert of a key below [`keys`](Self::keys).
    pub fn id_of(&self, key: usize) -> ExpertId {
        let bank = self.first_key.partition_point(|&first| first <= key) - 1;
        ExpertId {
            bank: bank as u32,
            expert: (key - self.first_key[bank]) as u32,
        }
    }

    /// The record of `id`.
    pub fn record(&self, id: ExpertId) -> Result<ExpertRecord, ExpertError> {
        self.key(id).map(|key| self.records[key])
    }
}

/// Which of `shards` shards owns `key`: a multiplicative hash, the same on every platform, so a tier and the cache shard
/// that owns it agree.
pub fn shard_of(key: usize, shards: usize) -> usize {
    let mut hash = (key as u32).wrapping_mul(0x9E37_79B1);
    hash ^= hash >> 15;
    ((hash & 0x7fff_ffff) as usize) % shards
}
