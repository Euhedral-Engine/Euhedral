//! The payload check of a P2E2 tensor (`P2e2Layout.validate`): Q3 entropy coded, rows of `k` codes (`k` a multiple of
//! 1024) in these planes, each starting on a 256-byte boundary:
//!
//! - the primary plane: 2 bits a code, 16 codes to a little-endian word, a row after another;
//! - the row-base plane: a word a row, the payload unit of the row's first BIG code (a code whose two bits are both set);
//! - the scale plane, which the check does not read;
//! - the payload plane: the BIG codes' units, then 80 zero words of padding.
//!
//! The kernels read the payload where the row bases say, so a base that does not follow from the primary plane, or a
//! payload shorter than the primary plane's BIG codes need, would make them read past the tensor. The check counts the
//! BIG codes of each row and compares the running count with the row bases and, in the end, with the payload's
//! capacity.
//!
//! The load cuts a tensor into pieces that arrive in any order, so the check is split in two: [`P2e2Geometry::scan`]
//! reads one piece on its own, and [`P2e2Validation::absorb`] puts the scans of a tensor's pieces together and judges the
//! tensor when its last byte has arrived.

const SLICE: u64 = 1024;
const PAD_WORDS: u64 = 80;
const ALIGNMENT: u64 = 256;
/// The low bit of each 2-bit code of a word.
const LOW_BITS: u32 = 0x5555_5555;

const fn align(value: u64) -> u64 {
    value.div_ceil(ALIGNMENT) * ALIGNMENT
}

/// The planes of one tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct P2e2Geometry {
    rows: u64,
    k: u64,
    bytes: u64,
    row_base: u64,
    payload: u64,
}

impl P2e2Geometry {
    /// The planes of a tensor of `rows` rows of `k` codes in `bytes` bytes, or why its size does not fit its shape.
    pub fn new(rows: u64, k: u64, bytes: u64) -> Result<Self, String> {
        let codes = rows
            .checked_mul(k)
            .filter(|codes| rows > 0 && *codes > 0 && k % SLICE == 0);
        let Some(codes) = codes else {
            return Err(format!(
                "P2E2 layout requires a rank-2 shape whose K is a multiple of {SLICE}, not {rows} by {k}"
            ));
        };
        let row_base = align(codes / 4);
        let scales = align(row_base + 4 * rows);
        let payload = align(scales + codes / 32);
        let minimum = payload + 4 * PAD_WORDS;
        let maximum = minimum + 4 * codes.div_ceil(16);
        if bytes < minimum || bytes > maximum || !(bytes - payload).is_multiple_of(4) {
            return Err("P2E2 tensor size does not match its shape".to_string());
        }
        Ok(Self {
            rows,
            k,
            bytes,
            row_base,
            payload,
        })
    }

    /// Bytes of the primary plane.
    fn primary_bytes(&self) -> u64 {
        self.rows * self.k / 4
    }

    /// Bytes of the payload plane's units: what the BIG codes may take.
    fn capacity_units(&self) -> u64 {
        ((self.bytes - self.payload) / 4 - PAD_WORDS) * 16
    }

    /// What the bytes `from..from + bytes.len()` of the tensor say: the BIG codes they hold for each row of the primary
    /// plane they reach, and the row bases they hold. `from` and the length are multiples of 4, as every piece of the
    /// load is.
    pub fn scan(&self, from: u64, bytes: &[u8]) -> Result<P2e2Scan, String> {
        let end = from + bytes.len() as u64;
        if !from.is_multiple_of(4) || !bytes.len().is_multiple_of(4) || end > self.bytes {
            return Err(format!(
                "P2E2 piece of {} bytes at {from} is not whole words of a tensor of {} bytes",
                bytes.len(),
                self.bytes
            ));
        }
        let word = |at: u64| {
            let at = (at - from) as usize;
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        let row_bytes = self.k / 4;
        let primary_end = end.min(self.primary_bytes());
        let (first_row, big) = if from < primary_end {
            let first = from / row_bytes;
            let last = (primary_end - 1) / row_bytes;
            let mut big = vec![0u64; (last - first + 1) as usize];
            let mut at = from;
            while at < primary_end {
                let value = word(at);
                big[(at / row_bytes - first) as usize] += u64::from((value & (value >> 1) & LOW_BITS).count_ones());
                at += 4;
            }
            (first, big)
        } else {
            (0, Vec::new())
        };
        let bases_from = from.max(self.row_base);
        let bases_end = end.min(self.row_base + 4 * self.rows);
        let first_base = (bases_from - self.row_base) / 4;
        let mut bases = Vec::new();
        if bases_from < bases_end {
            let mut at = bases_from;
            while at < bases_end {
                bases.push(word(at));
                at += 4;
            }
        }
        Ok(P2e2Scan {
            bytes: bytes.len() as u64,
            first_row,
            big,
            first_base,
            bases,
        })
    }
}

/// What one piece of a tensor says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct P2e2Scan {
    bytes: u64,
    first_row: u64,
    /// BIG codes of the primary plane's rows from `first_row` that the piece reaches.
    big: Vec<u64>,
    first_base: u64,
    /// Row bases from `first_base`.
    bases: Vec<u32>,
}

/// The scans of one tensor's pieces so far.
#[derive(Debug)]
pub struct P2e2Validation {
    geometry: P2e2Geometry,
    big: Vec<u64>,
    bases: Vec<u32>,
    received: u64,
}

impl P2e2Validation {
    pub fn new(geometry: P2e2Geometry) -> Self {
        let rows = geometry.rows as usize;
        Self {
            geometry,
            big: vec![0; rows],
            bases: vec![0; rows],
            received: 0,
        }
    }

    pub fn geometry(&self) -> P2e2Geometry {
        self.geometry
    }

    /// Takes in one piece's scan. When it is the tensor's last bytes the tensor is judged: the error is why it is not
    /// valid.
    pub fn absorb(&mut self, scan: P2e2Scan) -> Result<(), String> {
        for (offset, big) in scan.big.iter().enumerate() {
            self.big[scan.first_row as usize + offset] += big;
        }
        for (offset, base) in scan.bases.iter().enumerate() {
            self.bases[scan.first_base as usize + offset] = *base;
        }
        self.received += scan.bytes;
        if self.received < self.geometry.bytes {
            return Ok(());
        }
        let mut units = 0u64;
        for (row, (base, big)) in self.bases.iter().zip(&self.big).enumerate() {
            if u64::from(*base) != units {
                return Err(format!("P2E2 row base of row {row} does not match its primary plane"));
            }
            units += big;
        }
        if units > self.geometry.capacity_units() {
            return Err("P2E2 payload plane is shorter than its primary plane requires".to_string());
        }
        Ok(())
    }
}

/// The check of a whole tensor held in memory.
pub fn validate(payload: &[u8], rows: u64, k: u64) -> Result<(), String> {
    let geometry = P2e2Geometry::new(rows, k, payload.len() as u64)?;
    let scan = geometry.scan(0, payload)?;
    P2e2Validation::new(geometry).absorb(scan)
}

#[cfg(test)]
mod tests;
