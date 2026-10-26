use crate::error::EdrlError;

/// A big-endian cursor over caller-supplied bytes. Every read names the field it reads, so a short buffer says where.
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    pub(crate) fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    pub(crate) fn take(&mut self, length: usize, what: &'static str) -> Result<&'a [u8], EdrlError> {
        if length > self.remaining() {
            return Err(EdrlError::Truncated { what });
        }
        let taken = &self.bytes[self.position..self.position + length];
        self.position += length;
        Ok(taken)
    }

    fn array<const N: usize>(&mut self, what: &'static str) -> Result<[u8; N], EdrlError> {
        let mut out = [0; N];
        out.copy_from_slice(self.take(N, what)?);
        Ok(out)
    }

    pub(crate) fn u8(&mut self, what: &'static str) -> Result<u8, EdrlError> {
        Ok(self.array::<1>(what)?[0])
    }

    pub(crate) fn i32(&mut self, what: &'static str) -> Result<i32, EdrlError> {
        Ok(i32::from_be_bytes(self.array(what)?))
    }

    pub(crate) fn u32(&mut self, what: &'static str) -> Result<u32, EdrlError> {
        Ok(u32::from_be_bytes(self.array(what)?))
    }

    pub(crate) fn i64(&mut self, what: &'static str) -> Result<i64, EdrlError> {
        Ok(i64::from_be_bytes(self.array(what)?))
    }

    pub(crate) fn f64(&mut self, what: &'static str) -> Result<f64, EdrlError> {
        Ok(f64::from_be_bytes(self.array(what)?))
    }
}

/// `value` as a non-negative `u64`, or an error naming `what`: the Java `long` fields are never negative.
pub(crate) fn non_negative(value: i64, what: &'static str) -> Result<u64, EdrlError> {
    u64::try_from(value).map_err(|_| EdrlError::invalid(format!("{what} range overflows or is negative")))
}

/// The end of the byte range `[offset, offset + size)`, rejecting an overflow.
pub(crate) fn checked_end(offset: u64, size: u64, what: &'static str) -> Result<u64, EdrlError> {
    offset
        .checked_add(size)
        .ok_or_else(|| EdrlError::invalid(format!("{what} range overflows or is negative")))
}
