//! The device regions the dense part reads and writes: the residual streams and what feeds them, and the scratch of one
//! block. A session allocates one buffer of [`FlashRegions::bytes`] and carves it with [`FlashRegions::over`]; the
//! scratch of a block is carved from [`FlashRegions::layer_scratch`] with the block's own `carve`.
//!
//! Java allocates each region on its own (`Workspace`) and packs the scratch of a block back to back; here every region
//! and every scratch buffer starts at a multiple of [`ALIGNMENT`], so a BF16 linear's 16-byte alignment holds whatever
//! the row count and the low rank are (back to back it holds only when each extent is a multiple of 8 values). A
//! kernel's results do not depend on where its buffers lie.

use euhedral_inference_cuda::DevicePtr;

use super::FlashError;

/// The start of every region and scratch buffer.
pub const ALIGNMENT: u64 = 256;

const BF16: u64 = 2;

/// Hands out consecutive aligned ranges from a base address; with base 0 it sizes them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Carver {
    base: u64,
    used: u64,
}

impl Carver {
    pub(crate) fn new(base: DevicePtr) -> Self {
        Self {
            base: base.get(),
            used: 0,
        }
    }

    /// The next `bytes` bytes, at an aligned offset.
    pub(crate) fn take(&mut self, bytes: u64) -> DevicePtr {
        let start = self.used.next_multiple_of(ALIGNMENT);
        self.used = start + bytes;
        // SAFETY: an address inside the region the caller carves, or a size computed from base 0; carving never
        // dereferences it.
        unsafe { DevicePtr::from_raw(self.base + start) }
    }

    /// The bytes handed out so far, with the padding of the last.
    pub(crate) fn bytes(&self) -> u64 {
        self.used.next_multiple_of(ALIGNMENT)
    }
}

/// The shape of the residual streams the regions hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamShape {
    pub hidden: u32,
    pub streams: u32,
}

impl StreamShape {
    /// Streams of `hidden` values. The embedding kernel reads rows of multiples of 8 values, so `hidden` is one.
    pub fn new(hidden: u32, streams: u32) -> Result<Self, FlashError> {
        if streams == 0 || hidden == 0 || !hidden.is_multiple_of(8) {
            return Err(FlashError::refused(
                "stream shape",
                format!("{streams} streams of {hidden} values: both are positive and the width is a multiple of 8"),
            ));
        }
        u64::from(hidden)
            .checked_mul(u64::from(streams))
            .filter(|width| *width <= u64::from(u32::MAX))
            .ok_or_else(|| {
                FlashError::refused(
                    "stream shape",
                    format!("{streams} streams of {hidden} values exceed 32 bits"),
                )
            })?;
        Ok(Self { hidden, streams })
    }

    /// The values of one row of all streams.
    pub fn state_width(&self) -> u32 {
        self.hidden * self.streams
    }
}

/// The regions of one session's rows, each addressed for up to [`FlashRegions::capacity_rows`] rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashRegions {
    capacity_rows: u32,
    /// The residual streams: `[rows, streams, hidden]`.
    pub state: DevicePtr,
    /// The output of the PLE, added into `state`: `[rows, streams, hidden]`.
    pub ple_output: DevicePtr,
    /// The embedding of the tokens before the streams repeat it, and the mix that feeds a block: `[rows, hidden]`.
    pub embedded: DevicePtr,
    pub mixed: DevicePtr,
    /// What a block wrote, before it is injected into the streams: `[rows, hidden]`.
    pub block_output: DevicePtr,
    /// The last row after the final mixer, which the output head reads: `[hidden]`.
    pub final_mixed: DevicePtr,
    /// The scratch of the mixes, and the scratch of the one block in flight (as large as the largest block's).
    pub hyper_scratch: DevicePtr,
    pub layer_scratch: DevicePtr,
}

impl FlashRegions {
    /// The bytes of the regions for `capacity_rows` rows, with `hyper_scratch_bytes` for the mixes and
    /// `layer_scratch_bytes` for a block.
    pub fn bytes(capacity_rows: u32, shape: StreamShape, hyper_scratch_bytes: u64, layer_scratch_bytes: u64) -> u64 {
        Self::carve(
            DevicePtr::NULL,
            capacity_rows,
            shape,
            hyper_scratch_bytes,
            layer_scratch_bytes,
        )
        .1
    }

    /// The regions laid out over `base`, which holds [`FlashRegions::bytes`] bytes.
    pub fn over(
        base: DevicePtr,
        capacity_rows: u32,
        shape: StreamShape,
        hyper_scratch_bytes: u64,
        layer_scratch_bytes: u64,
    ) -> Result<Self, FlashError> {
        if capacity_rows == 0 || base.is_null() || base.get() % ALIGNMENT != 0 {
            return Err(FlashError::refused(
                "regions",
                format!(
                    "{capacity_rows} rows over {:#x}: rows are positive and the base is a multiple of {ALIGNMENT}",
                    base.get()
                ),
            ));
        }
        Ok(Self::carve(base, capacity_rows, shape, hyper_scratch_bytes, layer_scratch_bytes).0)
    }

    fn carve(
        base: DevicePtr,
        capacity_rows: u32,
        shape: StreamShape,
        hyper_scratch_bytes: u64,
        layer_scratch_bytes: u64,
    ) -> (Self, u64) {
        let rows = u64::from(capacity_rows);
        let streams = rows * u64::from(shape.state_width()) * BF16;
        let single = rows * u64::from(shape.hidden) * BF16;
        let mut carver = Carver::new(base);
        let regions = Self {
            capacity_rows,
            state: carver.take(streams),
            ple_output: carver.take(streams),
            embedded: carver.take(single),
            mixed: carver.take(single),
            block_output: carver.take(single),
            final_mixed: carver.take(u64::from(shape.hidden) * BF16),
            hyper_scratch: carver.take(hyper_scratch_bytes),
            layer_scratch: carver.take(layer_scratch_bytes),
        };
        (regions, carver.bytes())
    }

    /// The rows the regions are sized for.
    pub fn capacity_rows(&self) -> u32 {
        self.capacity_rows
    }
}

/// Bytes of a BF16 `[rows, width]` buffer.
pub(crate) fn bf16_bytes(rows: u32, width: u32) -> u64 {
    u64::from(rows) * u64::from(width) * BF16
}

/// Bytes of an FP32 `[rows, width]` buffer.
pub(crate) fn f32_bytes(rows: u32, width: u32) -> u64 {
    u64::from(rows) * u64::from(width) * 4
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(address: u64) -> DevicePtr {
        // SAFETY: the tests never dereference it.
        unsafe { DevicePtr::from_raw(address) }
    }

    #[test]
    fn regions_are_disjoint_aligned_and_end_inside_the_size() {
        let shape = StreamShape::new(24, 4).unwrap();
        let total = FlashRegions::bytes(5, shape, 1000, 3000);
        let regions = FlashRegions::over(at(0x10000), 5, shape, 1000, 3000).unwrap();
        let spans = [
            (regions.state, 5 * 96 * 2),
            (regions.ple_output, 5 * 96 * 2),
            (regions.embedded, 5 * 24 * 2),
            (regions.mixed, 5 * 24 * 2),
            (regions.block_output, 5 * 24 * 2),
            (regions.final_mixed, 24 * 2),
            (regions.hyper_scratch, 1000),
            (regions.layer_scratch, 3000),
        ];
        let mut end = 0x10000;
        for (start, bytes) in spans {
            assert_eq!(start.get() % ALIGNMENT, 0);
            assert!(start.get() >= end, "{:#x} overlaps the region before it", start.get());
            end = start.get() + bytes;
        }
        assert!(end <= 0x10000 + total);
        assert_eq!(total % ALIGNMENT, 0);
    }

    #[test]
    fn a_shape_or_a_base_that_a_kernel_cannot_read_is_refused() {
        assert!(
            StreamShape::new(20, 4)
                .unwrap_err()
                .to_string()
                .contains("4 streams of 20")
        );
        assert!(StreamShape::new(24, 0).is_err());
        let shape = StreamShape::new(24, 4).unwrap();
        assert!(
            FlashRegions::over(at(0x10008), 5, shape, 0, 0)
                .unwrap_err()
                .to_string()
                .contains("0x10008")
        );
        assert!(FlashRegions::over(DevicePtr::NULL, 5, shape, 0, 0).is_err());
        assert!(FlashRegions::over(at(0x10000), 0, shape, 0, 0).is_err());
    }
}
