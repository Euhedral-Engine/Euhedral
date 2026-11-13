//! The embedding launches: the host dispatch of `native/src/host/q3_embedding.c`, as plain functions from a shape to
//! the launch the wrapper would issue. The size checks of the C wrappers (the tensor geometry and the grid limit) are
//! ported with it.

use crate::DevicePtr;
use crate::kernel;
use crate::launch::PlanError;
use crate::launch::catalog::{Arguments, KernelId, Launch};

const EMBEDDING: KernelId = kernel!("euhedral_q3_embedding");
const EMBEDDING_P2E2: KernelId = kernel!("euhedral_q3_p2e2_embedding");

/// The grid limit of both wrappers (`euhedral_cuda_embed_q3` and `euhedral_cuda_embed_q3_p2e2` in `q3_embedding.c`).
const MAX_GRID: u64 = 2_147_483_647;

/// The shape of `euhedral_cuda_embed_q3` and `euhedral_cuda_embed_q3_p2e2` (`q3_embedding.c`).
#[derive(Clone, Copy, Debug)]
pub struct EmbedShape {
    pub token_ids: DevicePtr,
    pub embedding: DevicePtr,
    pub hidden: DevicePtr,
    pub token_count: u32,
    pub vocabulary_size: u32,
    pub hidden_size: u32,
    /// The byte size of the embedding tensor, checked against the size its geometry implies.
    pub embedding_bytes: u64,
}

/// `q3_embedding_geometry`, `q3_embedding.c`: the offset of the scale plane and the expected byte size, or
/// `None` when a product overflows 64 bits.
fn q3_geometry(vocabulary_size: u32, hidden_size: u32) -> Option<(u64, u64)> {
    let padded_hidden = u64::from(hidden_size).div_ceil(128) * 128;
    let groups_per_row = padded_hidden / 64;
    let group_count = u64::from(vocabulary_size).checked_mul(groups_per_row)?;
    let base_byte_size = group_count.checked_mul(24)?;
    let scale_byte_size = group_count.checked_mul(2)?;
    if base_byte_size > u64::MAX - 255 {
        return None;
    }
    let scale_offset = (base_byte_size + 255) & !255u64;
    if scale_offset > u64::MAX - scale_byte_size {
        return None;
    }
    Some((scale_offset, scale_offset + scale_byte_size))
}

/// `euhedral_cuda_embed_q3`, `q3_embedding.c`: one launch of `tokens * ceil(hidden / 64 / 4)` blocks of 128.
pub fn plan_embed_q3(shape: &EmbedShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    let s = shape;
    if s.token_ids.is_null()
        || s.embedding.is_null()
        || s.hidden.is_null()
        || s.token_count == 0
        || s.vocabulary_size == 0
        || s.hidden_size == 0
        || !s.hidden_size.is_multiple_of(64)
    {
        return Err(PlanError::InvalidArgument);
    }
    let Some((scale_offset, expected_bytes)) = q3_geometry(s.vocabulary_size, s.hidden_size) else {
        return Err(PlanError::SizeOverflow);
    };
    if s.embedding_bytes != expected_bytes {
        return Err(PlanError::FormatMismatch);
    }
    // euhedral_cuda_embed_q3 in q3_embedding.c.
    let groups_per_token = u64::from(s.hidden_size) / 64;
    let groups_per_block = groups_per_token.div_ceil(4);
    let grid_size = u64::from(s.token_count) * groups_per_block;
    if grid_size == 0 || grid_size > MAX_GRID {
        return Err(PlanError::SizeOverflow);
    }
    let args = Arguments::new()
        .pointer(s.token_ids)
        .pointer(s.embedding)
        .pointer(s.hidden)
        .u32(s.token_count)
        .u32(s.vocabulary_size)
        .u32(s.hidden_size)
        .u64(scale_offset);
    sink(Launch::new(EMBEDDING, [grid_size as u32, 1, 1], [128, 1, 1], 0, args));
    Ok(())
}

/// The P2E2 tensor slice (`EUHEDRAL_Q3_P2E2_SLICE`, `q3_p2e2_geometry.h`).
const P2E2_SLICE: u32 = 1024;
/// `EUHEDRAL_Q3_P2E2_PAD_WORDS`, `q3_p2e2_geometry.h`.
const P2E2_PAD_WORDS: u64 = 80;

const fn align256(value: u64) -> u64 {
    (value + 255) & !255u64
}

/// `euhedral_q3_p2e2_geometry`, `q3_p2e2_geometry.h`: a byte size is valid between an empty and a full payload.
fn p2e2_geometry(rows: u32, in_features: u32, byte_size: u64) -> Result<(), PlanError> {
    if rows == 0 || in_features == 0 {
        return Err(PlanError::InvalidArgument);
    }
    if !in_features.is_multiple_of(P2E2_SLICE) {
        return Err(PlanError::FormatMismatch);
    }
    // q3_p2e2_geometry.h.
    let codes = u64::from(rows) * u64::from(in_features);
    let base = align256(codes / 4);
    let scale = align256(base + 4 * u64::from(rows));
    let payload = align256(scale + codes / 32);
    let pad = 4 * P2E2_PAD_WORDS;
    let full = payload + 4 * codes.div_ceil(16) + pad;
    if byte_size < payload + pad || byte_size > full || !(byte_size - payload).is_multiple_of(4) {
        return Err(PlanError::FormatMismatch);
    }
    Ok(())
}

/// `euhedral_cuda_embed_q3_p2e2`, `q3_embedding.c`: one launch of `ceil(tokens / 4)` blocks of 128.
pub fn plan_embed_q3_p2e2(shape: &EmbedShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    let s = shape;
    if s.token_ids.is_null()
        || s.embedding.is_null()
        || s.hidden.is_null()
        || s.token_count == 0
        || s.vocabulary_size == 0
        || s.hidden_size == 0
        || s.hidden.get() & 15 != 0
        || s.embedding.get() & 15 != 0
    {
        return Err(PlanError::InvalidArgument);
    }
    p2e2_geometry(s.vocabulary_size, s.hidden_size, s.embedding_bytes)?;
    // euhedral_cuda_embed_q3_p2e2 in q3_embedding.c.
    let grid_size = u64::from(s.token_count).div_ceil(4);
    if grid_size > MAX_GRID {
        return Err(PlanError::SizeOverflow);
    }
    let args = Arguments::new()
        .pointer(s.token_ids)
        .pointer(s.embedding)
        .pointer(s.hidden)
        .u32(s.token_count)
        .u32(s.vocabulary_size)
        .u32(s.hidden_size);
    sink(Launch::new(
        EMBEDDING_P2E2,
        [grid_size as u32, 1, 1],
        [128, 1, 1],
        0,
        args,
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ptr(address: u64) -> DevicePtr {
        // SAFETY: the planner never dereferences a pointer.
        unsafe { DevicePtr::from_raw(address) }
    }

    /// The byte size of a Q3 tensor of `vocabulary` rows of `hidden` features, worked by hand for the tests below.
    fn q3(tokens: u32, vocabulary: u32, hidden: u32, bytes: u64) -> EmbedShape {
        EmbedShape {
            token_ids: ptr(0x100),
            embedding: ptr(0x200),
            hidden: ptr(0x300),
            token_count: tokens,
            vocabulary_size: vocabulary,
            hidden_size: hidden,
            embedding_bytes: bytes,
        }
    }

    fn plan_q3(shape: &EmbedShape) -> (Result<(), PlanError>, Vec<Launch>) {
        let mut launches = Vec::new();
        (plan_embed_q3(shape, &mut |l| launches.push(l)), launches)
    }

    fn plan_p2e2(shape: &EmbedShape) -> (Result<(), PlanError>, Vec<Launch>) {
        let mut launches = Vec::new();
        (plan_embed_q3_p2e2(shape, &mut |l| launches.push(l)), launches)
    }

    #[test]
    fn q3_geometry_by_hand() {
        // vocabulary 1000, hidden 192: padded 256, 4 groups per row, 4000 groups; base 4000 * 24 = 96000, which rounds
        // up to 96000 (375 * 256); scales 8000 bytes; total 104000.
        assert_eq!(q3_geometry(1000, 192), Some((96_000, 104_000)));
        // vocabulary 3, hidden 64: padded 128, 2 groups per row, 6 groups; base 144 rounds to 256; scales 12.
        assert_eq!(q3_geometry(3, 64), Some((256, 268)));
        // The products fit in 64 bits for any u32 shape, so only the final sums could overflow, and cannot either.
        assert!(q3_geometry(u32::MAX, u32::MAX).is_some());
    }

    #[test]
    fn embed_q3_grid_and_arguments() {
        // hidden 192: 3 groups per token, (3 + 3) / 4 = 1 block each; 5 tokens: 5 blocks.
        let (result, launches) = plan_q3(&q3(5, 1000, 192, 104_000));
        assert_eq!(result, Ok(()));
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].kernel, kernel!("euhedral_q3_embedding"));
        assert_eq!((launches[0].grid, launches[0].block), ([5, 1, 1], [128, 1, 1]));
        assert!(!launches[0].programmatic);
        // euhedral_cuda_embed_q3 in q3_embedding.c: three pointers, three u32, the scale offset as u64.
        assert_eq!(launches[0].args.words(), &[0x100, 0x200, 0x300, 5, 1000, 192, 96_000]);
        assert_eq!(launches[0].args.sizes(), &[8, 8, 8, 4, 4, 4, 8]);
        // hidden 256: 4 groups, 1 block; hidden 320: 5 groups, 2 blocks; hidden 5120: 80 groups, 20 blocks.
        for (hidden, per_token) in [(64u32, 1u32), (256, 1), (320, 2), (5120, 20)] {
            let groups = u64::from(hidden.div_ceil(128) * 2);
            let bytes = {
                let g = 7 * groups;
                (g * 24).div_ceil(256) * 256 + g * 2
            };
            let (result, launches) = plan_q3(&q3(3, 7, hidden, bytes));
            assert_eq!(result, Ok(()), "hidden {hidden}");
            assert_eq!(launches[0].grid, [3 * per_token, 1, 1], "hidden {hidden}");
        }
    }

    #[test]
    fn embed_q3_validation_and_size_limits() {
        let ok = q3(1, 3, 64, 268);
        assert_eq!(plan_q3(&ok).0, Ok(()));
        // A byte size one off is a format mismatch.
        assert_eq!(plan_q3(&q3(1, 3, 64, 267)), (Err(PlanError::FormatMismatch), vec![]));
        assert_eq!(plan_q3(&q3(1, 3, 64, 269)).0, Err(PlanError::FormatMismatch));
        // A hidden size that is not a multiple of 64 is refused first.
        assert_eq!(plan_q3(&q3(1, 3, 96, 268)).0, Err(PlanError::InvalidArgument));
        for bad in [
            EmbedShape {
                token_ids: DevicePtr::NULL,
                ..ok
            },
            EmbedShape {
                embedding: DevicePtr::NULL,
                ..ok
            },
            EmbedShape {
                hidden: DevicePtr::NULL,
                ..ok
            },
            EmbedShape { token_count: 0, ..ok },
            EmbedShape {
                vocabulary_size: 0,
                ..ok
            },
            EmbedShape { hidden_size: 0, ..ok },
        ] {
            assert_eq!(plan_q3(&bad), (Err(PlanError::InvalidArgument), vec![]));
        }
        // The grid is tokens * ceil(groups / 4): hidden 64 gives one block a token, so 2^31 - 1 tokens fit...
        let tokens = i32::MAX as u32;
        let (result, launches) = plan_q3(&q3(tokens, 1, 64, 256 + 4));
        // (one row: 2 groups, base 48 -> 256, scales 4 -> 260)
        assert_eq!(result, Ok(()));
        assert_eq!(launches[0].grid, [tokens, 1, 1]);
        // ...and 2^31 do not.
        assert_eq!(plan_q3(&q3(1 << 31, 1, 64, 260)).0, Err(PlanError::SizeOverflow));
        // Two blocks per token (hidden 512: 8 groups, 2 blocks) overflow at 2^30 tokens.
        // 8 groups: base 192 rounds up to 256, scales 16.
        let rows_bytes = 256 + 16;
        assert_eq!(
            plan_q3(&q3(1 << 30, 1, 512, rows_bytes)).0,
            Err(PlanError::SizeOverflow)
        );
        assert_eq!(plan_q3(&q3((1 << 30) - 1, 1, 512, rows_bytes)).0, Ok(()));
    }

    /// A P2E2 tensor of `rows` x `features`, with byte sizes worked by hand: `features` = 1024 and `rows` = 4 has 4096
    /// codes: base 1024, scale 1024 + 16 = 1040 -> 1280, payload 1280 + 128 = 1408 -> 1536; pad 320; full
    /// 1536 + 4 * 256 + 320 = 2880.
    fn p2e2(tokens: u32, bytes: u64) -> EmbedShape {
        EmbedShape {
            token_ids: ptr(0x100),
            embedding: ptr(0x200),
            hidden: ptr(0x300),
            token_count: tokens,
            vocabulary_size: 4,
            hidden_size: 1024,
            embedding_bytes: bytes,
        }
    }

    #[test]
    fn embed_p2e2_geometry_range_and_grid() {
        // q3_p2e2_geometry.h: valid from payload + pad (1536 + 320 = 1856) to full (2880), in steps of 4.
        for (bytes, ok) in [
            (1855, false),
            (1856, true),
            (1860, true),
            (1858, false),
            (2880, true),
            (2884, false),
            (2877, false),
        ] {
            let (result, launches) = plan_p2e2(&p2e2(1, bytes));
            assert_eq!(result.is_ok(), ok, "bytes {bytes}");
            if !ok {
                assert_eq!((result, launches.len()), (Err(PlanError::FormatMismatch), 0));
            }
        }
        // euhedral_cuda_embed_q3_p2e2 in q3_embedding.c: ceil(tokens / 4) blocks of 128.
        for (tokens, grid) in [(1, 1), (4, 1), (5, 2), (8, 2), (9, 3), (1000, 250)] {
            let (result, launches) = plan_p2e2(&p2e2(tokens, 2000));
            assert_eq!(result, Ok(()));
            assert_eq!(launches[0].kernel, kernel!("euhedral_q3_p2e2_embedding"));
            assert_eq!((launches[0].grid, launches[0].block), ([grid, 1, 1], [128, 1, 1]));
            assert_eq!(launches[0].shared_bytes, 0);
            assert!(!launches[0].programmatic);
        }
        let (_, launches) = plan_p2e2(&p2e2(9, 2000));
        // euhedral_cuda_embed_q3_p2e2 in q3_embedding.c: three pointers, three u32; no scale offset.
        assert_eq!(launches[0].args.words(), &[0x100, 0x200, 0x300, 9, 4, 1024]);
        assert_eq!(launches[0].args.sizes(), &[8, 8, 8, 4, 4, 4]);
        // A hidden size that is not a multiple of 1024 is a format mismatch; zero rows is invalid.
        assert_eq!(
            plan_p2e2(&EmbedShape {
                hidden_size: 1000,
                ..p2e2(1, 2000)
            })
            .0,
            Err(PlanError::FormatMismatch)
        );
        assert_eq!(
            plan_p2e2(&EmbedShape {
                vocabulary_size: 0,
                ..p2e2(1, 2000)
            })
            .0,
            Err(PlanError::InvalidArgument)
        );
    }

    #[test]
    fn embed_p2e2_alignment_and_nulls() {
        // euhedral_cuda_embed_q3_p2e2 in q3_embedding.c: the hidden and embedding pointers must be 16-byte aligned; token ids need not be.
        let base = p2e2(1, 2000);
        assert_eq!(
            plan_p2e2(&EmbedShape {
                token_ids: ptr(0x104),
                ..base
            })
            .0,
            Ok(())
        );
        assert_eq!(
            plan_p2e2(&EmbedShape {
                hidden: ptr(0x308),
                ..base
            })
            .0,
            Err(PlanError::InvalidArgument)
        );
        assert_eq!(
            plan_p2e2(&EmbedShape {
                embedding: ptr(0x208),
                ..base
            })
            .0,
            Err(PlanError::InvalidArgument)
        );
        for bad in [
            EmbedShape {
                token_ids: DevicePtr::NULL,
                ..base
            },
            EmbedShape {
                embedding: DevicePtr::NULL,
                ..base
            },
            EmbedShape {
                hidden: DevicePtr::NULL,
                ..base
            },
            EmbedShape { token_count: 0, ..base },
            EmbedShape { hidden_size: 0, ..base },
        ] {
            assert_eq!(plan_p2e2(&bad), (Err(PlanError::InvalidArgument), vec![]));
        }
        // Tokens past u32 cannot occur; the largest, u32::MAX, gives 2^30 blocks.
        let (result, launches) = plan_p2e2(&p2e2(u32::MAX, 2000));
        assert_eq!(result, Ok(()));
        assert_eq!(launches[0].grid, [1 << 30, 1, 1]);
    }
}
