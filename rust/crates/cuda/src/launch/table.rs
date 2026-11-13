//! The table-driven launches: `euhedral_cuda_linear_launch` (`native/src/host/linear_kernels.c`) and
//! `euhedral_cuda_qwen4_launch` (`native/src/host/qwen4_ops.c`), which both run
//! `euhedral_kernel_table_launch` (`native/src/host/kernel_table.c`).
//!
//! The caller of a table launch chooses the geometry; the launch takes its arguments as 8-byte words plus the byte
//! size (4 or 8) each is declared with, and the host narrows a 4-byte word to `u32`. The function maps a table entry,
//! a geometry and the words to a [`Launch`] of the catalog's kernel with typed arguments, and refuses what the table
//! launcher refuses without asking the device. Two refusals need the compiled kernel and stay with the consumer of the
//! launch: an argument count different from the kernel's parameter count, and a size different from the parameter's
//! (both `EUHEDRAL_CUDA_INVALID_ARGUMENT` in `euhedral_kernel_table_launch`), and a kernel the device lacks when the planner is told
//! the device has no architecture target.

use crate::launch::PlanError;
use crate::launch::catalog::{Arguments, KernelSpec, Launch, MAX_ARGUMENTS, linear_table, lookup, qwen4_table};

/// The kernel tables a launch can go to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Table {
    /// The shared linear table (`euhedral_cuda_linear_launch`).
    Linear,
    /// The Flash-Next table (`euhedral_cuda_qwen4_launch`), which holds the row-exact BF16 GEMMs.
    Qwen4,
}

/// The geometry of a table launch, chosen by its caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    /// Dynamic shared bytes.
    pub shared_bytes: u32,
}

impl Table {
    /// How many kernels the table has.
    pub fn len(self) -> usize {
        self.entries().len()
    }

    pub fn is_empty(self) -> bool {
        self.entries().is_empty()
    }

    fn entries(self) -> &'static [KernelSpec] {
        match self {
            Table::Linear => linear_table(),
            Table::Qwen4 => qwen4_table(),
        }
    }
}

/// `euhedral_kernel_table_launch` (`kernel_table.c`), as far as the host knows: the launch of entry `kernel` of
/// `table`, pushed into the sink.
///
/// `architecture_target` is whether the device compiled the architecture-specific modules; the kernel table, which
/// knows what loaded, supplies it.
pub fn plan_table_launch(
    table: Table,
    kernel: i32,
    geometry: Geometry,
    words: &[u64],
    sizes: &[u8],
    architecture_target: bool,
    sink: &mut impl FnMut(Launch),
) -> Result<(), PlanError> {
    let entries = table.entries();
    let [grid_x, grid_y, grid_z] = geometry.grid;
    let [block_x, block_y, block_z] = geometry.block;
    // euhedral_kernel_table_launch: the index, the arrays, the count (one length here) and every dimension.
    if kernel < 0
        || kernel as usize >= entries.len()
        || words.len() != sizes.len()
        || words.len() > MAX_ARGUMENTS
        || grid_x == 0
        || grid_y == 0
        || grid_z == 0
        || block_x == 0
        || block_y == 0
        || block_z == 0
    {
        return Err(PlanError::InvalidArgument);
    }
    let Some(id) = lookup(entries[kernel as usize].name) else {
        return Err(PlanError::InvalidArgument);
    };
    // `euhedral_kernel_table_launch` checks that the kernel is loaded before it checks the arguments, so a kernel
    // the device lacks answers `ROUTE_UNAVAILABLE` whatever the arguments are. The architecture-specific modules
    // exist on a device with the architecture target only.
    if id.spec().architecture_specific() && !architecture_target {
        return Err(PlanError::RouteUnavailable);
    }
    // `euhedral_kernel_table_launch`: the declared parameter is 4 or 8 bytes, so another size cannot match it.
    if sizes.iter().any(|&size| size != 4 && size != 8) {
        return Err(PlanError::InvalidArgument);
    }
    let mut args = Arguments::new();
    for (&word, &size) in words.iter().zip(sizes) {
        // euhedral_kernel_table_launch: a 4-byte argument is the word narrowed to u32.
        args = if size == 8 {
            args.u64(word)
        } else {
            args.u32(word as u32)
        };
    }
    sink(Launch::new(
        id,
        geometry.grid,
        geometry.block,
        geometry.shared_bytes,
        args,
    ));
    Ok(())
}

/// `euhedral_cuda_linear_launch` (`linear_kernels.c`). `architecture_target` is whether the device compiled the
/// architecture-specific modules; a table kernel of such a module is unavailable without it. The kernel table
/// supplies it, as the one that knows which modules loaded.
pub fn plan_linear_launch(
    kernel: i32,
    geometry: Geometry,
    words: &[u64],
    sizes: &[u8],
    architecture_target: bool,
    sink: &mut impl FnMut(Launch),
) -> Result<(), PlanError> {
    plan_table_launch(Table::Linear, kernel, geometry, words, sizes, architecture_target, sink)
}

/// `euhedral_cuda_qwen4_launch` (`qwen4_ops.c`). `architecture_target` is as for [`plan_linear_launch`].
pub fn plan_qwen4_launch(
    kernel: i32,
    geometry: Geometry,
    words: &[u64],
    sizes: &[u8],
    architecture_target: bool,
    sink: &mut impl FnMut(Launch),
) -> Result<(), PlanError> {
    plan_table_launch(Table::Qwen4, kernel, geometry, words, sizes, architecture_target, sink)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel;

    const GEOMETRY: Geometry = Geometry {
        grid: [3, 2, 1],
        block: [128, 1, 1],
        shared_bytes: 4096,
    };

    fn plan(
        table: Table,
        kernel: i32,
        geometry: Geometry,
        words: &[u64],
        sizes: &[u8],
    ) -> (Result<(), PlanError>, Vec<Launch>) {
        let mut launches = Vec::new();
        let result = plan_table_launch(table, kernel, geometry, words, sizes, true, &mut |l| launches.push(l));
        (result, launches)
    }

    #[test]
    fn table_sizes_are_the_native_tables() {
        // The kernel table in linear_kernels.c has 11 entries; the Flash-Next kernel table in qwen4_ops.c has 36.
        assert_eq!((Table::Linear.len(), Table::Qwen4.len()), (11, 36));
    }

    #[test]
    fn linear_entries_map_to_their_named_kernels() {
        // The table's order is the ABI (the kernel table in linear_kernels.c): index 4 is the epilogue, 7 the reference, 10 the
        // embedding.
        for (index, name) in [
            (0, "euhedral_trellis_prepass"),
            (1, "euhedral_trellis_mma"),
            (4, "euhedral_trellis_epilogue"),
            (7, "euhedral_trellis_reference"),
            (9, "euhedral_trellis_mma_rows64_fp8"),
            (10, "euhedral_nvfp4_embedding"),
        ] {
            let (result, launches) = plan(Table::Linear, index, GEOMETRY, &[], &[]);
            assert_eq!(result, Ok(()));
            assert_eq!(launches.len(), 1);
            assert_eq!(launches[0].kernel.name(), name);
            assert_eq!(launches[0].grid, [3, 2, 1]);
            assert_eq!(launches[0].block, [128, 1, 1]);
            assert_eq!(launches[0].shared_bytes, 4096);
            // Table kernels are never launched programmatically (euhedral_kernel_table_launch passes no attribute).
            assert!(!launches[0].programmatic);
        }
        // The first, the second and the last entry of the Flash-Next kernel table in qwen4_ops.c.
        for (index, name) in [
            (0, "euhedral_q4_linear_bf16"),
            (1, "euhedral_q4_grouped_rms_norm_bf16"),
            (35, "euhedral_q4_linear_split8_bf16"),
        ] {
            let (result, launches) = plan_qwen4(index);
            assert_eq!(result, Ok(()));
            assert_eq!(launches[0].kernel.name(), name);
        }
    }

    fn plan_qwen4(kernel: i32) -> (Result<(), PlanError>, Vec<Launch>) {
        plan(Table::Qwen4, kernel, GEOMETRY, &[], &[])
    }

    #[test]
    fn words_are_typed_by_size_and_narrowed() {
        // euhedral_kernel_table_launch: size 8 keeps the word, size 4 is `(uint32_t)word`.
        let words = [0x1234_5678_9abc_def0, 0x1_2345_6789, f32::to_bits(0.5) as u64, 7];
        let sizes = [8, 4, 4, 8];
        let (result, launches) = plan(Table::Linear, 3, GEOMETRY, &words, &sizes);
        assert_eq!(result, Ok(()));
        let args = launches[0].args;
        assert_eq!(args.words(), &[0x1234_5678_9abc_def0, 0x2345_6789, 0x3f00_0000, 7]);
        assert_eq!(args.sizes(), &[8, 4, 4, 8]);
        assert_eq!(args.f32_at(2), Some(0.5));
        assert_eq!(args.f64_at(2), None);
        // Sixteen kernel arguments of 32 are fine, 32 are the most, 33 are refused.
        for (count, expected) in [(16, Ok(())), (32, Ok(())), (33, Err(PlanError::InvalidArgument))] {
            let words = vec![1u64; count];
            let sizes = vec![4u8; count];
            let (result, launches) = plan(Table::Linear, 0, GEOMETRY, &words, &sizes);
            assert_eq!(result, expected);
            assert_eq!(launches.len(), usize::from(expected.is_ok()));
        }
    }

    #[test]
    fn the_launcher_refusals() {
        // euhedral_kernel_table_launch: the argument-count and the index checks.
        for (table, kernel, last_ok) in [(Table::Linear, 11, 10), (Table::Qwen4, 36, 35)] {
            assert_eq!(plan(table, last_ok, GEOMETRY, &[], &[]).0, Ok(()));
            assert_eq!(
                plan(table, kernel, GEOMETRY, &[], &[]),
                (Err(PlanError::InvalidArgument), vec![])
            );
            assert_eq!(
                plan(table, -1, GEOMETRY, &[], &[]),
                (Err(PlanError::InvalidArgument), vec![])
            );
        }
        for geometry in [
            Geometry {
                grid: [0, 1, 1],
                ..GEOMETRY
            },
            Geometry {
                grid: [1, 0, 1],
                ..GEOMETRY
            },
            Geometry {
                grid: [1, 1, 0],
                ..GEOMETRY
            },
            Geometry {
                block: [0, 1, 1],
                ..GEOMETRY
            },
            Geometry {
                block: [1, 0, 1],
                ..GEOMETRY
            },
            Geometry {
                block: [1, 1, 0],
                ..GEOMETRY
            },
        ] {
            assert_eq!(
                plan(Table::Linear, 0, geometry, &[], &[]),
                (Err(PlanError::InvalidArgument), vec![])
            );
        }
        // A width other than 4 or 8, or arrays of different lengths, cannot match a parameter.
        assert_eq!(
            plan(Table::Linear, 0, GEOMETRY, &[1], &[2]).0,
            Err(PlanError::InvalidArgument)
        );
        assert_eq!(
            plan(Table::Linear, 0, GEOMETRY, &[1], &[0]).0,
            Err(PlanError::InvalidArgument)
        );
        assert_eq!(
            plan(Table::Linear, 0, GEOMETRY, &[1, 2], &[4]).0,
            Err(PlanError::InvalidArgument)
        );
        // Shared bytes are the caller's and pass through untouched.
        let (_, launches) = plan(
            Table::Linear,
            7,
            Geometry {
                shared_bytes: 69632,
                ..GEOMETRY
            },
            &[],
            &[],
        );
        assert_eq!(launches[0].shared_bytes, 69632);
    }

    #[test]
    fn a_kernel_the_device_lacks_is_unavailable_before_its_arguments_are_checked() {
        // Linear entry 9 is an FP8 kernel of an architecture-specific module.
        let mut launches = Vec::new();
        let mut run = |target: bool, words: &[u64], sizes: &[u8]| {
            plan_linear_launch(9, GEOMETRY, words, sizes, target, &mut |l| launches.push(l))
        };
        assert_eq!(run(false, &[], &[]), Err(PlanError::RouteUnavailable));
        // A width no parameter has would be refused as an argument error on a device that has the kernel.
        assert_eq!(run(false, &[1], &[2]), Err(PlanError::RouteUnavailable));
        assert_eq!(run(true, &[1], &[2]), Err(PlanError::InvalidArgument));
        assert_eq!(run(true, &[], &[]), Ok(()));
        // A kernel of a portable module is unaffected.
        assert_eq!(plan(Table::Linear, 0, GEOMETRY, &[], &[]).0, Ok(()));
        assert!(launches.len() == 1);
    }

    #[test]
    fn the_two_entry_points_use_their_own_tables() {
        let mut linear = Vec::new();
        let mut qwen4 = Vec::new();
        plan_linear_launch(5, GEOMETRY, &[], &[], true, &mut |l| linear.push(l)).unwrap();
        plan_qwen4_launch(5, GEOMETRY, &[], &[], true, &mut |l| qwen4.push(l)).unwrap();
        assert_eq!(linear[0].kernel, kernel!("euhedral_trellis_decode_bf16"));
        assert_eq!(qwen4[0].kernel.name(), qwen4_table()[5].name);
        assert_ne!(linear[0].kernel, qwen4[0].kernel);
    }
}
