//! The launch catalog's table entries are the engine's own tables, in order.

use euhedral_inference_cuda::launch::catalog::linear_table;
use euhedral_inference_engine::formats::LinearKernel;

#[test]
fn the_linear_table_entries_equal_the_linear_kernels_in_order() {
    let table: Vec<&str> = linear_table().iter().map(|kernel| kernel.name).collect();
    let kernels: Vec<&str> = LinearKernel::ALL.iter().map(|kernel| kernel.symbol()).collect();
    assert_eq!(table, kernels);
    for (index, kernel) in LinearKernel::ALL.iter().enumerate() {
        assert_eq!(kernel.index(), index);
    }
}
