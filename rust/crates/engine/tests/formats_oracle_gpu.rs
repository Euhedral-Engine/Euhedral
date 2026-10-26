//! The library's process-wide exact-numerics switch (`exact_numerics_for_tests`) on the linear formats. It is a binary of
//! its own because the switch is seen by every launch of the process: a test beside it would run on the reference.
//! The tests are `#[ignore]`d and run locally in a memory-capped scope:
//! `systemd-run --user --scope -p MemoryMax=16G cargo test -p euhedral-inference-engine --test formats_oracle_gpu -- --ignored`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting calls: a test copies device memory and synchronizes"
)]

#[path = "common/linear_gpu.rs"]
mod linear_gpu;

use euhedral_inference_engine::formats::{LinearMode, Nvfp4Linear, TrellisLinear};
use linear_gpu::{Artifact, Random, Rig, activations, rig, synthetic_trellis, upload, values};

const PLAIN: LinearMode = LinearMode {
    row_exact: false,
    prefill: false,
};

/// One test runs both checks one after the other: the switch is process-wide, so two tests of this binary would see
/// each other's selection.
#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn exact_numerics_run_every_launch_on_the_scalar_reference() {
    let Some(rig) = rig() else { return };
    nvfp4_follows_the_switch(&rig);
    trellis_follows_the_switch(&rig);
}

fn nvfp4_follows_the_switch(rig: &Rig) {
    let Some(mut artifact) = Artifact::open() else { return };
    let (_weights, weight) = artifact.plain(&rig.library, "mtp/layer/attention/output");
    let (n, k) = (5120usize, 6144usize);
    let mut random = Random::new(5);
    let x = upload(&rig.library, &activations(&mut random, 64, k));
    let production = Nvfp4Linear::new(&rig.library);
    // The scalar reference computes each output element on its own, so its rows do not depend on how many run: above
    // 32 rows a device without the native route reaches it without the switch.
    let portable = Nvfp4Linear::with_native_route(false);
    let reference = values(&rig.linear(&portable, &weight, &x, 0, k, n, 64, PLAIN));
    let ordinary: Vec<Vec<f32>> = [1usize, 8, 20, 40]
        .iter()
        .map(|rows| values(&rig.linear(&production, &weight, &x, 0, k, n, *rows, PLAIN)))
        .collect();

    assert!(!rig.library.exact_numerics_for_tests(true), "the switch starts off");
    let exact: Vec<Vec<f32>> = [1usize, 8, 20, 40]
        .iter()
        .map(|rows| values(&rig.linear(&production, &weight, &x, 0, k, n, *rows, PLAIN)))
        .collect();
    assert!(rig.library.exact_numerics_for_tests(false), "and reports it was on");

    for ((rows, exact), ordinary) in [1usize, 8, 20, 40].iter().zip(&exact).zip(&ordinary) {
        assert_eq!(
            exact[..],
            reference[..rows * n],
            "{rows} rows under exact numerics are the reference, bit for bit"
        );
        assert_ne!(
            exact[..],
            ordinary[..],
            "{rows} rows differ from the production kernels somewhere"
        );
    }
    // Back off, the production kernels run again.
    let again = values(&rig.linear(&production, &weight, &x, 0, k, n, 8, PLAIN));
    assert_eq!(again, ordinary[1]);
}

fn trellis_follows_the_switch(rig: &Rig) {
    let mut random = Random::new(13);
    let object = synthetic_trellis(&rig.library, &mut random, 512, 1024, 56, &[128, 512]);
    let x = upload(&rig.library, &activations(&mut random, 8, 1024));
    // A format that was not built for the reference follows the library's selection, as Java's does.
    let production = TrellisLinear::new(140, 32, 128, 1024);
    let oracle = TrellisLinear::new(140, 32, 128, 1024).with_exact_numerics();
    let run = |format: &TrellisLinear| rig.linear(format, &object.weight, &x, 0, 1024, 512, 8, PLAIN);
    let ordinary = run(&production);
    let reference = run(&oracle);
    assert_ne!(
        ordinary, reference,
        "the tensor-core route sums in another order than the reference"
    );

    assert!(!rig.library.exact_numerics_for_tests(true), "the switch starts off");
    assert!(rig.library.exact_numerics());
    let selected = run(&production);
    assert!(rig.library.exact_numerics_for_tests(false), "and reports it was on");
    assert!(!rig.library.exact_numerics());
    assert_eq!(
        selected, reference,
        "under exact numerics the format runs the reference, bit for bit"
    );
    assert_eq!(run(&production), ordinary, "back off, the tensor-core route runs again");
}
