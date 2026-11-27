//! The inventory check of an artifact that holds the DFlash2 drafter, on tables of names, shapes and formats.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use euhedral_inference_edrl::TensorTable;
use euhedral_inference_engine::error::PlanError;
use euhedral_inference_engine::model::{ArtifactProfile, ModelConfig, Speculation, drafter_names, validate_inventory};

mod common;

use common::{dflash2_config, dflash2_table_without, drafter_rows};

fn refusal(table: &TensorTable, config: &ModelConfig) -> String {
    let error = validate_inventory(config, table).unwrap_err();
    assert!(
        matches!(error, PlanError::Inventory(_) | PlanError::Unsupported(_)),
        "{error:?}"
    );
    error.to_string()
}

#[test]
fn an_artifact_with_the_drafter_in_either_format_is_accepted() {
    let config = dflash2_config();
    for nvfp4 in [false, true] {
        let table = dflash2_table_without(nvfp4, |_| false);
        validate_inventory(&config, &table).unwrap();
        assert_eq!(
            ArtifactProfile::of(&table, &config).unwrap().speculation,
            Speculation::Dflash2
        );
    }
    let names = drafter_names(config.dflash2.as_ref().unwrap());
    assert_eq!(names.len(), 7 + 5 * 13);
    assert_eq!(drafter_rows(true).lines().count(), names.len());
}

#[test]
fn a_missing_drafter_object_is_named() {
    let config = dflash2_config();
    for gone in [
        "dflash2/layers/4/mlp_conv/base",
        "dflash2/selector/successor",
        "dflash2/fc",
    ] {
        let table = dflash2_table_without(true, |name| name == gone);
        assert!(refusal(&table, &config).contains(gone), "{gone}");
    }
}

#[test]
fn the_drafter_objects_without_their_configuration_are_refused() {
    let config = dflash2_config();
    let table = dflash2_table_without(true, |name| name == "dflash2/config");
    assert!(refusal(&table, &config).contains("dflash2/config"));
    let mut unread = config;
    unread.dflash2 = None;
    let table = dflash2_table_without(true, |_| false);
    assert!(refusal(&table, &unread).contains("was not read"));
}

/// The table with the rows of the objects named by `changed` replaced by the rows given.
fn with_rows(nvfp4: bool, changed: &[(&str, &str)]) -> euhedral_inference_edrl::TensorTable {
    let rows: String = drafter_rows(nvfp4)
        .lines()
        .map(|row| {
            let name = row.split('\t').next().unwrap();
            let row = changed
                .iter()
                .find(|(object, _)| *object == name)
                .map_or(row, |(_, row)| row);
            format!("{row}\n")
        })
        .collect();
    common::table_of(&(base_rows() + &rows), |_| false)
}

#[test]
fn a_drafter_object_of_the_wrong_shape_or_format_is_refused() {
    let config = dflash2_config();
    // A norm stored as FP16, a projection of the wrong rows, and a projection in a format the drafter has no kernel for.
    let table = with_rows(
        false,
        &[("dflash2/hidden_norm", "dflash2/hidden_norm\t5120\t1\t1\t0\t10240")],
    );
    assert!(refusal(&table, &config).contains("dflash2/hidden_norm"));
    let table = with_rows(false, &[("dflash2/fc", "dflash2/fc\t5121x25600\t0\t0\t0\t262195200")]);
    assert!(refusal(&table, &config).contains("dflash2/fc"));
    let table = with_rows(false, &[("dflash2/fc", "dflash2/fc\t5120x25600\t1\t1\t0\t262144000")]);
    assert!(refusal(&table, &config).contains("dflash2/fc"));
}

#[test]
fn an_unknown_drafter_object_is_refused() {
    let config = dflash2_config();
    let table = common::table_of(
        &(drafter_rows(true) + "dflash2/layers/0/extra\t5120\t0\t0\t0\t10240\n" + &base_rows()),
        |_| false,
    );
    assert!(refusal(&table, &config).contains("dflash2/layers/0/extra"));
}

#[test]
fn a_drafter_the_target_cannot_take_is_refused() {
    let mut config = dflash2_config();
    config.dflash2.as_mut().unwrap().taps[4] = 64;
    let table = dflash2_table_without(true, |_| false);
    assert!(refusal(&table, &config).contains("layer 64"));
}

fn base_rows() -> String {
    common::fixture()
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| format!("{line}\n"))
        .collect()
}
