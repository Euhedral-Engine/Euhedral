//! Which format each object is stored in, per recipe.

#![allow(clippy::unwrap_used, reason = "a test: it stops on a failed unwrap")]

use euhedral_inference_quantization::recipe::{Quantization, ROW_SPLIT, Recipe, RecipeError, SD4};

fn recipe(quantization: &str, compressed: bool, head: Option<&str>) -> Recipe {
    Recipe::new(Quantization::parse(quantization).unwrap(), compressed, head).unwrap()
}

#[test]
fn q3_widens_the_layers_at_the_ends_and_the_error_carrying_objects() {
    let q3 = recipe("q3", false, None);
    let name = |object: &str| q3.format(object).unwrap().name;
    assert_eq!(name("text/layers/5/mlp/down"), "Q4G64_F16S");
    assert_eq!(name("text/layers/6/mlp/down"), "Q3G64_F16S");
    assert_eq!(name("text/layers/57/mlp/down"), "Q3G64_F16S");
    assert_eq!(name("text/layers/58/mlp/down"), "Q4G64_F16S");
    assert_eq!(name("text/layers/9/gdn/query_key"), "Q4G64_F16S");
    assert_eq!(name("text/layers/9/attention/gate_value"), "Q5G64_F16S");
    assert_eq!(name("text/layers/9/mlp/gate_up"), "Q3G64_F16S");
    assert_eq!(name("text/output_head"), "NVFP4");
    assert_eq!(name("mtp/layer/mlp/down"), "NVFP4");
    assert_eq!(q3.format("text/layers/9/mlp/gate_up").unwrap().layout, ROW_SPLIT);
}

#[test]
fn compressed_q3_drops_the_lowest_code_except_for_down_projections() {
    let compressed = recipe("q3", true, None);
    let gate_up = compressed.format("text/layers/9/mlp/gate_up").unwrap();
    assert_eq!((gate_up.qmin, gate_up.qmax), (-3, 3));
    let down = compressed.format("text/layers/9/mlp/down").unwrap();
    assert_eq!((down.qmin, down.qmax), (-4, 3));
    // A Q4 object keeps its range: the limit applies to the Q3 format only.
    let query_key = compressed.format("text/layers/9/gdn/query_key").unwrap();
    assert_eq!((query_key.qmin, query_key.qmax), (-8, 7));
    let plain = recipe("q3", false, None).format("text/layers/9/mlp/gate_up").unwrap();
    assert_eq!(plain.qmin, -4);
}

#[test]
fn compressed_nvfp4_uses_the_sd4_layout_except_for_drafting() {
    let compressed = recipe("nvfp4", true, None);
    let layer = compressed.format("text/layers/9/mlp/down").unwrap();
    assert!(layer.nvfp4 && layer.sd4);
    assert_eq!(layer.layout, SD4);
    assert_eq!(layer.group, 16);
    for drafting in ["mtp/layer/mlp/down", "mtp/input_projection", "text/draft_head"] {
        let format = compressed.format(drafting).unwrap();
        assert!(format.nvfp4 && !format.sd4, "{drafting}");
        assert_eq!(format.layout, ROW_SPLIT);
    }
    assert!(
        !recipe("nvfp4", false, None)
            .format("text/layers/9/mlp/down")
            .unwrap()
            .sd4
    );
}

#[test]
fn an_output_head_format_replaces_only_the_heads_format() {
    let recipe = recipe("nvfp4", true, Some("Q5G64_F16S"));
    let head = recipe.format("text/output_head").unwrap();
    assert_eq!(
        (head.name.as_str(), head.layout.as_str(), head.qmin, head.qmax),
        ("Q5G64_F16S", ROW_SPLIT, -16, 15)
    );
    assert!(recipe.format("text/layers/9/mlp/down").unwrap().nvfp4);
    assert_eq!(
        Recipe::new(Quantization::Q3, false, Some("Q9")),
        Err(RecipeError::UnknownHeadFormat("Q9".to_owned()))
    );
}

#[test]
fn the_trellis_recipe_rounds_only_the_heads() {
    let trellis = recipe("trellis", false, None);
    for head in ["text/output_head", "text/draft_head"] {
        let format = trellis.format(head).unwrap();
        assert!(format.nvfp4 && !format.sd4, "{head}");
        assert_eq!(format.layout, ROW_SPLIT);
    }
    assert_eq!(
        trellis.format("text/layers/0/mlp/down"),
        Err(RecipeError::NotAHead("text/layers/0/mlp/down".to_owned()))
    );
    let overridden = self::recipe("trellis", false, Some("Q4G64_F16S"));
    assert_eq!(overridden.format("text/output_head").unwrap().name, "Q4G64_F16S");
    assert!(overridden.format("text/draft_head").unwrap().nvfp4);
    assert_eq!(
        Recipe::new(Quantization::Trellis, true, None),
        Err(RecipeError::TrellisCompressed)
    );
}

#[test]
fn unknown_names_are_refused() {
    assert_eq!(
        Quantization::parse("q4"),
        Err(RecipeError::UnknownQuantization("q4".to_owned()))
    );
    let q3 = recipe("q3", false, None);
    for object in [
        "text/layers/x/mlp/down",
        "text/layers/1/other",
        "text/layers/1",
        "nonsense",
    ] {
        assert_eq!(
            q3.format(object),
            Err(RecipeError::UnknownObject(object.to_owned())),
            "{object}"
        );
    }
}
