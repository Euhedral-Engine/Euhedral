use euhedral_inference_edrl::FlashConfig;
use euhedral_inference_edrl::tiny_flash::tiny_config;

use super::*;
use crate::flash::buffers::ALIGNMENT;
use crate::flash::startup::PaddedDown;

/// The tiny configuration with the heads the gated delta rule's kernel serves.
fn tiny() -> FlashConfig {
    let mut config = tiny_config();
    config.gdn.key_head_dim = 128;
    config.gdn.value_head_dim = 128;
    config.hyper_connection.lowrank = 8;
    config
}

fn at(address: u64) -> DevicePtr {
    // SAFETY: a test address, never dereferenced.
    unsafe { DevicePtr::from_raw(address) }
}

#[test]
fn the_sequence_keeps_gdn_state_for_each_gdn_layer_and_none_for_the_attention_layers_and_the_ngram_history() {
    let config = tiny();
    let spec = sequence_spec(&config, 64).unwrap();
    let (history, recurrent) = gdn_shape(&config).unwrap().state_bytes();
    assert_eq!(
        spec.layers,
        [
            LayerSizes::GatedDeltaNet {
                history_bytes: history,
                recurrent_bytes: recurrent
            },
            LayerSizes::Attention
        ]
    );
    assert_eq!(spec.max_tokens, 64);
    // (taps - 1) * dilation rows of the stream width, in BF16.
    let rows = u64::from((config.ple.conv_kernel_size - 1) * config.ngram.size);
    let width = u64::from(config.hyper_connection.count * config.text.hidden_size);
    assert_eq!(spec.ple_history_bytes, rows * width * 2);
}

#[test]
fn the_ngram_ids_follow_the_configuration_and_start_from_the_end_of_sequence_context() {
    let config = tiny();
    let ids = ngram_ids(&config).unwrap();
    assert_eq!(ids.heads() as u64, config.ngram.heads());
    assert!(ids.fresh_context().iter().all(|id| *id == config.text.eos_token_id));
}

#[test]
fn a_missing_weight_is_refused_naming_it_before_any_stage_exists() {
    let config = tiny();
    let spec = sequence_spec(&config, 64).unwrap();
    let layout = SequenceLayout::over(&spec, at(4 * ALIGNMENT)).unwrap();
    let weights = FixedWeights::empty();
    let padded = PaddedDown::none();
    let inputs = Ingredients {
        config: &config,
        weights: &weights,
        padded_down: &padded,
        tables: &[],
        banks: &[],
        layout: &layout,
        buckets: Buckets::new(8, config.moe.num_experts, config.moe.experts_per_token).unwrap(),
        device: Nvfp4Device::with_native_route(false, 1),
        max_tokens: 64,
    };
    let refusal = Blueprint::new(&inputs).unwrap_err().to_string();
    assert!(refusal.contains("text/layers/0/gdn/in_proj_qkv"), "{refusal}");
}

#[test]
fn a_layer_whose_state_is_of_the_other_kind_is_refused() {
    let config = tiny();
    let mut swapped = config.clone();
    swapped.text.layer_types.reverse();
    let spec = sequence_spec(&config, 64).unwrap();
    let layout = SequenceLayout::over(&spec, at(4 * ALIGNMENT)).unwrap();
    let weights = FixedWeights::empty();
    let padded = PaddedDown::none();
    let inputs = Ingredients {
        config: &swapped,
        weights: &weights,
        padded_down: &padded,
        tables: &[],
        banks: &[],
        layout: &layout,
        buckets: Buckets::new(8, config.moe.num_experts, config.moe.experts_per_token).unwrap(),
        device: Nvfp4Device::with_native_route(false, 1),
        max_tokens: 64,
    };
    let refusal = Blueprint::new(&inputs).unwrap_err().to_string();
    assert!(
        refusal.contains("layer 0") && refusal.contains("another kind"),
        "{refusal}"
    );
}
