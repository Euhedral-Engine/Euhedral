//! Whether an allocation can be served: the residency plan of its composite artifact at the target context keeps every
//! object on the device.

use euhedral_inference_edrl::TensorTable;
use euhedral_inference_engine::error::PlanError;
use euhedral_inference_engine::model::ModelConfig;
use euhedral_inference_engine::residency::planner::{PlanInputs, plan_closest};
use euhedral_inference_settings::Settings;

use super::allocation::Allocation;
use super::error::MixedError;

/// The verdict on one allocation: whether it fits, and the device bytes the plan needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Check {
    pub fits: bool,
    pub device_bytes: u64,
}

/// A judge of allocations against a device.
pub trait Feasibility {
    /// The device bytes the checks are made against.
    fn free_bytes(&self) -> u64;

    /// Whether `allocation` fits.
    fn check(&self, allocation: &Allocation) -> Result<Check, MixedError>;
}

/// The tensor table of the artifact an allocation makes: every object from the source its option names, the rest from
/// the base. It is the one piece the residency check needs that this module does not own.
pub trait Composites {
    fn table(&self, allocation: &Allocation) -> Result<TensorTable, MixedError>;
}

/// The engine's residency planner on the composite of an allocation, for a device with `free_bytes` free and
/// sequences of up to `context` tokens, the composite drafting `draft_depth` tokens per verification (a mixed
/// artifact stores no depth until its finalist measurement). It fits when the plan is made and nothing is held in
/// host memory or left in the file. The plan keeps the whole context's cache on the device and is given an unlimited
/// pinned host budget, so the verdict is about the weights; `settings` otherwise govern it, `memory.gpu-limit` among
/// them.
#[derive(Debug)]
pub struct PlannerFeasibility<C> {
    composites: C,
    config: ModelConfig,
    settings: Settings,
    free_bytes: u64,
}

impl<C: Composites> PlannerFeasibility<C> {
    pub fn new(
        composites: C,
        config: &ModelConfig,
        settings: &Settings,
        free_bytes: u64,
        context: u32,
        draft_depth: u8,
    ) -> PlannerFeasibility<C> {
        let mut config = config.clone();
        config.draft_depth = Some(draft_depth);
        let mut settings = settings.clone();
        settings.context.max_tokens = context;
        settings.context.kv_cache_on_gpu_tokens = context;
        PlannerFeasibility {
            composites,
            config,
            settings,
            free_bytes,
        }
    }
}

impl<C: Composites> Feasibility for PlannerFeasibility<C> {
    fn free_bytes(&self) -> u64 {
        self.free_bytes
    }

    fn check(&self, allocation: &Allocation) -> Result<Check, MixedError> {
        let table = self.composites.table(allocation)?;
        let inputs = PlanInputs {
            device_free: self.free_bytes,
            multiprocessors: None,
            host_pinned_bytes: Some(u64::MAX),
            sequences: 1,
        };
        match plan_closest(&self.config, &table, &self.settings, inputs) {
            Ok((plan, refusal)) => Ok(Check {
                fits: refusal.is_none() && plan.offload.is_empty(),
                device_bytes: plan.device_bytes,
            }),
            Err(failure) => Err(refused(failure)),
        }
    }
}

fn refused(failure: PlanError) -> MixedError {
    MixedError::Plan(failure.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use euhedral_inference_edrl::{
        ArtifactPlan, ModelMetadata, ObjectSpec, TensorDataType, WeightFormat, WeightLayout, encode_metadata,
    };
    use euhedral_inference_settings::ModelSettings;

    use super::*;
    use crate::mixed::option::QuantOption;

    /// The tensor table of a 27B-class hybrid model, one row per object: name, shape, container ordinals, bytes.
    const TABLE: &str = include_str!("../../../engine/tests/fixtures/rco_table.tsv");

    fn dtype(ordinal: u32) -> TensorDataType {
        [
            TensorDataType::Bf16,
            TensorDataType::Fp16,
            TensorDataType::Fp32,
            TensorDataType::Fp8E4m3,
            TensorDataType::Uint8,
            TensorDataType::Int8,
            TensorDataType::Int32,
        ][ordinal as usize]
    }

    fn format(ordinal: u32) -> WeightFormat {
        [
            WeightFormat::Bf16,
            WeightFormat::Fp16,
            WeightFormat::Fp32,
            WeightFormat::I32,
            WeightFormat::Q3G64Fp16,
            WeightFormat::Q4G64Fp16,
            WeightFormat::Q5G64Fp16,
            WeightFormat::Nvfp4,
            WeightFormat::Trellis,
        ][ordinal as usize]
    }

    fn layout(ordinal: u32) -> WeightLayout {
        [
            WeightLayout::ContiguousLeV1,
            WeightLayout::RowSplitK128V1,
            WeightLayout::RowSplitP2e2V1,
            WeightLayout::RowSplitK128Sd4V1,
            WeightLayout::RowInterleavedNvfp4V1,
            WeightLayout::TrellisTile16V1,
        ][ordinal as usize]
    }

    fn specs() -> Vec<ObjectSpec> {
        TABLE
            .lines()
            .filter(|line| !line.starts_with('#'))
            .map(|line| {
                let columns: Vec<&str> = line.split('\t').collect();
                ObjectSpec {
                    name: columns[0].to_string(),
                    shape: columns[1].split('x').map(|d| d.parse().unwrap()).collect(),
                    dtype: dtype(columns[2].parse().unwrap()),
                    format: format(columns[3].parse().unwrap()),
                    layout: layout(columns[4].parse().unwrap()),
                    bytes: columns[5].parse().unwrap(),
                }
            })
            .collect()
    }

    fn metadata() -> ModelMetadata {
        let mut metadata = euhedral_inference_edrl::ModelMetadata {
            vocab_size: 248_320,
            hidden_size: 5120,
            num_hidden_layers: 64,
            num_attention_heads: 24,
            num_key_value_heads: 4,
            attention_head_dim: 256,
            intermediate_size: 17_408,
            linear_num_key_heads: 16,
            linear_num_value_heads: 48,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            rms_norm_epsilon: 1e-6,
            rope_theta: 1e7,
            partial_rotary_factor: 0.25,
            max_position_embeddings: 262_144,
            hidden_activation: "silu".to_string(),
            layer_types: Vec::new(),
            num_experts: 0,
            num_experts_per_token: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            tie_word_embeddings: false,
            attention_output_gate: true,
            mtp_layer_count: 1,
            draft_depth: None,
        };
        metadata.layer_types = (0..64)
            .map(|layer| {
                if layer % 4 == 3 {
                    euhedral_inference_edrl::LayerType::FullAttention
                } else {
                    euhedral_inference_edrl::LayerType::GatedDeltaNet
                }
            })
            .collect();
        metadata
    }

    fn table_of(specs: &[ObjectSpec]) -> TensorTable {
        let metadata = encode_metadata(&metadata()).unwrap();
        let plan = ArtifactPlan::new(metadata.len() as u64, specs, 4096).unwrap();
        TensorTable::parse(&plan.table().unwrap(), plan.header(), plan.content_end()).unwrap()
    }

    /// Composites where an object an allocation gives `Trellis4_0` is stored with `extra` more bytes in each of its
    /// tiles' rows: `extra` times 2 times its tile count more payload bytes.
    struct Fat {
        extra: u64,
    }

    impl Composites for Fat {
        fn table(&self, allocation: &Allocation) -> Result<TensorTable, MixedError> {
            let mut specs = specs();
            for spec in &mut specs {
                if allocation.choices().get(&spec.name) == Some(&QuantOption::Trellis4_0) {
                    let tiles: u64 = spec.shape[..2].iter().product();
                    spec.shape[2] += self.extra;
                    spec.bytes += tiles * 2 * self.extra;
                }
            }
            Ok(table_of(&specs))
        }
    }

    fn judge(free: u64, extra: u64) -> PlannerFeasibility<Fat> {
        let config = ModelConfig::from(&metadata());
        let settings = Settings::with_model(ModelSettings::default());
        PlannerFeasibility::new(Fat { extra }, &config, &settings, free, 4096, 4)
    }

    fn one_heavy() -> Allocation {
        Allocation::new(BTreeMap::from([(
            "text/layers/10/mlp/down".to_string(),
            QuantOption::Trellis4_0,
        )]))
    }

    #[test]
    fn an_allocation_fits_when_the_device_holds_every_object() {
        let roomy = judge(64 << 30, 0).check(&Allocation::default()).unwrap();
        assert!(roomy.fits);
        assert!(roomy.device_bytes > 0 && roomy.device_bytes <= 64 << 30);
        assert_eq!(judge(64 << 30, 0).free_bytes(), 64 << 30);
    }

    #[test]
    fn a_device_one_byte_short_of_the_plan_does_not_fit_and_still_reports_the_bytes() {
        let need = judge(64 << 30, 0).check(&Allocation::default()).unwrap().device_bytes;
        let exact = judge(need, 0).check(&Allocation::default()).unwrap();
        assert!(exact.fits);
        let short = judge(need - 1, 0).check(&Allocation::default()).unwrap();
        assert!(!short.fits);
        assert!(short.device_bytes > 0);
    }

    #[test]
    fn a_bigger_object_raises_the_bytes_and_can_tip_the_verdict() {
        // 48 more bytes in each of the object's 1088 x 320 tiles' rows: 33423360 bytes.
        let added = 1088 * 320 * 2 * 48;
        let plain = judge(64 << 30, 0).check(&Allocation::default()).unwrap();
        let heavy = judge(64 << 30, 48).check(&one_heavy()).unwrap();
        assert!(heavy.device_bytes >= plain.device_bytes + added);
        let tight = judge(plain.device_bytes + added / 2, 48);
        assert!(tight.check(&Allocation::default()).unwrap().fits);
        assert!(!tight.check(&one_heavy()).unwrap().fits);
    }

    #[test]
    fn a_table_that_is_not_the_model_is_an_error_not_a_verdict() {
        struct Empty;
        impl Composites for Empty {
            fn table(&self, _allocation: &Allocation) -> Result<TensorTable, MixedError> {
                let mut specs = specs();
                specs.truncate(10);
                Ok(table_of(&specs))
            }
        }
        let config = ModelConfig::from(&metadata());
        let settings = Settings::with_model(ModelSettings::default());
        let judge = PlannerFeasibility::new(Empty, &config, &settings, 64 << 30, 4096, 4);
        assert!(matches!(judge.check(&Allocation::default()), Err(MixedError::Plan(_))));
    }
}
