use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display, Formatter};

use crate::error::EdrlError;
use crate::flash::config::{FlashConfig, FlashLayerType};
use crate::flash::tables::{ComponentGroup, ExpertBank, ExpertProjection, FlashTables};
use crate::format::{TensorDataType, WeightFormat, WeightLayout};

/// How many problems an error message lists.
const MAX_REPORTED: usize = 20;

/// One mismatch between an artifact's tables and the inventory its configuration implies. Every variant names the
/// object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InventoryProblem {
    Missing {
        name: String,
    },
    Extra {
        name: String,
    },
    Shape {
        name: String,
        found: Vec<u64>,
        expected: Vec<u64>,
    },
    DataType {
        name: String,
        found: TensorDataType,
        expected: TensorDataType,
    },
    Storage {
        name: String,
        found: (WeightFormat, WeightLayout),
        expected: (WeightFormat, WeightLayout),
    },
    Group {
        name: String,
        found: ComponentGroup,
        expected: ComponentGroup,
    },
    MissingBank {
        name: String,
    },
    ExtraBank {
        name: String,
    },
    /// A bank's layer, expert count, projections or record geometry differ from the expected.
    Bank {
        name: String,
        what: String,
    },
}

impl Display for InventoryProblem {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            InventoryProblem::Missing { name } => write!(f, "missing object {name}"),
            InventoryProblem::Extra { name } => write!(f, "unexpected object {name}"),
            InventoryProblem::Shape { name, found, expected } => {
                write!(f, "{name} has shape {found:?}, expected {expected:?}")
            }
            InventoryProblem::DataType { name, found, expected } => {
                write!(f, "{name} has data type {found:?}, expected {expected:?}")
            }
            InventoryProblem::Storage { name, found, expected } => {
                write!(
                    f,
                    "{name} is stored as {}/{}, expected {}/{}",
                    format_name(found.0),
                    layout_name(found.1),
                    format_name(expected.0),
                    layout_name(expected.1)
                )
            }
            InventoryProblem::Group { name, found, expected } => {
                write!(f, "{name} is in group {found:?}, expected {expected:?}")
            }
            InventoryProblem::MissingBank { name } => write!(f, "missing expert bank {name}"),
            InventoryProblem::ExtraBank { name } => write!(f, "unexpected expert bank {name}"),
            InventoryProblem::Bank { name, what } => write!(f, "{name} {what}"),
        }
    }
}

fn format_name(format: WeightFormat) -> String {
    format!("{format:?}")
}

fn layout_name(layout: WeightLayout) -> String {
    format!("{layout:?}")
}

/// The problems as an error message: the count, then the first few.
pub(crate) fn describe(problems: &[InventoryProblem]) -> String {
    let mut message = format!("artifact inventory is invalid ({} problems):", problems.len());
    for problem in problems.iter().take(MAX_REPORTED) {
        message.push_str("\n  ");
        message.push_str(&problem.to_string());
    }
    if problems.len() > MAX_REPORTED {
        message.push_str("\n  ...");
    }
    message
}

/// One expected fixed object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedTensor {
    pub shape: Vec<u64>,
    pub format: WeightFormat,
    pub layout: WeightLayout,
    pub group: ComponentGroup,
}

/// One expected expert bank: its projections' shapes, found by name (a record may store them in either order).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedBank {
    pub name: String,
    pub group: ComponentGroup,
    pub layer: i32,
    pub experts: usize,
    pub projections: Vec<(String, Vec<u64>)>,
}

/// What an inspection found: the object counts of a valid inventory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InventoryReport {
    pub tensors: usize,
    pub banks: usize,
    pub expert_records: usize,
}

/// The objects a `qwen4_exp` artifact must hold, derived from its configuration alone: every name, shape, storage
/// format and group. An artifact is valid only if its tables hold exactly this inventory.
///
/// Names: `text/...` for the text model (`text/layers/N/{attn_hc,mlp_hc,gdn,attention,moe,ple}/...`), `mtp/...` for the
/// multi-token-prediction layers and `vision/...` for the vision tower. NVFP4 shapes are the logical `[rows, K]`. Every
/// object's source data type is BF16.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedInventory {
    tensors: HashMap<String, ExpectedTensor>,
    banks: Vec<ExpectedBank>,
}

/// `a * b`, refusing an overflow: a configuration whose shapes overflow cannot match any file.
fn mul(a: u64, b: u64) -> Result<u64, EdrlError> {
    a.checked_mul(b)
        .ok_or_else(|| EdrlError::invalid("invalid configuration: a derived dimension overflows"))
}

struct Builder {
    tensors: HashMap<String, ExpectedTensor>,
    banks: Vec<ExpectedBank>,
}

impl Builder {
    fn add(&mut self, name: String, group: ComponentGroup, shape: &[u64]) {
        self.put(name, group, shape, WeightFormat::Bf16, WeightLayout::ContiguousLeV1);
    }

    fn quant(&mut self, name: String, group: ComponentGroup, rows: u64, k: u64) {
        self.put(
            name,
            group,
            &[rows, k],
            WeightFormat::Nvfp4,
            WeightLayout::RowSplitK128V1,
        );
    }

    fn put(&mut self, name: String, group: ComponentGroup, shape: &[u64], format: WeightFormat, layout: WeightLayout) {
        let expected = ExpectedTensor {
            shape: shape.to_vec(),
            format,
            layout,
            group,
        };
        // Expected names are generated per layer and never repeat; a repeat would be a bug in this derivation.
        assert!(
            self.tensors.insert(name.clone(), expected).is_none(),
            "duplicate expected object {name}"
        );
    }
}

impl ExpectedInventory {
    /// The inventory `config` implies.
    pub fn from_config(config: &FlashConfig) -> Result<ExpectedInventory, EdrlError> {
        config.validate()?;
        let mut b = Builder {
            tensors: HashMap::new(),
            banks: Vec::new(),
        };
        let hidden = u64::from(config.text.hidden_size);
        let width = mul(u64::from(config.hyper_connection.count), hidden)?;
        let vocab = u64::from(config.text.vocab_size);
        b.add(
            "text/token_embedding".into(),
            ComponentGroup::TokenEmbedding,
            &[vocab, hidden],
        );
        b.add("text/output_head".into(), ComponentGroup::OutputHead, &[vocab, hidden]);
        hyper_connection_mixer(
            &mut b,
            "text/hyper_connection_mixer",
            config,
            ComponentGroup::HyperConnection,
        )?;
        for (layer, kind) in config.text.layer_types.iter().enumerate() {
            let prefix = format!("text/layers/{layer}");
            layer_objects(
                &mut b,
                &prefix,
                layer,
                config,
                *kind,
                ComponentGroup::RoutedExpert,
                None,
            )?;
            if config.has_ngram(layer) {
                ngram_objects(&mut b, &format!("{prefix}/ple"), config, hidden, width)?;
            }
        }
        for (layer, kind) in config.mtp.layer_types.iter().enumerate() {
            let prefix = format!("mtp/layers/{layer}");
            layer_objects(
                &mut b,
                &prefix,
                layer,
                config,
                *kind,
                ComponentGroup::Mtp,
                Some(ComponentGroup::Mtp),
            )?;
        }
        if config.mtp.num_layers > 0 {
            b.quant("mtp/fc_embedding".into(), ComponentGroup::Mtp, hidden, hidden);
            b.quant("mtp/fc_hidden".into(), ComponentGroup::Mtp, hidden, hidden);
            b.add("mtp/pre_fc_norm_embedding".into(), ComponentGroup::Mtp, &[hidden]);
            b.add("mtp/pre_fc_norm_hidden".into(), ComponentGroup::Mtp, &[width]);
            hyper_connection_mixer(&mut b, "mtp/hyper_connection_mixer", config, ComponentGroup::Mtp)?;
        }
        vision_objects(&mut b, config)?;
        Ok(ExpectedInventory {
            tensors: b.tensors,
            banks: b.banks,
        })
    }

    /// The expected fixed objects by name.
    pub fn tensors(&self) -> &HashMap<String, ExpectedTensor> {
        &self.tensors
    }

    /// The expected expert banks.
    pub fn banks(&self) -> &[ExpectedBank] {
        &self.banks
    }

    /// Checks that `tables` hold exactly this inventory: no missing or unexpected object, and every shape, data
    /// type, format, layout and group as expected; every expert bank's layer, record count, projections and record
    /// geometry. Refuses with every problem found (the first few are in the message).
    pub fn validate(&self, tables: &FlashTables) -> Result<InventoryReport, EdrlError> {
        let mut problems = Vec::new();
        let mut names: Vec<&String> = self.tensors.keys().collect();
        names.sort();
        for name in names {
            let want = &self.tensors[name];
            let Some(tensor) = tables.tensor(name) else {
                problems.push(InventoryProblem::Missing { name: name.clone() });
                continue;
            };
            if tensor.shape != want.shape {
                problems.push(InventoryProblem::Shape {
                    name: name.clone(),
                    found: tensor.shape.clone(),
                    expected: want.shape.clone(),
                });
            }
            if tensor.dtype != TensorDataType::Bf16 {
                problems.push(InventoryProblem::DataType {
                    name: name.clone(),
                    found: tensor.dtype,
                    expected: TensorDataType::Bf16,
                });
            }
            if (tensor.format, tensor.layout) != (want.format, want.layout) {
                problems.push(InventoryProblem::Storage {
                    name: name.clone(),
                    found: (tensor.format, tensor.layout),
                    expected: (want.format, want.layout),
                });
            }
            if tensor.group != want.group {
                problems.push(InventoryProblem::Group {
                    name: name.clone(),
                    found: tensor.group,
                    expected: want.group,
                });
            }
        }
        for tensor in tables.tensors() {
            if !self.tensors.contains_key(&tensor.name) {
                problems.push(InventoryProblem::Extra {
                    name: tensor.name.clone(),
                });
            }
        }
        let mut seen = HashSet::new();
        for want in &self.banks {
            match tables.bank(&want.name) {
                Some(bank) => {
                    seen.insert(bank.name());
                    check_bank(bank, want, &mut problems);
                }
                None => problems.push(InventoryProblem::MissingBank {
                    name: want.name.clone(),
                }),
            }
        }
        for bank in tables.banks() {
            if !seen.contains(bank.name()) {
                problems.push(InventoryProblem::ExtraBank {
                    name: bank.name().to_owned(),
                });
            }
        }
        if !problems.is_empty() {
            return Err(EdrlError::Inventory { problems });
        }
        Ok(InventoryReport {
            tensors: tables.tensors().len(),
            banks: tables.banks().len(),
            expert_records: tables.record_count(),
        })
    }
}

fn check_bank(bank: &ExpertBank, want: &ExpectedBank, problems: &mut Vec<InventoryProblem>) {
    let mut bank_problem = |name: String, what: String| problems.push(InventoryProblem::Bank { name, what });
    if bank.group() != want.group {
        bank_problem(
            bank.name().to_owned(),
            format!("is in group {:?}, expected {:?}", bank.group(), want.group),
        );
    }
    if bank.layer() != want.layer {
        bank_problem(
            bank.name().to_owned(),
            format!("is for layer {}, expected {}", bank.layer(), want.layer),
        );
    }
    if bank.expert_count() != want.experts {
        bank_problem(
            bank.name().to_owned(),
            format!("has {} experts, expected {}", bank.expert_count(), want.experts),
        );
    }
    if bank.projections().len() != want.projections.len() {
        bank_problem(
            bank.name().to_owned(),
            format!(
                "has {} projections, expected {}",
                bank.projections().len(),
                want.projections.len()
            ),
        );
    }
    // Projections are found by name, in whatever order the record stores them, and never overlap.
    for (projection, shape) in &want.projections {
        let path = format!("{}/{projection}", bank.name());
        let Some(actual) = bank
            .projections()
            .iter()
            .find(|candidate| candidate.name == *projection)
        else {
            bank_problem(bank.name().to_owned(), format!("lacks projection {projection}"));
            continue;
        };
        if actual.shape != *shape {
            bank_problem(
                path.clone(),
                format!("has shape {:?}, expected {shape:?}", actual.shape),
            );
        }
        if (actual.format, actual.layout) != (WeightFormat::Nvfp4, WeightLayout::RowSplitK128V1) {
            bank_problem(
                path,
                format!(
                    "is stored as {}/{}, expected Nvfp4/RowSplitK128V1",
                    format_name(actual.format),
                    layout_name(actual.layout)
                ),
            );
        }
    }
    let mut by_offset: Vec<&ExpertProjection> = bank.projections().iter().collect();
    by_offset.sort_by_key(|projection| projection.record_offset);
    for pair in by_offset.windows(2) {
        let previous_end = pair[0].record_offset.saturating_add(pair[0].bytes);
        if pair[1].record_offset < previous_end {
            bank_problem(
                format!("{}/{}", bank.name(), pair[1].name),
                format!(
                    "starts at {} in the record, inside {} that ends at {previous_end}",
                    pair[1].record_offset, pair[0].name
                ),
            );
        }
    }
    // Expert `e` is the e-th record: records follow one another in expert order, so an id is its position.
    for (expert, pair) in bank.records().windows(2).enumerate() {
        if pair[1].offset < pair[0].offset.saturating_add(pair[0].bytes) {
            bank_problem(
                bank.name().to_owned(),
                format!(
                    "expert {} at {} does not follow expert {expert}, which ends at {}",
                    expert + 1,
                    pair[1].offset,
                    pair[0].offset.saturating_add(pair[0].bytes)
                ),
            );
            break;
        }
    }
}

fn hyper_connection_mixer(
    b: &mut Builder,
    prefix: &str,
    config: &FlashConfig,
    group: ComponentGroup,
) -> Result<(), EdrlError> {
    let width = mul(
        u64::from(config.hyper_connection.count),
        u64::from(config.text.hidden_size),
    )?;
    let lowrank = u64::from(config.hyper_connection.lowrank);
    b.add(format!("{prefix}/hc_norm"), group, &[width]);
    b.add(format!("{prefix}/input_mix_down"), group, &[lowrank, width]);
    b.add(format!("{prefix}/input_mix_up"), group, &[width, lowrank]);
    Ok(())
}

/// One layer of the text stack or of the MTP stack: `group_override` places every object of an MTP layer in the MTP
/// group.
fn layer_objects(
    b: &mut Builder,
    prefix: &str,
    layer: usize,
    config: &FlashConfig,
    kind: FlashLayerType,
    expert_group: ComponentGroup,
    group_override: Option<ComponentGroup>,
) -> Result<(), EdrlError> {
    let hidden = u64::from(config.text.hidden_size);
    let hc_count = u64::from(config.hyper_connection.count);
    let width = mul(hc_count, hidden)?;
    let lowrank = u64::from(config.hyper_connection.lowrank);
    let group = |own: ComponentGroup| group_override.unwrap_or(own);
    let hc = group(ComponentGroup::HyperConnection);
    let small = group(ComponentGroup::NormSmallState);
    for site in ["attn_hc", "mlp_hc"] {
        b.add(format!("{prefix}/{site}/hc_norm"), hc, &[width]);
        b.add(format!("{prefix}/{site}/input_mix_down"), hc, &[lowrank, width]);
        b.add(format!("{prefix}/{site}/input_mix_up"), hc, &[width, lowrank]);
        b.add(format!("{prefix}/{site}/block_inject"), hc, &[hc_count, width]);
    }
    match kind {
        FlashLayerType::GatedDeltaNet => {
            let g = &config.gdn;
            let gdn = group(ComponentGroup::Gdn);
            let value_heads = u64::from(g.num_value_heads);
            let values = mul(value_heads, u64::from(g.value_head_dim))?;
            b.add(format!("{prefix}/gdn/a_log"), small, &[value_heads]);
            b.add(format!("{prefix}/gdn/dt_bias"), small, &[value_heads]);
            b.add(
                format!("{prefix}/gdn/conv1d"),
                gdn,
                &[g.conv_channels(), 1, u64::from(g.conv_kernel_dim)],
            );
            b.add(format!("{prefix}/gdn/in_proj_a"), gdn, &[value_heads, hidden]);
            b.add(format!("{prefix}/gdn/in_proj_b"), gdn, &[value_heads, hidden]);
            b.quant(format!("{prefix}/gdn/in_proj_qkv"), gdn, g.conv_channels(), hidden);
            b.quant(format!("{prefix}/gdn/in_proj_z"), gdn, values, hidden);
            b.quant(format!("{prefix}/gdn/out_proj"), gdn, hidden, values);
            b.add(format!("{prefix}/gdn/norm"), small, &[u64::from(g.value_head_dim)]);
        }
        FlashLayerType::SparseAttention => {
            let a = &config.attention;
            let q = &config.qsa;
            let qsa = group(ComponentGroup::Qsa);
            let head_dim = u64::from(a.head_dim);
            let queries = mul(u64::from(a.num_heads), head_dim)?;
            let keys = mul(u64::from(a.num_kv_heads), head_dim)?;
            let indexer_rows = mul(u64::from(q.num_heads) + u64::from(q.kv_heads), u64::from(q.head_dim))?;
            b.quant(format!("{prefix}/attention/q_proj"), qsa, mul(2, queries)?, hidden);
            b.quant(format!("{prefix}/attention/k_proj"), qsa, keys, hidden);
            b.quant(format!("{prefix}/attention/v_proj"), qsa, keys, hidden);
            b.quant(format!("{prefix}/attention/o_proj"), qsa, hidden, queries);
            b.add(format!("{prefix}/attention/q_norm"), small, &[head_dim]);
            b.add(format!("{prefix}/attention/k_norm"), small, &[head_dim]);
            b.add(
                format!("{prefix}/attention/indexer/index_qk_proj"),
                qsa,
                &[indexer_rows, hidden],
            );
            b.add(
                format!("{prefix}/attention/indexer/q_layernorm"),
                small,
                &[u64::from(q.head_dim)],
            );
            b.add(
                format!("{prefix}/attention/indexer/k_layernorm"),
                small,
                &[u64::from(q.head_dim)],
            );
        }
    }
    let m = &config.moe;
    let router = group(ComponentGroup::Router);
    let shared = group(ComponentGroup::SharedExpert);
    let shared_size = u64::from(m.shared_expert_intermediate_size);
    let intermediate = u64::from(m.intermediate_size);
    b.add(
        format!("{prefix}/moe/router"),
        router,
        &[u64::from(m.num_experts), hidden],
    );
    b.quant(
        format!("{prefix}/moe/shared_expert/gate_proj"),
        shared,
        shared_size,
        hidden,
    );
    b.quant(
        format!("{prefix}/moe/shared_expert/up_proj"),
        shared,
        shared_size,
        hidden,
    );
    b.quant(
        format!("{prefix}/moe/shared_expert/down_proj"),
        shared,
        hidden,
        shared_size,
    );
    b.add(format!("{prefix}/moe/shared_expert_gate"), shared, &[1, hidden]);
    b.banks.push(ExpectedBank {
        name: format!("{prefix}/moe/experts"),
        group: expert_group,
        layer: i32::try_from(layer).map_err(|_| EdrlError::invalid("layer index is out of range"))?,
        experts: m.num_experts as usize,
        projections: vec![
            ("gate_up".into(), vec![mul(2, intermediate)?, hidden]),
            ("down".into(), vec![hidden, intermediate]),
        ],
    });
    Ok(())
}

/// The n-gram tables of one layer with the projections that feed them.
fn ngram_objects(
    b: &mut Builder,
    prefix: &str,
    config: &FlashConfig,
    hidden: u64,
    width: u64,
) -> Result<(), EdrlError> {
    b.quant(format!("{prefix}/key_proj"), ComponentGroup::FixedText, width, hidden);
    b.quant(
        format!("{prefix}/value_proj"),
        ComponentGroup::FixedText,
        hidden,
        hidden,
    );
    b.add(
        format!("{prefix}/conv1d"),
        ComponentGroup::FixedText,
        &[width, 1, u64::from(config.ple.conv_kernel_size)],
    );
    for norm in ["norm_conv", "norm_key", "norm_query"] {
        b.add(format!("{prefix}/{norm}"), ComponentGroup::NormSmallState, &[width]);
    }
    let row_width = hidden / config.ngram.heads();
    // The configuration is validated: the shard rows are positive.
    let rows = config.ngram.shard_rows.unsigned_abs();
    for shard in 0..config.ngram.split_parts {
        b.put(
            format!("{prefix}/ngram/{}", shard_name(shard)),
            ComponentGroup::Ngram,
            &[rows, row_width],
            WeightFormat::Nvfp4,
            WeightLayout::RowInterleavedNvfp4V1,
        );
    }
    Ok(())
}

/// `shard_000`, `shard_001`, ...
pub fn shard_name(shard: u32) -> String {
    format!("shard_{shard:03}")
}

fn vision_objects(b: &mut Builder, config: &FlashConfig) -> Result<(), EdrlError> {
    let v = &config.vision;
    let hidden = u64::from(v.hidden_size);
    let intermediate = u64::from(v.intermediate_size);
    let merge = u64::from(v.spatial_merge_size);
    let merged = mul(mul(hidden, merge)?, merge)?;
    let group = ComponentGroup::Vision;
    for block in 0..v.depth {
        let prefix = format!("vision/blocks/{block}");
        b.add(format!("{prefix}/attn/qkv/weight"), group, &[mul(3, hidden)?, hidden]);
        b.add(format!("{prefix}/attn/qkv/bias"), group, &[mul(3, hidden)?]);
        b.add(format!("{prefix}/attn/proj/weight"), group, &[hidden, hidden]);
        b.add(format!("{prefix}/attn/proj/bias"), group, &[hidden]);
        b.add(
            format!("{prefix}/mlp/linear_fc1/weight"),
            group,
            &[intermediate, hidden],
        );
        b.add(format!("{prefix}/mlp/linear_fc1/bias"), group, &[intermediate]);
        b.add(
            format!("{prefix}/mlp/linear_fc2/weight"),
            group,
            &[hidden, intermediate],
        );
        b.add(format!("{prefix}/mlp/linear_fc2/bias"), group, &[hidden]);
        for norm in ["norm1", "norm2"] {
            b.add(format!("{prefix}/{norm}/weight"), group, &[hidden]);
            b.add(format!("{prefix}/{norm}/bias"), group, &[hidden]);
        }
    }
    let out = u64::from(v.out_hidden_size);
    b.add("vision/merger/linear_fc1/weight".into(), group, &[merged, merged]);
    b.add("vision/merger/linear_fc1/bias".into(), group, &[merged]);
    b.add("vision/merger/linear_fc2/weight".into(), group, &[out, merged]);
    b.add("vision/merger/linear_fc2/bias".into(), group, &[out]);
    b.add("vision/merger/norm/weight".into(), group, &[hidden]);
    b.add("vision/merger/norm/bias".into(), group, &[hidden]);
    b.add(
        "vision/patch_embed/proj/weight".into(),
        group,
        &[
            hidden,
            u64::from(v.in_channels),
            u64::from(v.temporal_patch_size),
            u64::from(v.patch_size),
            u64::from(v.patch_size),
        ],
    );
    b.add("vision/patch_embed/proj/bias".into(), group, &[hidden]);
    b.add(
        "vision/pos_embed/weight".into(),
        group,
        &[u64::from(v.num_position_embeddings), hidden],
    );
    Ok(())
}
