//! The drafter's configuration, its parameters and where everything lives in the device arena.

use super::DrafterError;

/// What a parameter's name is, as the training and the export name it.
pub const INPUT_PROJECTION: &str = "mtp/input_projection";
pub const QUERY_KEY_GATE_VALUE: &str = "mtp/layer/attention/query_key_gate_value";
pub const OUTPUT: &str = "mtp/layer/attention/output";
pub const GATE_UP: &str = "mtp/layer/mlp/gate_up";
pub const DOWN: &str = "mtp/layer/mlp/down";
pub const EMBEDDING_NORM: &str = "mtp/embedding_norm";
pub const HIDDEN_NORM: &str = "mtp/hidden_norm";
pub const INPUT_NORM: &str = "mtp/layer/input_norm";
pub const POST_ATTENTION_NORM: &str = "mtp/layer/post_attention_norm";
pub const QUERY_NORM: &str = "mtp/layer/attention/query_norm";
pub const KEY_NORM: &str = "mtp/layer/attention/key_norm";
pub const FINAL_NORM: &str = "mtp/final_norm";

/// The projections, in parameter order: each is NVFP4-rounded in the forward pass.
pub const PROJECTIONS: [&str; 5] = [INPUT_PROJECTION, QUERY_KEY_GATE_VALUE, OUTPUT, GATE_UP, DOWN];
/// The norms, in parameter order, after the projections.
pub const NORMS: [&str; 7] = [
    EMBEDDING_NORM,
    HIDDEN_NORM,
    INPUT_NORM,
    POST_ATTENTION_NORM,
    QUERY_NORM,
    KEY_NORM,
    FINAL_NORM,
];

/// Dimensions of the drafter's layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    /// The width of the hidden state.
    pub hidden: u32,
    /// The width of the feed-forward's gate and up halves.
    pub intermediate: u32,
    /// Query heads.
    pub heads: u32,
    /// Key and value heads; divide the query heads.
    pub kv_heads: u32,
    /// The width of a head: a multiple of 32 up to 512.
    pub head_dim: u32,
    /// The leading dimensions of a head that rotate: even, at most the head.
    pub rotary_dim: u32,
    /// The rotation's base.
    pub theta: f64,
    /// Added to a norm's mean square.
    pub eps: f32,
}

impl Shape {
    /// The width of a row of queries and keys, `[q heads; k heads]`.
    pub fn qk_width(&self) -> u32 {
        (self.heads + self.kv_heads) * self.head_dim
    }

    /// The width of the attention's context, one `head_dim` per query head.
    pub fn mixer_width(&self) -> u32 {
        self.heads * self.head_dim
    }

    /// The width of a row of keys or of values.
    pub fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }
}

/// What a drafter is built for: its layer, how many drafts it unrolls and the most rows, keys and positions of a window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// The layer.
    pub shape: Shape,
    /// The drafts unrolled, from 1.
    pub depths: u32,
    /// The most rows of a window.
    pub rows: u32,
    /// The most keys a window attends.
    pub max_keys: u32,
    /// The most positions of a sequence, whose depth-1 keys and values are kept.
    pub max_positions: u32,
}

fn positive(what: &'static str, value: u32) -> Result<(), DrafterError> {
    if value == 0 {
        return Err(DrafterError::Config {
            what,
            value: u64::from(value),
        });
    }
    Ok(())
}

fn refuse(what: &'static str, value: u32) -> DrafterError {
    DrafterError::Config {
        what,
        value: u64::from(value),
    }
}

impl Config {
    /// Refuses a configuration the kernels cannot run: a zero dimension, query heads that do not divide into the key
    /// heads, a head the kernels' blocks cannot hold, an odd rotation, or a projection width that is not a multiple of
    /// 16 (the NVFP4 block).
    pub fn validate(&self) -> Result<(), DrafterError> {
        let shape = &self.shape;
        for (what, value) in [
            ("hidden", shape.hidden),
            ("intermediate", shape.intermediate),
            ("heads", shape.heads),
            ("kv_heads", shape.kv_heads),
            ("head_dim", shape.head_dim),
            ("depths", self.depths),
            ("rows", self.rows),
            ("max_keys", self.max_keys),
            ("max_positions", self.max_positions),
        ] {
            positive(what, value)?;
        }
        if !shape.heads.is_multiple_of(shape.kv_heads) {
            return Err(refuse("kv_heads", shape.kv_heads));
        }
        if !shape.head_dim.is_multiple_of(32) || shape.head_dim > 512 {
            return Err(refuse("head_dim", shape.head_dim));
        }
        if !shape.rotary_dim.is_multiple_of(2) || shape.rotary_dim > shape.head_dim {
            return Err(refuse("rotary_dim", shape.rotary_dim));
        }
        for (what, value) in [
            ("hidden", shape.hidden),
            ("intermediate", shape.intermediate),
            ("mixer width", shape.mixer_width()),
        ] {
            if !value.is_multiple_of(16) {
                return Err(refuse(what, value));
            }
        }
        if !(shape.theta.is_finite() && shape.theta > 0.0) {
            return Err(DrafterError::Config {
                what: "theta",
                value: shape.theta as u64,
            });
        }
        if !(shape.eps.is_finite() && shape.eps > 0.0) {
            return Err(DrafterError::Config { what: "eps", value: 0 });
        }
        // Every float offset of the arena and of a launch is a signed 32-bit value in the kernels.
        let most = u64::from(self.rows) * u64::from(shape.hidden.max(shape.intermediate * 2).max(shape.qk_width()));
        if most > i32::MAX as u64 {
            return Err(DrafterError::Config {
                what: "rows",
                value: u64::from(self.rows),
            });
        }
        Ok(())
    }
}

/// A trainable tensor in the arena: FP32 values, gradient and the optimiser's moments; a projection also the
/// device word that holds the bits of its largest magnitude, which is its NVFP4 global scale's source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parameter {
    /// The name, one of the constants of this module.
    pub name: &'static str,
    /// The rows of the tensor, 1 for a norm.
    pub rows: u32,
    /// The columns of the tensor.
    pub columns: u32,
    /// The values, `rows * columns` floats.
    pub values: u64,
    /// The gradient, which a backward pass adds into.
    pub gradient: u64,
    /// The optimiser's first moment.
    pub first: u64,
    /// The optimiser's second moment.
    pub second: u64,
    /// The bits of the largest magnitude of the values (one unsigned), which only a projection has.
    pub maximum: Option<u64>,
}

impl Parameter {
    /// The floats of the tensor.
    pub fn count(&self) -> u64 {
        u64::from(self.rows) * u64::from(self.columns)
    }
}

/// The floats an arena allocation is rounded to, so that every region is 256-byte aligned.
const ALIGN_FLOATS: u64 = 64;

/// A bump allocator over the arena: its address at a float offset, in order.
#[derive(Debug)]
pub(super) struct Arena {
    base: u64,
    floats: u64,
}

impl Arena {
    pub(super) fn new(base: u64) -> Self {
        Self { base, floats: 0 }
    }

    /// The address of a region of `floats` floats (at least one).
    pub(super) fn take(&mut self, floats: u64) -> u64 {
        let address = self.base + 4 * self.floats;
        self.floats += floats.max(1).next_multiple_of(ALIGN_FLOATS);
        address
    }

    /// The bytes taken so far.
    pub(super) fn bytes(&self) -> u64 {
        4 * self.floats
    }
}

/// The parameters of a drafter of `config` in parameter order, from `arena`.
pub(super) fn parameters(config: &Config, arena: &mut Arena) -> Vec<Parameter> {
    let shape = &config.shape;
    let (h, i) = (shape.hidden, shape.intermediate);
    let projections = [
        (INPUT_PROJECTION, h, 2 * h),
        (QUERY_KEY_GATE_VALUE, 2 * shape.qk_width(), h),
        (OUTPUT, h, shape.mixer_width()),
        (GATE_UP, 2 * i, h),
        (DOWN, h, i),
    ];
    let norms = NORMS.map(|name| {
        let width = if name == QUERY_NORM || name == KEY_NORM {
            shape.head_dim
        } else {
            h
        };
        (name, 1, width)
    });
    projections
        .into_iter()
        .map(|(name, rows, columns)| (name, rows, columns, true))
        .chain(
            norms
                .into_iter()
                .map(|(name, rows, columns)| (name, rows, columns, false)),
        )
        .map(|(name, rows, columns, projection)| {
            let count = u64::from(rows) * u64::from(columns);
            let values = arena.take(count);
            let gradient = arena.take(count);
            let first = arena.take(count);
            let second = arena.take(count);
            let maximum = projection.then(|| arena.take(1));
            Parameter {
                name,
                rows,
                columns,
                values,
                gradient,
                first,
                second,
                maximum,
            }
        })
        .collect()
}
