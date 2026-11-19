//! How each quantized object is stored in an artifact (the converter's recipes and inventory): the q3 artifact's
//! format per object, mapped to the recipe being built, and compressed q3's code limit. When an output head format is
//! set it is the LM head's format (row-split) in place of the recipe's. The trellis recipe stores only the two heads
//! through this module (NVFP4 row-split); its other objects are exllamav3 checkpoint linears.

use thiserror::Error;

/// The layout of one scale per row group of 128 columns.
pub const ROW_SPLIT: &str = "row-split-k128-v1";
/// The row-split layout with the stochastic-rounding table of four bits per value.
pub const SD4: &str = "row-split-k128-sd4-v1";

/// The formats an output head may take in place of the recipe's.
const HEAD_FORMATS: [&str; 4] = ["NVFP4", "Q3G64_F16S", "Q4G64_F16S", "Q5G64_F16S"];

/// Why a recipe or a format is refused.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum RecipeError {
    #[error("quantization is q3, nvfp4 or trellis, not {0}")]
    UnknownQuantization(String),
    #[error("the trellis recipe has no compressed form")]
    TrellisCompressed,
    #[error("unknown LM head format {0}")]
    UnknownHeadFormat(String),
    #[error("unknown format {0}")]
    UnknownFormat(String),
    #[error("unknown object {0}")]
    UnknownObject(String),
    #[error("the trellis recipe rounds only the LM and draft heads, not {0}")]
    NotAHead(String),
}

/// A storage format: grouped (Q3/Q4/Q5 G64, codes `qmin..=qmax`) or NVFP4 (block 16, optionally SD4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Format {
    pub name: String,
    pub layout: String,
    pub group: u32,
    pub qmin: i32,
    pub qmax: i32,
    pub nvfp4: bool,
    pub sd4: bool,
}

impl Format {
    /// The format `name` in `layout`. `lowest_code`, when set, raises a grouped format's lowest code.
    pub fn of(name: &str, layout: &str, lowest_code: Option<i32>) -> Result<Format, RecipeError> {
        let grouped = |qmin: i32, qmax: i32| Format {
            name: name.to_owned(),
            layout: layout.to_owned(),
            group: 64,
            qmin: lowest_code.map_or(qmin, |lowest| qmin.max(lowest)),
            qmax,
            nvfp4: false,
            sd4: false,
        };
        match name {
            "NVFP4" => Ok(Format {
                name: name.to_owned(),
                layout: layout.to_owned(),
                group: 16,
                qmin: 0,
                qmax: 0,
                nvfp4: true,
                sd4: layout.ends_with("sd4-v1"),
            }),
            "Q3G64_F16S" => Ok(grouped(-4, 3)),
            "Q4G64_F16S" => Ok(grouped(-8, 7)),
            "Q5G64_F16S" => Ok(grouped(-16, 15)),
            _ => Err(RecipeError::UnknownFormat(name.to_owned())),
        }
    }
}

/// The family of artifact a recipe builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantization {
    Q3,
    Nvfp4,
    Trellis,
}

impl Quantization {
    /// The command-line name: `q3`, `nvfp4` or `trellis`.
    pub fn parse(name: &str) -> Result<Quantization, RecipeError> {
        match name {
            "q3" => Ok(Quantization::Q3),
            "nvfp4" => Ok(Quantization::Nvfp4),
            "trellis" => Ok(Quantization::Trellis),
            _ => Err(RecipeError::UnknownQuantization(name.to_owned())),
        }
    }
}

/// The recipe of one artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recipe {
    quantization: Quantization,
    compressed: bool,
    output_head: Option<String>,
}

impl Recipe {
    /// A recipe; `output_head`, when set, is the LM head's format in place of the recipe's.
    pub fn new(quantization: Quantization, compressed: bool, output_head: Option<&str>) -> Result<Recipe, RecipeError> {
        if quantization == Quantization::Trellis && compressed {
            return Err(RecipeError::TrellisCompressed);
        }
        if let Some(head) = output_head
            && !HEAD_FORMATS.contains(&head)
        {
            return Err(RecipeError::UnknownHeadFormat(head.to_owned()));
        }
        Ok(Recipe {
            quantization,
            compressed,
            output_head: output_head.map(str::to_owned),
        })
    }

    pub fn quantization(&self) -> Quantization {
        self.quantization
    }

    pub fn compressed(&self) -> bool {
        self.compressed
    }

    /// The storage format of object `name` (`text/layers/<n>/<role>`, `text/output_head`, `mtp/...`).
    pub fn format(&self, name: &str) -> Result<Format, RecipeError> {
        if self.quantization == Quantization::Trellis {
            return match (name, &self.output_head) {
                ("text/output_head", Some(head)) => Format::of(head, ROW_SPLIT, None),
                ("text/output_head" | "text/draft_head", None) | ("text/draft_head", Some(_)) => {
                    Format::of("NVFP4", ROW_SPLIT, None)
                }
                _ => Err(RecipeError::NotAHead(name.to_owned())),
            };
        }
        if let (true, Some(head)) = (name == "text/output_head", &self.output_head) {
            return Format::of(head, ROW_SPLIT, None);
        }
        let q3 = q3_format(name)?;
        let (format, layout) = if self.quantization == Quantization::Q3 {
            (q3, ROW_SPLIT)
        } else {
            ("NVFP4", if self.compressed && !drafting(name) { SD4 } else { ROW_SPLIT })
        };
        // Compressed q3 stores its Q3 tensors other than the MLP down projections in P2E2, which holds -3..3.
        let lowest = (self.quantization == Quantization::Q3
            && self.compressed
            && format == "Q3G64_F16S"
            && !name.ends_with("/mlp/down"))
        .then_some(-3);
        Format::of(format, layout, lowest)
    }
}

/// The MTP layer and the draft head: they propose tokens the base model verifies.
pub fn drafting(name: &str) -> bool {
    name.starts_with("mtp/") || name == "text/draft_head"
}

/// The object's format in the q3 artifact.
fn q3_format(name: &str) -> Result<&'static str, RecipeError> {
    if name == "text/output_head" || name.starts_with("mtp/") {
        return Ok("NVFP4");
    }
    if name == "text/draft_head" || name == "text/token_embedding" {
        return Ok("Q3G64_F16S");
    }
    let unknown = || RecipeError::UnknownObject(name.to_owned());
    let (layer, role) = name.strip_prefix("text/layers/").and_then(|rest| rest.split_once('/')).ok_or_else(unknown)?;
    let layer: u32 = layer.parse().map_err(|_| unknown())?;
    match role {
        "attention/query_key" | "gdn/query_key" | "gdn/value_z" => Ok("Q4G64_F16S"),
        "attention/gate_value" => Ok("Q5G64_F16S"),
        "attention/output" | "gdn/output" | "mlp/gate_up" => Ok("Q3G64_F16S"),
        // The MLP down projections of the first and last six layers carry most of their family's error.
        "mlp/down" => Ok(if layer < 6 || layer >= 58 { "Q4G64_F16S" } else { "Q3G64_F16S" }),
        _ => Err(unknown()),
    }
}
