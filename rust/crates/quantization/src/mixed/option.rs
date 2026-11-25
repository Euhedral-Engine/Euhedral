use std::fmt;

use super::error::MixedError;

/// What a projection object can be stored as in a mixed artifact: NVFP4, or trellis at 2.5, 3.0, 3.5 or 4.0 bits per
/// weight. The order is the declaration order, which sorts allocation files and the cost table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QuantOption {
    Nvfp4,
    Trellis2_5,
    Trellis3_0,
    Trellis3_5,
    Trellis4_0,
}

impl QuantOption {
    /// Every option, in declaration order.
    pub const ALL: [QuantOption; 5] = [
        QuantOption::Nvfp4,
        QuantOption::Trellis2_5,
        QuantOption::Trellis3_0,
        QuantOption::Trellis3_5,
        QuantOption::Trellis4_0,
    ];

    /// The name of the option in allocation files and on the command line.
    pub fn key(self) -> &'static str {
        match self {
            QuantOption::Nvfp4 => "nvfp4",
            QuantOption::Trellis2_5 => "trellis-2.5",
            QuantOption::Trellis3_0 => "trellis-3.0",
            QuantOption::Trellis3_5 => "trellis-3.5",
            QuantOption::Trellis4_0 => "trellis-4.0",
        }
    }

    /// The option named `key`; the refusal lists the options.
    pub fn of(key: &str) -> Result<QuantOption, MixedError> {
        QuantOption::ALL
            .into_iter()
            .find(|option| option.key() == key)
            .ok_or_else(|| {
                let keys: Vec<&str> = QuantOption::ALL.iter().map(|option| option.key()).collect();
                MixedError::invalid(format!("unknown option '{key}', the options are {keys:?}"))
            })
    }
}

impl fmt::Display for QuantOption {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.write_str(self.key())
    }
}
