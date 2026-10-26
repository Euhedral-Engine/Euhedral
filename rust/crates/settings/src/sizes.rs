//! Sizes (`4GiB`, `700MiB`, plain bytes) and durations (`10s`, `30m`, `1000ms`).

use std::fmt;
use std::time::Duration;

/// A size in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ByteSize(pub u64);

impl ByteSize {
    /// The size in bytes.
    pub const fn bytes(self) -> u64 {
        self.0
    }

    /// A size of `mebibytes` MiB.
    pub const fn mib(mebibytes: u64) -> Self {
        Self(mebibytes << 20)
    }

    /// A size of `gibibytes` GiB.
    pub const fn gib(gibibytes: u64) -> Self {
        Self(gibibytes << 30)
    }

    /// Parses a whole number with an optional binary unit (`B`, `KiB`, `MiB`, `GiB`, `TiB`); no unit means bytes.
    pub fn parse(text: &str) -> Result<Self, String> {
        let text = text.trim();
        let digits_end = text.find(|c: char| !c.is_ascii_digit()).unwrap_or(text.len());
        let (digits, unit) = text.split_at(digits_end);
        let shift = match unit.trim_start() {
            "" | "B" => 0,
            "KiB" => 10,
            "MiB" => 20,
            "GiB" => 30,
            "TiB" => 40,
            _ => return Err(format!("not a size (whole number, optional B/KiB/MiB/GiB/TiB): {text}")),
        };
        if digits.is_empty() {
            return Err(format!("not a size (whole number, optional B/KiB/MiB/GiB/TiB): {text}"));
        }
        let value: u64 = digits.parse().map_err(|_| format!("number too large: {text}"))?;
        if value > u64::MAX >> shift {
            return Err(format!("size too large: {text}"));
        }
        Ok(Self(value << shift))
    }
}

impl fmt::Display for ByteSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}B", self.0)
    }
}

/// Parses a whole number with a unit (`ms`, `s`, `m`, `h`); a bare `0` is zero.
pub fn parse_duration(text: &str) -> Result<Duration, String> {
    let text = text.trim();
    if text == "0" {
        return Ok(Duration::ZERO);
    }
    let invalid = || format!("not a duration (whole number with ms/s/m/h): {text}");
    let digits_end = text.find(|c: char| !c.is_ascii_digit()).unwrap_or(text.len());
    let (digits, unit) = text.split_at(digits_end);
    if digits.is_empty() {
        return Err(invalid());
    }
    let value: u64 = digits.parse().map_err(|_| format!("number too large: {text}"))?;
    let too_large = || format!("duration too large: {text}");
    match unit.trim_start() {
        "ms" => Ok(Duration::from_millis(value)),
        "s" => Ok(Duration::from_secs(value)),
        "m" => value.checked_mul(60).map(Duration::from_secs).ok_or_else(too_large),
        "h" => value.checked_mul(3600).map(Duration::from_secs).ok_or_else(too_large),
        _ => Err(invalid()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_parse() {
        assert_eq!(ByteSize::parse("4GiB"), Ok(ByteSize(4_294_967_296)));
        assert_eq!(ByteSize::parse("700MiB"), Ok(ByteSize(734_003_200)));
        assert_eq!(ByteSize::parse("1048576"), Ok(ByteSize(1_048_576)));
        assert_eq!(ByteSize::parse("1KiB"), Ok(ByteSize(1024)));
        assert_eq!(ByteSize::parse("2 MiB"), Ok(ByteSize::mib(2)));
        assert_eq!(ByteSize::parse("0"), Ok(ByteSize(0)));
    }

    #[test]
    fn bad_sizes_are_errors() {
        for text in [
            "1.5GiB",
            "4GB",
            "99999999999TiB",
            "",
            "GiB",
            "-1",
            "1gib",
            "99999999999999999999",
        ] {
            assert!(ByteSize::parse(text).is_err(), "{text}");
        }
    }

    #[test]
    fn durations_parse() {
        assert_eq!(parse_duration("10s"), Ok(Duration::from_secs(10)));
        assert_eq!(parse_duration("30m"), Ok(Duration::from_secs(1800)));
        assert_eq!(parse_duration("1000ms"), Ok(Duration::from_secs(1)));
        assert_eq!(parse_duration("2h"), Ok(Duration::from_secs(7200)));
        assert_eq!(parse_duration("0"), Ok(Duration::ZERO));
    }

    #[test]
    fn bad_durations_are_errors() {
        for text in [
            "10",
            "1.5s",
            "1d",
            "",
            "s",
            "-1s",
            "99999999999999999999s",
            "18446744073709551615h",
        ] {
            assert!(parse_duration(text).is_err(), "{text}");
        }
    }
}
