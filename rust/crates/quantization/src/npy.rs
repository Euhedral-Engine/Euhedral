//! The `.npy` array format: little-endian, C order, format version 1.0 on write, versions 1 to 3 on read.
//!
//! The writer produces the same bytes for the same array on every run: the dictionary header written as
//! `{'descr': 'D', 'fortran_order': False, 'shape': (S), }`, padded with spaces so the header ends on a 64-byte
//! boundary, then the data.

use std::iter::repeat_n;
use std::str::from_utf8;

use thiserror::Error;

const MAGIC: [u8; 6] = [0x93, b'N', b'U', b'M', b'P', b'Y'];

/// Why an array could not be read or written.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum NpyError {
    #[error("not a .npy file: {0}")]
    Malformed(String),
    #[error("unsupported array: {0}")]
    Unsupported(String),
    #[error("the array is {found}, not {wanted}")]
    WrongType { found: String, wanted: String },
    #[error("the shape holds {expected} bytes but {found} are given")]
    DataLength { expected: u64, found: u64 },
}

/// An element type of an array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    I1,
    U1,
    I2,
    I4,
    I8,
    /// Half-precision floats, held as their 16 bits.
    F2,
    F4,
    /// A unicode string of this many code points, four bytes each (`<U<n>`).
    Unicode(usize),
}

impl Dtype {
    /// The bytes of one element.
    pub fn item_size(self) -> usize {
        match self {
            Dtype::I1 | Dtype::U1 => 1,
            Dtype::I2 | Dtype::F2 => 2,
            Dtype::I4 | Dtype::F4 => 4,
            Dtype::I8 => 8,
            Dtype::Unicode(length) => 4 * length,
        }
    }

    /// The `descr` text the writer puts in the header.
    pub fn descr(self) -> String {
        match self {
            Dtype::I1 => "|i1".to_string(),
            Dtype::U1 => "|u1".to_string(),
            Dtype::I2 => "<i2".to_string(),
            Dtype::I4 => "<i4".to_string(),
            Dtype::I8 => "<i8".to_string(),
            Dtype::F2 => "<f2".to_string(),
            Dtype::F4 => "<f4".to_string(),
            Dtype::Unicode(length) => format!("<U{length}"),
        }
    }

    /// The type a `descr` names. A one-byte type may be written with `|` or `<`.
    pub fn parse(descr: &str) -> Result<Dtype, NpyError> {
        let dtype = match descr {
            "|i1" | "<i1" => Dtype::I1,
            "|u1" | "<u1" => Dtype::U1,
            "<i2" => Dtype::I2,
            "<i4" => Dtype::I4,
            "<i8" => Dtype::I8,
            "<f2" => Dtype::F2,
            "<f4" => Dtype::F4,
            other => match other.strip_prefix("<U").map(str::parse::<usize>) {
                Some(Ok(length)) => Dtype::Unicode(length),
                _ => return Err(NpyError::Unsupported(format!("element type '{other}'"))),
            },
        };
        Ok(dtype)
    }
}

/// A Rust type that is one element of an array.
pub trait Element: Copy {
    const DTYPE: Dtype;
    /// Reads one element from `Self::DTYPE.item_size()` little-endian bytes.
    fn read(bytes: &[u8]) -> Self;
    /// Appends the element's little-endian bytes.
    fn write(self, out: &mut Vec<u8>);
}

/// A half-precision float, held as its 16 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Half(pub u16);

macro_rules! element {
    ($type:ty, $dtype:expr) => {
        impl Element for $type {
            const DTYPE: Dtype = $dtype;
            fn read(bytes: &[u8]) -> Self {
                let mut buffer = [0u8; size_of::<$type>()];
                buffer.copy_from_slice(bytes);
                <$type>::from_le_bytes(buffer)
            }
            fn write(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
        }
    };
}

element!(i8, Dtype::I1);
element!(u8, Dtype::U1);
element!(i16, Dtype::I2);
element!(i32, Dtype::I4);
element!(i64, Dtype::I8);
element!(f32, Dtype::F4);

impl Element for Half {
    const DTYPE: Dtype = Dtype::F2;
    fn read(bytes: &[u8]) -> Self {
        Half(u16::from_le_bytes([bytes[0], bytes[1]]))
    }
    fn write(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0.to_le_bytes());
    }
}

/// The bytes of an array of `shape`, or `None` when they do not fit in 64 bits.
fn byte_count(dtype: Dtype, shape: &[u64]) -> Option<u64> {
    shape.iter().try_fold(dtype.item_size() as u64, |total, &dimension| {
        total.checked_mul(dimension)
    })
}

/// The version 1.0 header of an array: magic, version, length and the padded dictionary.
pub fn header(dtype: Dtype, shape: &[u64]) -> Vec<u8> {
    let mut dimensions = String::from("(");
    for (index, dimension) in shape.iter().enumerate() {
        dimensions.push_str(&dimension.to_string());
        if shape.len() == 1 {
            dimensions.push(',');
        } else if index + 1 < shape.len() {
            dimensions.push_str(", ");
        }
    }
    dimensions.push(')');
    let mut dictionary = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {dimensions}, }}",
        dtype.descr()
    );
    let unpadded = 10 + dictionary.len() + 1;
    let padding = (64 - unpadded % 64) % 64;
    dictionary.extend(repeat_n(' ', padding));
    dictionary.push('\n');
    let mut bytes = Vec::with_capacity(10 + dictionary.len());
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&[1, 0]);
    bytes.extend_from_slice(&(dictionary.len() as u16).to_le_bytes());
    bytes.extend_from_slice(dictionary.as_bytes());
    bytes
}

/// A whole `.npy` file: the header and `data`, which must be exactly the bytes of the shape.
pub fn encode(dtype: Dtype, shape: &[u64], data: &[u8]) -> Result<Vec<u8>, NpyError> {
    let expected = byte_count(dtype, shape).ok_or_else(|| NpyError::Unsupported("a shape beyond 64 bits".into()))?;
    if expected != data.len() as u64 {
        return Err(NpyError::DataLength {
            expected,
            found: data.len() as u64,
        });
    }
    let mut bytes = header(dtype, shape);
    bytes.extend_from_slice(data);
    Ok(bytes)
}

/// A whole `.npy` file of `values`.
pub fn encode_elements<T: Element>(shape: &[u64], values: &[T]) -> Result<Vec<u8>, NpyError> {
    let mut data = Vec::with_capacity(values.len() * T::DTYPE.item_size());
    for &value in values {
        value.write(&mut data);
    }
    encode(T::DTYPE, shape, &data)
}

/// A whole `.npy` file holding `value` as a 0-d unicode array.
pub fn encode_text(value: &str) -> Vec<u8> {
    let codes: Vec<u8> = value.chars().flat_map(|c| (c as u32).to_le_bytes()).collect();
    let mut bytes = header(Dtype::Unicode(value.chars().count()), &[]);
    bytes.extend_from_slice(&codes);
    bytes
}

/// A parsed `.npy` file: the type, the shape and the data, which stays in the caller's bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Npy<'a> {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub data: &'a [u8],
}

impl<'a> Npy<'a> {
    /// Parses a `.npy` file of format version 1, 2 or 3. An array in Fortran order, a type outside [`Dtype`] and data
    /// whose length differs from the shape's are errors.
    pub fn parse(bytes: &'a [u8]) -> Result<Npy<'a>, NpyError> {
        if bytes.len() < 10 || bytes[..6] != MAGIC {
            return Err(NpyError::Malformed("no magic".into()));
        }
        let (length, start): (usize, usize) = match bytes[6] {
            1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
            2 | 3 => {
                let field = bytes
                    .get(8..12)
                    .ok_or_else(|| NpyError::Malformed("the header is cut".into()))?;
                (
                    u32::from_le_bytes([field[0], field[1], field[2], field[3]]) as usize,
                    12,
                )
            }
            other => return Err(NpyError::Unsupported(format!("format version {other}"))),
        };
        let text = start
            .checked_add(length)
            .and_then(|end| bytes.get(start..end))
            .ok_or_else(|| NpyError::Malformed("the header is cut".into()))?;
        let text = from_utf8(text).map_err(|_| NpyError::Malformed("the header is not text".into()))?;
        let dtype = Dtype::parse(quoted(text, "'descr'")?)?;
        if value_of(text, "'fortran_order'")?.starts_with("True") {
            return Err(NpyError::Unsupported("an array in Fortran order".into()));
        }
        let shape = shape_of(text)?;
        let data = &bytes[start + length..];
        let expected = byte_count(dtype, &shape).ok_or_else(|| NpyError::Malformed("a shape beyond 64 bits".into()))?;
        if expected != data.len() as u64 {
            return Err(NpyError::DataLength {
                expected,
                found: data.len() as u64,
            });
        }
        Ok(Npy { dtype, shape, data })
    }

    /// The number of elements.
    pub fn len(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }

    /// Whether the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The elements, when the array's type is `T`'s.
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>, NpyError> {
        if self.dtype != T::DTYPE {
            return Err(NpyError::WrongType {
                found: self.dtype.descr(),
                wanted: T::DTYPE.descr(),
            });
        }
        Ok(self.data.chunks_exact(T::DTYPE.item_size()).map(T::read).collect())
    }

    /// The text of a 0-d unicode array: its code points, without the NUL padding.
    pub fn text(&self) -> Result<String, NpyError> {
        if !matches!(self.dtype, Dtype::Unicode(_)) {
            return Err(NpyError::WrongType {
                found: self.dtype.descr(),
                wanted: "<U".into(),
            });
        }
        let mut text = String::new();
        for chunk in self.data.as_chunks::<4>().0 {
            let code = u32::from_le_bytes(*chunk);
            if code != 0 {
                text.push(char::from_u32(code).ok_or_else(|| NpyError::Malformed("a bad code point".into()))?);
            }
        }
        Ok(text)
    }
}

/// The text after `key:` in the header dictionary.
fn value_of<'t>(text: &'t str, key: &str) -> Result<&'t str, NpyError> {
    let at = text
        .find(key)
        .ok_or_else(|| NpyError::Malformed(format!("the header has no {key}")))?;
    let rest = text[at + key.len()..].trim_start();
    rest.strip_prefix(':')
        .map(str::trim_start)
        .ok_or_else(|| NpyError::Malformed(format!("the header's {key} has no value")))
}

/// The quoted text after `key:`.
fn quoted<'t>(text: &'t str, key: &str) -> Result<&'t str, NpyError> {
    let rest = value_of(text, key)?
        .strip_prefix('\'')
        .ok_or_else(|| NpyError::Malformed(format!("the header's {key} is not quoted")))?;
    rest.split('\'')
        .next()
        .ok_or_else(|| NpyError::Malformed(format!("the header's {key} is not closed")))
}

fn shape_of(text: &str) -> Result<Vec<u64>, NpyError> {
    let rest = value_of(text, "'shape'")?
        .strip_prefix('(')
        .ok_or_else(|| NpyError::Malformed("the shape is not a tuple".into()))?;
    let inside = rest
        .split(')')
        .next()
        .ok_or_else(|| NpyError::Malformed("the shape is not closed".into()))?;
    inside
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<u64>()
                .map_err(|_| NpyError::Malformed(format!("the dimension '{part}' is not a number")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_of_a_matrix_is_padded_to_64_bytes() {
        let bytes = header(Dtype::F4, &[2, 3]);
        assert_eq!(bytes.len() % 64, 0);
        assert_eq!(&bytes[..10], &[0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0, 118, 0]);
        let text = from_utf8(&bytes[10..]).unwrap();
        assert!(text.starts_with("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }"));
        assert!(text.ends_with(" \n"));
    }

    #[test]
    fn one_dimension_has_a_trailing_comma_and_a_scalar_none() {
        let vector = header(Dtype::I8, &[5]);
        assert!(from_utf8(&vector[10..]).unwrap().contains("'shape': (5,), }"));
        let scalar = header(Dtype::F4, &[]);
        assert!(from_utf8(&scalar[10..]).unwrap().contains("'shape': (), }"));
    }

    #[test]
    fn every_type_round_trips() {
        let bytes = encode_elements(&[2, 2], &[1i16, -2, 3, i16::MIN]).unwrap();
        assert_eq!(
            Npy::parse(&bytes).unwrap().to_vec::<i16>().unwrap(),
            [1, -2, 3, i16::MIN]
        );
        let bytes = encode_elements(&[3], &[Half(0x3c00), Half(0), Half(0xffff)]).unwrap();
        assert_eq!(
            Npy::parse(&bytes).unwrap().to_vec::<Half>().unwrap(),
            [Half(0x3c00), Half(0), Half(0xffff)]
        );
        let bytes = encode_elements(&[], &[1.5f32]).unwrap();
        assert_eq!(Npy::parse(&bytes).unwrap().to_vec::<f32>().unwrap(), [1.5]);
        let bytes = encode_elements(&[2], &[i64::MIN, 7]).unwrap();
        assert_eq!(Npy::parse(&bytes).unwrap().to_vec::<i64>().unwrap(), [i64::MIN, 7]);
        let bytes = encode_elements(&[2], &[-5i32, 9]).unwrap();
        assert_eq!(Npy::parse(&bytes).unwrap().to_vec::<i32>().unwrap(), [-5, 9]);
        let bytes = encode_elements(&[2], &[-5i8, 9]).unwrap();
        assert_eq!(Npy::parse(&bytes).unwrap().to_vec::<i8>().unwrap(), [-5, 9]);
        let bytes = encode_elements(&[2], &[250u8, 9]).unwrap();
        assert_eq!(Npy::parse(&bytes).unwrap().to_vec::<u8>().unwrap(), [250, 9]);
    }

    #[test]
    fn text_round_trips() {
        let bytes = encode_text("NVFP4");
        let npy = Npy::parse(&bytes).unwrap();
        assert_eq!(npy.dtype, Dtype::Unicode(5));
        assert!(npy.shape.is_empty());
        assert_eq!(npy.text().unwrap(), "NVFP4");
        assert_eq!(Npy::parse(&encode_text("")).unwrap().text().unwrap(), "");
    }

    #[test]
    fn a_wrong_type_or_length_is_an_error() {
        let bytes = encode_elements(&[2], &[1i32, 2]).unwrap();
        assert!(matches!(
            Npy::parse(&bytes).unwrap().to_vec::<f32>(),
            Err(NpyError::WrongType { .. })
        ));
        assert!(matches!(
            encode(Dtype::I4, &[3], &[0; 8]),
            Err(NpyError::DataLength { expected: 12, found: 8 })
        ));
        assert!(matches!(
            Npy::parse(&bytes[..bytes.len() - 1]),
            Err(NpyError::DataLength { .. })
        ));
    }

    #[test]
    fn broken_headers_are_errors() {
        assert!(Npy::parse(b"hello world, not an array").is_err());
        assert!(Npy::parse(&[0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0, 255, 255]).is_err());
        let mut fortran = header(Dtype::F4, &[0]);
        let text = String::from_utf8(fortran.split_off(10))
            .unwrap()
            .replace("False", "True ");
        fortran.extend_from_slice(text.as_bytes());
        assert!(matches!(Npy::parse(&fortran), Err(NpyError::Unsupported(_))));
        let mut other = header(Dtype::F4, &[0]);
        let text = String::from_utf8(other.split_off(10)).unwrap().replace("<f4", "<c8");
        other.extend_from_slice(text.as_bytes());
        assert!(matches!(Npy::parse(&other), Err(NpyError::Unsupported(_))));
    }

    #[test]
    fn version_two_headers_are_read() {
        let v1 = encode_elements(&[2], &[1i32, 2]).unwrap();
        let length = usize::from(u16::from_le_bytes([v1[8], v1[9]]));
        let mut v2 = v1[..6].to_vec();
        v2.extend_from_slice(&[2, 0]);
        v2.extend_from_slice(&(length as u32).to_le_bytes());
        v2.extend_from_slice(&v1[10..]);
        assert_eq!(Npy::parse(&v2).unwrap().to_vec::<i32>().unwrap(), [1, 2]);
    }
}
