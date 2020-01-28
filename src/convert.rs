use std::convert::TryInto;
use std::fmt::Debug;

/// Convert `n` to `u32` using `TryFrom` or panic.
///
/// # Panics
/// Panics if the conversion returns an error.
pub fn cast_u32<T>(n: T) -> u32
where
    T: TryInto<u32>,
    <T as TryInto<u32>>::Error: Debug,
{
    n.try_into().expect("Expected N to fit in u32")
}

/// Convert `n` to `i32` using `TryFrom` or panic.
///
/// # Panics
/// Panics if the conversion returns an error.
pub fn cast_i32<T>(n: T) -> i32
where
    T: TryInto<i32>,
    <T as TryInto<i32>>::Error: Debug,
{
    n.try_into().expect("Expected N to fit in i32")
}

/// Convert `n` to `usize` using `TryFrom` or panic.
///
/// # Panics
/// Panics if the conversion returns an error.
pub fn cast_usize<T>(n: T) -> usize
where
    T: TryInto<usize>,
    <T as TryInto<usize>>::Error: Debug,
{
    n.try_into().expect("Expected N to fit in usize")
}

/// Convert u32 to i16 clamping to max value of i16 if necessary.
pub fn clamp_cast_u32_to_i16(n: u32) -> i16 {
    if n > i16::max_value() as u32 {
        i16::max_value()
    } else {
        n as i16
    }
}

/// Convert u32 to i32 clamping to max value of i32 if necessary.
pub fn clamp_cast_u32_to_i32(n: u32) -> i32 {
    if n > i32::max_value() as u32 {
        i32::max_value()
    } else {
        n as i32
    }
}

/// Convert i32 to u32 clamping to 0 if necessary.
pub fn clamp_cast_i32_to_u32(n: i32) -> u32 {
    if n < 0 {
        0
    } else {
        n as u32
    }
}

/// Convert u8 based color array to f32 based color array with
/// floating point domain `[0..1]`.
pub fn cast_u8_color_to_f32(color: [u8; 4]) -> [f32; 4] {
    [
        color[0] as f32 / 255.0,
        color[1] as f32 / 255.0,
        color[2] as f32 / 255.0,
        color[3] as f32 / 255.0,
    ]
}

/// Convert u8 based color array to f64 based color array with
/// floating point domain `[0..1]`.
pub fn cast_u8_color_to_f64(color: [u8; 4]) -> [f64; 4] {
    [
        color[0] as f64 / 255.0,
        color[1] as f64 / 255.0,
        color[2] as f64 / 255.0,
        color[3] as f64 / 255.0,
    ]
}
