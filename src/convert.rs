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
