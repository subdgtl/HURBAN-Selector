use std::ops::{Add, Div, Mul, Sub};

use num_traits::{Bounded, FromPrimitive, ToPrimitive};

/// Interval is a set of real numbers lying between two numbers, the extremities
/// (left and right) of the interval.
/// # Source
/// https://en.wikipedia.org/wiki/Interval_(mathematics)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval<T> {
    left: T,
    right: T,
    left_infinite: bool,
    right_infinite: bool,
}

impl<
        T: Add<Output = T>
            + Bounded
            + Copy
            + Div<Output = T>
            + FromPrimitive
            + Mul<Output = T>
            + PartialOrd
            + Sub<Output = T>
            + ToPrimitive,
    > Interval<T>
{
    /// Creates a new interval from two extremities. Left and right extremity
    /// don't need to be ordered.
    #[allow(dead_code)]
    pub fn new(left: T, right: T) -> Self {
        Interval {
            left,
            right,
            left_infinite: false,
            right_infinite: false,
        }
    }

    /// Creates a new interval from right extremity, with left extremity
    /// infinite.
    #[allow(dead_code)]
    pub fn new_left_infinite(right: T) -> Self {
        Interval {
            left: T::min_value(),
            right,
            left_infinite: true,
            right_infinite: false,
        }
    }

    /// Creates a new interval from left extremity, with right extremity
    /// infinite.
    #[allow(dead_code)]
    pub fn new_right_infinite(left: T) -> Self {
        Interval {
            left,
            right: T::min_value(),
            left_infinite: false,
            right_infinite: true,
        }
    }

    /// Creates a new closed interval from an iterator of numbers as bounds of
    /// these numbers. The interval will contain all the input numbers, the
    /// lowest number will become the left extremity and the highest number will
    /// become the right extremity.
    #[allow(dead_code)]
    pub fn from_values<I>(values: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut left = T::max_value();
        let mut right = T::min_value();
        for v in values {
            if v < left {
                left = v;
            }
            if v > right {
                right = v;
            }
        }
        Interval {
            left,
            right,
            left_infinite: false,
            right_infinite: false,
        }
    }

    /// Computes the length of the interval. Returns none if any of the bounds
    /// are infinite.
    #[allow(dead_code)]
    pub fn length(&self) -> Option<T> {
        if self.left_infinite || self.right_infinite {
            None
        } else {
            let (min, max) = if self.right > self.left {
                (self.left, self.right)
            } else {
                (self.right, self.left)
            };
            Some(max - min)
        }
    }

    /// Returns the left limit of the interval or None if left is infinite.
    pub fn left(&self) -> Option<T> {
        if self.left_infinite {
            None
        } else {
            Some(self.left)
        }
    }

    /// Returns the right limit of the interval or None if right is infinite.
    pub fn right(&self) -> Option<T> {
        if self.right_infinite {
            None
        } else {
            Some(self.right)
        }
    }

    /// Computes the length of the interval. Returns none if any of the bounds
    /// are infinite.
    #[allow(dead_code)]
    pub fn centre(&self) -> Option<T> {
        if self.left_infinite || self.right_infinite {
            None
        } else {
            let (min, max) = if self.right > self.left {
                (self.left, self.right)
            } else {
                (self.right, self.left)
            };
            Some((min + max) / T::from_f32(2_f32).unwrap())
        }
    }

    /// Checks if the open interval includes the input value. If the value
    /// equals any of the extremities, it is not considered included.
    #[allow(dead_code)]
    pub fn includes_open(&self, value: T) -> bool {
        if self.left_infinite && self.right_infinite {
            return true;
        }
        if self.left_infinite {
            return value < self.right;
        }
        if self.right_infinite {
            return value > self.left;
        }
        let (min, max) = if self.right > self.left {
            (self.left, self.right)
        } else {
            (self.right, self.left)
        };
        value > min && value < max
    }

    /// Checks if the closed interval includes the input value. If the value
    /// equals any of the extremities, it is considered included.
    #[allow(dead_code)]
    pub fn includes_closed(&self, value: T) -> bool {
        if self.left_infinite && self.right_infinite {
            return true;
        }
        if self.left_infinite {
            return value <= self.right;
        }
        if self.right_infinite {
            return value >= self.left;
        }
        let (min, max) = if self.right > self.left {
            (self.left, self.right)
        } else {
            (self.right, self.left)
        };
        value >= min && value <= max
    }

    /// Checks if the left-closed interval includes the input value. If the
    /// value equals the right extremity, it is considered included. If the
    /// value equals the left extremity, it is not considered included.
    #[allow(dead_code)]
    pub fn includes_left_open_right_closed(&self, value: T) -> bool {
        if self.left_infinite && self.right_infinite {
            return true;
        }
        if self.left_infinite {
            return value <= self.right;
        }
        if self.right_infinite {
            return value > self.left;
        }
        let (min, max) = if self.right > self.left {
            (self.left, self.right)
        } else {
            (self.right, self.left)
        };
        value > min && value <= max
    }

    /// Checks if the right-closed interval includes the input value. If the
    /// value equals the left extremity, it is considered included. If the value
    /// equals the right extremity, it is not considered included.
    #[allow(dead_code)]
    pub fn includes_left_closed_right_open(&self, value: T) -> bool {
        if self.left_infinite && self.right_infinite {
            return true;
        }
        if self.left_infinite {
            return value < self.right;
        }
        if self.right_infinite {
            return value >= self.left;
        }
        let (min, max) = if self.right > self.left {
            (self.left, self.right)
        } else {
            (self.right, self.left)
        };
        value >= min && value < max
    }

    #[allow(dead_code)]
    pub fn remap_to(&self, value: T, target_interval: Interval<T>) -> Option<T> {
        if self.left_infinite
            || self.right_infinite
            || target_interval.left_infinite
            || target_interval.right_infinite
        {
            None
        } else {
            let left1 = self.left.to_f64().expect("Can't convert to f64");
            let right1 = self.right.to_f64().expect("Can't convert to f64");
            let left2 = target_interval.left.to_f64().expect("Can't convert to f64");
            let v = value.to_f64().expect("Can't convert to f64");
            let right2 = target_interval
                .right
                .to_f64()
                .expect("Can't convert to f64");

            let remapped = left2 + ((v - left1) / (right1 - left1)) * (right2 - left2);

            Some(T::from_f64(remapped).expect("Can't convert from f64"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_new_i32() {
        Interval::new(0_i32, 1_i32);
    }

    #[test]
    fn test_interval_new_u32() {
        Interval::new(0_u32, 1_u32);
    }

    #[test]
    fn test_interval_new_usize() {
        Interval::new(0_usize, 1_usize);
    }

    #[test]
    fn test_interval_new_f32() {
        Interval::new(0_f32, 1_f32);
    }

    #[test]
    fn test_interval_new_left_infinite_i32() {
        Interval::new_left_infinite(1_i32);
    }

    #[test]
    fn test_interval_new_left_infinite_u32() {
        Interval::new_left_infinite(1_u32);
    }

    #[test]
    fn test_interval_new_left_infinite_usize() {
        Interval::new_left_infinite(1_usize);
    }

    #[test]
    fn test_interval_new_left_infinite_f32() {
        Interval::new_left_infinite(1_f32);
    }

    #[test]
    fn test_interval_new_right_infinite_i32() {
        Interval::new_right_infinite(1_i32);
    }

    #[test]
    fn test_interval_new_right_infinite_u32() {
        Interval::new_right_infinite(1_u32);
    }

    #[test]
    fn test_interval_new_right_infinite_usize() {
        Interval::new_right_infinite(1_usize);
    }

    #[test]
    fn test_interval_new_right_infinite_f32() {
        Interval::new_right_infinite(1_f32);
    }

    #[test]
    fn test_interval_from_values_i32() {
        let values = 1..10_i32;
        let interval = Interval::from_values(values);
        let interval_correct = Interval::new(1_i32, 9_i32);
        assert_eq!(interval, interval_correct);
    }

    #[test]
    fn test_interval_from_values_u32() {
        let values = 1..10_u32;
        let interval = Interval::from_values(values);
        let interval_correct = Interval::new(1_u32, 9_u32);
        assert_eq!(interval, interval_correct);
    }

    #[test]
    fn test_interval_from_values_usize() {
        let values = 1..10_usize;
        let interval = Interval::from_values(values);
        let interval_correct = Interval::new(1_usize, 9_usize);
        assert_eq!(interval, interval_correct);
    }

    #[test]
    fn test_interval_from_values_f32() {
        let values = (1..10).map(|v| v as f32);
        let interval = Interval::from_values(values);
        let interval_correct = Interval::new(1_f32, 9_f32);
        assert_eq!(interval, interval_correct);
    }

    #[test]
    fn test_interval_centre_i32() {
        let interval = Interval::new(1_i32, 2_i32);
        let centre = interval.centre().unwrap();
        let centre_correct = 1_i32;
        assert_eq!(centre, centre_correct);
    }

    #[test]
    fn test_interval_centre_i32_negative() {
        let interval = Interval::new(-2_i32, -1_i32);
        let centre = interval.centre().unwrap();
        let centre_correct = -1_i32;
        assert_eq!(centre, centre_correct);
    }

    #[test]
    fn test_interval_centre_u32() {
        let interval = Interval::new(1_u32, 2_u32);
        let centre = interval.centre().unwrap();
        let centre_correct = 1_u32;
        assert_eq!(centre, centre_correct);
    }

    #[test]
    fn test_interval_centre_usize() {
        let interval = Interval::new(1_usize, 2_usize);
        let centre = interval.centre().unwrap();
        let centre_correct = 1_usize;
        assert_eq!(centre, centre_correct);
    }

    #[test]
    fn test_interval_centre_f32() {
        let interval = Interval::new(1_f32, 2_f32);
        let centre = interval.centre().unwrap();
        let centre_correct = 1.5_f32;
        assert_eq!(centre, centre_correct);
    }

    #[test]
    fn test_interval_centre_f32_negative() {
        let interval = Interval::new(-2_f32, -1_f32);
        let centre = interval.centre().unwrap();
        let centre_correct = -1.5_f32;
        assert_eq!(centre, centre_correct);
    }

    #[test]
    fn test_interval_includes_closed_i32() {
        let interval = Interval::new(1_i32, 2_i32);
        assert!(interval.includes_closed(1_i32));
        assert!(interval.includes_closed(2_i32));
    }

    #[test]
    fn test_interval_includes_closed_u32() {
        let interval = Interval::new(1_u32, 2_u32);
        assert!(interval.includes_closed(1_u32));
        assert!(interval.includes_closed(2_u32));
    }

    #[test]
    fn test_interval_includes_closed_usize() {
        let interval = Interval::new(1_usize, 2_usize);
        assert!(interval.includes_closed(1_usize));
        assert!(interval.includes_closed(2_usize));
    }

    #[test]
    fn test_interval_includes_closed_f32() {
        let interval = Interval::new(1_f32, 2_f32);
        assert!(interval.includes_closed(1_f32));
        assert!(interval.includes_closed(2_f32));
    }

    #[test]
    fn test_interval_includes_closed_does_not_include_i32() {
        let interval = Interval::new(1_i32, 2_i32);
        assert!(!interval.includes_closed(0_i32));
        assert!(!interval.includes_closed(3_i32));
    }

    #[test]
    fn test_interval_includes_closed_does_not_include_u32() {
        let interval = Interval::new(1_u32, 2_u32);
        assert!(!interval.includes_closed(0_u32));
        assert!(!interval.includes_closed(3_u32));
    }

    #[test]
    fn test_interval_includes_closed_does_not_include_usize() {
        let interval = Interval::new(1_usize, 2_usize);
        assert!(!interval.includes_closed(0_usize));
        assert!(!interval.includes_closed(3_usize));
    }

    #[test]
    fn test_interval_includes_closed_does_not_include_f32() {
        let interval = Interval::new(1_f32, 2_f32);
        assert!(!interval.includes_closed(0_f32));
        assert!(!interval.includes_closed(3_f32));
    }

    #[test]
    fn test_interval_includes_open_i32() {
        let interval = Interval::new(1_i32, 3_i32);
        assert!(!interval.includes_open(1_i32));
        assert!(interval.includes_open(2_i32));
        assert!(!interval.includes_open(3_i32));
    }

    #[test]
    fn test_interval_includes_open_u32() {
        let interval = Interval::new(1_u32, 3_u32);
        assert!(!interval.includes_open(1_u32));
        assert!(interval.includes_open(2_u32));
        assert!(!interval.includes_open(3_u32));
    }

    #[test]
    fn test_interval_includes_open_usize() {
        let interval = Interval::new(1_usize, 3_usize);
        assert!(!interval.includes_open(1_usize));
        assert!(interval.includes_open(2_usize));
        assert!(!interval.includes_open(3_usize));
    }

    #[test]
    fn test_interval_includes_open_f32() {
        let interval = Interval::new(1_f32, 3_f32);
        assert!(!interval.includes_open(1_f32));
        assert!(interval.includes_open(2_f32));
        assert!(!interval.includes_open(3_f32));
    }

    #[test]
    fn test_interval_includes_open_does_not_include_i32() {
        let interval = Interval::new(1_i32, 2_i32);
        assert!(!interval.includes_open(0_i32));
        assert!(!interval.includes_open(3_i32));
    }

    #[test]
    fn test_interval_includes_open_does_not_include_u32() {
        let interval = Interval::new(1_u32, 2_u32);
        assert!(!interval.includes_open(0_u32));
        assert!(!interval.includes_open(3_u32));
    }

    #[test]
    fn test_interval_includes_open_does_not_include_usize() {
        let interval = Interval::new(1_usize, 2_usize);
        assert!(!interval.includes_open(0_usize));
        assert!(!interval.includes_open(3_usize));
    }

    #[test]
    fn test_interval_includes_open_does_not_include_f32() {
        let interval = Interval::new(1_f32, 2_f32);
        assert!(!interval.includes_open(0_f32));
        assert!(!interval.includes_open(3_f32));
    }

    #[test]
    fn test_interval_includes_left_closed_right_open_i32() {
        let interval = Interval::new(1_i32, 2_i32);
        assert!(interval.includes_left_closed_right_open(1_i32));
        assert!(!interval.includes_left_closed_right_open(2_i32));
    }

    #[test]
    fn test_interval_includes_left_closed_right_open_u32() {
        let interval = Interval::new(1_u32, 2_u32);
        assert!(interval.includes_left_closed_right_open(1_u32));
        assert!(!interval.includes_left_closed_right_open(2_u32));
    }

    #[test]
    fn test_interval_includes_left_closed_right_open_usize() {
        let interval = Interval::new(1_usize, 2_usize);
        assert!(interval.includes_left_closed_right_open(1_usize));
        assert!(!interval.includes_left_closed_right_open(2_usize));
    }

    #[test]
    fn test_interval_includes_left_closed_right_open_f32() {
        let interval = Interval::new(1_f32, 2_f32);
        assert!(interval.includes_left_closed_right_open(1_f32));
        assert!(!interval.includes_left_closed_right_open(2_f32));
    }

    #[test]
    fn test_interval_includes_left_open_right_closed_i32() {
        let interval = Interval::new(1_i32, 2_i32);
        assert!(!interval.includes_left_open_right_closed(1_i32));
        assert!(interval.includes_left_open_right_closed(2_i32));
    }

    #[test]
    fn test_interval_includes_left_open_right_closed_u32() {
        let interval = Interval::new(1_u32, 2_u32);
        assert!(!interval.includes_left_open_right_closed(1_u32));
        assert!(interval.includes_left_open_right_closed(2_u32));
    }

    #[test]
    fn test_interval_includes_left_open_right_closed_usize() {
        let interval = Interval::new(1_usize, 2_usize);
        assert!(!interval.includes_left_open_right_closed(1_usize));
        assert!(interval.includes_left_open_right_closed(2_usize));
    }

    #[test]
    fn test_interval_includes_left_open_right_closed_f32() {
        let interval = Interval::new(1_f32, 2_f32);
        assert!(!interval.includes_left_open_right_closed(1_f32));
        assert!(interval.includes_left_open_right_closed(2_f32));
    }

    #[test]
    fn test_interval_length_i32() {
        let interval = Interval::new(0_i32, 1_i32);
        // Note: This is equivalent to Grasshopper
        assert_eq!(interval.length().unwrap(), 1_i32);
    }

    #[test]
    fn test_interval_length_u32() {
        let interval = Interval::new(0_u32, 1_u32);
        // Note: This is equivalent to Grasshopper
        assert_eq!(interval.length().unwrap(), 1_u32);
    }

    #[test]
    fn test_interval_length_usize() {
        let interval = Interval::new(0_usize, 1_usize);
        // Note: This is equivalent to Grasshopper
        assert_eq!(interval.length().unwrap(), 1_usize);
    }

    #[test]
    fn test_interval_length_f32() {
        let interval = Interval::new(0_f32, 1_f32);
        // Note: This is equivalent to Grasshopper
        assert_eq!(interval.length().unwrap(), 1_f32);
    }

    #[test]
    fn test_interval_length_infinite_i32() {
        let interval = Interval::new_right_infinite(0_i32);
        assert_eq!(interval.length(), None);
    }

    #[test]
    fn test_interval_length_infinite_u32() {
        let interval = Interval::new_right_infinite(0_u32);
        assert_eq!(interval.length(), None);
    }

    #[test]
    fn test_interval_length_infinite_usize() {
        let interval = Interval::new_right_infinite(0_usize);
        assert_eq!(interval.length(), None);
    }

    #[test]
    fn test_interval_length_infinite_f32() {
        let interval = Interval::new_right_infinite(0_f32);
        assert_eq!(interval.length(), None);
    }

    #[test]
    fn test_interval_remap_interpolated_f32() {
        let interval_source = Interval::new(0_f32, 2_f32);
        let interval_target = Interval::new(10_f32, 20_f32);
        assert_eq!(
            interval_source.remap_to(1_f32, interval_target).unwrap(),
            15_f32
        );
    }

    #[test]
    fn test_interval_remap_interpolated_reverted_f32() {
        let interval_source = Interval::new(0_f32, 2_f32);
        let interval_target = Interval::new(20_f32, 10_f32);
        assert_eq!(
            interval_source.remap_to(1_f32, interval_target).unwrap(),
            15_f32
        );
    }

    #[test]
    fn test_interval_remap_extrapolated_left_f32() {
        let interval_source = Interval::new(10_f32, 20_f32);
        let interval_target = Interval::new(10_f32, 30_f32);
        assert_eq!(
            interval_source.remap_to(9_f32, interval_target).unwrap(),
            8_f32
        );
    }

    #[test]
    fn test_interval_remap_extrapolated_right_f32() {
        let interval_source = Interval::new(10_f32, 20_f32);
        let interval_target = Interval::new(10_f32, 30_f32);
        assert_eq!(
            interval_source.remap_to(21_f32, interval_target).unwrap(),
            32_f32
        );
    }

    #[test]
    fn test_interval_remap_interpolated_i32() {
        let interval_source = Interval::new(0_i32, 2_i32);
        let interval_target = Interval::new(10_i32, 20_i32);
        assert_eq!(
            interval_source.remap_to(1_i32, interval_target).unwrap(),
            15_i32
        );
    }

    #[test]
    fn test_interval_remap_interpolated_reverted_i32() {
        let interval_source = Interval::new(0_i32, 2_i32);
        let interval_target = Interval::new(20_i32, 10_i32);
        assert_eq!(
            interval_source.remap_to(1_i32, interval_target).unwrap(),
            15_i32
        );
    }

    #[test]
    fn test_interval_remap_extrapolated_left_i32() {
        let interval_source = Interval::new(10_i32, 20_i32);
        let interval_target = Interval::new(10_i32, 30_i32);
        assert_eq!(
            interval_source.remap_to(9_i32, interval_target).unwrap(),
            8_i32
        );
    }

    #[test]
    fn test_interval_remap_extrapolated_right_i32() {
        let interval_source = Interval::new(10_i32, 20_i32);
        let interval_target = Interval::new(10_i32, 30_i32);
        assert_eq!(
            interval_source.remap_to(21_i32, interval_target).unwrap(),
            32_i32
        );
    }

    #[test]
    fn test_interval_remap_interpolated_u32() {
        let interval_source = Interval::new(0_u32, 2_u32);
        let interval_target = Interval::new(10_u32, 20_u32);
        assert_eq!(
            interval_source.remap_to(1_u32, interval_target).unwrap(),
            15_u32
        );
    }

    #[test]
    fn test_interval_remap_interpolated_reverted_u32() {
        let interval_source = Interval::new(0_u32, 2_u32);
        let interval_target = Interval::new(20_u32, 10_u32);
        assert_eq!(
            interval_source.remap_to(1_u32, interval_target).unwrap(),
            15_u32
        );
    }

    #[test]
    fn test_interval_remap_extrapolated_left_u32() {
        let interval_source = Interval::new(10_u32, 20_u32);
        let interval_target = Interval::new(10_u32, 30_u32);
        assert_eq!(
            interval_source.remap_to(9_u32, interval_target).unwrap(),
            8_u32
        );
    }

    #[test]
    fn test_interval_remap_extrapolated_right_u32() {
        let interval_source = Interval::new(10_u32, 20_u32);
        let interval_target = Interval::new(10_u32, 30_u32);
        assert_eq!(
            interval_source.remap_to(21_u32, interval_target).unwrap(),
            32_u32
        );
    }

    #[test]
    fn test_interval_remap_interpolated_usize() {
        let interval_source = Interval::new(0_usize, 2_usize);
        let interval_target = Interval::new(10_usize, 20_usize);
        assert_eq!(
            interval_source.remap_to(1_usize, interval_target).unwrap(),
            15_usize
        );
    }

    #[test]
    fn test_interval_remap_interpolated_reverted_usize() {
        let interval_source = Interval::new(0_usize, 2_usize);
        let interval_target = Interval::new(20_usize, 10_usize);
        assert_eq!(
            interval_source.remap_to(1_usize, interval_target).unwrap(),
            15_usize
        );
    }

    #[test]
    fn test_interval_remap_extrapolated_left_usize() {
        let interval_source = Interval::new(10_usize, 20_usize);
        let interval_target = Interval::new(10_usize, 30_usize);
        assert_eq!(
            interval_source.remap_to(9_usize, interval_target).unwrap(),
            8_usize
        );
    }

    #[test]
    fn test_interval_remap_extrapolated_right_usize() {
        let interval_source = Interval::new(10_usize, 20_usize);
        let interval_target = Interval::new(10_usize, 30_usize);
        assert_eq!(
            interval_source.remap_to(21_usize, interval_target).unwrap(),
            32_usize
        );
    }
}
