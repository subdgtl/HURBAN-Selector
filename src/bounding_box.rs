use arrayvec::ArrayVec;
use nalgebra::{Point3, Scalar, Vector3};
use num_traits::{Bounded, Zero};

use crate::convert::clamp_cast_i32_to_u32;

/// World-origin-based axis-aligned bounding box contains the entire given
/// geometry and defines an envelope aligned to the world (euclidean) coordinate
/// system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox<T>
where
    T: Scalar,
{
    minimum_point: Point3<T>,
    maximum_point: Point3<T>,
}

impl<T: Bounded + Scalar + Zero + PartialOrd> BoundingBox<T> {
    /// Creates a new bounding box. The two input points will be deconstructed
    /// and a new couple of points will be created: minimum point with minimum
    /// values of x, y, z and maximum point with maximum values of x, y, z. The
    /// resulting bounding box will be defined in the units of the input points.
    pub fn new(box_corner1: &Point3<T>, box_corner2: &Point3<T>) -> Self {
        BoundingBox {
            minimum_point: Point3::new(
                if box_corner1.x < box_corner2.x {
                    box_corner1.x
                } else {
                    box_corner2.x
                },
                if box_corner1.y < box_corner2.y {
                    box_corner1.y
                } else {
                    box_corner2.y
                },
                if box_corner1.z < box_corner2.z {
                    box_corner1.z
                } else {
                    box_corner2.z
                },
            ),
            maximum_point: Point3::new(
                if box_corner1.x > box_corner2.x {
                    box_corner1.x
                } else {
                    box_corner2.x
                },
                if box_corner1.y > box_corner2.y {
                    box_corner1.y
                } else {
                    box_corner2.y
                },
                if box_corner1.z > box_corner2.z {
                    box_corner1.z
                } else {
                    box_corner2.z
                },
            ),
        }
    }

    /// Creates a new bounding box from an iterator of points. The resulting
    /// bounding box will encompass all the input points. The resulting bounding
    /// box will be defined in the units of the input points. If the input list
    /// is empty, a zero size box at the word origin will be created.
    pub fn from_points<'a, I>(points: I) -> Option<Self>
    where
        I: IntoIterator<Item = &'a Point3<T>> + Clone,
    {
        if points.clone().into_iter().peekable().peek().is_none() {
            return None;
        }

        let mut min_x = T::max_value();
        let mut min_y = T::max_value();
        let mut min_z = T::max_value();
        let mut max_x = T::min_value();
        let mut max_y = T::min_value();
        let mut max_z = T::min_value();

        for point in points {
            if point.x < min_x {
                min_x = point.x;
            }
            if point.y < min_y {
                min_y = point.y;
            }
            if point.z < min_z {
                min_z = point.z;
            }
            if point.x > max_x {
                max_x = point.x;
            }
            if point.y > max_y {
                max_y = point.y;
            }
            if point.z > max_z {
                max_z = point.z;
            }
        }

        Some(BoundingBox {
            minimum_point: Point3::new(min_x, min_y, min_z),
            maximum_point: Point3::new(max_x, max_y, max_z),
        })
    }

    /// Creates a new bounding box encompassing all the input bounding boxes.
    /// The resulting bounding box will be defined in the units of the input
    /// bounding boxes.
    pub fn union<'a, I>(bounding_boxes: I) -> Option<Self>
    where
        I: IntoIterator<Item = &'a BoundingBox<T>>,
    {
        // FIXME: @Optimization Remove the allocation, try from_fn or successors
        let points: Vec<_> = bounding_boxes
            .into_iter()
            .flat_map(|b_box| ArrayVec::from(b_box.corners()).into_iter())
            .collect();
        BoundingBox::from_points(points.iter())
    }

    /// Creates a new bounding box so that it encloses the block of space
    /// common to all the input bounding boxes, including the current one.
    ///
    /// Returns None if there is no intersection.
    pub fn intersection<'a, I>(bounding_boxes: I) -> Option<Self>
    where
        I: IntoIterator<Item = &'a BoundingBox<T>> + Clone,
    {
        if bounding_boxes
            .clone()
            .into_iter()
            .peekable()
            .peek()
            .is_none()
        {
            return None;
        }

        let mut min_x = T::min_value();
        let mut min_y = T::min_value();
        let mut min_z = T::min_value();
        let mut max_x = T::max_value();
        let mut max_y = T::max_value();
        let mut max_z = T::max_value();

        for bounding_box in bounding_boxes {
            if bounding_box.minimum_point.x > min_x {
                min_x = bounding_box.minimum_point.x;
            }
            if bounding_box.minimum_point.y > min_y {
                min_y = bounding_box.minimum_point.y;
            }
            if bounding_box.minimum_point.z > min_z {
                min_z = bounding_box.minimum_point.z;
            }
            if bounding_box.maximum_point.x < max_x {
                max_x = bounding_box.maximum_point.x;
            }
            if bounding_box.maximum_point.y < max_y {
                max_y = bounding_box.maximum_point.y;
            }
            if bounding_box.maximum_point.z < max_z {
                max_z = bounding_box.maximum_point.z;
            }

            if min_x > max_x || min_y > max_y || min_z > max_z {
                return None;
            }
        }

        Some(BoundingBox::new(
            &Point3::new(min_x, min_y, min_z),
            &Point3::new(max_x, max_y, max_z),
        ))
    }

    /// Gets the minimum point of the bounding box. All the components of the
    /// minimum point are the minimum values of the bounding box coordinates.
    pub fn minimum_point(&self) -> Point3<T> {
        self.minimum_point
    }

    /// Gets the minimum point of the bounding box. All the components of the
    /// minimum point are the maximum values of the bounding box coordinates.
    pub fn maximum_point(&self) -> Point3<T> {
        self.maximum_point
    }

    /// Collects all 8 corners of the bounding box as points defined in the
    /// units of the bounding box.
    pub fn corners(&self) -> [Point3<T>; 8] {
        [
            Point3::new(
                self.minimum_point.x,
                self.minimum_point.y,
                self.minimum_point.z,
            ),
            Point3::new(
                self.minimum_point.x,
                self.minimum_point.y,
                self.maximum_point.z,
            ),
            Point3::new(
                self.maximum_point.x,
                self.minimum_point.y,
                self.maximum_point.z,
            ),
            Point3::new(
                self.maximum_point.x,
                self.minimum_point.y,
                self.minimum_point.z,
            ),
            Point3::new(
                self.minimum_point.x,
                self.maximum_point.y,
                self.minimum_point.z,
            ),
            Point3::new(
                self.minimum_point.x,
                self.maximum_point.y,
                self.maximum_point.z,
            ),
            Point3::new(
                self.maximum_point.x,
                self.maximum_point.y,
                self.maximum_point.z,
            ),
            Point3::new(
                self.maximum_point.x,
                self.maximum_point.y,
                self.minimum_point.z,
            ),
        ]
    }
}

// Implementation specific to units defined in f32.
impl BoundingBox<f32> {
    /// Computes center of the current bounding box.
    pub fn center(&self) -> Point3<f32> {
        nalgebra::center(&self.minimum_point, &self.maximum_point)
    }

    /// Computes the diagonal vector of the current bounding box.
    pub fn diagonal(&self) -> Vector3<f32> {
        self.maximum_point - self.minimum_point
    }
}

// Implementation specific to units defined in i32.
impl BoundingBox<i32> {
    /// Computes center of the current bounding box.
    pub fn center(&self) -> Point3<i32> {
        Point3::new(
            ((self.minimum_point.x as f32 + self.maximum_point.x as f32) / 2.0).round() as i32,
            ((self.minimum_point.y as f32 + self.maximum_point.y as f32) / 2.0).round() as i32,
            ((self.minimum_point.z as f32 + self.maximum_point.z as f32) / 2.0).round() as i32,
        )
    }

    /// Computes the diagonal vector of the current bounding box.
    /// # Warning
    /// The diagonal dimensions are `maximum point coordinates - minimum point
    /// coordinates + 1`. For a singularity bounding box, this returns diagonal
    /// size 1, 1, 1 because that is the size of a single voxel.
    pub fn diagonal(&self) -> Vector3<u32> {
        let diagonal_i32 = self.maximum_point - self.minimum_point;
        Vector3::new(
            clamp_cast_i32_to_u32(diagonal_i32.x) + 1,
            clamp_cast_i32_to_u32(diagonal_i32.y) + 1,
            clamp_cast_i32_to_u32(diagonal_i32.z) + 1,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;

    #[test]
    fn test_bounding_box_construct_f32() {
        BoundingBox::new(
            &Point3::new(1_f32, 2_f32, 3_f32),
            &Point3::new(1_f32, 2_f32, 3_f32),
        );
    }

    #[test]
    fn test_bounding_box_construct_i32() {
        BoundingBox::new(
            &Point3::new(1_i32, 2_i32, 3_i32),
            &Point3::new(1_i32, 2_i32, 3_i32),
        );
    }

    #[test]
    fn test_bounding_box_construct_u32() {
        BoundingBox::new(
            &Point3::new(1_u32, 2_u32, 3_u32),
            &Point3::new(1_u32, 2_u32, 3_u32),
        );
    }

    #[test]
    fn test_bounding_box_set_minimum_and_maximum_correctly_f32() {
        let bb = BoundingBox::new(
            &Point3::new(4_f32, 14_f32, 24_f32),
            &Point3::new(2_f32, 22_f32, 33_f32),
        );

        assert!(approx::relative_eq!(
            bb.minimum_point,
            Point3::new(2_f32, 14_f32, 24_f32)
        ));
        assert!(approx::relative_eq!(
            bb.maximum_point,
            Point3::new(4_f32, 22_f32, 33_f32)
        ));
    }

    #[test]
    fn test_bounding_box_set_minimum_and_maximum_correctly_i32() {
        let bb = BoundingBox::new(
            &Point3::new(4_i32, 14_i32, 24_i32),
            &Point3::new(2_i32, 22_i32, 33_i32),
        );

        assert_eq!(bb.minimum_point, Point3::new(2_i32, 14_i32, 24_i32));
        assert_eq!(bb.maximum_point, Point3::new(4_i32, 22_i32, 33_i32));
    }

    #[test]
    fn test_bounding_box_from_no_points_is_none_i32() {
        let points: Vec<Point3<i32>> = Vec::new();

        let bb = BoundingBox::from_points(&points);

        assert_eq!(bb, None);
    }

    #[test]
    fn test_bounding_box_from_points_i32() {
        let mut points: Vec<Point3<i32>> = Vec::new();

        for x in 0..5_i32 {
            for y in (1..6_i32).rev() {
                for z in 2..7_i32 {
                    points.push(Point3::new(x, y, z));
                }
            }
        }

        let bb = BoundingBox::from_points(&points).unwrap();

        assert_eq!(bb.minimum_point, Point3::new(0_i32, 1_i32, 2_i32));
        assert_eq!(bb.maximum_point, Point3::new(4_i32, 5_i32, 6_i32));
    }

    #[test]
    fn test_bounding_box_union_empty_is_none_i32() {
        let boxes: Vec<BoundingBox<i32>> = Vec::new();

        let bb = BoundingBox::union(&boxes);

        assert_eq!(bb, None);
    }

    #[test]
    fn test_bounding_box_union_single_i32() {
        let bb = BoundingBox::new(
            &Point3::new(4_i32, 14_i32, 24_i32),
            &Point3::new(2_i32, 22_i32, 33_i32),
        );

        let bb_union = BoundingBox::union(iter::once(&bb)).unwrap();

        assert_eq!(bb_union.minimum_point, Point3::new(2_i32, 14_i32, 24_i32));
        assert_eq!(bb_union.maximum_point, Point3::new(4_i32, 22_i32, 33_i32));
    }

    #[test]
    fn test_bounding_box_union_two_i32() {
        let bb1 = BoundingBox::new(
            &Point3::new(4_i32, 14_i32, 24_i32),
            &Point3::new(2_i32, 22_i32, 33_i32),
        );
        let bb2 = BoundingBox::new(
            &Point3::new(5_i32, 2_i32, 28_i32),
            &Point3::new(1_i32, 15_i32, 40_i32),
        );

        let bb_union = BoundingBox::union([bb1, bb2].iter()).unwrap();

        assert_eq!(bb_union.minimum_point, Point3::new(1_i32, 2_i32, 24_i32));
        assert_eq!(bb_union.maximum_point, Point3::new(5_i32, 22_i32, 40_i32));
    }

    #[test]
    fn test_bounding_box_intersection_empty_is_none_i32() {
        let boxes: Vec<BoundingBox<i32>> = Vec::new();

        let bb = BoundingBox::intersection(&boxes);

        assert_eq!(bb, None);
    }

    #[test]
    fn test_bounding_box_intersection_single_i32() {
        let bb = BoundingBox::new(
            &Point3::new(4_i32, 14_i32, 24_i32),
            &Point3::new(2_i32, 22_i32, 33_i32),
        );

        let bb_intersection = BoundingBox::intersection(iter::once(&bb)).unwrap();

        assert_eq!(
            bb_intersection.minimum_point,
            Point3::new(2_i32, 14_i32, 24_i32)
        );
        assert_eq!(
            bb_intersection.maximum_point,
            Point3::new(4_i32, 22_i32, 33_i32)
        );
    }

    #[test]
    fn test_bounding_box_intersection_two_i32() {
        let bb1 = BoundingBox::new(
            &Point3::new(4_i32, 14_i32, 24_i32),
            &Point3::new(2_i32, 22_i32, 33_i32),
        );
        let bb2 = BoundingBox::new(
            &Point3::new(5_i32, 2_i32, 28_i32),
            &Point3::new(1_i32, 15_i32, 40_i32),
        );

        let bb_intersection = BoundingBox::intersection([bb1, bb2].iter()).unwrap();

        assert_eq!(
            bb_intersection.minimum_point,
            Point3::new(2_i32, 14_i32, 28_i32)
        );
        assert_eq!(
            bb_intersection.maximum_point,
            Point3::new(4_i32, 15_i32, 33_i32)
        );
    }

    #[test]
    fn test_bounding_box_intersection_two_non_intersecting_i32() {
        let bb1 = BoundingBox::new(
            &Point3::new(4_i32, 14_i32, 24_i32),
            &Point3::new(2_i32, 22_i32, 33_i32),
        );
        let bb2 = BoundingBox::new(
            &Point3::new(5_i32, 2_i32, 28_i32),
            &Point3::new(1_i32, 15_i32, 40_i32),
        );

        let bb_intersection = BoundingBox::intersection([bb1, bb2].iter()).unwrap();

        assert_eq!(
            bb_intersection.minimum_point,
            Point3::new(2_i32, 14_i32, 28_i32)
        );
        assert_eq!(
            bb_intersection.maximum_point,
            Point3::new(4_i32, 15_i32, 33_i32)
        );
    }

    #[test]
    fn test_bounding_box_corners_i32() {
        let bb = BoundingBox::new(
            &Point3::new(-1_i32, -1_i32, -1_i32),
            &Point3::new(1_i32, 1_i32, 1_i32),
        );

        let corners_correct = [
            Point3::new(-1_i32, -1_i32, -1_i32),
            Point3::new(-1_i32, -1_i32, 1_i32),
            Point3::new(1_i32, -1_i32, 1_i32),
            Point3::new(1_i32, -1_i32, -1_i32),
            Point3::new(-1_i32, 1_i32, -1_i32),
            Point3::new(-1_i32, 1_i32, 1_i32),
            Point3::new(1_i32, 1_i32, 1_i32),
            Point3::new(1_i32, 1_i32, -1_i32),
        ];

        assert_eq!(bb.corners(), corners_correct);
    }

    #[test]
    fn test_bounding_box_center_zero_i32() {
        let bb = BoundingBox::new(
            &Point3::new(-1_i32, -1_i32, -1_i32),
            &Point3::new(1_i32, 1_i32, 1_i32),
        );

        let center = bb.center();
        let center_correct = Point3::new(0_i32, 0_i32, 0_i32);

        assert_eq!(center, center_correct);
    }

    #[test]
    fn test_bounding_box_center_non_zero_i32() {
        let bb = BoundingBox::new(
            &Point3::new(-1_i32, -1_i32, -1_i32),
            &Point3::new(3_i32, 3_i32, 3_i32),
        );

        let center = bb.center();
        let center_correct = Point3::new(1_i32, 1_i32, 1_i32);

        assert_eq!(center, center_correct);
    }

    #[test]
    fn test_bounding_box_center_non_zero_rounded_i32() {
        let bb = BoundingBox::new(
            &Point3::new(-1_i32, -1_i32, -1_i32),
            &Point3::new(2_i32, 2_i32, 2_i32),
        );

        let center = bb.center();
        let center_correct = Point3::new(1_i32, 1_i32, 1_i32);

        assert_eq!(center, center_correct);
    }

    #[test]
    fn test_bounding_box_center_zero_f32() {
        let bb = BoundingBox::new(
            &Point3::new(-1_f32, -1_f32, -1_f32),
            &Point3::new(1_f32, 1_f32, 1_f32),
        );

        let center = bb.center();
        let center_correct = Point3::new(0_f32, 0_f32, 0_f32);

        assert_eq!(center, center_correct);
    }

    #[test]
    fn test_bounding_box_center_non_zero_f32() {
        let bb = BoundingBox::new(
            &Point3::new(-1_f32, -1_f32, -1_f32),
            &Point3::new(3_f32, 3_f32, 3_f32),
        );

        let center = bb.center();
        let center_correct = Point3::new(1_f32, 1_f32, 1_f32);

        assert_eq!(center, center_correct);
    }

    #[test]
    fn test_bounding_box_center_non_zero_not_rounded_f32() {
        let bb = BoundingBox::new(
            &Point3::new(-1_f32, -1_f32, -1_f32),
            &Point3::new(2_f32, 2_f32, 2_f32),
        );

        let center = bb.center();
        let center_correct = Point3::new(0.5_f32, 0.5_f32, 0.5_f32);

        assert_eq!(center, center_correct);
    }

    #[test]
    fn test_bounding_box_diagonal_i32() {
        let bb = BoundingBox::new(
            &Point3::new(-1_i32, -1_i32, -1_i32),
            &Point3::new(1_i32, 2_i32, 3_i32),
        );

        let diagonal = bb.diagonal();
        let diagonal_correct = Vector3::new(3_u32, 4_u32, 5_u32);

        assert_eq!(diagonal, diagonal_correct);
    }

    #[test]
    fn test_bounding_box_diagonal_f32() {
        let bb = BoundingBox::new(
            &Point3::new(-1_f32, -1_f32, -1_f32),
            &Point3::new(1_f32, 2_f32, 3_f32),
        );

        let diagonal = bb.diagonal();
        let diagonal_correct = Vector3::new(2_f32, 3_f32, 4_f32);

        assert_eq!(diagonal, diagonal_correct);
    }
}
