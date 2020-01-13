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

    // Creates a new bounding box from an iterator of points. The resulting
    // bounding box will encompass all the input points. The resulting bounding
    // box will be defined in the units of the input points.
    pub fn from_points<'a, I>(points: I) -> Self
    where
        I: IntoIterator<Item = &'a Point3<T>>,
    {
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

        BoundingBox {
            minimum_point: Point3::new(min_x, min_y, min_z),
            maximum_point: Point3::new(max_x, max_y, max_z),
        }
    }

    /// Creates a new bounding box encompassing all the input bounding boxes.
    /// The resulting bounding box will be defined in the units of the input
    /// bounding boxes.
    pub fn from_bounding_boxes_union<'a, I>(bounding_boxes: I) -> Self
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
    /// If there is no intersection, the resulting bounding box will have a zero
    /// volume and therefore will be a singularity.
    pub fn from_bounding_boxes_intersection<'a, I>(bounding_boxes: I) -> Self
    where
        I: IntoIterator<Item = &'a BoundingBox<T>>,
    {
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
                let singular_point = Point3::new(min_x, min_y, min_z);
                return BoundingBox::new(&singular_point, &singular_point);
            }
        }

        BoundingBox::new(
            &Point3::new(min_x, min_y, min_z),
            &Point3::new(max_x, max_y, max_z),
        )
    }

    /// Sets the current bounding box to have zero volume.
    pub fn set_singularity(&mut self) {
        self.maximum_point = self.minimum_point;
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

    /// Grows the current bounding box so that it contains also the input
    /// points. This doesn't shrink the existing bounding box.
    pub fn grow_to_contain_points<'a, I>(&mut self, points: I)
    where
        I: IntoIterator<Item = &'a Point3<T>>,
    {
        for point in points {
            if point.x < self.minimum_point.x {
                self.minimum_point.x = point.x;
            }
            if point.y < self.minimum_point.y {
                self.minimum_point.y = point.y;
            }
            if point.z < self.minimum_point.z {
                self.minimum_point.z = point.z;
            }
            if point.x > self.maximum_point.x {
                self.maximum_point.x = point.x;
            }
            if point.y > self.maximum_point.y {
                self.maximum_point.y = point.y;
            }
            if point.z > self.maximum_point.z {
                self.maximum_point.z = point.z;
            }
        }
    }

    /// Resizes the current bounding box so that it contains the input points.
    /// This may shrink, grow, move or keep the existing bounding box.
    pub fn fit_to_points<'a, I>(&mut self, points: I)
    where
        I: IntoIterator<Item = &'a Point3<T>>,
    {
        for point in points {
            if point.x < self.minimum_point.x {
                self.minimum_point.x = point.x;
            }
            if point.y < self.minimum_point.y {
                self.minimum_point.y = point.y;
            }
            if point.z < self.minimum_point.z {
                self.minimum_point.z = point.z;
            }
            if point.x > self.maximum_point.x {
                self.maximum_point.x = point.x;
            }
            if point.y > self.maximum_point.y {
                self.maximum_point.y = point.y;
            }
            if point.z > self.maximum_point.z {
                self.maximum_point.z = point.z;
            }
        }
    }

    /// Resizes the current bounding box so that it contains also the other
    /// input bounding boxes. Results in a bounding box encompassing the
    /// original and the other bounding boxes.
    pub fn union_with<'a, I>(&mut self, bounding_boxes: I)
    where
        I: IntoIterator<Item = &'a BoundingBox<T>>,
    {
        for bounding_box in bounding_boxes {
            self.grow_to_contain_points(&bounding_box.corners());
        }
    }

    /// Resizes the current bounding box so that it encloses the block of space
    /// common to all the input bounding boxes, including the current one.
    ///
    /// If there is no intersection, the resulting bounding box will have a zero
    /// volume and therefore will be a singularity.

    pub fn intersect_with<'a, I>(&mut self, bounding_boxes: I)
    where
        I: IntoIterator<Item = &'a BoundingBox<T>>,
    {
        for bounding_box in bounding_boxes {
            if bounding_box.minimum_point.x > self.minimum_point.x {
                self.minimum_point.x = bounding_box.minimum_point.x;
            }
            if bounding_box.minimum_point.y > self.minimum_point.y {
                self.minimum_point.y = bounding_box.minimum_point.y;
            }
            if bounding_box.minimum_point.z > self.minimum_point.z {
                self.minimum_point.z = bounding_box.minimum_point.z;
            }
            if bounding_box.maximum_point.x < self.maximum_point.x {
                self.maximum_point.x = bounding_box.maximum_point.x;
            }
            if bounding_box.maximum_point.y < self.maximum_point.y {
                self.maximum_point.y = bounding_box.maximum_point.y;
            }
            if bounding_box.maximum_point.z < self.maximum_point.z {
                self.maximum_point.z = bounding_box.maximum_point.z;
            }

            if self.minimum_point.x > self.maximum_point.x
                || self.minimum_point.y > self.maximum_point.y
                || self.minimum_point.z > self.maximum_point.z
            {
                self.set_singularity();
                return;
            }
        }
    }

    /// Checks if the two bounding boxes intersect / share any portion
    /// of space.
    ///
    /// # Sources
    /// https://math.stackexchange.com/questions/2651710/simplest-way-to-determine-if-two-3d-boxes-intersect
    pub fn intersects_with(&self, other: &BoundingBox<T>) -> bool {
        let self_min_x = self.minimum_point.x;
        let self_min_y = self.minimum_point.y;
        let self_min_z = self.minimum_point.z;
        let self_max_x = self.minimum_point.x;
        let self_max_y = self.minimum_point.y;
        let self_max_z = self.minimum_point.z;
        let other_min_x = other.minimum_point.x;
        let other_min_y = other.minimum_point.y;
        let other_min_z = other.minimum_point.z;
        let other_max_x = other.minimum_point.x;
        let other_max_y = other.minimum_point.y;
        let other_max_z = other.minimum_point.z;

        ((self_min_x <= other_min_x && other_min_x <= self_max_x)
            || (self_min_x <= other_max_x && other_max_x <= self_max_x)
            || (other_min_x <= self_min_x && self_min_x <= other_max_x)
            || (other_min_x <= self_max_x && self_max_x <= other_max_x))
            && ((self_min_y <= other_min_y && other_min_y <= self_max_y)
                || (self_min_y <= other_max_y && other_max_y <= self_max_y)
                || (other_min_y <= self_min_y && self_min_y <= other_max_y)
                || (other_min_y <= self_max_y && self_max_y <= other_max_y))
            && ((self_min_z <= other_min_z && other_min_z <= self_max_z)
                || (self_min_z <= other_max_z && other_max_z <= self_max_z)
                || (other_min_z <= self_min_z && self_min_z <= other_max_z)
                || (other_min_z <= self_max_z && self_max_z <= other_max_z))
    }
}

// Implementation specific to units defined in f32.
impl BoundingBox<f32> {
    /// Computes center of the current bounding box.
    pub fn center(&self) -> Point3<f32> {
        nalgebra::center(&self.minimum_point, &self.maximum_point)
    }

    /// Computes the length of the box diagonal.
    pub fn diagonal_length(&self) -> f32 {
        nalgebra::distance(&self.minimum_point, &self.maximum_point)
    }

    /// Computes the diagonal vector of the current bounding box.
    pub fn diagonal(&self) -> Vector3<f32> {
        self.maximum_point - self.minimum_point
    }

    /// Checks if the current bounding box is a singularity (has zero volume,
    /// minimum and maximum points are identical).

    pub fn is_singularity(&self) -> bool {
        approx::relative_eq!(self.minimum_point(), self.maximum_point())
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

    /// Computes the length of the box diagonal. The result is in the whole
    /// number of units.
    pub fn diagonal_length(&self) -> u32 {
        (((self.maximum_point.x - self.minimum_point.x).pow(2)
            + (self.maximum_point.y - self.minimum_point.y).pow(2)
            + (self.maximum_point.z - self.minimum_point.z).pow(2)) as f32)
            .sqrt()
            .round() as u32
    }

    /// Computes the diagonal vector of the current bounding box.
    pub fn diagonal(&self) -> Vector3<u32> {
        let diagonal_i32 = self.maximum_point - self.minimum_point;
        Vector3::new(
            clamp_cast_i32_to_u32(diagonal_i32.x) + 1,
            clamp_cast_i32_to_u32(diagonal_i32.y) + 1,
            clamp_cast_i32_to_u32(diagonal_i32.z) + 1,
        )
    }

    /// Checks if the current bounding box is a singularity (has zero volume,
    /// minimum and maximum points are identical).

    pub fn is_singularity(&self) -> bool {
        self.minimum_point() == self.maximum_point()
    }
}
