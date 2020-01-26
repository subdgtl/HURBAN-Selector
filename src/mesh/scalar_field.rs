use std::collections::VecDeque;
use std::f32;
use std::ops::{Add, Div, Mul, Neg, Sub};

use nalgebra::{Matrix4, Point3, Rotation3, Vector2, Vector3};
use num_traits::{Bounded, FromPrimitive, ToPrimitive};

use crate::bounding_box::BoundingBox;
use crate::convert::{cast_i32, cast_u32, cast_usize, clamp_cast_i32_to_u32};
use crate::geometry;
use crate::interval::Interval;
use crate::plane::Plane;

use super::{primitive, tools, Face, Mesh};

/// Scalar field is an abstract representation of points in a block of space.
/// Each point is a center of a voxel - an abstract box of given dimensions in a
/// discrete spatial grid.
///
/// The voxels contain a value, which can be read in various ways: as a scalar
/// charge field, as a distance field or as any arbitrary discrete value grid.
/// There is always a constant value (largest number of the given type) for
/// empty voxels.
///
/// The scalar field is meant to be materialized into a mesh - voxels within a
/// certain value interval will become mesh boxes.
///
/// The block of voxel space stored in the scalar field is delimited by its
/// beginning and its dimensions, both in the units of the voxels. All voxels
/// have the same dimensions, which can be different in each direction.
///
/// The voxel space is a discrete grid and can't start half way in a voxel,
/// therefore its beginning as well as the voxel positions are defined in the
/// voxel-space coordinates. The voxel space starts at the cartesian space
/// origin with voxel coordinates 0, 0, 0.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ScalarField<T> {
    block_start: Point3<i32>,
    block_dimensions: Vector3<u32>,
    voxel_dimensions: Vector3<f32>,
    values: Vec<T>,
}

impl<
        T: Add<Output = T>
            + Bounded
            + Copy
            + Div<Output = T>
            + FromPrimitive
            + Mul<Output = T>
            + Neg<Output = T>
            + PartialOrd
            + Sub<Output = T>
            + ToPrimitive,
    > ScalarField<T>
{
    /// Gets the internal value for an empty voxel in the current scalar field.
    pub fn empty_value<U>() -> U
    where
        U: Add<Output = T>
            + Bounded
            + Copy
            + Div<Output = T>
            + FromPrimitive
            + Mul<Output = T>
            + Neg<Output = T>
            + PartialOrd
            + Sub<Output = T>
            + ToPrimitive,
    {
        U::max_value()
    }

    /// Define a new empty block of voxel space, which begins at
    /// `block_start`(in discrete voxel units), has dimensions
    /// `block_dimensions` (in discrete voxel units) and contains voxels sized
    /// `voxel_dimensions` (in model space units).
    pub fn new(
        block_start: &Point3<i32>,
        block_dimensions: &Vector3<u32>,
        voxel_dimensions: &Vector3<f32>,
    ) -> Self {
        assert!(
            voxel_dimensions.x > 0.0 && voxel_dimensions.y > 0.0 && voxel_dimensions.z > 0.0,
            "One or more voxel dimensions are 0.0"
        );
        let map_length = block_dimensions.x * block_dimensions.y * block_dimensions.z;
        let values: Vec<T> = vec![ScalarField::empty_value(); cast_usize(map_length)];

        ScalarField {
            block_start: *block_start,
            block_dimensions: *block_dimensions,
            voxel_dimensions: *voxel_dimensions,
            values,
        }
    }

    /// Creates a new empty voxel space from a bounding box defined in cartesian
    /// units.
    ///
    /// # Panics
    /// Panics if any of the voxel dimensions is below or equal to zero.
    pub fn from_cartesian_bounding_box(
        bounding_box: &BoundingBox<f32>,
        voxel_dimensions: &Vector3<f32>,
    ) -> Self {
        assert!(
            voxel_dimensions.x > 0.0 && voxel_dimensions.y > 0.0 && voxel_dimensions.z > 0.0,
            "One or more voxel dimensions are 0.0"
        );
        let min_point = &bounding_box.minimum_point();
        let max_point = &bounding_box.maximum_point();
        let min_x_index = (min_point.x.min(max_point.x) / voxel_dimensions.x).floor() as i32;
        let max_x_index = (min_point.x.max(max_point.x) / voxel_dimensions.x).ceil() as i32;
        let min_y_index = (min_point.y.min(max_point.y) / voxel_dimensions.y).floor() as i32;
        let max_y_index = (min_point.y.max(max_point.y) / voxel_dimensions.y).ceil() as i32;
        let min_z_index = (min_point.z.min(max_point.z) / voxel_dimensions.z).floor() as i32;
        let max_z_index = (min_point.z.max(max_point.z) / voxel_dimensions.z).ceil() as i32;

        let block_start = Point3::new(min_x_index, min_y_index, min_z_index);
        let block_dimensions = Vector3::new(
            cast_u32(max_x_index - min_x_index) + 1,
            cast_u32(max_y_index - min_y_index) + 1,
            cast_u32(max_z_index - min_z_index) + 1,
        );

        ScalarField::new(&block_start, &block_dimensions, voxel_dimensions)
    }

    /// Creates a scalar field from an existing mesh. The voxels intersecting
    /// the mesh (volume voxels) will be set to `value_on_mesh_surface`, the
    /// empty voxels (void voxels) will be set to the default empty value. The
    /// `growth_offset` defines how much bigger will the scalar field be. This
    /// is useful if the distance field is about to be calculated.
    ///
    /// # Panics
    /// Panics if the value of volume voxels is equal to the value reserved for
    /// the void voxels.
    ///
    /// Panics if any of the voxel dimensions is below or equal to zero.
    pub fn from_mesh(
        mesh: &Mesh,
        voxel_dimensions: &Vector3<f32>,
        value_on_mesh_surface: T,
        growth_offset: u32,
    ) -> Self {
        assert!(
            value_on_mesh_surface != ScalarField::empty_value(),
            "The value on mesh surface is equal to the empty marker."
        );
        assert!(
            voxel_dimensions.x > 0.0 && voxel_dimensions.y > 0.0 && voxel_dimensions.z > 0.0,
            "One or more voxel dimensions are 0.0."
        );

        // Determine the needed block of voxel space.
        let bounding_box_tight = mesh.bounding_box();
        let growth_offset_vector_in_cartesian_coordinates = Vector3::new(
            voxel_dimensions.x * growth_offset as f32,
            voxel_dimensions.y * growth_offset as f32,
            voxel_dimensions.z * growth_offset as f32,
        );
        let bounding_box_offset =
            bounding_box_tight.offset(growth_offset_vector_in_cartesian_coordinates);

        // Target scalar field to be marked with points on the mesh.
        let mut scalar_field =
            ScalarField::from_cartesian_bounding_box(&bounding_box_offset, voxel_dimensions);

        // Going to populate the mesh with points as dense as the smallest voxel
        // dimension.
        let shortest_voxel_dimension = voxel_dimensions
            .x
            .min(voxel_dimensions.y.min(voxel_dimensions.z));

        for face in mesh.faces() {
            match face {
                Face::Triangle(f) => {
                    let point_a = &mesh.vertices()[cast_usize(f.vertices.0)];
                    let point_b = &mesh.vertices()[cast_usize(f.vertices.1)];
                    let point_c = &mesh.vertices()[cast_usize(f.vertices.2)];
                    // Compute the density of points on the respective face.
                    let ab_distance_sq = nalgebra::distance_squared(point_a, point_b);
                    let bc_distance_sq = nalgebra::distance_squared(point_b, point_c);
                    let ca_distance_sq = nalgebra::distance_squared(point_c, point_a);
                    let longest_edge_len = ab_distance_sq
                        .max(bc_distance_sq.max(ca_distance_sq))
                        .sqrt();
                    // Number of face divisions (points) in each direction.
                    let divisions = (longest_edge_len / shortest_voxel_dimension).ceil() as usize;
                    let divisions_f32 = divisions as f32;

                    for ui in 0..=divisions {
                        for wi in 0..=divisions {
                            let u_normalized = ui as f32 / divisions_f32;
                            let w_normalized = wi as f32 / divisions_f32;
                            let v_normalized = 1.0 - u_normalized - w_normalized;
                            if v_normalized >= 0.0 {
                                let barycentric =
                                    Point3::new(u_normalized, v_normalized, w_normalized);
                                // Compute point position in model space
                                let cartesian = geometry::barycentric_to_cartesian(
                                    &barycentric,
                                    &point_a,
                                    &point_b,
                                    &point_c,
                                );
                                // and set a voxel containing the point to be on
                                let absolute_coordinate = cartesian_to_absolute_voxel_coordinate(
                                    &cartesian,
                                    voxel_dimensions,
                                );
                                scalar_field.set_value_at_absolute_voxel_coordinate(
                                    &absolute_coordinate,
                                    value_on_mesh_surface,
                                );
                            }
                        }
                    }
                }
            }
        }

        scalar_field
    }

    /// Creates a new scalar field with arbitrary voxel dimensions from another
    /// scalar field.
    ///
    /// # Panics
    /// Panics if any of the voxel dimensions is below or equal to zero.
    pub fn from_scalar_field(
        source_scalar_field: &ScalarField<T>,
        volume_value_interval: Interval<T>,
        voxel_dimensions: &Vector3<f32>,
    ) -> Option<Self> {
        assert!(
            voxel_dimensions.x > 0.0 && voxel_dimensions.y > 0.0 && voxel_dimensions.z > 0.0,
            "One mor more voxel dimensions is below or equal to zero"
        );

        source_scalar_field
            .mesh_volume_bounding_box_cartesian(volume_value_interval)
            .map(|current_sf_bounding_box| {
                // Make a bounding box of the source bounding box's mesh volume.
                // This will be the volume to be scanned for any voxels.

                // New scalar field that can encompass the source
                // scalar field's mesh.
                let mut target_scalar_field = ScalarField::from_cartesian_bounding_box(
                    &current_sf_bounding_box,
                    &voxel_dimensions,
                );

                for (one_dimensional, voxel) in target_scalar_field.values.iter_mut().enumerate() {
                    let cartesian_coordinate = one_dimensional_to_cartesian_coordinate(
                        one_dimensional,
                        &target_scalar_field.block_start,
                        &target_scalar_field.block_dimensions,
                        &target_scalar_field.voxel_dimensions,
                    )
                    .expect("Index out of bounds");

                    // Set the new voxel state according to a sampled value from
                    // the source scalar field.
                    let absolute_coordinate = cartesian_to_absolute_voxel_coordinate(
                        &cartesian_coordinate,
                        voxel_dimensions,
                    );
                    *voxel = source_scalar_field
                        .value_at_absolute_voxel_coordinate(&absolute_coordinate)
                        .unwrap_or(ScalarField::empty_value());
                }

                // FIXME: @Optimization Due to overly safe
                // mesh_volume_bounding_box_cartesian, the scalar field may be
                // unnecessarily big.
                target_scalar_field.shrink_to_fit(volume_value_interval);

                target_scalar_field
            })
    }

    /// Creates a new scalar field from another scalar field transformed
    /// (scaled, rotated, moved) in a cartesian space.
    ///
    /// # Panics
    /// Panics if the voxel dimension is below or equal to zero.
    pub fn from_scalar_field_transformed(
        source_scalar_field: &ScalarField<T>,
        volume_value_interval: Interval<T>,
        voxel_dimension: f32,
        cartesian_translation: &Vector3<f32>,
        rotation: &Rotation3<f32>,
        scale_factor: &Vector3<f32>,
    ) -> Option<Self> {
        let euler_angles = rotation.euler_angles();
        if approx::relative_eq!(cartesian_translation, &Vector3::zeros())
            && approx::relative_eq!(euler_angles.0, 0.0)
            && approx::relative_eq!(euler_angles.1, 0.0)
            && approx::relative_eq!(euler_angles.2, 0.0)
            && approx::relative_eq!(scale_factor, &Vector3::new(1.0, 1.0, 1.0))
        {
            if approx::relative_eq!(voxel_dimension, source_scalar_field.voxel_dimensions.x)
                && approx::relative_eq!(voxel_dimension, source_scalar_field.voxel_dimensions.y)
                && approx::relative_eq!(voxel_dimension, source_scalar_field.voxel_dimensions.z)
            {
                return Some(ScalarField {
                    block_start: source_scalar_field.block_start,
                    block_dimensions: source_scalar_field.block_dimensions,
                    voxel_dimensions: source_scalar_field.voxel_dimensions,
                    values: source_scalar_field.values.to_vec(),
                });
            } else {
                return ScalarField::from_scalar_field(
                    source_scalar_field,
                    volume_value_interval,
                    &Vector3::new(voxel_dimension, voxel_dimension, voxel_dimension),
                );
            }
        }

        assert!(
            voxel_dimension > 0.0,
            "Voxel dimension is below or equal to zero"
        );

        // Make a bounding box of the source bounding box's mesh volume. This
        // will be the volume to be scanned for any voxels.
        if let Some(source_sf_bounding_box) =
            source_scalar_field.mesh_volume_bounding_box_cartesian(volume_value_interval)
        {
            // FIXME: Heterogenous voxels (with different with, height and
            // depth) require non-trivial compensation of final scalar field
            // position after rotating if the volume equilibrium is not at the
            // world origin.
            let voxel_dimensions = Vector3::new(voxel_dimension, voxel_dimension, voxel_dimension);

            let vector_to_origin = Vector3::zeros() - source_sf_bounding_box.center().coords;

            // Transform the source mesh volume and calculate a new bounding box
            // that will encompass the transformed source scalar field.
            let transformation_to_origin = Matrix4::new_translation(&vector_to_origin);
            let compound_transformation =
                Matrix4::from(*rotation) * Matrix4::new_nonuniform_scaling(scale_factor);

            let source_sf_bounding_box_corners = source_sf_bounding_box.corners();
            let transformed_bounding_box_corners = source_sf_bounding_box_corners.iter().map(|v| {
                let v1 = transformation_to_origin.transform_point(v);
                compound_transformation.transform_point(&v1)
            });

            let transformed_bounding_box =
                BoundingBox::from_points(transformed_bounding_box_corners)
                    .expect("No input bounding box");

            // New scalar field that can encompass the transformed source scalar
            // field's mesh.
            let mut target_scalar_field = ScalarField::from_cartesian_bounding_box(
                &transformed_bounding_box,
                &voxel_dimensions,
            );

            // Transform the target voxels inverse to the user transformation so
            // that it is possible to sample the source scalar field.
            if let Ok(reversed_user_transformation) =
                compound_transformation.pseudo_inverse(f32::EPSILON)
            {
                for (one_dimensional, voxel) in target_scalar_field.values.iter_mut().enumerate() {
                    let cartesian_coordinate = one_dimensional_to_cartesian_coordinate(
                        one_dimensional,
                        &target_scalar_field.block_start,
                        &target_scalar_field.block_dimensions,
                        &target_scalar_field.voxel_dimensions,
                    )
                    .expect("Index out of bounds");

                    // Transform each new voxel inverse to the user
                    // transformation.
                    let transformed_voxel_center_cartesian = reversed_user_transformation
                        .transform_point(&cartesian_coordinate)
                        - vector_to_origin;

                    // Set the new voxel state according to a sampled value from
                    // the source scalar field.
                    let absolute_coordinate = cartesian_to_absolute_voxel_coordinate(
                        &transformed_voxel_center_cartesian,
                        &voxel_dimensions,
                    );
                    *voxel = source_scalar_field
                        .value_at_absolute_voxel_coordinate(&absolute_coordinate)
                        .unwrap_or(ScalarField::empty_value());
                }

                let cartesian_final_translation_vector = cartesian_translation - vector_to_origin;

                let voxel_final_translation_vector = cartesian_to_absolute_voxel_coordinate(
                    &Point3::from(cartesian_final_translation_vector),
                    &voxel_dimensions,
                )
                .coords;

                target_scalar_field.block_start += voxel_final_translation_vector;

                // FIXME: @Optimization Due to overly safe
                // mesh_volume_bounding_box_cartesian, the scalar field may
                // be unnecessarily big.
                target_scalar_field.shrink_to_fit(volume_value_interval);

                return Some(target_scalar_field);
            }
        }

        None
    }

    /// Clears the scalar field, sets its block dimensions to zero.
    pub fn wipe(&mut self) {
        self.block_start = Point3::origin();
        self.block_dimensions = Vector3::zeros();
        self.values.resize(0, ScalarField::empty_value());
    }

    /// Returns scalar field block end in absolute voxel coordinates.
    #[allow(dead_code)]
    fn block_end(&self) -> Point3<i32> {
        Point3::new(
            self.block_start.x + cast_i32(self.block_dimensions.x) - 1,
            self.block_start.y + cast_i32(self.block_dimensions.y) - 1,
            self.block_start.z + cast_i32(self.block_dimensions.z) - 1,
        )
    }

    /// Returns single voxel dimensions in cartesian units.
    pub fn voxel_dimensions(&self) -> Vector3<f32> {
        self.voxel_dimensions
    }

    /// Checks if the scalar field contains any volume (non-empty) voxel.
    #[allow(dead_code)]
    pub fn contains_voxels(&self) -> bool {
        self.values.iter().any(|v| *v != ScalarField::empty_value())
    }

    /// Checks if the scalar field contains any voxel with a value from the
    /// given interval.
    pub fn contains_voxels_within_interval(&self, volume_value_interval: Interval<T>) -> bool {
        self.values
            .iter()
            .any(|v| volume_value_interval.includes_closed(*v))
    }

    /// Computes boolean intersection (logical AND operation) of the current and
    /// another scalar field. The current scalar field will be mutated and
    /// resized to the size and position of an intersection of the two scalar
    /// fields' volumes.
    pub fn boolean_intersection(
        &mut self,
        volume_value_interval_self: Interval<T>,
        other: &ScalarField<T>,
        volume_value_interval_other: Interval<T>,
    ) {
        // Find volume common to both scalar fields.
        if let Some(self_volume_bounding_box) = self.volume_bounding_box(volume_value_interval_self)
        {
            if let Some(other_volume_bounding_box) =
                other.volume_bounding_box(volume_value_interval_other)
            {
                if let Some(bounding_box) = BoundingBox::intersection(
                    [self_volume_bounding_box, other_volume_bounding_box]
                        .iter()
                        .copied(),
                ) {
                    // Resize (keep or shrink) the existing scalar field so that that can
                    // possibly contain intersection voxels.
                    self.resize_to_voxel_space_bounding_box(&bounding_box);

                    let block_start = bounding_box.minimum_point();
                    let diagonal = bounding_box.diagonal();
                    let block_dimensions = Vector3::new(
                        cast_u32(diagonal.x),
                        cast_u32(diagonal.y),
                        cast_u32(diagonal.z),
                    );
                    // Iterate through the block of space common to both scalar fields.
                    for i in 0..self.values.len() {
                        let cartesian_coordinate = one_dimensional_to_cartesian_coordinate(
                            i,
                            &block_start,
                            &block_dimensions,
                            &self.voxel_dimensions,
                        )
                        .expect("The current voxel map out of bounds");

                        // Perform boolean AND on voxel states of both scalar fields.
                        if volume_value_interval_self.includes_closed(self.values[i]) {
                            let absolute_coordinate = cartesian_to_absolute_voxel_coordinate(
                                &cartesian_coordinate,
                                &other.voxel_dimensions,
                            );
                            if let Some(value) =
                                other.value_at_absolute_voxel_coordinate(&absolute_coordinate)
                            {
                                if !volume_value_interval_other.includes_closed(value) {
                                    self.values[i] = ScalarField::empty_value();
                                }
                            }
                        }
                    }
                    self.shrink_to_fit(volume_value_interval_self);
                    // Return here because any other option needs to wipe the
                    // current scalar field.
                    return;
                }
            }
        }

        // If the two scalar fields don't intersect or one of them is empty,
        // then wipe the resulting scalar field.
        self.wipe();
    }

    /// Computes boolean union (logical OR operation) of two scalar fields. The
    /// current scalar field will be mutated and resized to contain both input
    /// scalar fields' volumes.
    ///
    /// # Warning
    /// If the input scalar fields are far apart, the resulting scalar field may
    /// be huge.
    pub fn boolean_union(
        &mut self,
        volume_value_interval_self: Interval<T>,
        other: &ScalarField<T>,
        volume_value_interval_other: Interval<T>,
    ) {
        let bounding_boxes = [
            self.volume_bounding_box(volume_value_interval_self),
            other.volume_bounding_box(volume_value_interval_other),
        ];

        let valid_bounding_boxes_iter = bounding_boxes.iter().filter_map(|b| *b);
        if let Some(bounding_box) = BoundingBox::union(valid_bounding_boxes_iter) {
            // Resize (keep or grow) the existing scalar field to a block that
            // can possibly contain union voxels.
            self.resize_to_voxel_space_bounding_box(&bounding_box);

            let block_start = bounding_box.minimum_point();
            let diagonal = bounding_box.diagonal();
            let block_dimensions = Vector3::new(
                cast_u32(diagonal.x),
                cast_u32(diagonal.y),
                cast_u32(diagonal.z),
            );
            // Iterate through the block of space containing both scalar fields.
            for i in 0..self.values.len() {
                let cartesian_coordinate = one_dimensional_to_cartesian_coordinate(
                    i,
                    &block_start,
                    &block_dimensions,
                    &self.voxel_dimensions,
                )
                .expect("The current voxel map out of bounds");
                // If the other scalar field exists on the current absolute
                // coordinate, perform boolean OR, otherwise don't change the
                // existing voxel.
                let absolute_coordinate = cartesian_to_absolute_voxel_coordinate(
                    &cartesian_coordinate,
                    &other.voxel_dimensions,
                );
                if let Some(value) = other.value_at_absolute_voxel_coordinate(&absolute_coordinate)
                {
                    if volume_value_interval_other.includes_closed(value) {
                        self.values[i] = volume_value_interval_other
                            .remap_to(value, volume_value_interval_self)
                            .expect("One of the intervals is infinite.");
                    }
                }
            }
            self.shrink_to_fit(volume_value_interval_self);
        } else {
            self.wipe();
        }
    }

    /// Computes boolean difference of the current scalar field minus the other
    /// scalar field. The current scalar field will be modified so that voxels,
    /// that are on in both scalar fields will be turned off, while the rest
    /// remains intact.
    pub fn boolean_difference(
        &mut self,
        volume_value_interval_self: Interval<T>,
        other: &ScalarField<T>,
        volume_value_interval_other: Interval<T>,
    ) {
        // Iterate through the target scalar field
        for i in 0..self.values.len() {
            let self_cartesian_coordinate = one_dimensional_to_cartesian_coordinate(
                i,
                &self.block_start,
                &self.block_dimensions,
                &self.voxel_dimensions,
            )
            .expect("The current voxel map out of bounds");

            if let Some(other_one_dimensional) = cartesian_to_one_dimensional_coordinate(
                &self_cartesian_coordinate,
                &other.block_start,
                &other.block_dimensions,
                &other.voxel_dimensions(),
            ) {
                // If the other scalar fields contains a voxel at the position,
                // remove the existing voxel from the target scalar field
                if let Some(value) = other.values.get(other_one_dimensional) {
                    if volume_value_interval_other.includes_closed(*value) {
                        self.values[i] = ScalarField::empty_value();
                    }
                }
            }
        }
        self.shrink_to_fit(volume_value_interval_self)
    }

    /// Gets the state of a voxel defined in absolute voxel coordinates
    /// (relative to the voxel space origin).
    pub fn value_at_absolute_voxel_coordinate(
        &self,
        absolute_coordinate: &Point3<i32>,
    ) -> Option<T> {
        absolute_voxel_to_one_dimensional_coordinate(
            absolute_coordinate,
            &self.block_start,
            &self.block_dimensions,
        )
        .map(|index| self.values[index])
    }

    /// Sets the state of a voxel defined in absolute voxel coordinates
    /// (relative to the voxel space origin).
    pub fn set_value_at_absolute_voxel_coordinate(
        &mut self,
        absolute_coordinate: &Point3<i32>,
        state: T,
    ) {
        let index = absolute_voxel_to_one_dimensional_coordinate(
            absolute_coordinate,
            &self.block_start,
            &self.block_dimensions,
        )
        .expect("Coordinates out of bounds");
        self.values[cast_usize(index)] = state;
    }

    /// Fills the current scalar field with the given value.
    #[allow(dead_code)]
    pub fn fill_with(&mut self, value: T) {
        for v in self.values.iter_mut() {
            *v = value;
        }
    }

    /// Resize the scalar field block to match new block start and block
    /// dimensions.
    ///
    /// This clips the outstanding parts of the original scalar field.
    pub fn resize(
        &mut self,
        resized_block_start: &Point3<i32>,
        resized_block_dimensions: &Vector3<u32>,
    ) {
        if resized_block_start != &self.block_start
            || resized_block_dimensions != &self.block_dimensions
        {
            // Short cut if resizing to an empty scalar field.
            if resized_block_dimensions == &Vector3::zeros() {
                self.wipe();
                return;
            }

            let original_values = self.values.clone();
            let original_block_start = self.block_start;
            let original_block_dimensions = self.block_dimensions;

            self.block_start = *resized_block_start;
            self.block_dimensions = *resized_block_dimensions;

            let resized_values_len = cast_usize(
                resized_block_dimensions.x
                    * resized_block_dimensions.y
                    * resized_block_dimensions.z,
            );

            self.values
                .resize(resized_values_len, ScalarField::empty_value());

            for resized_index in 0..self.values.len() {
                let absolute_coordinate = one_dimensional_to_absolute_voxel_coordinate(
                    resized_index,
                    resized_block_start,
                    resized_block_dimensions,
                )
                .expect("Index out of bounds");

                self.values[resized_index] = match absolute_voxel_to_one_dimensional_coordinate(
                    &absolute_coordinate,
                    &original_block_start,
                    &original_block_dimensions,
                ) {
                    Some(original_index) => original_values[original_index],
                    _ => ScalarField::empty_value(),
                }
            }
        }
    }

    /// Resize the current scalar field to match the input bounding box in
    /// voxel-space units
    pub fn resize_to_voxel_space_bounding_box(&mut self, bounding_box: &BoundingBox<i32>) {
        let diagonal = bounding_box.diagonal();
        let block_dimensions = Vector3::new(
            cast_u32(diagonal.x),
            cast_u32(diagonal.y),
            cast_u32(diagonal.z),
        );
        self.resize(&bounding_box.minimum_point(), &block_dimensions);
    }

    /// Resize the scalar field block to exactly fit the volumetric geometry.
    /// Returns None for empty the scalar field.
    pub fn shrink_to_fit(&mut self, volume_value_interval: Interval<T>) {
        if let Some((shrunk_block_start, shrunk_block_dimensions)) =
            self.compute_volume_boundaries(volume_value_interval)
        {
            self.resize(&shrunk_block_start, &shrunk_block_dimensions);
        } else {
            self.wipe();
        }
    }

    /// Computes a simple triangulated welded mesh from the current state of the
    /// scalar field.
    ///
    /// For watertight meshes this creates both, outer and inner boundary mesh.
    /// There is also a high risk of generating a non-manifold mesh if some
    /// voxels touch only diagonally.
    pub fn to_mesh(&self, volume_value_interval: Interval<T>) -> Option<Mesh> {
        if self.block_dimensions.x == 0
            || self.block_dimensions.y == 0
            || self.block_dimensions.z == 0
        {
            return None;
        }

        // A collection of rectangular meshes (two triangles) defining an outer
        // envelope of volumes stored in the scalar field
        let mut plane_meshes: Vec<Mesh> = Vec::new();

        struct VoxelMeshHelper {
            plane: Plane,
            direction_to_wall: Vector3<f32>,
            direction_to_neighbor: Vector3<i32>,
            voxel_dimensions: Vector2<f32>,
        }

        // Pre-computed geometry helpers
        let neighbor_helpers = [
            VoxelMeshHelper {
                //top
                plane: Plane::new(
                    &Point3::origin(),
                    &Vector3::new(1.0, 0.0, 0.0),
                    &Vector3::new(0.0, 1.0, 0.0),
                ),
                direction_to_wall: Vector3::new(0.0, 0.0, self.voxel_dimensions.z / 2.0),
                direction_to_neighbor: Vector3::new(0, 0, 1),
                voxel_dimensions: Vector2::new(self.voxel_dimensions.x, self.voxel_dimensions.y),
            },
            VoxelMeshHelper {
                //bottom
                plane: Plane::new(
                    &Point3::origin(),
                    &Vector3::new(1.0, 0.0, 0.0),
                    &Vector3::new(0.0, -1.0, 0.0),
                ),
                direction_to_wall: Vector3::new(0.0, 0.0, -self.voxel_dimensions.z / 2.0),
                direction_to_neighbor: Vector3::new(0, 0, -1),
                voxel_dimensions: Vector2::new(self.voxel_dimensions.x, self.voxel_dimensions.y),
            },
            VoxelMeshHelper {
                //right
                plane: Plane::new(
                    &Point3::origin(),
                    &Vector3::new(0.0, 1.0, 0.0),
                    &Vector3::new(0.0, 0.0, 1.0),
                ),
                direction_to_wall: Vector3::new(self.voxel_dimensions.x / 2.0, 0.0, 0.0),
                direction_to_neighbor: Vector3::new(1, 0, 0),
                voxel_dimensions: Vector2::new(self.voxel_dimensions.y, self.voxel_dimensions.z),
            },
            VoxelMeshHelper {
                //left
                plane: Plane::new(
                    &Point3::origin(),
                    &Vector3::new(0.0, -1.0, 0.0),
                    &Vector3::new(0.0, 0.0, 1.0),
                ),
                direction_to_wall: Vector3::new(-self.voxel_dimensions.x / 2.0, 0.0, 0.0),
                direction_to_neighbor: Vector3::new(-1, 0, 0),
                voxel_dimensions: Vector2::new(self.voxel_dimensions.y, self.voxel_dimensions.z),
            },
            VoxelMeshHelper {
                //front
                plane: Plane::new(
                    &Point3::origin(),
                    &Vector3::new(1.0, 0.0, 0.0),
                    &Vector3::new(0.0, 0.0, 1.0),
                ),
                direction_to_wall: Vector3::new(0.0, -self.voxel_dimensions.y / 2.0, 0.0),
                direction_to_neighbor: Vector3::new(0, -1, 0),
                voxel_dimensions: Vector2::new(self.voxel_dimensions.x, self.voxel_dimensions.z),
            },
            VoxelMeshHelper {
                //rear
                plane: Plane::new(
                    &Point3::origin(),
                    &Vector3::new(-1.0, 0.0, 0.0),
                    &Vector3::new(0.0, 0.0, 1.0),
                ),
                direction_to_wall: Vector3::new(0.0, self.voxel_dimensions.y / 2.0, 0.0),
                direction_to_neighbor: Vector3::new(0, 1, 0),
                voxel_dimensions: Vector2::new(self.voxel_dimensions.x, self.voxel_dimensions.z),
            },
        ];

        // Iterate through the scalar field
        for (one_dimensional_coordinate, voxel) in self.values.iter().enumerate() {
            // If the current voxel is a volume voxel
            if volume_value_interval.includes_closed(*voxel) {
                let voxel_coordinate = one_dimensional_to_relative_voxel_coordinate(
                    one_dimensional_coordinate,
                    &self.block_dimensions,
                )
                .expect("Out of bounds");
                // compute the position of its center in model space coordinates
                let voxel_center = relative_voxel_to_cartesian_coordinate(
                    &voxel_coordinate,
                    &self.block_start,
                    &self.voxel_dimensions,
                );
                // and check if there is any voxel around it.
                for helper in &neighbor_helpers {
                    let absolute_neighbor_coordinate = relative_voxel_to_absolute_voxel_coordinate(
                        &(voxel_coordinate + helper.direction_to_neighbor),
                        &self.block_start,
                    );
                    match self.value_at_absolute_voxel_coordinate(&absolute_neighbor_coordinate) {
                        Some(neighbor_value) => {
                            // If there is a neighbor next to the current voxel,
                            // and it is not within the volume interval, the
                            // boundary side of the voxel box should be
                            // materialized.
                            if !volume_value_interval.includes_closed(neighbor_value) {
                                // Add a rectangle
                                plane_meshes.push(primitive::create_mesh_plane(
                                    Plane::from_origin_and_plane(
                                        // around the voxel center half way the
                                        // respective dimension of the voxel,
                                        &(voxel_center + helper.direction_to_wall),
                                        // align it properly
                                        &helper.plane,
                                    ),
                                    // and set its size to match the dimensions
                                    // of the respective side of a voxel.
                                    helper.voxel_dimensions,
                                ));
                            }
                        }
                        // Also materialize the boundary side of the voxel box
                        // if there is no neighbor - it means the current voxel
                        // is at the boundary of the scalar field.
                        None => {
                            plane_meshes.push(primitive::create_mesh_plane(
                                Plane::from_origin_and_plane(
                                    &(voxel_center + helper.direction_to_wall),
                                    &helper.plane,
                                ),
                                helper.voxel_dimensions,
                            ));
                        }
                    }
                }
            }
        }

        // Join separate mesh planes into one mesh
        let joined_voxel_mesh = tools::join_multiple_meshes(&plane_meshes);
        let min_voxel_dimension = self
            .voxel_dimensions
            .x
            .min(self.voxel_dimensions.y.min(self.voxel_dimensions.z));
        // and weld naked edges.
        tools::weld(&joined_voxel_mesh, (min_voxel_dimension as f32) / 4.0)
    }

    /// Returns the bounding box of the mesh produced by `ScalarField::to_mesh`
    /// for this scalar field in world space cartesian units.
    pub fn mesh_volume_bounding_box_cartesian(
        &self,
        volume_value_interval: Interval<T>,
    ) -> Option<BoundingBox<f32>> {
        let voxel_dimensions = self.voxel_dimensions;
        self.compute_volume_boundaries(volume_value_interval).map(
            |(volume_start, volume_dimensions)| {
                BoundingBox::new(
                    &Point3::new(
                        (volume_start.x as f32 - 0.5) * voxel_dimensions.x,
                        (volume_start.y as f32 - 0.5) * voxel_dimensions.y,
                        (volume_start.z as f32 - 0.5) * voxel_dimensions.z,
                    ),
                    &Point3::new(
                        (volume_start.x as f32 + volume_dimensions.x as f32 + 0.5)
                            * voxel_dimensions.x,
                        (volume_start.y as f32 + volume_dimensions.y as f32 + 0.5)
                            * voxel_dimensions.y,
                        (volume_start.z as f32 + volume_dimensions.z as f32 + 0.5)
                            * voxel_dimensions.z,
                    ),
                )
            },
        )
    }

    /// Returns the bounding box in cartesian units of the current scalar field
    /// after shrinking to fit just the nonempty voxels.
    #[allow(dead_code)]
    pub fn volume_bounding_box_cartesian(
        &self,
        volume_value_interval: Interval<T>,
    ) -> Option<BoundingBox<f32>> {
        let voxel_dimensions = self.voxel_dimensions;
        self.compute_volume_boundaries(volume_value_interval).map(
            |(volume_start, volume_dimensions)| {
                BoundingBox::new(
                    &Point3::new(
                        (volume_start.x as f32) * voxel_dimensions.x,
                        (volume_start.y as f32) * voxel_dimensions.y,
                        (volume_start.z as f32) * voxel_dimensions.z,
                    ),
                    &Point3::new(
                        (volume_start.x as f32 + volume_dimensions.x as f32) * voxel_dimensions.x,
                        (volume_start.y as f32 + volume_dimensions.y as f32) * voxel_dimensions.y,
                        (volume_start.z as f32 + volume_dimensions.z as f32) * voxel_dimensions.z,
                    ),
                )
            },
        )
    }

    /// Returns the bounding box in voxel units of the current scalar field
    /// after shrinking to fit just the nonempty voxels.
    pub fn volume_bounding_box(
        &self,
        volume_value_interval: Interval<T>,
    ) -> Option<BoundingBox<i32>> {
        self.compute_volume_boundaries(volume_value_interval).map(
            |(volume_start, volume_dimensions)| {
                let volume_end = volume_start
                    + Vector3::new(
                        cast_i32(volume_dimensions.x),
                        cast_i32(volume_dimensions.y),
                        cast_i32(volume_dimensions.z),
                    );
                BoundingBox::new(&volume_start, &volume_end)
            },
        )
    }

    /// Compute discrete distance field. Each voxel will be set a value equal to
    /// its distance from the original volume. The voxels that were originally
    /// volume voxels, will be set to value 0. Voxels inside the closed volumes
    /// will have a value with a negative sign.
    pub fn compute_distance_filed(&mut self, volume_value_interval: Interval<T>) {
        // Lookup table of neighbor coordinates
        let neighbor_offsets = [
            Vector3::new(-1, 0, 0),
            Vector3::new(1, 0, 0),
            Vector3::new(0, -1, 0),
            Vector3::new(0, 1, 0),
            Vector3::new(0, 0, -1),
            Vector3::new(0, 0, 1),
        ];

        // Contains indices into the voxel map
        let mut queue_to_find_outer: VecDeque<usize> = VecDeque::new();
        // Contains indices to the voxel map and their distance value
        let mut queue_to_compute_distance: VecDeque<(usize, T)> = VecDeque::new();
        // Match the voxel map length
        let mut discovered_as_outer_and_empty = vec![false; self.values.len()];
        let mut discovered_as_empty = vec![false; self.values.len()];

        // Scan for void voxels at the boundaries of the scalar field and for
        // volume voxels anywhere.
        for (one_dimensional, voxel_value) in self.values.iter().enumerate() {
            let relative_coordinate = one_dimensional_to_relative_voxel_coordinate(
                one_dimensional,
                &self.block_dimensions,
            )
            .expect("Coord out of bounds");
            // If the voxel is void
            if !volume_value_interval.includes_closed(*voxel_value) {
                // If any of these is true, the coordinate is at the boundary of the
                // scalar field block
                if relative_coordinate.x == 0
                    || relative_coordinate.y == 0
                    || relative_coordinate.z == 0
                    || relative_coordinate.x == cast_i32(self.block_dimensions.x) - 1
                    || relative_coordinate.y == cast_i32(self.block_dimensions.y) - 1
                    || relative_coordinate.z == cast_i32(self.block_dimensions.z) - 1
                {
                    // put it into the queue for finding outer empty voxels
                    queue_to_find_outer.push_back(one_dimensional);
                    // and mark it discovered.
                    discovered_as_outer_and_empty[one_dimensional] = true;
                }
            } else {
                // If the voxel is a part of the volume
                let absolute_coordinate = one_dimensional_to_absolute_voxel_coordinate(
                    one_dimensional,
                    &self.block_start,
                    &self.block_dimensions,
                )
                .expect("Coord out of bounds");

                // Check if any of his neighbors are void
                for neighbor_offset in &neighbor_offsets {
                    let neighbor_absolute_coordinate = absolute_coordinate + neighbor_offset;
                    if let Some(neighbor_value) =
                        self.value_at_absolute_voxel_coordinate(&neighbor_absolute_coordinate)
                    {
                        // if they are void, add them to the processing queue
                        if !volume_value_interval.includes_closed(neighbor_value) {
                            if let Some(one_dimensional_neighbor) =
                                absolute_voxel_to_one_dimensional_coordinate(
                                    &neighbor_absolute_coordinate,
                                    &self.block_start,
                                    &self.block_dimensions,
                                )
                            {
                                // with the current distance from the volume 1
                                queue_to_compute_distance.push_back((
                                    one_dimensional_neighbor,
                                    T::from_u32(1).expect("Conversion from u32 failed"),
                                ));
                                // and mark them discovered
                                discovered_as_empty[one_dimensional_neighbor] = true;
                            }
                        }
                    }
                }
            }
        }

        // Process the queue to find the outer void voxels
        while let Some(one_dimensional) = queue_to_find_outer.pop_front() {
            // Calculate the absolute coord of the currently processed voxel.
            // It will be needed to calculate its neighbors.
            let absolute_coordinate = one_dimensional_to_absolute_voxel_coordinate(
                one_dimensional,
                &self.block_start,
                &self.block_dimensions,
            )
            .expect("Coord out of bounds");

            // Check all the neighbors
            for neighbor_offset in &neighbor_offsets {
                let neighbor_absolute_coordinate = absolute_coordinate + neighbor_offset;
                // If the neighbor exists (is not out of bounds)
                if let Some(neighbor_value) =
                    self.value_at_absolute_voxel_coordinate(&neighbor_absolute_coordinate)
                {
                    // and doesn't contain any volume
                    if !volume_value_interval.includes_closed(neighbor_value) {
                        let neighbor_one_dimensional =
                            absolute_voxel_to_one_dimensional_coordinate(
                                &neighbor_absolute_coordinate,
                                &self.block_start,
                                &self.block_dimensions,
                            )
                            .expect("Coord out of bounds");
                        // Check if it hasn't been discovered yet
                        if !discovered_as_outer_and_empty[neighbor_one_dimensional] {
                            // Put it to the processing queue
                            queue_to_find_outer.push_back(neighbor_one_dimensional);
                            // and mark it discovered.
                            discovered_as_outer_and_empty[neighbor_one_dimensional] = true;
                        }
                    }
                }
            }
        }

        // Now when we know which voxels are outside, let's scan for distance.

        // Process the queue to set the voxel distance from the volume
        while let Some((one_dimensional, distance)) = queue_to_compute_distance.pop_front() {
            // Needed to calculate neighbors' coordinates
            let absolute_coordinate = one_dimensional_to_absolute_voxel_coordinate(
                one_dimensional,
                &self.block_start,
                &self.block_dimensions,
            )
            .expect("Coord out of bounds");

            // Check each neighbor
            for neighbor_offset in &neighbor_offsets {
                let neighbor_absolute_coordinate = absolute_coordinate + neighbor_offset;
                // If the neighbor does exist
                if let Some(one_dimensional_neighbor) = absolute_voxel_to_one_dimensional_coordinate(
                    &neighbor_absolute_coordinate,
                    &self.block_start,
                    &self.block_dimensions,
                ) {
                    // and hasn't been discovered yet
                    if !discovered_as_empty[one_dimensional_neighbor] {
                        let neighbor_value = self
                            .value_at_absolute_voxel_coordinate(&neighbor_absolute_coordinate)
                            .expect("The neighbor voxel doesn't exist");
                        // and is void,
                        if !volume_value_interval.includes_closed(neighbor_value) {
                            // put it into the processing queue with the
                            // distance one higher than the current
                            queue_to_compute_distance.push_back((
                                one_dimensional_neighbor,
                                distance + T::from_u32(1).expect("Conversion from u32 failed."),
                            ));
                            // and mark it discovered.
                            discovered_as_empty[one_dimensional_neighbor] = true;
                        }
                    }
                }
            }

            // Process the current voxel. If it is outside the volumes, set its
            // value to be positive, if it's inside, set it to negative.
            self.values[one_dimensional] = if discovered_as_outer_and_empty[one_dimensional] {
                distance
            } else {
                -distance
            };
        }

        // The actual volume voxels remained intact. Scan the scalar field and
        // set the volume voxel distance to 0.
        for (one_dimensional, voxel_value) in self.values.iter_mut().enumerate() {
            if !discovered_as_empty[one_dimensional] {
                *voxel_value = T::from_u32(0).expect("Conversion from u32 failed");
            }
        }
    }

    /// Computes boundaries of volumes contained in scalar field. Returns tuple
    /// (block_start, block_dimensions). For empty scalar fields returns the
    /// original block start and zero block dimensions.
    fn compute_volume_boundaries(
        &self,
        volume_value_interval: Interval<T>,
    ) -> Option<(Point3<i32>, Vector3<u32>)> {
        let mut min: Vector3<i32> =
            Vector3::new(i32::max_value(), i32::max_value(), i32::max_value());
        let mut max: Vector3<i32> =
            Vector3::new(i32::min_value(), i32::min_value(), i32::min_value());
        for (i, v) in self.values.iter().enumerate() {
            if volume_value_interval.includes_closed(*v) {
                let relative_coordinate =
                    one_dimensional_to_relative_voxel_coordinate(i, &self.block_dimensions)
                        .expect("Out of bounds");
                if relative_coordinate.x < min.x {
                    min.x = relative_coordinate.x;
                }
                if relative_coordinate.x > max.x {
                    max.x = relative_coordinate.x;
                }
                if relative_coordinate.y < min.y {
                    min.y = relative_coordinate.y;
                }
                if relative_coordinate.y > max.y {
                    max.y = relative_coordinate.y;
                }
                if relative_coordinate.z < min.z {
                    min.z = relative_coordinate.z;
                }
                if relative_coordinate.z > max.z {
                    max.z = relative_coordinate.z;
                }
            }
        }
        // It's enough to check one of the values because if anything is found,
        // all the values would change.
        if min.x == i32::max_value() {
            assert_eq!(
                min.y,
                i32::max_value(),
                "scalar field emptiness check failed"
            );
            assert_eq!(
                min.z,
                i32::max_value(),
                "scalar field emptiness check failed"
            );
            assert_eq!(
                max.x,
                i32::min_value(),
                "scalar field emptiness check failed"
            );
            assert_eq!(
                max.y,
                i32::min_value(),
                "scalar field emptiness check failed"
            );
            assert_eq!(
                max.z,
                i32::min_value(),
                "scalar field emptiness check failed"
            );
            None
        } else {
            let block_dimensions = Vector3::new(
                clamp_cast_i32_to_u32(max.x - min.x + 1),
                clamp_cast_i32_to_u32(max.y - min.y + 1),
                clamp_cast_i32_to_u32(max.z - min.z + 1),
            );
            Some(((self.block_start + min), block_dimensions))
        }
    }
}

/// Computes a voxel position relative to the block start (relative coordinate)
/// from an index to the linear representation of the voxel block.
///
/// Returns None if out of bounds.
fn one_dimensional_to_relative_voxel_coordinate(
    one_dimensional_coordinate: usize,
    block_dimensions: &Vector3<u32>,
) -> Option<Point3<i32>> {
    let values_len = cast_usize(block_dimensions.x * block_dimensions.y * block_dimensions.z);
    if one_dimensional_coordinate < values_len {
        let one_dimensional_i32 = cast_i32(one_dimensional_coordinate);
        let horizontal_area_i32 = cast_i32(block_dimensions.x * block_dimensions.y);
        let x_dimension_i32 = cast_i32(block_dimensions.x);
        let z = one_dimensional_i32 / horizontal_area_i32;
        let y = (one_dimensional_i32 % horizontal_area_i32) / x_dimension_i32;
        let x = one_dimensional_i32 % x_dimension_i32;
        Some(Point3::new(x, y, z))
    } else {
        None
    }
}

/// Computes a voxel position relative to the model space origin (absolute
/// coordinate) from an index to the linear representation of the voxel block.
///
/// Returns None if out of bounds.
fn one_dimensional_to_absolute_voxel_coordinate(
    one_dimensional_coordinate: usize,
    block_start: &Point3<i32>,
    block_dimensions: &Vector3<u32>,
) -> Option<Point3<i32>> {
    one_dimensional_to_relative_voxel_coordinate(one_dimensional_coordinate, block_dimensions)
        .map(|relative| relative_voxel_to_absolute_voxel_coordinate(&relative, block_start))
}

/// Computes a voxel position in world space cartesian units from an index to
/// the linear representation of the voxel block.
///
/// Returns None if out of bounds.
fn one_dimensional_to_cartesian_coordinate(
    one_dimensional_coordinate: usize,
    block_start: &Point3<i32>,
    block_dimensions: &Vector3<u32>,
    voxel_dimensions: &Vector3<f32>,
) -> Option<Point3<f32>> {
    one_dimensional_to_relative_voxel_coordinate(one_dimensional_coordinate, block_dimensions).map(
        |relative| relative_voxel_to_cartesian_coordinate(&relative, block_start, voxel_dimensions),
    )
}

/// Computes an index to the linear representation of the voxel block from
/// voxel coordinates relative to the voxel space block start.
///
/// Returns None if out of bounds.
fn relative_voxel_to_one_dimensional_coordinate(
    relative_coordinate: &Point3<i32>,
    block_dimensions: &Vector3<u32>,
) -> Option<usize> {
    if relative_coordinate
        .iter()
        .enumerate()
        .all(|(i, coordinate)| *coordinate >= 0 && *coordinate < cast_i32(block_dimensions[i]))
    {
        let index =
            relative_coordinate.z * cast_i32(block_dimensions.x) * cast_i32(block_dimensions.y)
                + relative_coordinate.y * cast_i32(block_dimensions.x)
                + relative_coordinate.x;
        Some(cast_usize(index))
    } else {
        None
    }
}

/// Computes the center of a voxel in absolute voxel units from voxel
/// coordinates relative to the voxel block start.
fn relative_voxel_to_absolute_voxel_coordinate(
    relative_coordinate: &Point3<i32>,
    block_start: &Point3<i32>,
) -> Point3<i32> {
    relative_coordinate + block_start.coords
}

/// Computes the center of a voxel in worlds space cartesian units from voxel
/// coordinates relative to the voxel block start.
///
/// # Panics
/// Panics if any of the voxel dimensions is zero.
fn relative_voxel_to_cartesian_coordinate(
    relative_coordinate: &Point3<i32>,
    block_start: &Point3<i32>,
    voxel_dimensions: &Vector3<f32>,
) -> Point3<f32> {
    assert!(
        !approx::relative_eq!(voxel_dimensions.x, 0.0)
            && !approx::relative_eq!(voxel_dimensions.y, 0.0)
            && !approx::relative_eq!(voxel_dimensions.z, 0.0),
        "Voxel dimensions can't be 0.0"
    );
    Point3::new(
        (relative_coordinate.x + block_start.x) as f32 * voxel_dimensions.x,
        (relative_coordinate.y + block_start.y) as f32 * voxel_dimensions.y,
        (relative_coordinate.z + block_start.z) as f32 * voxel_dimensions.z,
    )
}

/// Computes the absolute voxel space coordinate of a voxel containing the input
/// point.
///
/// # Panics
/// Panics if any of the voxel dimensions is equal or below zero.
fn cartesian_to_absolute_voxel_coordinate(
    point: &Point3<f32>,
    voxel_dimensions: &Vector3<f32>,
) -> Point3<i32> {
    assert!(
        voxel_dimensions.x > 0.0 && voxel_dimensions.y > 0.0 && voxel_dimensions.z > 0.0,
        "Voxel dimensions can't be below or equal to zero"
    );
    Point3::new(
        (point.x / voxel_dimensions.x).round() as i32,
        (point.y / voxel_dimensions.y).round() as i32,
        (point.z / voxel_dimensions.z).round() as i32,
    )
}

/// Computes an index to the linear representation of the voxel block from a
/// cartesian coordinate.
///
/// # Panics
/// Panics if any of the voxel dimensions is equal or below zero.
fn cartesian_to_one_dimensional_coordinate(
    point: &Point3<f32>,
    block_start: &Point3<i32>,
    block_dimensions: &Vector3<u32>,
    voxel_dimensions: &Vector3<f32>,
) -> Option<usize> {
    absolute_voxel_to_one_dimensional_coordinate(
        &cartesian_to_absolute_voxel_coordinate(point, voxel_dimensions),
        block_start,
        block_dimensions,
    )
}

/// Computes an index to the linear representation of the voxel block from
/// absolute voxel coordinates (relative to the voxel space origin).
///
/// Returns None if out of bounds.
fn absolute_voxel_to_one_dimensional_coordinate(
    absolute_coordinate: &Point3<i32>,
    block_start: &Point3<i32>,
    block_dimensions: &Vector3<u32>,
) -> Option<usize> {
    let relative_coordinate = absolute_coordinate - block_start.coords;
    relative_voxel_to_one_dimensional_coordinate(&relative_coordinate, block_dimensions)
}

#[cfg(test)]
mod tests {
    use nalgebra::Rotation3;

    use crate::mesh::{analysis, topology, NormalStrategy};

    use super::*;

    fn torus() -> (Vec<(u32, u32, u32)>, Vec<Point3<f32>>) {
        let vertices = vec![
            Point3::new(0.566987, -1.129e-11, 0.25),
            Point3::new(-0.716506, 1.241025, 0.25),
            Point3::new(-0.283494, 0.491025, 0.25),
            Point3::new(-0.716506, -1.241025, 0.25),
            Point3::new(-0.283494, -0.491025, 0.25),
            Point3::new(1.0, -1.129e-11, -0.5),
            Point3::new(1.433013, -1.129e-11, 0.25),
            Point3::new(-0.5, 0.866025, -0.5),
            Point3::new(-0.5, -0.866025, -0.5),
        ];

        let faces = vec![
            (4, 3, 6),
            (0, 6, 2),
            (2, 1, 3),
            (8, 4, 0),
            (3, 8, 6),
            (5, 0, 7),
            (6, 5, 7),
            (7, 2, 4),
            (1, 7, 8),
            (4, 6, 0),
            (6, 1, 2),
            (2, 3, 4),
            (8, 0, 5),
            (8, 5, 6),
            (0, 2, 7),
            (6, 7, 1),
            (7, 4, 8),
            (1, 8, 3),
        ];

        (faces, vertices)
    }

    #[test]
    fn test_scalar_field_from_mesh_for_torus() {
        let (faces, vertices) = torus();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );
        let scalar_field = ScalarField::from_mesh(&mesh, &Vector3::new(1.0, 1.0, 1.0), 0, 0);

        insta::assert_json_snapshot!("torus_after_voxelization_into_scalar_field", &scalar_field);
    }

    #[test]
    fn test_scalar_field_from_mesh_for_sphere() {
        let mesh = primitive::create_uv_sphere(
            Point3::origin(),
            Rotation3::identity(),
            Vector3::new(1.0, 1.0, 1.0),
            10,
            10,
            NormalStrategy::Sharp,
        );

        let scalar_field = ScalarField::from_mesh(&mesh, &Vector3::new(0.5, 0.5, 0.5), 0, 0);

        insta::assert_json_snapshot!("sphere_after_voxelization_into_scalar_field", &scalar_field);
    }

    #[test]
    fn test_scalar_field_three_dimensional_to_one_dimensional_and_back_relative() {
        let scalar_field: ScalarField<i16> = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(3, 4, 5),
            &Vector3::new(1.5, 2.5, 3.5),
        );
        for z in 0..scalar_field.block_dimensions.z {
            for y in 0..scalar_field.block_dimensions.y {
                for x in 0..scalar_field.block_dimensions.x {
                    let relative_position = Point3::new(cast_i32(x), cast_i32(y), cast_i32(z));
                    let one_dimensional = relative_voxel_to_one_dimensional_coordinate(
                        &relative_position,
                        &scalar_field.block_dimensions,
                    )
                    .unwrap();
                    let three_dimensional = one_dimensional_to_relative_voxel_coordinate(
                        one_dimensional,
                        &scalar_field.block_dimensions,
                    )
                    .unwrap();
                    assert_eq!(relative_position, three_dimensional);
                }
            }
        }
    }

    #[test]
    fn test_scalar_field_boolean_intersection_all_true() {
        let mut sf_a = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(3, 3, 3),
            &Vector3::new(0.5, 0.5, 0.5),
        );
        let mut sf_b = ScalarField::new(
            &Point3::new(2, 1, 1),
            &Vector3::new(3, 3, 3),
            &Vector3::new(0.5, 0.5, 0.5),
        );
        let mut sf_correct = ScalarField::new(
            &Point3::new(2, 1, 1),
            &Vector3::new(1, 2, 2),
            &Vector3::new(0.5, 0.5, 0.5),
        );

        sf_a.fill_with(0);
        sf_b.fill_with(0);
        sf_correct.fill_with(0);

        sf_a.boolean_intersection(Interval::new(0, 0), &sf_b, Interval::new(0, 0));

        assert_eq!(sf_a, sf_correct);
    }

    #[test]
    fn test_scalar_field_boolean_intersection_one_false() {
        let mut sf_a = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(3, 3, 3),
            &Vector3::new(0.5, 0.5, 0.5),
        );
        let mut sf_b = ScalarField::new(
            &Point3::new(1, 1, 1),
            &Vector3::new(3, 3, 3),
            &Vector3::new(0.5, 0.5, 0.5),
        );
        let mut sf_correct = ScalarField::new(
            &Point3::new(1, 1, 1),
            &Vector3::new(2, 2, 2),
            &Vector3::new(0.5, 0.5, 0.5),
        );

        sf_a.fill_with(0);
        sf_b.fill_with(0);
        sf_b.set_value_at_absolute_voxel_coordinate(
            &Point3::new(2, 2, 2),
            ScalarField::empty_value(),
        );
        sf_correct.fill_with(0);
        sf_correct.set_value_at_absolute_voxel_coordinate(
            &Point3::new(2, 2, 2),
            ScalarField::empty_value(),
        );

        sf_a.boolean_intersection(Interval::new(0, 0), &sf_b, Interval::new(0, 0));

        assert_eq!(sf_a, sf_correct);
    }

    #[test]
    fn test_scalar_field_boolean_union_one_false() {
        let mut sf_a = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(3, 3, 3),
            &Vector3::new(0.5, 0.5, 0.5),
        );
        let mut sf_b = ScalarField::new(
            &Point3::new(1, 1, 1),
            &Vector3::new(3, 3, 3),
            &Vector3::new(0.5, 0.5, 0.5),
        );
        let mut sf_correct = ScalarField::new(
            &Point3::new(0, 0, 0),
            &Vector3::new(4, 4, 4),
            &Vector3::new(0.5, 0.5, 0.5),
        );

        sf_a.fill_with(0);
        sf_b.fill_with(0);
        sf_b.set_value_at_absolute_voxel_coordinate(
            &Point3::new(2, 2, 2),
            ScalarField::empty_value(),
        );
        sf_correct.fill_with(0);
        sf_correct.set_value_at_absolute_voxel_coordinate(
            &Point3::new(3, 0, 0),
            ScalarField::empty_value(),
        );
        sf_correct.set_value_at_absolute_voxel_coordinate(
            &Point3::new(3, 1, 0),
            ScalarField::empty_value(),
        );
        sf_correct.set_value_at_absolute_voxel_coordinate(
            &Point3::new(3, 2, 0),
            ScalarField::empty_value(),
        );
        sf_correct.set_value_at_absolute_voxel_coordinate(
            &Point3::new(3, 3, 0),
            ScalarField::empty_value(),
        );
        sf_correct.set_value_at_absolute_voxel_coordinate(
            &Point3::new(3, 0, 1),
            ScalarField::empty_value(),
        );
        sf_correct.set_value_at_absolute_voxel_coordinate(
            &Point3::new(3, 0, 2),
            ScalarField::empty_value(),
        );
        sf_correct.set_value_at_absolute_voxel_coordinate(
            &Point3::new(3, 0, 3),
            ScalarField::empty_value(),
        );
        sf_correct.set_value_at_absolute_voxel_coordinate(
            &Point3::new(2, 0, 3),
            ScalarField::empty_value(),
        );
        sf_correct.set_value_at_absolute_voxel_coordinate(
            &Point3::new(1, 0, 3),
            ScalarField::empty_value(),
        );
        sf_correct.set_value_at_absolute_voxel_coordinate(
            &Point3::new(0, 0, 3),
            ScalarField::empty_value(),
        );
        sf_correct.set_value_at_absolute_voxel_coordinate(
            &Point3::new(0, 1, 3),
            ScalarField::empty_value(),
        );
        sf_correct.set_value_at_absolute_voxel_coordinate(
            &Point3::new(0, 2, 3),
            ScalarField::empty_value(),
        );
        sf_correct.set_value_at_absolute_voxel_coordinate(
            &Point3::new(0, 3, 3),
            ScalarField::empty_value(),
        );
        sf_correct.set_value_at_absolute_voxel_coordinate(
            &Point3::new(0, 3, 2),
            ScalarField::empty_value(),
        );
        sf_correct.set_value_at_absolute_voxel_coordinate(
            &Point3::new(0, 3, 1),
            ScalarField::empty_value(),
        );
        sf_correct.set_value_at_absolute_voxel_coordinate(
            &Point3::new(0, 3, 0),
            ScalarField::empty_value(),
        );
        sf_correct.set_value_at_absolute_voxel_coordinate(
            &Point3::new(1, 3, 0),
            ScalarField::empty_value(),
        );
        sf_correct.set_value_at_absolute_voxel_coordinate(
            &Point3::new(2, 3, 0),
            ScalarField::empty_value(),
        );

        sf_a.boolean_union(Interval::new(0, 0), &sf_b, Interval::new(0, 0));

        assert_eq!(sf_a, sf_correct);
    }

    #[test]
    fn test_scalar_field_get_set_for_single_voxel() {
        let mut scalar_field: ScalarField<i16> = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(1, 1, 1),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        let before = scalar_field
            .value_at_absolute_voxel_coordinate(&Point3::new(0, 0, 0))
            .unwrap();
        scalar_field.set_value_at_absolute_voxel_coordinate(&Point3::new(0, 0, 0), 0);
        let after = scalar_field
            .value_at_absolute_voxel_coordinate(&Point3::new(0, 0, 0))
            .unwrap();
        let empty: i16 = ScalarField::empty_value();
        assert_eq!(before, empty);
        assert_eq!(after, 0);
    }

    #[test]
    fn test_scalar_field_single_voxel_to_mesh_produces_synchronized_mesh() {
        let mut scalar_field = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(1, 1, 1),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        scalar_field.set_value_at_absolute_voxel_coordinate(&Point3::new(0, 0, 0), 0);

        let voxel_mesh = scalar_field.to_mesh(Interval::new(0, 0)).unwrap();

        let v2f = topology::compute_vertex_to_face_topology(&voxel_mesh);
        let f2f = topology::compute_face_to_face_topology(&voxel_mesh, &v2f);
        let voxel_mesh_synced = tools::synchronize_mesh_winding(&voxel_mesh, &f2f);

        assert!(analysis::are_similar(&voxel_mesh, &voxel_mesh_synced));
    }

    #[test]
    fn test_scalar_field_resize_zero_to_nonzero_all_false() {
        let mut scalar_field: ScalarField<i16> = ScalarField::new(
            &Point3::origin(),
            &Vector3::zeros(),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        scalar_field.resize(&Point3::origin(), &Vector3::new(1, 1, 1));

        let voxel = scalar_field
            .value_at_absolute_voxel_coordinate(&Point3::new(0, 0, 0))
            .unwrap();

        let empty: i16 = ScalarField::empty_value();
        assert_eq!(voxel, empty);
    }

    #[test]
    fn test_scalar_field_resize_zero_to_nonzero_correct_start_and_dimensions() {
        let mut scalar_field: ScalarField<i16> = ScalarField::new(
            &Point3::origin(),
            &Vector3::zeros(),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        let new_origin = Point3::new(1, 2, 3);
        let new_block_dimensions = Vector3::new(4, 5, 6);
        scalar_field.resize(&new_origin, &new_block_dimensions);

        assert_eq!(scalar_field.block_start, new_origin);
        assert_eq!(scalar_field.block_dimensions, new_block_dimensions);
        assert_eq!(scalar_field.values.len(), 4 * 5 * 6);
    }

    #[test]
    fn test_scalar_field_resize_nonzero_to_zero_correct_start_and_dimensions() {
        let mut scalar_field: ScalarField<i16> = ScalarField::new(
            &Point3::new(1, 2, 3),
            &Vector3::new(4, 5, 6),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        let new_origin = Point3::origin();
        let new_block_dimensions = Vector3::zeros();
        scalar_field.resize(&new_origin, &new_block_dimensions);

        assert_eq!(scalar_field.block_start, new_origin);
        assert_eq!(scalar_field.block_dimensions, new_block_dimensions);
        assert_eq!(scalar_field.values.len(), 0);
    }

    #[test]
    fn test_scalar_field_resize_nonzero_to_smaller_nonzero_correct_start_and_dimensions() {
        let mut scalar_field: ScalarField<i16> = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(4, 5, 6),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        let new_origin = Point3::new(1, 2, 3);
        let new_block_dimensions = Vector3::new(1, 2, 3);
        scalar_field.resize(&new_origin, &new_block_dimensions);

        assert_eq!(scalar_field.block_start, new_origin);
        assert_eq!(scalar_field.block_dimensions, new_block_dimensions);
        assert_eq!(scalar_field.values.len(), 1 * 2 * 3);
    }

    #[test]
    fn test_scalar_field_resize_nonzero_to_larger_nonzero_correct_start_and_dimensions() {
        let mut scalar_field: ScalarField<i16> = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(1, 2, 3),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        let new_origin = Point3::new(1, 2, 3);
        let new_block_dimensions = Vector3::new(4, 5, 6);
        scalar_field.resize(&new_origin, &new_block_dimensions);

        assert_eq!(scalar_field.block_start, new_origin);
        assert_eq!(scalar_field.block_dimensions, new_block_dimensions);
        assert_eq!(scalar_field.values.len(), 4 * 5 * 6);
    }

    #[test]
    fn test_scalar_field_resize_nonzero_to_larger_nonzero_grown_contains_false_rest_original() {
        let original_origin = Point3::new(0i32, 0i32, 0i32);
        let original_block_dimensions = Vector3::new(1u32, 10u32, 3u32);
        let mut scalar_field: ScalarField<i16> = ScalarField::new(
            &original_origin,
            &original_block_dimensions,
            &Vector3::new(1.0, 1.0, 1.0),
        );
        let original_block_end = scalar_field.block_end();

        scalar_field.fill_with(0);

        let new_origin = Point3::new(-1, 2, 3);
        let new_block_dimensions = Vector3::new(4, 5, 6);
        scalar_field.resize(&new_origin, &new_block_dimensions);

        let empty: i16 = ScalarField::empty_value();
        
        for (i, v) in scalar_field.values.iter().enumerate() {
            let coordinate = one_dimensional_to_absolute_voxel_coordinate(
                i,
                &scalar_field.block_start,
                &scalar_field.block_dimensions,
            )
            .unwrap();

            if coordinate.x < original_origin.x
                || coordinate.y < original_origin.y
                || coordinate.z < original_origin.z
                || coordinate.x > original_block_end.x
                || coordinate.y > original_block_end.y
                || coordinate.z > original_block_end.z
            {
                assert_eq!(*v, empty);
            } else {
                assert_eq!(*v, 0);
            }
        }
    }

    #[test]
    fn test_scalar_field_shrink_to_volume() {
        let mut scalar_field: ScalarField<i16> = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(4, 5, 6),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        scalar_field.set_value_at_absolute_voxel_coordinate(&Point3::new(1, 1, 1), 0);
        scalar_field.shrink_to_fit(Interval::new(0, 0));

        assert_eq!(scalar_field.block_start, Point3::new(1, 1, 1));
        assert_eq!(scalar_field.block_dimensions, Vector3::new(1, 1, 1));
        assert_eq!(scalar_field.values.len(), 1);
        assert_eq!(
            scalar_field
                .value_at_absolute_voxel_coordinate(&Point3::new(1, 1, 1))
                .unwrap(),
            0
        );
    }

    #[test]
    fn test_scalar_field_shrink_to_empty() {
        let mut scalar_field: ScalarField<i16> = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(4, 5, 6),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        scalar_field.shrink_to_fit(Interval::new(0, 0));

        assert_eq!(scalar_field.block_start, Point3::origin());
        assert_eq!(scalar_field.block_dimensions, Vector3::new(0, 0, 0));
        assert_eq!(scalar_field.values.len(), 0);
    }

    #[test]
    fn test_scalar_field_transform_box_at_origin_identity() {
        let mesh = primitive::create_box(
            Point3::origin(),
            Rotation3::from_euler_angles(0.0, 0.0, 0.0),
            Vector3::new(1.0, 2.0, 3.0),
        );
        let scalar_field: ScalarField<i16> = ScalarField::from_mesh(&mesh, &Vector3::new(0.25, 0.25, 0.25), 0, 0);
        let transformed_scalar_field = ScalarField::from_scalar_field_transformed(
            &scalar_field,
            Interval::new(0, 0),
            0.25,
            &Vector3::zeros(),
            &Rotation3::from_euler_angles(0.0, 0.0, 0.0),
            &Vector3::new(1.0, 1.0, 1.0),
        )
        .unwrap();

        assert_eq!(transformed_scalar_field, scalar_field);
    }

    #[test]
    fn test_scalar_field_transform_box_at_random_location_identity() {
        let mesh = primitive::create_box(
            Point3::new(5.1, 6.2, 7.3),
            Rotation3::from_euler_angles(0.0, 0.0, 0.0),
            Vector3::new(1.0, 2.0, 3.0),
        );
        let scalar_field: ScalarField<i16> = ScalarField::from_mesh(&mesh, &Vector3::new(0.25, 0.25, 0.25), 0, 0);
        let transformed_scalar_field = ScalarField::from_scalar_field_transformed(
            &scalar_field,
            Interval::new(0, 0),
            0.25,
            &Vector3::zeros(),
            &Rotation3::from_euler_angles(0.0, 0.0, 0.0),
            &Vector3::new(1.0, 1.0, 1.0),
        )
        .unwrap();

        assert_eq!(transformed_scalar_field, scalar_field);
    }

    #[test]
    fn test_scalar_field_transform_box_at_random_location_rotated_identity() {
        let mesh = primitive::create_box(
            Point3::new(5.1, 6.2, 7.3),
            Rotation3::from_euler_angles(1.1, 2.2, 3.3),
            Vector3::new(1.0, 2.0, 3.0),
        );
        let scalar_field: ScalarField<i16> = ScalarField::from_mesh(&mesh, &Vector3::new(0.25, 0.25, 0.25), 0, 0);
        let transformed_scalar_field = ScalarField::from_scalar_field_transformed(
            &scalar_field,
            Interval::new(0, 0),
            0.25,
            &Vector3::zeros(),
            &Rotation3::from_euler_angles(0.0, 0.0, 0.0),
            &Vector3::new(1.0, 1.0, 1.0),
        )
        .unwrap();

        assert_eq!(transformed_scalar_field, scalar_field);
    }

    #[test]
    fn test_scalar_field_transform_sub_voxel_translation() {
        let mesh = primitive::create_box(
            Point3::new(5.1, 6.2, 7.3),
            Rotation3::from_euler_angles(1.1, 2.2, 3.3),
            Vector3::new(1.0, 2.0, 3.0),
        );
        let scalar_field: ScalarField<i16> = ScalarField::from_mesh(&mesh, &Vector3::new(0.25, 0.25, 0.25), 0, 0);
        let transformed_scalar_field = ScalarField::from_scalar_field_transformed(
            &scalar_field,
            Interval::new(0, 0),
            0.25,
            &Vector3::new(0.1, 0.1, 0.1),
            &Rotation3::from_euler_angles(0.0, 0.0, 0.0),
            &Vector3::new(1.0, 1.0, 1.0),
        )
        .unwrap();

        insta::assert_json_snapshot!(
            "scalar_field, transform_sub_voxel_translation",
            &transformed_scalar_field
        );
    }

    #[test]
    fn test_scalar_field_transform_voxel_size_translation() {
        let mesh = primitive::create_box(
            Point3::new(5.1, 6.2, 7.3),
            Rotation3::from_euler_angles(1.1, 2.2, 3.3),
            Vector3::new(1.0, 2.0, 3.0),
        );
        let scalar_field: ScalarField<i16> = ScalarField::from_mesh(&mesh, &Vector3::new(0.25, 0.25, 0.25), 0, 0);
        let transformed_scalar_field = ScalarField::from_scalar_field_transformed(
            &scalar_field,
            Interval::new(0, 0),
            0.25,
            &Vector3::new(0.0, 0.0, 0.25),
            &Rotation3::from_euler_angles(0.0, 0.0, 0.0),
            &Vector3::new(1.0, 1.0, 1.0),
        )
        .unwrap();

        insta::assert_json_snapshot!(
            "scalar_field_transform_voxel_size_translation",
            &transformed_scalar_field
        );
    }

    #[test]
    fn test_scalar_field_transform_arbitrary_translation() {
        let mesh = primitive::create_box(
            Point3::new(5.1, 6.2, 7.3),
            Rotation3::from_euler_angles(1.1, 2.2, 3.3),
            Vector3::new(1.0, 2.0, 3.0),
        );
        let scalar_field: ScalarField<i16> = ScalarField::from_mesh(&mesh, &Vector3::new(0.25, 0.25, 0.25), 0, 0);
        let transformed_scalar_field = ScalarField::from_scalar_field_transformed(
            &scalar_field,
            Interval::new(0, 0),
            0.25,
            &Vector3::new(0.4, 0.6, 0.7),
            &Rotation3::from_euler_angles(0.0, 0.0, 0.0),
            &Vector3::new(1.0, 1.0, 1.0),
        )
        .unwrap();

        insta::assert_json_snapshot!(
            "scalar_field_transform_arbitrary_translation",
            &transformed_scalar_field
        );
    }

    #[test]
    fn test_scalar_field_arbitrary_transform() {
        let mesh = primitive::create_box(
            Point3::new(5.1, 6.2, 7.3),
            Rotation3::from_euler_angles(1.1, 2.2, 3.3),
            Vector3::new(1.0, 2.0, 3.0),
        );
        let scalar_field: ScalarField<i16> = ScalarField::from_mesh(&mesh, &Vector3::new(0.25, 0.25, 0.25), 0, 0);
        let transformed_scalar_field = ScalarField::from_scalar_field_transformed(
            &scalar_field,
            Interval::new(0, 0),
            0.25,
            &Vector3::new(0.4, 0.6, 0.7),
            &Rotation3::from_euler_angles(25.0, 37.0, 42.0),
            &Vector3::new(1.5, 1.76, 0.5),
        )
        .unwrap();

        insta::assert_json_snapshot!(
            "scalar_field_arbitrary_transform",
            &transformed_scalar_field
        );
    }
}
