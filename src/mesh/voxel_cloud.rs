use std::collections::VecDeque;
use std::f32;

use nalgebra::{Matrix4, Point3, Rotation3, Vector2, Vector3};

use crate::bounding_box::BoundingBox;
use crate::convert::{cast_i32, cast_u32, cast_usize, clamp_cast_i32_to_u32};
use crate::geometry;
use crate::plane::Plane;

use super::{primitive, tools, Face, Mesh};

/// Voxel cloud is an abstract representation of points in a block of
/// space. The block is delimited by its beginning and its dimensions, both in
/// the units of the voxels. All voxels have the same dimensions, which can be
/// different in each direction.
///
/// The voxel space is discrete and you can't start it half way in a voxel,
/// therefore its beginning as well as the voxel positions are defined in the
/// voxel-space coordinates. The voxel space starts at the cartesian space
/// origin with voxel coordinates 0, 0, 0. Voxel clouds with the same voxel size
/// are compatible and collateral operations be performed on them.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct VoxelCloud {
    block_start: Point3<i32>,
    block_dimensions: Vector3<u32>,
    voxel_dimensions: Vector3<f32>,
    // FIXME: @Optimization Change this to a bit vector.
    voxel_map: Vec<bool>,
}

impl VoxelCloud {
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
        VoxelCloud {
            block_start: *block_start,
            block_dimensions: *block_dimensions,
            voxel_dimensions: *voxel_dimensions,
            voxel_map: vec![false; cast_usize(map_length)],
        }
    }

    /// Creates a new empty voxel space from a bounding box defined in cartesian
    /// units.
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

        VoxelCloud::new(&block_start, &block_dimensions, voxel_dimensions)
    }

    /// Creates a voxel cloud from an existing mesh with computed
    /// occupied voxels.
    pub fn from_mesh(mesh: &Mesh, voxel_dimensions: &Vector3<f32>) -> Self {
        assert!(
            voxel_dimensions.x > 0.0 && voxel_dimensions.y > 0.0 && voxel_dimensions.z > 0.0,
            "One or more voxel dimensions are 0.0"
        );
        // Determine the needed block of voxel space.
        let b_box = mesh.bounding_box();

        let mut voxel_cloud = VoxelCloud::from_cartesian_bounding_box(&b_box, voxel_dimensions);

        // Going to populate the mesh with points as dense as the smallest voxel dimension.
        let shortest_voxel_dimension = voxel_dimensions
            .x
            .min(voxel_dimensions.y.min(voxel_dimensions.z));

        for face in mesh.faces() {
            match face {
                Face::Triangle(f) => {
                    let point_a = &mesh.vertices()[cast_usize(f.vertices.0)];
                    let point_b = &mesh.vertices()[cast_usize(f.vertices.1)];
                    let point_c = &mesh.vertices()[cast_usize(f.vertices.2)];
                    // Compute the density of points on the respective face
                    let ab_distance_sq = nalgebra::distance_squared(point_a, point_b);
                    let bc_distance_sq = nalgebra::distance_squared(point_b, point_c);
                    let ca_distance_sq = nalgebra::distance_squared(point_c, point_a);
                    let longest_edge_len = ab_distance_sq
                        .max(bc_distance_sq.max(ca_distance_sq))
                        .sqrt();
                    // Number of face divisions (points) in each direction
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
                                voxel_cloud.set_voxel_at_cartesian_coords(&cartesian, true);
                            }
                        }
                    }
                }
            }
        }

        voxel_cloud
    }

    /// Creates a new voxel cloud with arbitrary voxel dimensions from another
    /// voxel cloud.
    ///
    /// # Panics
    /// Panics if any of the voxel dimensions is below or equal to zero.
    pub fn from_voxel_cloud(
        source_voxel_cloud: &VoxelCloud,
        voxel_dimensions: &Vector3<f32>,
    ) -> Option<Self> {
        assert!(
            voxel_dimensions.x > 0.0 && voxel_dimensions.y > 0.0 && voxel_dimensions.z > 0.0,
            "One mor more voxel dimensions is below or equal to zero"
        );

        // New voxel cloud that can encompass the source voxel cloud's mesh.
        source_voxel_cloud
            .mesh_volume_bounding_box_cartesian()
            .map(|source_vc_bounding_box| {
                // Make a bounding box of the source bounding box's mesh volume.
                // This will be the volume to be scanned for any voxels.

                // New voxel cloud that can encompass the source
                // voxel cloud's mesh.
                let mut target_voxel_cloud = VoxelCloud::from_cartesian_bounding_box(
                    &source_vc_bounding_box,
                    &voxel_dimensions,
                );

                for (one_dimensional, voxel) in target_voxel_cloud.voxel_map.iter_mut().enumerate()
                {
                    let cartesian_coordinate = one_dimensional_to_cartesian_coordinate(
                        one_dimensional,
                        &target_voxel_cloud.block_start,
                        &target_voxel_cloud.block_dimensions,
                        &target_voxel_cloud.voxel_dimensions,
                    )
                    .expect("Index out of bounds");

                    // Set the new voxel state according to a sampled value from
                    // the source voxel cloud.
                    *voxel = source_voxel_cloud
                        .voxel_at_cartesian_coords(&cartesian_coordinate)
                        .unwrap_or(false);
                }

                // FIXME: @Optimization Due to overly safe
                // mesh_volume_bounding_box_cartesian, the voxel cloud may be
                // unnecessarily big.
                target_voxel_cloud.shrink_to_fit();

                target_voxel_cloud
            })
    }

    /// Creates a new voxel cloud from another voxel cloud transformed (scaled,
    /// rotated, moved) in a cartesian space.
    ///
    /// # Panics
    /// Panics if any of the voxel dimensions is below or equal to zero.
    pub fn from_voxel_cloud_transformed(
        source_voxel_cloud: &VoxelCloud,
        voxel_dimension: f32,
        cartesian_translation: &Vector3<f32>,
        rotation: &Rotation3<f32>,
        scale: &Vector3<f32>,
    ) -> Option<Self> {
        let euler_angles = rotation.euler_angles();
        if approx::relative_eq!(cartesian_translation, &Vector3::zeros())
            && approx::relative_eq!(euler_angles.0, 0.0)
            && approx::relative_eq!(euler_angles.1, 0.0)
            && approx::relative_eq!(euler_angles.2, 0.0)
            && approx::relative_eq!(scale, &Vector3::new(1.0, 1.0, 1.0))
        {
            if approx::relative_eq!(voxel_dimension, source_voxel_cloud.voxel_dimensions.x)
                && approx::relative_eq!(voxel_dimension, source_voxel_cloud.voxel_dimensions.y)
                && approx::relative_eq!(voxel_dimension, source_voxel_cloud.voxel_dimensions.z)
            {
                return Some(VoxelCloud {
                    block_start: source_voxel_cloud.block_start,
                    block_dimensions: source_voxel_cloud.block_dimensions,
                    voxel_dimensions: source_voxel_cloud.voxel_dimensions,
                    voxel_map: source_voxel_cloud.voxel_map.to_vec(),
                });
            } else {
                return VoxelCloud::from_voxel_cloud(
                    source_voxel_cloud,
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
        if let Some(source_vc_bounding_box) =
            source_voxel_cloud.mesh_volume_bounding_box_cartesian()
        {
            // FIXME: Heterogenous voxels require non-trivial compensation
            // of final voxel cloud position after rotating if the volume
            // equilibrium is not at the world origin.
            let voxel_dimensions = Vector3::new(voxel_dimension, voxel_dimension, voxel_dimension);

            let vector_to_origin = Vector3::zeros() - source_vc_bounding_box.center().coords;

            // Transform the source mesh volume and calculate a new bounding
            // box that will encompass the transformed source voxel cloud.
            let transformation_to_origin = Matrix4::new_translation(&vector_to_origin);
            let compound_transformation =
                Matrix4::from(*rotation) * Matrix4::new_nonuniform_scaling(scale);

            let source_vc_bounding_box_corners = source_vc_bounding_box.corners();
            let transformed_bounding_box_corners = source_vc_bounding_box_corners.iter().map(|v| {
                let v1 = transformation_to_origin.transform_point(v);
                compound_transformation.transform_point(&v1)
            });

            let transformed_bounding_box =
                BoundingBox::from_points(transformed_bounding_box_corners)
                    .expect("No input bounding box");

            // New voxel cloud that can encompass the transformed source
            // voxel cloud's mesh.
            let mut target_voxel_cloud = VoxelCloud::from_cartesian_bounding_box(
                &transformed_bounding_box,
                &voxel_dimensions,
            );

            // Transform the target voxels inverse to the user
            // transformation so that it is possible to sample the source
            // voxel cloud.
            match compound_transformation.pseudo_inverse(f32::EPSILON) {
                Ok(reversed_user_transformation) => {
                    for (one_dimensional, voxel) in
                        target_voxel_cloud.voxel_map.iter_mut().enumerate()
                    {
                        let cartesian_coordinate = one_dimensional_to_cartesian_coordinate(
                            one_dimensional,
                            &target_voxel_cloud.block_start,
                            &target_voxel_cloud.block_dimensions,
                            &target_voxel_cloud.voxel_dimensions,
                        )
                        .expect("Index out of bounds");

                        // Transform each new voxel inverse to the user
                        // transformation.
                        let transformed_voxel_center_cartesian = reversed_user_transformation
                            .transform_point(&cartesian_coordinate)
                            - vector_to_origin;

                        // Set the new voxel state according to a sampled value from
                        // the source voxel cloud.
                        *voxel = source_voxel_cloud
                            .voxel_at_cartesian_coords(&transformed_voxel_center_cartesian)
                            .unwrap_or(false);
                    }

                    let cartesian_final_translation_vector =
                        cartesian_translation - vector_to_origin;

                    let voxel_final_translation_vector = cartesian_to_absolute_voxel_coordinate(
                        &Point3::from(cartesian_final_translation_vector),
                        &voxel_dimensions,
                    )
                    .coords;

                    target_voxel_cloud.block_start += voxel_final_translation_vector;

                    // FIXME: @Optimization Due to overly safe
                    // mesh_volume_bounding_box_cartesian, the voxel cloud may
                    // be unnecessarily big.
                    target_voxel_cloud.shrink_to_fit();

                    Some(target_voxel_cloud)
                }
                _ => None,
            }
        } else {
            None
        }
    }

    /// Clears the voxel cloud, sets its block dimensions to zero.
    pub fn wipe(&mut self) {
        self.block_start = Point3::origin();
        self.block_dimensions = Vector3::zeros();
        self.voxel_map.resize(0, false);
    }

    /// Returns voxel cloud block end in absolute voxel coordinates.
    fn block_end(&self) -> Point3<i32> {
        Point3::new(
            self.block_start.x + cast_i32(self.block_dimensions.x) - 1,
            self.block_start.y + cast_i32(self.block_dimensions.y) - 1,
            self.block_start.z + cast_i32(self.block_dimensions.z) - 1,
        )
    }

    /// Returns single voxel dimensions in model space units.
    #[allow(dead_code)]
    pub fn voxel_dimensions(&self) -> Vector3<f32> {
        self.voxel_dimensions
    }

    /// Checks if the voxel cloud contains any voxel / volume
    #[allow(dead_code)]
    pub fn contains_voxels(&self) -> bool {
        self.voxel_map.iter().any(|v| *v)
    }

    /// For each existing voxel turn on all neighbor voxels to grow (offset) the
    /// volumes stored in the voxel cloud.

    // FIXME: This is not the most efficient way of doing this, but this
    // function will become obsolete with Distance field.
    pub fn grow_volume(&mut self) {
        let neighbor_offsets = [
            Vector3::new(0, 0, 0),  //self
            Vector3::new(-1, 0, 0), //neighbors
            Vector3::new(1, 0, 0),
            Vector3::new(0, -1, 0),
            Vector3::new(0, 1, 0),
            Vector3::new(0, 0, -1),
            Vector3::new(0, 0, 1),
        ];

        // If the voxels in the existing voxel cloud reach the boundaries of the
        // block, it's needed to grow the block by 1 in each direction.
        let grown_block_start = self.block_start - Vector3::new(1, 1, 1);
        let grown_block_dimensions = self.block_dimensions + Vector3::new(2, 2, 2);

        let original_voxel_map = self.voxel_map.clone();
        let original_block_start = self.block_start;
        let original_block_dimensions = self.block_dimensions;

        self.block_start = grown_block_start;
        self.block_dimensions = grown_block_dimensions;

        let grown_voxel_map_len = cast_usize(
            grown_block_dimensions.x * grown_block_dimensions.y * grown_block_dimensions.z,
        );

        // The original voxel map is shorter, therefore wipe it before resizing.
        self.fill_with(false);

        self.voxel_map.resize(grown_voxel_map_len, false);

        for grown_index in 0..self.voxel_map.len() {
            let absolute_coords = one_dimensional_to_absolute_voxel_coordinate(
                grown_index,
                &grown_block_start,
                &grown_block_dimensions,
            )
            .expect("Index out of bounds");

            if let Some(original_index) = absolute_voxel_to_one_dimensional_coordinate(
                &absolute_coords,
                &original_block_start,
                &original_block_dimensions,
            ) {
                if original_voxel_map[original_index] {
                    // set self an also its neighbors to be on
                    for neighbor_offset in &neighbor_offsets {
                        self.set_voxel_at_absolute_coords(
                            &(absolute_coords + neighbor_offset),
                            true,
                        );
                    }
                }
            }
        }
    }

    /// Computes boolean intersection (logical AND operation) of the current and
    /// another Voxel cloud. The current Voxel cloud will be mutated and resized
    /// to the size and position of an intersection of the two Voxel clouds'
    /// volumes.
    pub fn boolean_intersection(&mut self, other: &VoxelCloud) {
        // Find volume common to both voxel clouds.
        if let Some(self_volume_bounding_box) = self.volume_bounding_box() {
            if let Some(other_volume_bounding_box) = other.volume_bounding_box() {
                if let Some(bounding_box) = BoundingBox::intersection(
                    [self_volume_bounding_box, other_volume_bounding_box]
                        .iter()
                        .copied(),
                ) {
                    // Resize (keep or shrink) the existing voxel cloud so that that can
                    // possibly contain intersection voxels.
                    self.resize_to_voxel_space_bounding_box(&bounding_box);

                    let block_start = bounding_box.minimum_point();
                    let diagonal = bounding_box.diagonal();
                    let block_dimensions = Vector3::new(
                        cast_u32(diagonal.x),
                        cast_u32(diagonal.y),
                        cast_u32(diagonal.z),
                    );
                    // Iterate through the block of space common to both voxel clouds.
                    for i in 0..self.voxel_map.len() {
                        let cartesian_coords = one_dimensional_to_cartesian_coordinate(
                            i,
                            &block_start,
                            &block_dimensions,
                            &self.voxel_dimensions,
                        )
                        .expect("The current voxel map out of bounds");

                        // Perform boolean AND on voxel states of both voxel clouds.
                        self.voxel_map[i] &= other
                            .voxel_at_cartesian_coords(&cartesian_coords)
                            .unwrap_or(false);
                    }
                    self.shrink_to_fit();
                    // Return here because any other option needs to wipe the
                    // current voxel cloud.
                    return;
                }
            }
        }

        // If the two voxel clouds don't intersect or one of them is empty, then
        // wipe the resulting voxel cloud.
        self.wipe();
    }

    /// Computes boolean union (logical OR operation) of two Voxel clouds. The
    /// current Voxel cloud will be mutated and resized to contain both input
    /// Voxel clouds' volumes.
    ///
    /// # Warning
    /// If the input Voxel clouds are far apart, the resulting voxel cloud may
    /// be huge.
    pub fn boolean_union(&mut self, other: &VoxelCloud) {
        let bounding_boxes = [self.volume_bounding_box(), other.volume_bounding_box()];

        let valid_bounding_boxes_iter = bounding_boxes.iter().filter_map(|b| *b);
        if let Some(bounding_box) = BoundingBox::union(valid_bounding_boxes_iter) {
            // Resize (keep or grow) the existing voxel cloud to a block that can
            // possibly contain union voxels.
            self.resize_to_voxel_space_bounding_box(&bounding_box);

            let block_start = bounding_box.minimum_point();
            let diagonal = bounding_box.diagonal();
            let block_dimensions = Vector3::new(
                cast_u32(diagonal.x),
                cast_u32(diagonal.y),
                cast_u32(diagonal.z),
            );
            // Iterate through the block of space containing both voxel clouds.
            for i in 0..self.voxel_map.len() {
                let cartesian_coords = one_dimensional_to_cartesian_coordinate(
                    i,
                    &block_start,
                    &block_dimensions,
                    &self.voxel_dimensions,
                )
                .expect("The current voxel map out of bounds");
                // If the other voxel cloud exists on the current absolute
                // coordinate, perform boolean OR, otherwise don't change the
                // existing voxel.
                if let Some(true) = other.voxel_at_cartesian_coords(&cartesian_coords) {
                    self.voxel_map[i] = true;
                }
            }
            self.shrink_to_fit();
        } else {
            self.wipe();
        }
    }

    /// Computes boolean difference of the current Voxel cloud minus the other
    /// Voxel cloud. The current Voxel cloud will be modified so that voxels,
    /// that are on in both Voxel clouds will be turned off, while the rest
    /// remains intact.
    pub fn boolean_difference(&mut self, other: &VoxelCloud) {
        // Iterate through the target voxel cloud
        for i in 0..self.voxel_map.len() {
            let cartesian_coords = one_dimensional_to_cartesian_coordinate(
                i,
                &self.block_start,
                &self.block_dimensions,
                &self.voxel_dimensions,
            )
            .expect("The current voxel map out of bounds");

            // If the other voxel clouds contains a voxel at the position,
            // remove the existing voxel from the target voxel cloud
            if let Some(true) = other.voxel_at_cartesian_coords(&cartesian_coords) {
                self.voxel_map[i] = false;
            }
        }
        self.shrink_to_fit()
    }

    /// Gets the state of a voxel defined in voxel coordinates relative to the
    /// voxel block start.
    pub fn voxel_at_relative_coords(&self, relative_coords: &Point3<i32>) -> Option<bool> {
        relative_voxel_to_one_dimensional_coordinate(relative_coords, &self.block_dimensions)
            .map(|index| self.voxel_map[index])
    }

    /// Gets the state of a voxel defined in absolute voxel coordinates
    /// (relative to the voxel space origin).
    pub fn voxel_at_absolute_coords(&self, absolute_coords: &Point3<i32>) -> Option<bool> {
        absolute_voxel_to_one_dimensional_coordinate(
            absolute_coords,
            &self.block_start,
            &self.block_dimensions,
        )
        .map(|index| self.voxel_map[index])
    }

    /// Gets the state of a voxel containing the input point defined in model
    /// space coordinates.
    pub fn voxel_at_cartesian_coords(&self, point: &Point3<f32>) -> Option<bool> {
        let voxel_coords = cartesian_to_absolute_voxel_coordinate(point, &self.voxel_dimensions);
        self.voxel_at_absolute_coords(&voxel_coords)
    }

    /// Sets the state of a voxel defined in voxel coordinates relative to the
    /// voxel block start.
    #[allow(dead_code)]
    pub fn set_voxel_at_relative_coords(&mut self, relative_coords: &Point3<i32>, state: bool) {
        let index =
            relative_voxel_to_one_dimensional_coordinate(relative_coords, &self.block_dimensions)
                .expect("Coordinates out of bounds");
        self.voxel_map[cast_usize(index)] = state;
    }

    /// Sets the state of a voxel defined in absolute voxel coordinates
    /// (relative to the voxel space origin).
    pub fn set_voxel_at_absolute_coords(&mut self, absolute_coords: &Point3<i32>, state: bool) {
        let index = absolute_voxel_to_one_dimensional_coordinate(
            absolute_coords,
            &self.block_start,
            &self.block_dimensions,
        )
        .expect("Coordinates out of bounds");
        self.voxel_map[cast_usize(index)] = state;
    }

    /// Sets the state of a voxel containing the input point defined in model
    /// space coordinates.
    pub fn set_voxel_at_cartesian_coords(&mut self, point: &Point3<f32>, state: bool) {
        let voxel_coords = cartesian_to_absolute_voxel_coordinate(point, &self.voxel_dimensions);
        self.set_voxel_at_absolute_coords(&voxel_coords, state);
    }

    /// Fills the current Voxel cloud with the given value.
    pub fn fill_with(&mut self, value: bool) {
        for v in self.voxel_map.iter_mut() {
            *v = value;
        }
    }

    /// Resize the voxel cloud block to match new block start and block dimensions.
    ///
    /// This clips the outstanding parts of the original voxel cloud.
    pub fn resize(
        &mut self,
        resized_block_start: &Point3<i32>,
        resized_block_dimensions: &Vector3<u32>,
    ) {
        if resized_block_start != &self.block_start
            || resized_block_dimensions != &self.block_dimensions
        {
            // Short cut if resizing to an empty voxel cloud.
            if resized_block_dimensions == &Vector3::zeros() {
                self.wipe();
                return;
            }

            let original_voxel_map = self.voxel_map.clone();
            let original_block_start = self.block_start;
            let original_block_dimensions = self.block_dimensions;

            self.block_start = *resized_block_start;
            self.block_dimensions = *resized_block_dimensions;

            let resized_voxel_map_len = cast_usize(
                resized_block_dimensions.x
                    * resized_block_dimensions.y
                    * resized_block_dimensions.z,
            );

            self.voxel_map.resize(resized_voxel_map_len, false);

            for resized_index in 0..self.voxel_map.len() {
                let absolute_coords = one_dimensional_to_absolute_voxel_coordinate(
                    resized_index,
                    resized_block_start,
                    resized_block_dimensions,
                )
                .expect("Index out of bounds");

                self.voxel_map[resized_index] = match absolute_voxel_to_one_dimensional_coordinate(
                    &absolute_coords,
                    &original_block_start,
                    &original_block_dimensions,
                ) {
                    Some(original_index) => original_voxel_map[original_index],
                    _ => false,
                }
            }
        }
    }

    /// Resize the current voxel cloud to match the input bounding box in
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

    /// Resize the voxel cloud block to exactly fit the volumetric geometry.
    /// Returns None for empty the voxel cloud.
    pub fn shrink_to_fit(&mut self) {
        if let Some((shrunk_block_start, shrunk_block_dimensions)) =
            self.compute_volume_boundaries()
        {
            self.resize(&shrunk_block_start, &shrunk_block_dimensions);
        } else {
            self.wipe();
        }
    }

    /// Computes a simple triangulated welded mesh from the current state of
    /// the voxel cloud.
    ///
    /// For watertight meshes this creates both, outer and inner boundary mesh.
    /// There is also a high risk of generating a non-manifold mesh if some
    /// voxels touch only diagonally.
    pub fn to_mesh(&self) -> Option<Mesh> {
        if self.block_dimensions.x == 0
            || self.block_dimensions.y == 0
            || self.block_dimensions.z == 0
        {
            return None;
        }

        // A collection of rectangular meshes (two triangles) defining an outer
        // envelope of volumes stored in the voxel cloud
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

        // Iterate through the voxel cloud
        for (one_dimensional_coord, voxel) in self.voxel_map.iter().enumerate() {
            // If the current voxel is on
            if *voxel {
                let voxel_coords = one_dimensional_to_relative_voxel_coordinate(
                    one_dimensional_coord,
                    &self.block_dimensions,
                )
                .expect("Out of bounds");
                // compute the position of its center in model space coordinates
                let voxel_center = relative_voxel_to_cartesian_coordinate(
                    &voxel_coords,
                    &self.block_start,
                    &self.voxel_dimensions,
                );
                // and check if there is any voxel around it
                for helper in &neighbor_helpers {
                    match self
                        .voxel_at_relative_coords(&(voxel_coords + helper.direction_to_neighbor))
                    {
                        // if there isn't or if the current voxel is on the
                        // boundary of the voxel space block
                        Some(false) | None => {
                            // add a horizontal rectangle
                            plane_meshes.push(primitive::create_mesh_plane(
                                Plane::from_origin_and_plane(
                                    // above the voxel center half way the height of the voxel
                                    &(voxel_center + helper.direction_to_wall),
                                    // align it properly
                                    &helper.plane,
                                ),
                                // and set its size to match the
                                // dimensions of the top side of a voxel
                                helper.voxel_dimensions,
                            ));
                        }
                        // if there is a neighbor above the current voxel,
                        // it means no boundary side of the voxel box should
                        // be materialized
                        _ => {}
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
        // and weld naked edges
        tools::weld(&joined_voxel_mesh, (min_voxel_dimension as f32) / 4.0)
    }

    /// Returns the bounding box of this voxel cloud in world space cartesian
    /// units.
    #[allow(dead_code)]
    pub fn bounding_box_cartesian(&self) -> BoundingBox<f32> {
        let voxel_dimensions = self.voxel_dimensions;
        let block_start = self.block_start;
        let block_end = self.block_end();
        BoundingBox::new(
            &Point3::new(
                block_start.x as f32 * voxel_dimensions.x,
                block_start.y as f32 * voxel_dimensions.y,
                block_start.z as f32 * voxel_dimensions.z,
            ),
            &Point3::new(
                block_end.x as f32 * voxel_dimensions.x,
                block_end.y as f32 * voxel_dimensions.y,
                block_end.z as f32 * voxel_dimensions.z,
            ),
        )
    }

    /// Returns the bounding box of this voxel cloud in voxel units.
    #[allow(dead_code)]
    pub fn bounding_box(&self) -> BoundingBox<i32> {
        BoundingBox::new(&self.block_start, &self.block_end())
    }

    /// Returns the bounding box of the mesh produced by `VoxelCloud::to_mesh`
    /// for this voxel cloud in world space cartesian units.
    pub fn mesh_volume_bounding_box_cartesian(&self) -> Option<BoundingBox<f32>> {
        let voxel_dimensions = self.voxel_dimensions;
        self.compute_volume_boundaries()
            .map(|(volume_start, volume_dimensions)| {
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
            })
    }

    /// Returns the bounding box in world space cartesian units of the current
    /// voxel cloud after shrinking to fit just the nonempty voxels.
    #[allow(dead_code)]
    pub fn volume_bounding_box_cartesian(&self) -> Option<BoundingBox<f32>> {
        let voxel_dimensions = self.voxel_dimensions;
        self.compute_volume_boundaries()
            .map(|(volume_start, volume_dimensions)| {
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
            })
    }

    /// Returns the bounding box in voxel units of the current voxel cloud after
    /// shrinking to fit just the nonempty voxels.
    pub fn volume_bounding_box(&self) -> Option<BoundingBox<i32>> {
        self.compute_volume_boundaries()
            .map(|(volume_start, volume_dimensions)| {
                let volume_end = volume_start
                    + Vector3::new(
                        cast_i32(volume_dimensions.x),
                        cast_i32(volume_dimensions.y),
                        cast_i32(volume_dimensions.z),
                    );
                BoundingBox::new(&volume_start, &volume_end)
            })
    }

    /// Fill hollow volumes in voxel cloud. The original voxel cloud will be
    /// mutated. The method flood-fills the outer space with void voxels,
    /// leaving everything inside volume voxels filled.
    ///
    /// The method scans the entire boundaries of the voxel cloud and starts
    /// flood-filling with void voxels wherever there is a void voxel. The flood
    /// fill stops at volume voxels.
    pub fn fill_volumes(&mut self) {
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
        let mut queue_to_process: VecDeque<usize> = VecDeque::new();
        // Matches the voxel map length
        let mut discovered = vec![false; self.voxel_map.len()];

        // Scan for void voxels at the boundaries of the voxel cloud. For
        // optimization and readability reasons this scans the entire voxel
        // cloud and filters out coordinates inside the voxel cloud block.
        for (one_dimensional, voxel) in self.voxel_map.iter().enumerate() {
            let coord = one_dimensional_to_relative_voxel_coordinate(
                one_dimensional,
                &self.block_dimensions,
            )
            .expect("Coord out of bounds");
            // If any of these is true, the coordinate is at the boundary of the
            // voxel cloud block
            if coord.x == 0
                || coord.y == 0
                || coord.z == 0
                || coord.x == cast_i32(self.block_dimensions.x) - 1
                || coord.y == cast_i32(self.block_dimensions.y) - 1
                || coord.z == cast_i32(self.block_dimensions.z) - 1
            {
                // If the voxel is void and hasn't been discovered yet
                if !voxel && !discovered[one_dimensional] {
                    // put it into the processing queue
                    queue_to_process.push_back(one_dimensional);
                    // and mark it discovered.
                    discovered[one_dimensional] = true;
                }
            }
        }

        // Process the queue
        while let Some(one_dimensional) = queue_to_process.pop_front() {
            // Calculate the relative coord of the currently processed voxel.
            // Will be needed to calculate its neighbors.
            let coord = one_dimensional_to_relative_voxel_coordinate(
                one_dimensional,
                &self.block_dimensions,
            )
            .expect("Coord out of bounds");
            // Check all the neighbors
            for neighbor_offset in &neighbor_offsets {
                let neighbor_coord = coord + neighbor_offset;
                // If the neighbor exists (is not out of bounds)
                if let Some(false) = self.voxel_at_relative_coords(&neighbor_coord) {
                    let neighbor_one_dimensional = relative_voxel_to_one_dimensional_coordinate(
                        &neighbor_coord,
                        &self.block_dimensions,
                    )
                    .expect("Coord out of bounds");
                    // Check if it is void and hasn't been discovered yet
                    if !discovered[neighbor_one_dimensional] {
                        // Put it to the processing queue
                        queue_to_process.push_back(neighbor_one_dimensional);
                        // and mark it discovered.
                        discovered[neighbor_one_dimensional] = true;
                    }
                }
            }
        }

        // All the discovered voxels were empty and connected with the voxel
        // cloud boundaries. Therefore they are part of the outer void space and
        // everything else is a filled volume.
        for (i, v) in self.voxel_map.iter_mut().enumerate() {
            *v = !discovered[i];
        }
    }

    /// Computes boundaries of volumes contained in voxel cloud. Returns tuple
    /// (block_start, block_dimensions). For empty voxel clouds returns the
    /// original block start and zero block dimensions.
    fn compute_volume_boundaries(&self) -> Option<(Point3<i32>, Vector3<u32>)> {
        let mut min: Vector3<i32> =
            Vector3::new(i32::max_value(), i32::max_value(), i32::max_value());
        let mut max: Vector3<i32> =
            Vector3::new(i32::min_value(), i32::min_value(), i32::min_value());
        for (i, v) in self.voxel_map.iter().enumerate() {
            if *v {
                let relative_coords =
                    one_dimensional_to_relative_voxel_coordinate(i, &self.block_dimensions)
                        .expect("Out of bounds");
                if relative_coords.x < min.x {
                    min.x = relative_coords.x;
                }
                if relative_coords.x > max.x {
                    max.x = relative_coords.x;
                }
                if relative_coords.y < min.y {
                    min.y = relative_coords.y;
                }
                if relative_coords.y > max.y {
                    max.y = relative_coords.y;
                }
                if relative_coords.z < min.z {
                    min.z = relative_coords.z;
                }
                if relative_coords.z > max.z {
                    max.z = relative_coords.z;
                }
            }
        }
        // It's enough to check one of the values because if anything is found,
        // all the values would change.
        if min.x == i32::max_value() {
            assert_eq!(
                min.y,
                i32::max_value(),
                "Voxel cloud emptiness check failed"
            );
            assert_eq!(
                min.z,
                i32::max_value(),
                "Voxel cloud emptiness check failed"
            );
            assert_eq!(
                max.x,
                i32::min_value(),
                "Voxel cloud emptiness check failed"
            );
            assert_eq!(
                max.y,
                i32::min_value(),
                "Voxel cloud emptiness check failed"
            );
            assert_eq!(
                max.z,
                i32::min_value(),
                "Voxel cloud emptiness check failed"
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
    let voxel_map_len = cast_usize(block_dimensions.x * block_dimensions.y * block_dimensions.z);
    if one_dimensional_coordinate < voxel_map_len {
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
        .all(|(i, coord)| *coord >= 0 && *coord < cast_i32(block_dimensions[i]))
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

/// Computes an index to the linear representation of the voxel block from
/// absolute voxel coordinates (relative to the voxel space origin).
///
/// Returns None if out of bounds.
fn absolute_voxel_to_one_dimensional_coordinate(
    absolute_coords: &Point3<i32>,
    block_start: &Point3<i32>,
    block_dimensions: &Vector3<u32>,
) -> Option<usize> {
    let relative_coords = absolute_coords - block_start.coords;
    relative_voxel_to_one_dimensional_coordinate(&relative_coords, block_dimensions)
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
    fn test_voxel_cloud_from_mesh_for_torus() {
        let (faces, vertices) = torus();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );
        let voxel_cloud = VoxelCloud::from_mesh(&mesh, &Vector3::new(1.0, 1.0, 1.0));

        insta::assert_json_snapshot!("torus_after_voxelization", &voxel_cloud);
    }

    #[test]
    fn test_voxel_cloud_from_mesh_for_sphere() {
        let mesh = primitive::create_uv_sphere(
            Point3::origin(),
            Rotation3::identity(),
            Vector3::new(1.0, 1.0, 1.0),
            10,
            10,
            NormalStrategy::Sharp,
        );

        let voxel_cloud = VoxelCloud::from_mesh(&mesh, &Vector3::new(0.5, 0.5, 0.5));

        insta::assert_json_snapshot!("sphere_after_voxelization", &voxel_cloud);
    }

    #[test]
    fn test_voxel_cloud_three_dimensional_to_one_dimensional_and_back_relative() {
        let voxel_cloud = VoxelCloud::new(
            &Point3::origin(),
            &Vector3::new(3, 4, 5),
            &Vector3::new(1.5, 2.5, 3.5),
        );
        for z in 0..voxel_cloud.block_dimensions.z {
            for y in 0..voxel_cloud.block_dimensions.y {
                for x in 0..voxel_cloud.block_dimensions.x {
                    let relative_position = Point3::new(cast_i32(x), cast_i32(y), cast_i32(z));
                    let one_dimensional = relative_voxel_to_one_dimensional_coordinate(
                        &relative_position,
                        &voxel_cloud.block_dimensions,
                    )
                    .unwrap();
                    let three_dimensional = one_dimensional_to_relative_voxel_coordinate(
                        one_dimensional,
                        &voxel_cloud.block_dimensions,
                    )
                    .unwrap();
                    assert_eq!(relative_position, three_dimensional);
                }
            }
        }
    }

    #[test]
    fn test_voxel_cloud_boolean_intersection_all_true() {
        let mut vc_a = VoxelCloud::new(
            &Point3::origin(),
            &Vector3::new(3, 3, 3),
            &Vector3::new(0.5, 0.5, 0.5),
        );
        let mut vc_b = VoxelCloud::new(
            &Point3::new(2, 1, 1),
            &Vector3::new(3, 3, 3),
            &Vector3::new(0.5, 0.5, 0.5),
        );
        let mut vc_correct = VoxelCloud::new(
            &Point3::new(2, 1, 1),
            &Vector3::new(1, 2, 2),
            &Vector3::new(0.5, 0.5, 0.5),
        );

        vc_a.fill_with(true);
        vc_b.fill_with(true);
        vc_correct.fill_with(true);

        vc_a.boolean_intersection(&vc_b);

        assert_eq!(vc_a, vc_correct);
    }

    #[test]
    fn test_voxel_cloud_boolean_intersection_one_false() {
        let mut vc_a = VoxelCloud::new(
            &Point3::origin(),
            &Vector3::new(3, 3, 3),
            &Vector3::new(0.5, 0.5, 0.5),
        );
        let mut vc_b = VoxelCloud::new(
            &Point3::new(1, 1, 1),
            &Vector3::new(3, 3, 3),
            &Vector3::new(0.5, 0.5, 0.5),
        );
        let mut vc_correct = VoxelCloud::new(
            &Point3::new(1, 1, 1),
            &Vector3::new(2, 2, 2),
            &Vector3::new(0.5, 0.5, 0.5),
        );

        vc_a.fill_with(true);
        vc_b.fill_with(true);
        vc_b.set_voxel_at_relative_coords(&Point3::new(1, 1, 1), false);
        vc_correct.fill_with(true);
        vc_correct.set_voxel_at_relative_coords(&Point3::new(1, 1, 1), false);

        vc_a.boolean_intersection(&vc_b);

        assert_eq!(vc_a, vc_correct);
    }

    #[test]
    fn test_voxel_cloud_boolean_union_one_false() {
        let mut vc_a = VoxelCloud::new(
            &Point3::origin(),
            &Vector3::new(3, 3, 3),
            &Vector3::new(0.5, 0.5, 0.5),
        );
        let mut vc_b = VoxelCloud::new(
            &Point3::new(1, 1, 1),
            &Vector3::new(3, 3, 3),
            &Vector3::new(0.5, 0.5, 0.5),
        );
        let mut vc_correct = VoxelCloud::new(
            &Point3::new(0, 0, 0),
            &Vector3::new(4, 4, 4),
            &Vector3::new(0.5, 0.5, 0.5),
        );

        vc_a.fill_with(true);
        vc_b.fill_with(true);
        vc_b.set_voxel_at_relative_coords(&Point3::new(1, 1, 1), false);
        vc_correct.fill_with(true);
        vc_correct.set_voxel_at_relative_coords(&Point3::new(3, 0, 0), false);
        vc_correct.set_voxel_at_relative_coords(&Point3::new(3, 1, 0), false);
        vc_correct.set_voxel_at_relative_coords(&Point3::new(3, 2, 0), false);
        vc_correct.set_voxel_at_relative_coords(&Point3::new(3, 3, 0), false);
        vc_correct.set_voxel_at_relative_coords(&Point3::new(3, 0, 1), false);
        vc_correct.set_voxel_at_relative_coords(&Point3::new(3, 0, 2), false);
        vc_correct.set_voxel_at_relative_coords(&Point3::new(3, 0, 3), false);
        vc_correct.set_voxel_at_relative_coords(&Point3::new(2, 0, 3), false);
        vc_correct.set_voxel_at_relative_coords(&Point3::new(1, 0, 3), false);
        vc_correct.set_voxel_at_relative_coords(&Point3::new(0, 0, 3), false);
        vc_correct.set_voxel_at_relative_coords(&Point3::new(0, 1, 3), false);
        vc_correct.set_voxel_at_relative_coords(&Point3::new(0, 2, 3), false);
        vc_correct.set_voxel_at_relative_coords(&Point3::new(0, 3, 3), false);
        vc_correct.set_voxel_at_relative_coords(&Point3::new(0, 3, 2), false);
        vc_correct.set_voxel_at_relative_coords(&Point3::new(0, 3, 1), false);
        vc_correct.set_voxel_at_relative_coords(&Point3::new(0, 3, 0), false);
        vc_correct.set_voxel_at_relative_coords(&Point3::new(1, 3, 0), false);
        vc_correct.set_voxel_at_relative_coords(&Point3::new(2, 3, 0), false);

        vc_a.boolean_union(&vc_b);

        assert_eq!(vc_a, vc_correct);
    }

    #[test]
    fn test_voxel_cloud_get_set_for_single_voxel() {
        let mut voxel_cloud = VoxelCloud::new(
            &Point3::origin(),
            &Vector3::new(1, 1, 1),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        let before = voxel_cloud
            .voxel_at_relative_coords(&Point3::new(0, 0, 0))
            .unwrap();
        voxel_cloud.set_voxel_at_relative_coords(&Point3::new(0, 0, 0), true);
        let after = voxel_cloud
            .voxel_at_relative_coords(&Point3::new(0, 0, 0))
            .unwrap();
        assert!(!before);
        assert!(after);
    }

    #[test]
    fn test_voxel_cloud_single_voxel_to_mesh_produces_synchronized_mesh() {
        let mut voxel_cloud = VoxelCloud::new(
            &Point3::origin(),
            &Vector3::new(1, 1, 1),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        voxel_cloud.set_voxel_at_relative_coords(&Point3::new(0, 0, 0), true);

        let voxel_mesh = voxel_cloud.to_mesh().unwrap();

        let v2f = topology::compute_vertex_to_face_topology(&voxel_mesh);
        let f2f = topology::compute_face_to_face_topology(&voxel_mesh, &v2f);
        let voxel_mesh_synced = tools::synchronize_mesh_winding(&voxel_mesh, &f2f);

        assert!(analysis::are_similar(&voxel_mesh, &voxel_mesh_synced));
    }

    #[test]
    fn test_voxel_cloud_resize_zero_to_nonzero_all_false() {
        let mut voxel_cloud = VoxelCloud::new(
            &Point3::origin(),
            &Vector3::zeros(),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        voxel_cloud.resize(&Point3::origin(), &Vector3::new(1, 1, 1));

        let voxel = voxel_cloud
            .voxel_at_relative_coords(&Point3::new(0, 0, 0))
            .unwrap();

        assert!(!voxel);
    }

    #[test]
    fn test_voxel_cloud_resize_zero_to_nonzero_correct_start_and_dimensions() {
        let mut voxel_cloud = VoxelCloud::new(
            &Point3::origin(),
            &Vector3::zeros(),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        let new_origin = Point3::new(1, 2, 3);
        let new_block_dimensions = Vector3::new(4, 5, 6);
        voxel_cloud.resize(&new_origin, &new_block_dimensions);

        assert_eq!(voxel_cloud.block_start, new_origin);
        assert_eq!(voxel_cloud.block_dimensions, new_block_dimensions);
        assert_eq!(voxel_cloud.voxel_map.len(), 4 * 5 * 6);
    }

    #[test]
    fn test_voxel_cloud_resize_nonzero_to_zero_correct_start_and_dimensions() {
        let mut voxel_cloud = VoxelCloud::new(
            &Point3::new(1, 2, 3),
            &Vector3::new(4, 5, 6),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        let new_origin = Point3::origin();
        let new_block_dimensions = Vector3::zeros();
        voxel_cloud.resize(&new_origin, &new_block_dimensions);

        assert_eq!(voxel_cloud.block_start, new_origin);
        assert_eq!(voxel_cloud.block_dimensions, new_block_dimensions);
        assert_eq!(voxel_cloud.voxel_map.len(), 0);
    }

    #[test]
    fn test_voxel_cloud_resize_nonzero_to_smaller_nonzero_correct_start_and_dimensions() {
        let mut voxel_cloud = VoxelCloud::new(
            &Point3::origin(),
            &Vector3::new(4, 5, 6),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        let new_origin = Point3::new(1, 2, 3);
        let new_block_dimensions = Vector3::new(1, 2, 3);
        voxel_cloud.resize(&new_origin, &new_block_dimensions);

        assert_eq!(voxel_cloud.block_start, new_origin);
        assert_eq!(voxel_cloud.block_dimensions, new_block_dimensions);
        assert_eq!(voxel_cloud.voxel_map.len(), 1 * 2 * 3);
    }

    #[test]
    fn test_voxel_cloud_resize_nonzero_to_larger_nonzero_correct_start_and_dimensions() {
        let mut voxel_cloud = VoxelCloud::new(
            &Point3::origin(),
            &Vector3::new(1, 2, 3),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        let new_origin = Point3::new(1, 2, 3);
        let new_block_dimensions = Vector3::new(4, 5, 6);
        voxel_cloud.resize(&new_origin, &new_block_dimensions);

        assert_eq!(voxel_cloud.block_start, new_origin);
        assert_eq!(voxel_cloud.block_dimensions, new_block_dimensions);
        assert_eq!(voxel_cloud.voxel_map.len(), 4 * 5 * 6);
    }

    #[test]
    fn test_voxel_cloud_resize_nonzero_to_larger_nonzero_grown_contains_false_rest_original() {
        let original_origin = Point3::new(0i32, 0i32, 0i32);
        let original_block_dimensions = Vector3::new(1u32, 10u32, 3u32);
        let mut voxel_cloud = VoxelCloud::new(
            &original_origin,
            &original_block_dimensions,
            &Vector3::new(1.0, 1.0, 1.0),
        );
        let original_block_end = voxel_cloud.block_end();

        voxel_cloud.fill_with(true);

        let new_origin = Point3::new(-1, 2, 3);
        let new_block_dimensions = Vector3::new(4, 5, 6);
        voxel_cloud.resize(&new_origin, &new_block_dimensions);

        for (i, v) in voxel_cloud.voxel_map.iter().enumerate() {
            let coords = one_dimensional_to_absolute_voxel_coordinate(
                i,
                &voxel_cloud.block_start,
                &voxel_cloud.block_dimensions,
            )
            .unwrap();

            if coords.x < original_origin.x
                || coords.y < original_origin.y
                || coords.z < original_origin.z
                || coords.x > original_block_end.x
                || coords.y > original_block_end.y
                || coords.z > original_block_end.z
            {
                assert!(!v);
            } else {
                assert!(v);
            }
        }
    }

    #[test]
    fn test_voxel_cloud_shrink_to_volume() {
        let mut voxel_cloud = VoxelCloud::new(
            &Point3::origin(),
            &Vector3::new(4, 5, 6),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        voxel_cloud.set_voxel_at_relative_coords(&Point3::new(1, 1, 1), true);
        voxel_cloud.shrink_to_fit();

        assert_eq!(voxel_cloud.block_start, Point3::new(1, 1, 1));
        assert_eq!(voxel_cloud.block_dimensions, Vector3::new(1, 1, 1));
        assert_eq!(voxel_cloud.voxel_map.len(), 1);
        assert!(voxel_cloud
            .voxel_at_relative_coords(&Point3::new(0, 0, 0))
            .unwrap());
    }

    #[test]
    fn test_voxel_cloud_shrink_to_empty() {
        let mut voxel_cloud = VoxelCloud::new(
            &Point3::origin(),
            &Vector3::new(4, 5, 6),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        voxel_cloud.shrink_to_fit();

        assert_eq!(voxel_cloud.block_start, Point3::origin());
        assert_eq!(voxel_cloud.block_dimensions, Vector3::new(0, 0, 0));
        assert_eq!(voxel_cloud.voxel_map.len(), 0);
    }

    #[test]
    fn test_voxel_cloud_transform_box_at_origin_identity() {
        let mesh = primitive::create_box(
            Point3::origin(),
            Rotation3::from_euler_angles(0.0, 0.0, 0.0),
            Vector3::new(1.0, 2.0, 3.0),
        );
        let voxel_cloud = VoxelCloud::from_mesh(&mesh, &Vector3::new(0.25, 0.25, 0.25));
        let transformed_voxel_cloud = VoxelCloud::from_voxel_cloud_transformed(
            &voxel_cloud,
            0.25,
            &Vector3::zeros(),
            &Rotation3::from_euler_angles(0.0, 0.0, 0.0),
            &Vector3::new(1.0, 1.0, 1.0),
        )
        .unwrap();

        assert_eq!(transformed_voxel_cloud, voxel_cloud);
    }

    #[test]
    fn test_voxel_cloud_transform_box_at_random_location_identity() {
        let mesh = primitive::create_box(
            Point3::new(5.1, 6.2, 7.3),
            Rotation3::from_euler_angles(0.0, 0.0, 0.0),
            Vector3::new(1.0, 2.0, 3.0),
        );
        let voxel_cloud = VoxelCloud::from_mesh(&mesh, &Vector3::new(0.25, 0.25, 0.25));
        let transformed_voxel_cloud = VoxelCloud::from_voxel_cloud_transformed(
            &voxel_cloud,
            0.25,
            &Vector3::zeros(),
            &Rotation3::from_euler_angles(0.0, 0.0, 0.0),
            &Vector3::new(1.0, 1.0, 1.0),
        )
        .unwrap();

        assert_eq!(transformed_voxel_cloud, voxel_cloud);
    }

    #[test]
    fn test_voxel_cloud_transform_box_at_random_location_rotated_identity() {
        let mesh = primitive::create_box(
            Point3::new(5.1, 6.2, 7.3),
            Rotation3::from_euler_angles(1.1, 2.2, 3.3),
            Vector3::new(1.0, 2.0, 3.0),
        );
        let voxel_cloud = VoxelCloud::from_mesh(&mesh, &Vector3::new(0.25, 0.25, 0.25));
        let transformed_voxel_cloud = VoxelCloud::from_voxel_cloud_transformed(
            &voxel_cloud,
            0.25,
            &Vector3::zeros(),
            &Rotation3::from_euler_angles(0.0, 0.0, 0.0),
            &Vector3::new(1.0, 1.0, 1.0),
        )
        .unwrap();

        assert_eq!(transformed_voxel_cloud, voxel_cloud);
    }

    #[test]
    fn test_voxel_cloud_transform_sub_voxel_translation() {
        let mesh = primitive::create_box(
            Point3::new(5.1, 6.2, 7.3),
            Rotation3::from_euler_angles(1.1, 2.2, 3.3),
            Vector3::new(1.0, 2.0, 3.0),
        );
        let voxel_cloud = VoxelCloud::from_mesh(&mesh, &Vector3::new(0.25, 0.25, 0.25));
        let transformed_voxel_cloud = VoxelCloud::from_voxel_cloud_transformed(
            &voxel_cloud,
            0.25,
            &Vector3::new(0.1, 0.1, 0.1),
            &Rotation3::from_euler_angles(0.0, 0.0, 0.0),
            &Vector3::new(1.0, 1.0, 1.0),
        )
        .unwrap();

        insta::assert_json_snapshot!(
            "voxel_cloud, transform_sub_voxel_translation",
            &transformed_voxel_cloud
        );
    }

    #[test]
    fn test_voxel_cloud_transform_voxel_size_translation() {
        let mesh = primitive::create_box(
            Point3::new(5.1, 6.2, 7.3),
            Rotation3::from_euler_angles(1.1, 2.2, 3.3),
            Vector3::new(1.0, 2.0, 3.0),
        );
        let voxel_cloud = VoxelCloud::from_mesh(&mesh, &Vector3::new(0.25, 0.25, 0.25));
        let transformed_voxel_cloud = VoxelCloud::from_voxel_cloud_transformed(
            &voxel_cloud,
            0.25,
            &Vector3::new(0.0, 0.0, 0.25),
            &Rotation3::from_euler_angles(0.0, 0.0, 0.0),
            &Vector3::new(1.0, 1.0, 1.0),
        )
        .unwrap();

        insta::assert_json_snapshot!(
            "voxel_cloud_transform_voxel_size_translation",
            &transformed_voxel_cloud
        );
    }

    #[test]
    fn test_voxel_cloud_transform_arbitrary_translation() {
        let mesh = primitive::create_box(
            Point3::new(5.1, 6.2, 7.3),
            Rotation3::from_euler_angles(1.1, 2.2, 3.3),
            Vector3::new(1.0, 2.0, 3.0),
        );
        let voxel_cloud = VoxelCloud::from_mesh(&mesh, &Vector3::new(0.25, 0.25, 0.25));
        let transformed_voxel_cloud = VoxelCloud::from_voxel_cloud_transformed(
            &voxel_cloud,
            0.25,
            &Vector3::new(0.4, 0.6, 0.7),
            &Rotation3::from_euler_angles(0.0, 0.0, 0.0),
            &Vector3::new(1.0, 1.0, 1.0),
        )
        .unwrap();

        insta::assert_json_snapshot!(
            "voxel_cloud_transform_arbitrary_translation",
            &transformed_voxel_cloud
        );
    }

    #[test]
    fn test_voxel_cloud_arbitrary_transform() {
        let mesh = primitive::create_box(
            Point3::new(5.1, 6.2, 7.3),
            Rotation3::from_euler_angles(1.1, 2.2, 3.3),
            Vector3::new(1.0, 2.0, 3.0),
        );
        let voxel_cloud = VoxelCloud::from_mesh(&mesh, &Vector3::new(0.25, 0.25, 0.25));
        let transformed_voxel_cloud = VoxelCloud::from_voxel_cloud_transformed(
            &voxel_cloud,
            0.25,
            &Vector3::new(0.4, 0.6, 0.7),
            &Rotation3::from_euler_angles(25.0, 37.0, 42.0),
            &Vector3::new(1.5, 1.76, 0.5),
        )
        .unwrap();

        insta::assert_json_snapshot!("voxel_cloud_arbitrary_transform", &transformed_voxel_cloud);
    }
}
