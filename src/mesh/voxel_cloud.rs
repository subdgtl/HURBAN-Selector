use std::collections::VecDeque;
use std::f32;
use std::ops::RangeBounds;

use nalgebra::{Point3, Vector2, Vector3};

use crate::bounding_box::BoundingBox;
use crate::convert::{cast_i32, cast_u32, cast_usize};
use crate::geometry;
use crate::plane::Plane;

use super::{primitive, tools, Face, Mesh};

/// Discrete Scalar field is an abstract representation of points in a block of
/// space. Each point is a center of a voxel - an abstract box of given
/// dimensions in a discrete spatial grid.
///
/// The voxels contain a value, which can be read in various ways: as a scalar
/// charge field, as a distance from a volume or as any arbitrary discrete value
/// grid. The voxels can also contain no value at all (None).
///
/// The scalar field is meant to be materialized into a mesh - voxels within a
/// certain value range will become mesh boxes.
///
/// The block of voxel space stored in the scalar field is delimited by its
/// beginning and its dimensions, both defined in discrete voxel units - counts
/// of voxels in each direction. All voxels in one field have the same
/// dimensions, which can be different in each direction. The voxel space is a
/// discrete grid and can't start half way in a voxel. The voxel space starts at
/// the cartesian space origin with absolute voxel coordinates `[0, 0, 0]`.
///
/// The Scalar field manifests itself as infinite, however an attempt to set a
/// value outside of the block will cause the program to panic. Reading a value
/// from beyond the bounds returns None, which is also a valid value even inside
/// the block.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ScalarField {
    block_start: Point3<i32>,
    block_dimensions: Vector3<u32>,
    voxel_dimensions: Vector3<f32>,
    voxels: Vec<Option<f32>>,
}

impl ScalarField {
    /// Define a new empty block of voxel space, which begins at
    /// `block_start`(in discrete absolute voxel units), has dimensions
    /// `block_dimensions` (in discrete voxel units) and contains voxels sized
    /// `voxel_dimensions` (in cartesian model-space units).
    ///
    /// # Panics
    ///
    /// Panics if any of the voxel dimensions is below or equal to zero.
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
        let voxels: Vec<Option<f32>> = vec![None; cast_usize(map_length)];

        ScalarField {
            block_start: *block_start,
            block_dimensions: *block_dimensions,
            voxel_dimensions: *voxel_dimensions,
            voxels,
        }
    }

    /// Creates a new empty voxel space from a bounding box defined in cartesian
    /// units.
    ///
    /// # Panics
    ///
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

        let min_x_index = (min_point.x / voxel_dimensions.x).floor() as i32;
        let min_y_index = (min_point.y / voxel_dimensions.y).floor() as i32;
        let min_z_index = (min_point.z / voxel_dimensions.z).floor() as i32;

        let max_x_index = (max_point.x / voxel_dimensions.x).ceil() as i32;
        let max_y_index = (max_point.y / voxel_dimensions.y).ceil() as i32;
        let max_z_index = (max_point.z / voxel_dimensions.z).ceil() as i32;

        let block_start = Point3::new(min_x_index, min_y_index, min_z_index);
        let block_dimensions = Vector3::new(
            cast_u32(max_x_index - min_x_index) + 1,
            cast_u32(max_y_index - min_y_index) + 1,
            cast_u32(max_z_index - min_z_index) + 1,
        );

        ScalarField::new(&block_start, &block_dimensions, voxel_dimensions)
    }

    /// Creates a scalar field from an existing mesh.
    ///
    /// The voxels intersecting the mesh (volume voxels) will be set to
    /// `value_on_mesh_surface`, the empty voxels (void voxels) will be set to
    /// None. The `growth_offset` defines how much bigger the scalar field be
    /// when initialized. This is useful if the distance field is about to be
    /// calculated for purposes of voxel growth.
    ///
    /// # Panics
    ///
    /// Panics if any of the voxel dimensions is below or equal to zero.
    pub fn from_mesh(
        mesh: &Mesh,
        voxel_dimensions: &Vector3<f32>,
        value_on_mesh_surface: f32,
        growth_offset: u32,
    ) -> Self {
        assert!(
            voxel_dimensions.x > 0.0 && voxel_dimensions.y > 0.0 && voxel_dimensions.z > 0.0,
            "One or more voxel dimensions are 0.0."
        );

        // Determine the needed block of voxel space.
        let bounding_box_tight = mesh.bounding_box();
        let growth_offset_vector_in_cartesian_units = Vector3::new(
            voxel_dimensions.x * growth_offset as f32,
            voxel_dimensions.y * growth_offset as f32,
            voxel_dimensions.z * growth_offset as f32,
        );
        let bounding_box_offset =
            bounding_box_tight.offset(growth_offset_vector_in_cartesian_units);

        // Target scalar field to be filled with points on the mesh surface.
        let mut scalar_field =
            ScalarField::from_cartesian_bounding_box(&bounding_box_offset, voxel_dimensions);

        // Going to populate the mesh with points as dense as the smallest voxel
        // dimension.
        let smallest_voxel_dimension = voxel_dimensions
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
                    let divisions = (longest_edge_len / smallest_voxel_dimension).ceil() as usize;
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
                                // and set a voxel containing the point to the
                                // volume value `value_on_mesh_surface`
                                let absolute_coordinate = cartesian_to_absolute_voxel_coordinate(
                                    &cartesian,
                                    voxel_dimensions,
                                );
                                scalar_field.set_value_at_absolute_voxel_coordinate(
                                    &absolute_coordinate,
                                    Some(value_on_mesh_surface),
                                );
                            }
                        }
                    }
                }
            }
        }

        scalar_field
    }

    /// Checks if the scalar field contains any voxel with a value from the
    /// given range.
    ///
    /// The `volume_value_range` is an interval defining which values of the
    /// scalar field should be considered to be a volume. The
    /// `ScalarField::from_mesh` generates a scalar field, which marks volume
    /// voxels with value `0`. `compute_distance_field` marks each voxel with a
    /// value representing the voxel's distance from the original volume,
    /// therefore the voxels right at the shell of the volume are marked 0, the
    /// layer around them is marked 1 or -1 (inside closed volumes) etc. Once
    /// the scalar field is populated with meaningful values, it is possible to
    /// treat (perform boolean operations or materialize into mesh) on various
    /// numerical ranges. Such range is specified ad-hoc by parameter
    /// `volume_value_range`.
    pub fn contains_voxels_within_range<U>(&self, volume_value_range: &U) -> bool
    where
        U: RangeBounds<f32>,
    {
        self.voxels
            .iter()
            .any(|voxel| is_voxel_within_range(*voxel, volume_value_range))
    }

    /// Gets the value of a voxel on absolute voxel coordinates (relative to the
    /// voxel space origin).
    ///
    /// Returns None if voxel is empty or out of bounds
    pub fn value_at_absolute_voxel_coordinate(
        &self,
        absolute_coordinate: &Point3<i32>,
    ) -> Option<f32> {
        match absolute_voxel_to_one_dimensional_coordinate(
            absolute_coordinate,
            &self.block_start,
            &self.block_dimensions,
        ) {
            Some(index) => self.voxels[index],
            _ => None,
        }
    }

    /// Returns true if the value of a voxel on absolute voxel coordinates
    /// (relative to the voxel space origin) is within given range.
    ///
    /// The `volume_value_range` is an interval defining which values of the
    /// scalar field should be considered to be a volume. The
    /// `ScalarField::from_mesh` generates a scalar field, which marks volume
    /// voxels with value `0`. `compute_distance_field` marks each voxel with a
    /// value representing the voxel's distance from the original volume,
    /// therefore the voxels right at the shell of the volume are marked 0, the
    /// layer around them is marked 1 or -1 (inside closed volumes) etc. Once
    /// the scalar field is populated with meaningful values, it is possible to
    /// treat (perform boolean operations or materialize into mesh) on various
    /// numerical ranges. Such range is specified ad-hoc by parameter
    /// `volume_value_range`.
    pub fn is_value_at_absolute_voxel_coordinate_within_range<U>(
        &self,
        absolute_coordinate: &Point3<i32>,
        volume_value_range: &U,
    ) -> bool
    where
        U: RangeBounds<f32>,
    {
        is_voxel_within_range(
            self.value_at_absolute_voxel_coordinate(absolute_coordinate),
            volume_value_range,
        )
    }

    /// Sets the value of a voxel defined in absolute voxel coordinates
    /// (relative to the voxel space origin).
    ///
    /// # Panics
    ///
    /// Panics if absolute coordinate out of bounds
    pub fn set_value_at_absolute_voxel_coordinate(
        &mut self,
        absolute_coordinate: &Point3<i32>,
        value: Option<f32>,
    ) {
        let index = absolute_voxel_to_one_dimensional_coordinate(
            absolute_coordinate,
            &self.block_start,
            &self.block_dimensions,
        )
        .expect("Coordinates out of bounds");
        self.voxels[index] = value;
    }

    /// Fills the current scalar field with the given value.
    #[allow(dead_code)]
    pub fn fill_with(&mut self, value: Option<f32>) {
        for v in self.voxels.iter_mut() {
            *v = value;
        }
    }

    /// Computes a simple triangulated welded mesh from the current state of the
    /// scalar field. The mesh will be made of orthogonal voxels.
    ///
    /// For watertight volumetric geometry (i.e. from a watertight source mesh)
    /// this creates both, outer and inner boundary mesh. There is also a high
    /// risk of generating a non-manifold mesh if some voxels touch only
    /// diagonally.
    ///
    /// The `volume_value_range` is an interval defining which values of the
    /// scalar field should be considered to be a volume. The
    /// `ScalarField::from_mesh` generates a scalar field, which marks volume
    /// voxels with value `0`. `compute_distance_field` marks each voxel with a
    /// value representing the voxel's distance from the original volume,
    /// therefore the voxels right at the shell of the volume are marked 0, the
    /// layer around them is marked 1 or -1 (inside closed volumes) etc. Once
    /// the scalar field is populated with meaningful values, it is possible to
    /// treat (perform boolean operations or materialize into mesh) on various
    /// numerical ranges. Such range is specified ad-hoc by parameter
    /// `volume_value_range`.
    pub fn to_mesh<U>(&self, volume_value_range: &U) -> Option<Mesh>
    where
        U: RangeBounds<f32>,
    {
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
        for (one_dimensional, voxel) in self.voxels.iter().enumerate() {
            // If the current voxel is a volume voxel
            if is_voxel_within_range(*voxel, volume_value_range) {
                let absolute_coordinate = one_dimensional_to_absolute_voxel_coordinate(
                    one_dimensional,
                    &self.block_start,
                    &self.block_dimensions,
                );

                // compute the position of its center in model space coordinates
                let voxel_center_cartesian = one_dimensional_to_cartesian_coordinate(
                    one_dimensional,
                    &self.block_start,
                    &self.block_dimensions,
                    &self.voxel_dimensions,
                );
                // and check if there is any voxel around it.
                for helper in &neighbor_helpers {
                    let absolute_neighbor_coordinate =
                        absolute_coordinate + helper.direction_to_neighbor;
                    let neighbor_voxel =
                        self.value_at_absolute_voxel_coordinate(&absolute_neighbor_coordinate);
                    // If the neighbor voxel is not within the volume range,
                    // the boundary side of the voxel box should be
                    // materialized.
                    if !is_voxel_within_range(neighbor_voxel, volume_value_range) {
                        // Add a rectangle
                        plane_meshes.push(primitive::create_mesh_plane(
                            Plane::from_origin_and_plane(
                                // around the voxel center half way the
                                // respective dimension of the voxel,
                                &(voxel_center_cartesian + helper.direction_to_wall),
                                // align it properly
                                &helper.plane,
                            ),
                            // and set its size to match the dimensions
                            // of the respective side of a voxel.
                            helper.voxel_dimensions,
                        ));
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

    /// Compute discrete distance field.
    ///
    /// Each voxel will be set a value equal to its distance from the original
    /// volume. The voxels that were originally volume voxels, will be set to
    /// value 0. Voxels inside the closed volumes will have the distance value
    /// with a negative sign.
    ///
    /// The `volume_value_range` is an interval defining which values of the
    /// scalar field should be considered to be a volume. The
    /// `ScalarField::from_mesh` generates a scalar field, which marks volume
    /// voxels with value `0`. `compute_distance_field` marks each voxel with a
    /// value representing the voxel's distance from the original volume,
    /// therefore the voxels right at the shell of the volume are marked 0, the
    /// layer around them is marked 1 or -1 (inside closed volumes) etc. Once
    /// the scalar field is populated with meaningful values, it is possible to
    /// treat (perform boolean operations or materialize into mesh) on various
    /// numerical ranges. Such range is specified ad-hoc by parameter
    /// `volume_value_range`.
    pub fn compute_distance_field<U>(&mut self, volume_value_range: &U)
    where
        U: RangeBounds<f32>,
    {
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
        let mut queue_to_compute_distance: VecDeque<(usize, f32)> = VecDeque::new();
        // Match the voxel map length
        let mut discovered_as_outer_and_empty = vec![false; self.voxels.len()];
        let mut discovered_for_distance_field = vec![false; self.voxels.len()];

        // Scan for void voxels at the boundaries of the scalar field
        // and at the same time for volume voxels anywhere.
        for (one_dimensional, voxel) in self.voxels.iter().enumerate() {
            let relative_coordinate = one_dimensional_to_relative_voxel_coordinate(
                one_dimensional,
                &self.block_dimensions,
            );

            // If the voxel is void
            if !is_voxel_within_range(*voxel, volume_value_range) {
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
                // Add the current voxel to the queue for distance field
                // processing
                queue_to_compute_distance.push_back((one_dimensional, 0.0));
                // and mark it discovered.
                discovered_for_distance_field[one_dimensional] = true;
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
            );

            // Check all the neighbors
            for neighbor_offset in &neighbor_offsets {
                let neighbor_absolute_coordinate = absolute_coordinate + neighbor_offset;
                // If the neighbor doesn't contain any volume
                if !self.is_value_at_absolute_voxel_coordinate_within_range(
                    &neighbor_absolute_coordinate,
                    volume_value_range,
                ) {
                    // and is not out of bounds
                    if let Some(neighbor_one_dimensional) =
                        absolute_voxel_to_one_dimensional_coordinate(
                            &neighbor_absolute_coordinate,
                            &self.block_start,
                            &self.block_dimensions,
                        )
                    {
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
            );

            // Check each neighbor
            for neighbor_offset in &neighbor_offsets {
                let neighbor_absolute_coordinate = absolute_coordinate + neighbor_offset;
                // If the neighbor does exist
                if let Some(one_dimensional_neighbor) = absolute_voxel_to_one_dimensional_coordinate(
                    &neighbor_absolute_coordinate,
                    &self.block_start,
                    &self.block_dimensions,
                ) {
                    // and hasn't been discovered yet and is void,
                    if !discovered_for_distance_field[one_dimensional_neighbor]
                        && !self.is_value_at_absolute_voxel_coordinate_within_range(
                            &neighbor_absolute_coordinate,
                            volume_value_range,
                        )
                    {
                        // put it into the processing queue with the distance
                        // one higher than the current
                        queue_to_compute_distance
                            .push_back((one_dimensional_neighbor, distance + 1.0));
                        // and mark it discovered.
                        discovered_for_distance_field[one_dimensional_neighbor] = true;
                    }
                }
            }

            // Process the current voxel. If it is outside the volumes, set its
            // value to be positive, if it's inside, set it to negative.
            self.voxels[one_dimensional] = if discovered_as_outer_and_empty[one_dimensional] {
                Some(distance)
            } else {
                Some(-distance)
            };
        }
    }
}

/// Returns number of voxels created when `ScalarField::from_mesh()` called.
pub fn evaluate_voxel_count(
    mesh_bounding_box: &BoundingBox<f32>,
    voxel_dimensions: &Vector3<f32>,
) -> u32 {
    let min_absolute = cartesian_to_absolute_voxel_coordinate(
        &mesh_bounding_box.minimum_point(),
        voxel_dimensions,
    );
    let max_absolute = cartesian_to_absolute_voxel_coordinate(
        &mesh_bounding_box.maximum_point(),
        voxel_dimensions,
    );
    let diagonal_absolute = max_absolute - min_absolute.coords + Vector3::new(1, 1, 1);
    cast_u32(diagonal_absolute.x * diagonal_absolute.y * diagonal_absolute.z)
}

/// Computes voxel dimensions with similar proportions to
/// `current_voxel_dimensions` so that the `mesh_bounding_box` contains roughly
/// `voxel_count_threshold` voxels.
pub fn suggest_voxel_size_to_fit_bbox_within_voxel_count(
    voxel_count: u32,
    current_voxel_dimensions: &Vector3<f32>,
    voxel_count_threshold: u32,
) -> Vector3<f32> {
    let voxel_scaling_ratio_3d = voxel_count as f32 / voxel_count_threshold as f32;
    let voxel_scaling_ratio_1d = voxel_scaling_ratio_3d.cbrt();
    // When changing the voxel dimensions, also the bounding box dimensions
    // change, therefore the equation is not simple. Therefore a safe buffer of
    // 1.1 is a quick fix.
    // FIXME: Come up with a precise equation
    current_voxel_dimensions * voxel_scaling_ratio_1d * 1.1
}

/// Returns `true` if the value of a voxel is within given value range. Returns
/// `false` if the voxel value is not within the `value_range` or if the voxel
/// does not exist or is out of scalar field's bounds.
fn is_voxel_within_range<U>(voxel: Option<f32>, value_range: &U) -> bool
where
    U: RangeBounds<f32>,
{
    match voxel {
        Some(value) => value_range.contains(&value),
        None => false,
    }
}

/// Computes a voxel position relative to the block start (relative coordinate)
/// from an index to the linear representation of the voxel block.
fn one_dimensional_to_relative_voxel_coordinate(
    one_dimensional_coordinate: usize,
    block_dimensions: &Vector3<u32>,
) -> Point3<i32> {
    let one_dimensional_i32 = cast_i32(one_dimensional_coordinate);
    let horizontal_area_i32 = cast_i32(block_dimensions.x * block_dimensions.y);
    let x_dimension_i32 = cast_i32(block_dimensions.x);
    let z = one_dimensional_i32 / horizontal_area_i32;
    let y = (one_dimensional_i32 % horizontal_area_i32) / x_dimension_i32;
    let x = one_dimensional_i32 % x_dimension_i32;
    Point3::new(x, y, z)
}

/// Computes a voxel position relative to the model space origin (absolute
/// coordinate) from an index to the linear representation of the voxel block.
fn one_dimensional_to_absolute_voxel_coordinate(
    one_dimensional_coordinate: usize,
    block_start: &Point3<i32>,
    block_dimensions: &Vector3<u32>,
) -> Point3<i32> {
    let relative =
        one_dimensional_to_relative_voxel_coordinate(one_dimensional_coordinate, block_dimensions);
    relative_voxel_to_absolute_voxel_coordinate(&relative, block_start)
}

/// Computes a voxel position in world space cartesian units from an index to
/// the linear representation of the voxel block.
fn one_dimensional_to_cartesian_coordinate(
    one_dimensional_coordinate: usize,
    block_start: &Point3<i32>,
    block_dimensions: &Vector3<u32>,
    voxel_dimensions: &Vector3<f32>,
) -> Point3<f32> {
    let relative =
        one_dimensional_to_relative_voxel_coordinate(one_dimensional_coordinate, block_dimensions);
    relative_voxel_to_cartesian_coordinate(&relative, block_start, voxel_dimensions)
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
///
/// Panics if any of the voxel dimensions is equal or below zero.
fn relative_voxel_to_cartesian_coordinate(
    relative_coordinate: &Point3<i32>,
    block_start: &Point3<i32>,
    voxel_dimensions: &Vector3<f32>,
) -> Point3<f32> {
    assert!(
        voxel_dimensions.x > 0.0 && voxel_dimensions.y > 0.0 && voxel_dimensions.z > 0.0,
        "Voxel dimensions can't be below or equal to zero"
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
///
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
        let scalar_field = ScalarField::from_mesh(&mesh, &Vector3::new(1.0, 1.0, 1.0), 0.0, 0);

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

        let scalar_field = ScalarField::from_mesh(&mesh, &Vector3::new(0.5, 0.5, 0.5), 0.0, 0);

        insta::assert_json_snapshot!("sphere_after_voxelization_into_scalar_field", &scalar_field);
    }

    #[test]
    fn test_scalar_field_three_dimensional_to_one_dimensional_and_back_relative() {
        let block_dimensions = Vector3::new(3, 4, 5);

        for z in 0..block_dimensions.z {
            for y in 0..block_dimensions.y {
                for x in 0..block_dimensions.x {
                    let relative_position = Point3::new(cast_i32(x), cast_i32(y), cast_i32(z));
                    let one_dimensional = relative_voxel_to_one_dimensional_coordinate(
                        &relative_position,
                        &block_dimensions,
                    )
                    .unwrap();
                    let three_dimensional = one_dimensional_to_relative_voxel_coordinate(
                        one_dimensional,
                        &block_dimensions,
                    );

                    assert_eq!(relative_position, three_dimensional);
                }
            }
        }
    }

    #[test]
    fn test_scalar_field_get_set_for_single_voxel() {
        let mut scalar_field: ScalarField = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(1, 1, 1),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        let before = scalar_field.value_at_absolute_voxel_coordinate(&Point3::new(0, 0, 0));
        scalar_field.set_value_at_absolute_voxel_coordinate(&Point3::new(0, 0, 0), Some(0.0));
        let after = scalar_field
            .value_at_absolute_voxel_coordinate(&Point3::new(0, 0, 0))
            .unwrap();

        assert_eq!(before, None);
        assert_eq!(after, 0.0);
    }

    #[test]
    fn test_scalar_field_single_voxel_to_mesh_produces_synchronized_mesh() {
        let mut scalar_field = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(1, 1, 1),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        scalar_field.set_value_at_absolute_voxel_coordinate(&Point3::new(0, 0, 0), Some(0.0));

        let voxel_mesh = scalar_field.to_mesh(&(0.0..=0.0)).unwrap();

        let v2f = topology::compute_vertex_to_face_topology(&voxel_mesh);
        let f2f = topology::compute_face_to_face_topology(&voxel_mesh, &v2f);
        let voxel_mesh_synced = tools::synchronize_mesh_winding(&voxel_mesh, &f2f);

        assert!(analysis::are_similar(&voxel_mesh, &voxel_mesh_synced));
    }
}
