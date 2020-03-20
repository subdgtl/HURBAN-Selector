use std::collections::VecDeque;
use std::f32;
use std::ops::RangeBounds;

use nalgebra::{Point3, Vector2, Vector3};

use crate::bounding_box::BoundingBox;
use crate::convert::{cast_i32, cast_u32, cast_usize};
use crate::geometry;
use crate::math;
use crate::plane::Plane;

use super::{primitive, tools, Face, Mesh, NormalStrategy};

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
///
/// In this struct's method parameters, `volume_value_range` is an interval
/// defining which values of the scalar field should be considered to be a
/// volume. The `ScalarField::from_mesh` generates a scalar field, which marks
/// volume voxels with value `0`. `compute_distance_field` marks each voxel with
/// a value representing the voxel's distance from the original volume,
/// therefore the voxels right at the shell of the volume are marked 0, the
/// layer around them is marked 1 or -1 (inside closed volumes) etc. Once the
/// scalar field is populated with meaningful values, it is possible to treat
/// (perform boolean operations or materialize into mesh) on various numerical
/// ranges. Such range is specified ad-hoc by parameter `volume_value_range`.
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

    /// Clears the scalar field, sets its block dimensions to zero.
    pub fn wipe(&mut self) {
        self.block_start = Point3::origin();
        self.block_dimensions = Vector3::zeros();
        self.voxels.resize(0, None);
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

    /// Checks if the scalar field contains any voxel with a value from the
    /// given range.
    pub fn contains_voxels_within_range<U>(&self, volume_value_range: &U) -> bool
    where
        U: RangeBounds<f32>,
    {
        self.voxels.iter().any(|voxel| {
            voxel
                .map(|value| volume_value_range.contains(&value))
                .unwrap_or(false)
        })
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
            if voxel
                .map(|value| volume_value_range.contains(&value))
                .unwrap_or(false)
            {
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
                    if neighbor_voxel
                        .map(|value| !volume_value_range.contains(&value))
                        .unwrap_or(true)
                    {
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

    /// Computes a relatively smooth triangulated welded mesh from the current
    /// state of the scalar field.
    ///
    /// For watertight volumetric geometry (i.e. from a watertight source mesh)
    /// this creates both, outer and inner boundary mesh. There is also a high
    /// risk of generating a non-manifold mesh.
    pub fn to_marching_cubes<U>(&self, volume_value_range: &U) -> Option<Mesh>
    where
        U: RangeBounds<f32>,
    {
        if self.block_dimensions.x == 0
            || self.block_dimensions.y == 0
            || self.block_dimensions.z == 0
            || !self.contains_voxels_within_range(volume_value_range)
        {
            return None;
        }

        let marching_cubes_lookup_table = marching_cubes_lookup_table();

        // A collection of mesh patches defining an outer envelope of volumes
        // stored in the scalar field
        let mut marching_cubes: Vec<Mesh> = Vec::new();

        let block_end = self.block_end();

        // Iterate through the scalar field. Consider 8 neighboring voxels to be
        // corners of the test cube. Start iterating at position [-1, -1, -1]
        // relative to the `block_start` and finish at [1, 1, 1] relative to the
        // `block_end` to catch the block boundaries.
        for z in (self.block_start.z - 1)..=block_end.z {
            for y in (self.block_start.y - 1)..=block_end.y {
                for x in (self.block_start.x - 1)..=block_end.x {
                    //                 v4_______e4_____________v5
                    //                  /|                    /|
                    //                 / |                   / |
                    //              e7/  |                e5/  |
                    //               /___|______e6_________/   |
                    //            v7|    |                 |v6 |e9
                    //              |    |                 |   |
                    //              |    |e8               |e10|
                    //           e11|    |                 |   |
                    //              |    |_________________|___|
                    //              |   / v0    e0         |   /v1
                    //              |  /                   |  /
                    //              | /e3                  | /e1
                    //              |/_____________________|/
                    //              v3         e2          v2
                    //
                    // the current voxel center is `v3`, the other vertices are centers of
                    // the neighboring voxels
                    let corners_absolute_coordinates = [
                        Point3::new(x, y + 1, z),
                        Point3::new(x + 1, y + 1, z),
                        Point3::new(x + 1, y, z),
                        Point3::new(x, y, z),
                        Point3::new(x, y + 1, z + 1),
                        Point3::new(x + 1, y + 1, z + 1),
                        Point3::new(x + 1, y, z + 1),
                        Point3::new(x, y, z + 1),
                    ];
                    let corners_cartesian_coordinates: Vec<_> = corners_absolute_coordinates
                        .iter()
                        .map(|a| absolute_voxel_to_cartesian_coordinate(a, &self.voxel_dimensions))
                        .collect();

                    // These points will become vertices of mesh patches forming
                    // the outer envelope of the scalar field.
                    #[rustfmt::skip]
                    let edge_midpoints = vec![
                        Point3::from(corners_cartesian_coordinates[0].coords.lerp(&corners_cartesian_coordinates[1].coords, 0.5)), //e0
                        Point3::from(corners_cartesian_coordinates[1].coords.lerp(&corners_cartesian_coordinates[2].coords, 0.5)), //e1
                        Point3::from(corners_cartesian_coordinates[2].coords.lerp(&corners_cartesian_coordinates[3].coords, 0.5)), //e2
                        Point3::from(corners_cartesian_coordinates[3].coords.lerp(&corners_cartesian_coordinates[0].coords, 0.5)), //e3
                        Point3::from(corners_cartesian_coordinates[4].coords.lerp(&corners_cartesian_coordinates[5].coords, 0.5)), //e4
                        Point3::from(corners_cartesian_coordinates[5].coords.lerp(&corners_cartesian_coordinates[6].coords, 0.5)), //e5
                        Point3::from(corners_cartesian_coordinates[6].coords.lerp(&corners_cartesian_coordinates[7].coords, 0.5)), //e6
                        Point3::from(corners_cartesian_coordinates[7].coords.lerp(&corners_cartesian_coordinates[4].coords, 0.5)), //e7
                        Point3::from(corners_cartesian_coordinates[0].coords.lerp(&corners_cartesian_coordinates[4].coords, 0.5)), //e8
                        Point3::from(corners_cartesian_coordinates[1].coords.lerp(&corners_cartesian_coordinates[5].coords, 0.5)), //e9
                        Point3::from(corners_cartesian_coordinates[2].coords.lerp(&corners_cartesian_coordinates[6].coords, 0.5)), //e10
                        Point3::from(corners_cartesian_coordinates[3].coords.lerp(&corners_cartesian_coordinates[7].coords, 0.5)), //e11
                    ];

                    // Boolean vector marking each of the voxels either volume
                    // (`true`) or void(`false`)
                    let corners_inside_volume_pattern: Vec<_> = corners_absolute_coordinates
                        .iter()
                        .map(|a| {
                            self.value_at_absolute_voxel_coordinate(&a)
                                .map(|value| volume_value_range.contains(&value))
                                .unwrap_or(false)
                        })
                        .collect();

                    // The marching cubes lookup table contains 256 possible
                    // samples of mesh patches. The sample index specifies the
                    // proper sample for the current voxel group.
                    let mut sample_index = 0_u8;
                    if corners_inside_volume_pattern[0] {
                        sample_index |= 1;
                    }
                    if corners_inside_volume_pattern[1] {
                        sample_index |= 2;
                    }
                    if corners_inside_volume_pattern[2] {
                        sample_index |= 4;
                    }
                    if corners_inside_volume_pattern[3] {
                        sample_index |= 8;
                    }
                    if corners_inside_volume_pattern[4] {
                        sample_index |= 16;
                    }
                    if corners_inside_volume_pattern[5] {
                        sample_index |= 32;
                    }
                    if corners_inside_volume_pattern[6] {
                        sample_index |= 64;
                    }
                    if corners_inside_volume_pattern[7] {
                        sample_index |= 128;
                    }

                    // The marching cubes lookup table sample, that contains
                    // triplets of indices to the edge midpoints (currently
                    // already vertices) that will become mesh patch faces.
                    let marching_cubes_faces =
                        marching_cubes_lookup_table[cast_usize(sample_index)].clone();

                    // In some cases (when all the voxels are inside or outside
                    // a volume), there is no mesh patch to be created.
                    if !marching_cubes_faces.is_empty() {
                        marching_cubes.push(
                        // Generate new mesh path from all edge midpoint
                        // vertices and only those faces that should be created
                        // according th the marching cubes lookup table.
                        Mesh::from_triangle_faces_with_vertices_and_computed_normals_remove_orphans(
                            marching_cubes_faces,
                            edge_midpoints,
                            // Sharp is more efficient and gets lost by welding anyway
                            NormalStrategy::Sharp,
                        ),
                    );
                    }
                }
            }
        }

        // Join separate mesh planes into one mesh
        let joined_voxel_mesh = tools::join_multiple_meshes(&marching_cubes);
        let min_voxel_dimension = self
            .voxel_dimensions
            .x
            .min(self.voxel_dimensions.y.min(self.voxel_dimensions.z));
        // and weld naked edges.
        tools::weld(&joined_voxel_mesh, (min_voxel_dimension as f32) / 4.0)
    }

    /// Computes boolean intersection (logical AND operation) of the current and
    /// another scalar field. The current scalar field will be mutated and
    /// resized to the size and position of an intersection of the two scalar
    /// fields' volumes. The two scalar fields do not have to contain voxels of
    /// the same size.
    pub fn boolean_intersection<U>(
        &mut self,
        volume_value_range_self: &U,
        other: &ScalarField,
        volume_value_range_other: &U,
    ) where
        U: RangeBounds<f32>,
    {
        // Find volume common to both scalar fields.
        if let (Some(self_volume_bounding_box), Some(other_volume_bounding_box)) = (
            self.volume_voxel_space_bounding_box(volume_value_range_self),
            other.volume_voxel_space_bounding_box(volume_value_range_other),
        ) {
            if let Some(bounding_box) = BoundingBox::intersection(
                [self_volume_bounding_box, other_volume_bounding_box]
                    .iter()
                    .copied(),
            ) {
                // Resize (keep or shrink) the existing scalar field so that
                // that can possibly contain intersection voxels.
                self.resize_to_voxel_space_bounding_box(&bounding_box);

                let block_start = bounding_box.minimum_point();
                let diagonal = bounding_box.diagonal();
                let block_dimensions = Vector3::new(
                    cast_u32(diagonal.x),
                    cast_u32(diagonal.y),
                    cast_u32(diagonal.z),
                );
                // Iterate through the block of space common to both scalar fields.
                for (one_dimensional, voxel) in self.voxels.iter_mut().enumerate() {
                    // Perform boolean AND on voxel values of both scalar fields.
                    let cartesian_coordinate = one_dimensional_to_cartesian_coordinate(
                        one_dimensional,
                        &block_start,
                        &block_dimensions,
                        &self.voxel_dimensions,
                    );
                    let absolute_coordinate_other = cartesian_to_absolute_voxel_coordinate(
                        &cartesian_coordinate,
                        &other.voxel_dimensions,
                    );

                    if other
                        .value_at_absolute_voxel_coordinate(&absolute_coordinate_other)
                        .map(|value| !volume_value_range_other.contains(&value))
                        .unwrap_or(true)
                    {
                        *voxel = None;
                    }
                }
                self.shrink_to_fit(volume_value_range_self);
                // Return here because any other option needs to wipe the
                // current scalar field.
                return;
            }
        }
        // If the two scalar fields do not intersect or one of them is empty,
        // then wipe the resulting scalar field.
        self.wipe();
    }

    /// Computes boolean union (logical OR operation) of two scalar fields. The
    /// current scalar field will be mutated and resized to contain both input
    /// scalar fields' volumes. The values from the other scalar field which are
    /// considered a volume, will be remapped to the volume value range of the
    /// source scalar field. The two scalar fields do not have to contain voxels
    /// of the same size.
    ///
    /// # Panics
    ///
    /// Panics if one of the volume value ranges is infinite.
    ///
    /// # Warning
    ///
    /// If the input scalar fields are far apart, the resulting scalar field may
    /// be huge.
    pub fn boolean_union<U>(
        &mut self,
        volume_value_range_self: &U,
        other: &ScalarField,
        volume_value_range_other: &U,
    ) where
        U: RangeBounds<f32>,
    {
        use std::ops::Bound::*;

        // FIXME: This discards the `other` scalar field's fine-grain value
        // information, such as distance field. This is not a rare situation,
        // because the ranges are infinite when the volumes should be considered
        // filled. On the other hand, if the distance field values only make
        // sense if the volume value ranges are the same for both scalar fields,
        // which is na edge case now resolved in the remap function - it returns
        // the unchanged value even if the ranges are identical, even if they
        // are unbounded.
        //
        // Define a value that will be used for a volume voxel coming from the
        // `other` scalar field in case the `volume_value_range_other` is
        // unbounded (infinite) and therefore its values can't be remapped.
        let self_certain_volume_value = if let Included(self_start) | Excluded(self_start) =
            volume_value_range_self.start_bound()
        {
            // Prefer the start value of `volume_value_range_self`.
            *self_start as f64
        } else if let Included(self_end) | Excluded(self_end) = volume_value_range_self.end_bound()
        {
            // If the start of the `volume_value_range_self` is unbounded, then use its end.
            *self_end as f64
        } else {
            // If the `volume_value_range_self` is unbounded on both ends,
            // use zero because it certainly is in the range and is in its
            // middle.
            0_f64
        };

        let bounding_box_self = self.volume_voxel_space_bounding_box(volume_value_range_self);
        let bounding_box_other = other.volume_voxel_space_bounding_box(volume_value_range_other);

        // Early return if the other scalar field doesn't contain any voxels
        // (there are no voxels to be added to self).
        if bounding_box_other == None {
            return;
        }

        let bounding_boxes = [bounding_box_self, bounding_box_other];

        // Unwrap the bounding box options. the other bounding box must be valid
        // at this point and the self can be None. In that case, all the volume
        // voxels from the other scalar field will be remapped to the current
        // scalar field.
        let valid_bounding_boxes_iter = bounding_boxes.iter().filter_map(|b| *b);

        if let Some(bounding_box) = BoundingBox::union(valid_bounding_boxes_iter) {
            // Resize (keep or grow) the current scalar field to a block that
            // will contain union voxels.
            self.resize_to_voxel_space_bounding_box(&bounding_box);

            // Iterate through the block of space containing volume voxels from
            // both scalar fields. Iterate through the units of the current
            // scalar field.
            for (one_dimensional, voxel) in self.voxels.iter_mut().enumerate() {
                // If the current scalar field doesn't contain a volume voxel at
                // the current position
                if voxel
                    .map(|value| !volume_value_range_self.contains(&value))
                    .unwrap_or(true)
                {
                    let cartesian_coordinate = one_dimensional_to_cartesian_coordinate(
                        one_dimensional,
                        &self.block_start,
                        &self.block_dimensions,
                        &self.voxel_dimensions,
                    );
                    let absolute_coordinate_other = cartesian_to_absolute_voxel_coordinate(
                        &cartesian_coordinate,
                        &other.voxel_dimensions,
                    );

                    // If the other scalar field contains a voxel on the
                    // cartesian coordinate of the current voxel, then remap the
                    // other value to the volume value range of the current
                    // scalar field and set the voxel to the value.
                    if let Some(voxel_other) =
                        other.value_at_absolute_voxel_coordinate(&absolute_coordinate_other)
                    {
                        if volume_value_range_other.contains(&voxel_other) {
                            // If the remap fails, the program should panic.
                            *voxel = Some(
                                math::remap(
                                    voxel_other,
                                    volume_value_range_other,
                                    volume_value_range_self,
                                )
                                .unwrap_or(self_certain_volume_value)
                                    as f32,
                            );
                        }
                    }
                }
            }
        } else {
            // Wipe the current scalar field if none of the scalar fields
            // contained any volume voxels.
            self.wipe();
        }
    }

    /// Computes boolean difference of the current scalar field minus the other
    /// scalar field. The current scalar field will be modified so that voxels,
    /// that are within volume value range in both scalar fields will be set to
    /// None in the current scalar field, while the rest remains intact. The two
    /// scalar fields do not have to contain voxels of the same size.
    pub fn boolean_difference<U>(
        &mut self,
        volume_value_range_self: &U,
        other: &ScalarField,
        volume_value_range_other: &U,
    ) where
        U: RangeBounds<f32>,
    {
        // Iterate through the target scalar field
        for (one_dimensional, voxel) in self.voxels.iter_mut().enumerate() {
            // If the current scalar field contains a volume voxel at the
            // current position
            if voxel
                .map(|value| volume_value_range_self.contains(&value))
                .unwrap_or(false)
            {
                let cartesian_coordinate = one_dimensional_to_cartesian_coordinate(
                    one_dimensional,
                    &self.block_start,
                    &self.block_dimensions,
                    &self.voxel_dimensions,
                );

                let absolute_coordinate_other = cartesian_to_absolute_voxel_coordinate(
                    &cartesian_coordinate,
                    &other.voxel_dimensions,
                );
                // and so does the other scalar field
                if other
                    .value_at_absolute_voxel_coordinate(&absolute_coordinate_other)
                    .map(|value| volume_value_range_other.contains(&value))
                    .unwrap_or(false)
                {
                    // then remove the voxel from the current scalar field
                    *voxel = None;
                }
            }
        }
        self.shrink_to_fit(volume_value_range_self)
    }

    /// Resize the scalar field block to match new block start and block
    /// dimensions.
    ///
    /// This clips the outstanding parts of the original scalar field and fills
    /// the newly added parts with None (no voxel).
    pub fn resize(
        &mut self,
        resized_block_start: &Point3<i32>,
        resized_block_dimensions: &Vector3<u32>,
    ) {
        // Don't resize if the scalar field dimensions haven't changed.
        if resized_block_start == &self.block_start
            && resized_block_dimensions == &self.block_dimensions
        {
            return;
        }

        // Wipe if resizing to an empty scalar field.
        if resized_block_dimensions == &Vector3::zeros() {
            self.wipe();
            return;
        }

        let resized_values_len = cast_usize(
            resized_block_dimensions.x * resized_block_dimensions.y * resized_block_dimensions.z,
        );

        let mut new_voxels: Vec<Option<f32>> = Vec::with_capacity(resized_values_len);

        for resized_one_dimensional in 0..resized_values_len {
            let absolute_coordinate = one_dimensional_to_absolute_voxel_coordinate(
                resized_one_dimensional,
                resized_block_start,
                resized_block_dimensions,
            );

            let new_voxel = match absolute_voxel_to_one_dimensional_coordinate(
                &absolute_coordinate,
                &self.block_start,
                &self.block_dimensions,
            ) {
                Some(original_one_dimensional) => self.voxels[original_one_dimensional],
                None => None,
            };
            new_voxels.push(new_voxel);
        }

        self.voxels = new_voxels;
        self.block_start = *resized_block_start;
        self.block_dimensions = *resized_block_dimensions;
    }

    /// Resize the scalar field block to exactly fit the volumetric geometry.
    pub fn shrink_to_fit<U>(&mut self, volume_value_range: &U)
    where
        U: RangeBounds<f32>,
    {
        if let Some((shrunk_block_start, shrunk_block_dimensions)) =
            self.compute_volume_boundaries(volume_value_range)
        {
            self.resize(&shrunk_block_start, &shrunk_block_dimensions);
        } else {
            self.wipe();
        }
    }

    /// Compute discrete distance field.
    ///
    /// Each voxel will be set a value equal to its distance from the original
    /// volume. The voxels that were originally volume voxels, will be set to
    /// value 0. Voxels inside the closed volumes will have the distance value
    /// with a negative sign.
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
            if voxel
                .map(|value| !volume_value_range.contains(&value))
                .unwrap_or(true)
            {
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
                if self
                    .value_at_absolute_voxel_coordinate(&neighbor_absolute_coordinate)
                    .map(|value| !volume_value_range.contains(&value))
                    .unwrap_or(true)
                {
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
                        && self
                            .value_at_absolute_voxel_coordinate(&neighbor_absolute_coordinate)
                            .map(|value| !volume_value_range.contains(&value))
                            .unwrap_or(true)
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

    /// Returns the bounding box in voxel units of the current scalar field
    /// after shrinking to fit just the nonempty voxels.
    pub fn volume_voxel_space_bounding_box<U>(
        &self,
        volume_value_range: &U,
    ) -> Option<BoundingBox<i32>>
    where
        U: RangeBounds<f32>,
    {
        self.compute_volume_boundaries(volume_value_range).map(
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

    /// Computes boundaries of volumes contained in scalar field. Returns tuple
    /// `(block_start, block_dimensions)`. For empty scalar fields returns the
    /// original block start and zero block dimensions.
    fn compute_volume_boundaries<U>(
        &self,
        volume_value_range: &U,
    ) -> Option<(Point3<i32>, Vector3<u32>)>
    where
        U: RangeBounds<f32>,
    {
        let mut absolute_min: Point3<i32> =
            Point3::new(i32::max_value(), i32::max_value(), i32::max_value());
        let mut absolute_max: Point3<i32> =
            Point3::new(i32::min_value(), i32::min_value(), i32::min_value());
        for (one_dimensional, voxel) in self.voxels.iter().enumerate() {
            if voxel
                .map(|value| volume_value_range.contains(&value))
                .unwrap_or(false)
            {
                let absolute_coordinate = one_dimensional_to_absolute_voxel_coordinate(
                    one_dimensional,
                    &self.block_start,
                    &self.block_dimensions,
                );

                if absolute_coordinate.x < absolute_min.x {
                    absolute_min.x = absolute_coordinate.x;
                }
                if absolute_coordinate.x > absolute_max.x {
                    absolute_max.x = absolute_coordinate.x;
                }
                if absolute_coordinate.y < absolute_min.y {
                    absolute_min.y = absolute_coordinate.y;
                }
                if absolute_coordinate.y > absolute_max.y {
                    absolute_max.y = absolute_coordinate.y;
                }
                if absolute_coordinate.z < absolute_min.z {
                    absolute_min.z = absolute_coordinate.z;
                }
                if absolute_coordinate.z > absolute_max.z {
                    absolute_max.z = absolute_coordinate.z;
                }
            }
        }
        // If the scalar field doesn't contain any voxels, all of the min/max
        // values should remain unchanged. It's enough to check one of the
        // values because if anything is found, all the values would change.
        if absolute_min.x == i32::max_value() {
            assert_eq!(
                absolute_min.y,
                i32::max_value(),
                "scalar field emptiness check failed"
            );
            assert_eq!(
                absolute_min.z,
                i32::max_value(),
                "scalar field emptiness check failed"
            );
            assert_eq!(
                absolute_max.x,
                i32::min_value(),
                "scalar field emptiness check failed"
            );
            assert_eq!(
                absolute_max.y,
                i32::min_value(),
                "scalar field emptiness check failed"
            );
            assert_eq!(
                absolute_max.z,
                i32::min_value(),
                "scalar field emptiness check failed"
            );
            None
        } else {
            let block_dimensions = Vector3::new(
                cast_u32(absolute_max.x - absolute_min.x + 1),
                cast_u32(absolute_max.y - absolute_min.y + 1),
                cast_u32(absolute_max.z - absolute_min.z + 1),
            );
            Some((absolute_min, block_dimensions))
        }
    }
}

/// Lookup table of triangle faces suitable to the sample situation. Each vector
/// item contains a vector of vertex indices. The vector indices correcpond wit
/// the edge index.
///
///  ## Corner and edge indicex scheme
///
///                 v4_______e4_____________v5
///                  /|                    /|
///                 / |                   / |
///              e7/  |                e5/  |
///               /___|______e6_________/   |
///            v7|    |                 |v6 |e9
///              |    |                 |   |
///              |    |e8               |e10|
///           e11|    |                 |   |
///              |    |_________________|___|
///              |   / v0    e0         |   /v1
///              |  /                   |  /
///              | /e3                  | /e1
///              |/_____________________|/
///              v3         e2          v2
///
/// # Source
///
/// http://paulbourke.net/geometry/polygonise/
fn marching_cubes_lookup_table() -> Vec<Vec<(u32, u32, u32)>> {
    vec![
        vec![],
        vec![(0, 8, 3)],
        vec![(0, 1, 9)],
        vec![(1, 8, 3), (9, 8, 1)],
        vec![(1, 2, 10)],
        vec![(0, 8, 3), (1, 2, 10)],
        vec![(9, 2, 10), (0, 2, 9)],
        vec![(2, 8, 3), (2, 10, 8), (10, 9, 8)],
        vec![(3, 11, 2)],
        vec![(0, 11, 2), (8, 11, 0)],
        vec![(1, 9, 0), (2, 3, 11)],
        vec![(1, 11, 2), (1, 9, 11), (9, 8, 11)],
        vec![(3, 10, 1), (11, 10, 3)],
        vec![(0, 10, 1), (0, 8, 10), (8, 11, 10)],
        vec![(3, 9, 0), (3, 11, 9), (11, 10, 9)],
        vec![(9, 8, 10), (10, 8, 11)],
        vec![(4, 7, 8)],
        vec![(4, 3, 0), (7, 3, 4)],
        vec![(0, 1, 9), (8, 4, 7)],
        vec![(4, 1, 9), (4, 7, 1), (7, 3, 1)],
        vec![(1, 2, 10), (8, 4, 7)],
        vec![(3, 4, 7), (3, 0, 4), (1, 2, 10)],
        vec![(9, 2, 10), (9, 0, 2), (8, 4, 7)],
        vec![(2, 10, 9), (2, 9, 7), (2, 7, 3), (7, 9, 4)],
        vec![(8, 4, 7), (3, 11, 2)],
        vec![(11, 4, 7), (11, 2, 4), (2, 0, 4)],
        vec![(9, 0, 1), (8, 4, 7), (2, 3, 11)],
        vec![(4, 7, 11), (9, 4, 11), (9, 11, 2), (9, 2, 1)],
        vec![(3, 10, 1), (3, 11, 10), (7, 8, 4)],
        vec![(1, 11, 10), (1, 4, 11), (1, 0, 4), (7, 11, 4)],
        vec![(4, 7, 8), (9, 0, 11), (9, 11, 10), (11, 0, 3)],
        vec![(4, 7, 11), (4, 11, 9), (9, 11, 10)],
        vec![(9, 5, 4)],
        vec![(9, 5, 4), (0, 8, 3)],
        vec![(0, 5, 4), (1, 5, 0)],
        vec![(8, 5, 4), (8, 3, 5), (3, 1, 5)],
        vec![(1, 2, 10), (9, 5, 4)],
        vec![(3, 0, 8), (1, 2, 10), (4, 9, 5)],
        vec![(5, 2, 10), (5, 4, 2), (4, 0, 2)],
        vec![(2, 10, 5), (3, 2, 5), (3, 5, 4), (3, 4, 8)],
        vec![(9, 5, 4), (2, 3, 11)],
        vec![(0, 11, 2), (0, 8, 11), (4, 9, 5)],
        vec![(0, 5, 4), (0, 1, 5), (2, 3, 11)],
        vec![(2, 1, 5), (2, 5, 8), (2, 8, 11), (4, 8, 5)],
        vec![(10, 3, 11), (10, 1, 3), (9, 5, 4)],
        vec![(4, 9, 5), (0, 8, 1), (8, 10, 1), (8, 11, 10)],
        vec![(5, 4, 0), (5, 0, 11), (5, 11, 10), (11, 0, 3)],
        vec![(5, 4, 8), (5, 8, 10), (10, 8, 11)],
        vec![(9, 7, 8), (5, 7, 9)],
        vec![(9, 3, 0), (9, 5, 3), (5, 7, 3)],
        vec![(0, 7, 8), (0, 1, 7), (1, 5, 7)],
        vec![(1, 5, 3), (3, 5, 7)],
        vec![(9, 7, 8), (9, 5, 7), (10, 1, 2)],
        vec![(10, 1, 2), (9, 5, 0), (5, 3, 0), (5, 7, 3)],
        vec![(8, 0, 2), (8, 2, 5), (8, 5, 7), (10, 5, 2)],
        vec![(2, 10, 5), (2, 5, 3), (3, 5, 7)],
        vec![(7, 9, 5), (7, 8, 9), (3, 11, 2)],
        vec![(9, 5, 7), (9, 7, 2), (9, 2, 0), (2, 7, 11)],
        vec![(2, 3, 11), (0, 1, 8), (1, 7, 8), (1, 5, 7)],
        vec![(11, 2, 1), (11, 1, 7), (7, 1, 5)],
        vec![(9, 5, 8), (8, 5, 7), (10, 1, 3), (10, 3, 11)],
        vec![(5, 7, 0), (5, 0, 9), (7, 11, 0), (1, 0, 10), (11, 10, 0)],
        vec![(11, 10, 0), (11, 0, 3), (10, 5, 0), (8, 0, 7), (5, 7, 0)],
        vec![(11, 10, 5), (7, 11, 5)],
        vec![(10, 6, 5)],
        vec![(0, 8, 3), (5, 10, 6)],
        vec![(9, 0, 1), (5, 10, 6)],
        vec![(1, 8, 3), (1, 9, 8), (5, 10, 6)],
        vec![(1, 6, 5), (2, 6, 1)],
        vec![(1, 6, 5), (1, 2, 6), (3, 0, 8)],
        vec![(9, 6, 5), (9, 0, 6), (0, 2, 6)],
        vec![(5, 9, 8), (5, 8, 2), (5, 2, 6), (3, 2, 8)],
        vec![(2, 3, 11), (10, 6, 5)],
        vec![(11, 0, 8), (11, 2, 0), (10, 6, 5)],
        vec![(0, 1, 9), (2, 3, 11), (5, 10, 6)],
        vec![(5, 10, 6), (1, 9, 2), (9, 11, 2), (9, 8, 11)],
        vec![(6, 3, 11), (6, 5, 3), (5, 1, 3)],
        vec![(0, 8, 11), (0, 11, 5), (0, 5, 1), (5, 11, 6)],
        vec![(3, 11, 6), (0, 3, 6), (0, 6, 5), (0, 5, 9)],
        vec![(6, 5, 9), (6, 9, 11), (11, 9, 8)],
        vec![(5, 10, 6), (4, 7, 8)],
        vec![(4, 3, 0), (4, 7, 3), (6, 5, 10)],
        vec![(1, 9, 0), (5, 10, 6), (8, 4, 7)],
        vec![(10, 6, 5), (1, 9, 7), (1, 7, 3), (7, 9, 4)],
        vec![(6, 1, 2), (6, 5, 1), (4, 7, 8)],
        vec![(1, 2, 5), (5, 2, 6), (3, 0, 4), (3, 4, 7)],
        vec![(8, 4, 7), (9, 0, 5), (0, 6, 5), (0, 2, 6)],
        vec![(7, 3, 9), (7, 9, 4), (3, 2, 9), (5, 9, 6), (2, 6, 9)],
        vec![(3, 11, 2), (7, 8, 4), (10, 6, 5)],
        vec![(5, 10, 6), (4, 7, 2), (4, 2, 0), (2, 7, 11)],
        vec![(0, 1, 9), (4, 7, 8), (2, 3, 11), (5, 10, 6)],
        vec![(9, 2, 1), (9, 11, 2), (9, 4, 11), (7, 11, 4), (5, 10, 6)],
        vec![(8, 4, 7), (3, 11, 5), (3, 5, 1), (5, 11, 6)],
        vec![(5, 1, 11), (5, 11, 6), (1, 0, 11), (7, 11, 4), (0, 4, 11)],
        vec![(0, 5, 9), (0, 6, 5), (0, 3, 6), (11, 6, 3), (8, 4, 7)],
        vec![(6, 5, 9), (6, 9, 11), (4, 7, 9), (7, 11, 9)],
        vec![(10, 4, 9), (6, 4, 10)],
        vec![(4, 10, 6), (4, 9, 10), (0, 8, 3)],
        vec![(10, 0, 1), (10, 6, 0), (6, 4, 0)],
        vec![(8, 3, 1), (8, 1, 6), (8, 6, 4), (6, 1, 10)],
        vec![(1, 4, 9), (1, 2, 4), (2, 6, 4)],
        vec![(3, 0, 8), (1, 2, 9), (2, 4, 9), (2, 6, 4)],
        vec![(0, 2, 4), (4, 2, 6)],
        vec![(8, 3, 2), (8, 2, 4), (4, 2, 6)],
        vec![(10, 4, 9), (10, 6, 4), (11, 2, 3)],
        vec![(0, 8, 2), (2, 8, 11), (4, 9, 10), (4, 10, 6)],
        vec![(3, 11, 2), (0, 1, 6), (0, 6, 4), (6, 1, 10)],
        vec![(6, 4, 1), (6, 1, 10), (4, 8, 1), (2, 1, 11), (8, 11, 1)],
        vec![(9, 6, 4), (9, 3, 6), (9, 1, 3), (11, 6, 3)],
        vec![(8, 11, 1), (8, 1, 0), (11, 6, 1), (9, 1, 4), (6, 4, 1)],
        vec![(3, 11, 6), (3, 6, 0), (0, 6, 4)],
        vec![(6, 4, 8), (11, 6, 8)],
        vec![(7, 10, 6), (7, 8, 10), (8, 9, 10)],
        vec![(0, 7, 3), (0, 10, 7), (0, 9, 10), (6, 7, 10)],
        vec![(10, 6, 7), (1, 10, 7), (1, 7, 8), (1, 8, 0)],
        vec![(10, 6, 7), (10, 7, 1), (1, 7, 3)],
        vec![(1, 2, 6), (1, 6, 8), (1, 8, 9), (8, 6, 7)],
        vec![(2, 6, 9), (2, 9, 1), (6, 7, 9), (0, 9, 3), (7, 3, 9)],
        vec![(7, 8, 0), (7, 0, 6), (6, 0, 2)],
        vec![(7, 3, 2), (6, 7, 2)],
        vec![(2, 3, 11), (10, 6, 8), (10, 8, 9), (8, 6, 7)],
        vec![(2, 0, 7), (2, 7, 11), (0, 9, 7), (6, 7, 10), (9, 10, 7)],
        vec![(1, 8, 0), (1, 7, 8), (1, 10, 7), (6, 7, 10), (2, 3, 11)],
        vec![(11, 2, 1), (11, 1, 7), (10, 6, 1), (6, 7, 1)],
        vec![(8, 9, 6), (8, 6, 7), (9, 1, 6), (11, 6, 3), (1, 3, 6)],
        vec![(0, 9, 1), (11, 6, 7)],
        vec![(7, 8, 0), (7, 0, 6), (3, 11, 0), (11, 6, 0)],
        vec![(7, 11, 6)],
        vec![(7, 6, 11)],
        vec![(3, 0, 8), (11, 7, 6)],
        vec![(0, 1, 9), (11, 7, 6)],
        vec![(8, 1, 9), (8, 3, 1), (11, 7, 6)],
        vec![(10, 1, 2), (6, 11, 7)],
        vec![(1, 2, 10), (3, 0, 8), (6, 11, 7)],
        vec![(2, 9, 0), (2, 10, 9), (6, 11, 7)],
        vec![(6, 11, 7), (2, 10, 3), (10, 8, 3), (10, 9, 8)],
        vec![(7, 2, 3), (6, 2, 7)],
        vec![(7, 0, 8), (7, 6, 0), (6, 2, 0)],
        vec![(2, 7, 6), (2, 3, 7), (0, 1, 9)],
        vec![(1, 6, 2), (1, 8, 6), (1, 9, 8), (8, 7, 6)],
        vec![(10, 7, 6), (10, 1, 7), (1, 3, 7)],
        vec![(10, 7, 6), (1, 7, 10), (1, 8, 7), (1, 0, 8)],
        vec![(0, 3, 7), (0, 7, 10), (0, 10, 9), (6, 10, 7)],
        vec![(7, 6, 10), (7, 10, 8), (8, 10, 9)],
        vec![(6, 8, 4), (11, 8, 6)],
        vec![(3, 6, 11), (3, 0, 6), (0, 4, 6)],
        vec![(8, 6, 11), (8, 4, 6), (9, 0, 1)],
        vec![(9, 4, 6), (9, 6, 3), (9, 3, 1), (11, 3, 6)],
        vec![(6, 8, 4), (6, 11, 8), (2, 10, 1)],
        vec![(1, 2, 10), (3, 0, 11), (0, 6, 11), (0, 4, 6)],
        vec![(4, 11, 8), (4, 6, 11), (0, 2, 9), (2, 10, 9)],
        vec![(10, 9, 3), (10, 3, 2), (9, 4, 3), (11, 3, 6), (4, 6, 3)],
        vec![(8, 2, 3), (8, 4, 2), (4, 6, 2)],
        vec![(0, 4, 2), (4, 6, 2)],
        vec![(1, 9, 0), (2, 3, 4), (2, 4, 6), (4, 3, 8)],
        vec![(1, 9, 4), (1, 4, 2), (2, 4, 6)],
        vec![(8, 1, 3), (8, 6, 1), (8, 4, 6), (6, 10, 1)],
        vec![(10, 1, 0), (10, 0, 6), (6, 0, 4)],
        vec![(4, 6, 3), (4, 3, 8), (6, 10, 3), (0, 3, 9), (10, 9, 3)],
        vec![(10, 9, 4), (6, 10, 4)],
        vec![(4, 9, 5), (7, 6, 11)],
        vec![(0, 8, 3), (4, 9, 5), (11, 7, 6)],
        vec![(5, 0, 1), (5, 4, 0), (7, 6, 11)],
        vec![(11, 7, 6), (8, 3, 4), (3, 5, 4), (3, 1, 5)],
        vec![(9, 5, 4), (10, 1, 2), (7, 6, 11)],
        vec![(6, 11, 7), (1, 2, 10), (0, 8, 3), (4, 9, 5)],
        vec![(7, 6, 11), (5, 4, 10), (4, 2, 10), (4, 0, 2)],
        vec![(3, 4, 8), (3, 5, 4), (3, 2, 5), (10, 5, 2), (11, 7, 6)],
        vec![(7, 2, 3), (7, 6, 2), (5, 4, 9)],
        vec![(9, 5, 4), (0, 8, 6), (0, 6, 2), (6, 8, 7)],
        vec![(3, 6, 2), (3, 7, 6), (1, 5, 0), (5, 4, 0)],
        vec![(6, 2, 8), (6, 8, 7), (2, 1, 8), (4, 8, 5), (1, 5, 8)],
        vec![(9, 5, 4), (10, 1, 6), (1, 7, 6), (1, 3, 7)],
        vec![(1, 6, 10), (1, 7, 6), (1, 0, 7), (8, 7, 0), (9, 5, 4)],
        vec![(4, 0, 10), (4, 10, 5), (0, 3, 10), (6, 10, 7), (3, 7, 10)],
        vec![(7, 6, 10), (7, 10, 8), (5, 4, 10), (4, 8, 10)],
        vec![(6, 9, 5), (6, 11, 9), (11, 8, 9)],
        vec![(3, 6, 11), (0, 6, 3), (0, 5, 6), (0, 9, 5)],
        vec![(0, 11, 8), (0, 5, 11), (0, 1, 5), (5, 6, 11)],
        vec![(6, 11, 3), (6, 3, 5), (5, 3, 1)],
        vec![(1, 2, 10), (9, 5, 11), (9, 11, 8), (11, 5, 6)],
        vec![(0, 11, 3), (0, 6, 11), (0, 9, 6), (5, 6, 9), (1, 2, 10)],
        vec![(11, 8, 5), (11, 5, 6), (8, 0, 5), (10, 5, 2), (0, 2, 5)],
        vec![(6, 11, 3), (6, 3, 5), (2, 10, 3), (10, 5, 3)],
        vec![(5, 8, 9), (5, 2, 8), (5, 6, 2), (3, 8, 2)],
        vec![(9, 5, 6), (9, 6, 0), (0, 6, 2)],
        vec![(1, 5, 8), (1, 8, 0), (5, 6, 8), (3, 8, 2), (6, 2, 8)],
        vec![(1, 5, 6), (2, 1, 6)],
        vec![(1, 3, 6), (1, 6, 10), (3, 8, 6), (5, 6, 9), (8, 9, 6)],
        vec![(10, 1, 0), (10, 0, 6), (9, 5, 0), (5, 6, 0)],
        vec![(0, 3, 8), (5, 6, 10)],
        vec![(10, 5, 6)],
        vec![(11, 5, 10), (7, 5, 11)],
        vec![(11, 5, 10), (11, 7, 5), (8, 3, 0)],
        vec![(5, 11, 7), (5, 10, 11), (1, 9, 0)],
        vec![(10, 7, 5), (10, 11, 7), (9, 8, 1), (8, 3, 1)],
        vec![(11, 1, 2), (11, 7, 1), (7, 5, 1)],
        vec![(0, 8, 3), (1, 2, 7), (1, 7, 5), (7, 2, 11)],
        vec![(9, 7, 5), (9, 2, 7), (9, 0, 2), (2, 11, 7)],
        vec![(7, 5, 2), (7, 2, 11), (5, 9, 2), (3, 2, 8), (9, 8, 2)],
        vec![(2, 5, 10), (2, 3, 5), (3, 7, 5)],
        vec![(8, 2, 0), (8, 5, 2), (8, 7, 5), (10, 2, 5)],
        vec![(9, 0, 1), (5, 10, 3), (5, 3, 7), (3, 10, 2)],
        vec![(9, 8, 2), (9, 2, 1), (8, 7, 2), (10, 2, 5), (7, 5, 2)],
        vec![(1, 3, 5), (3, 7, 5)],
        vec![(0, 8, 7), (0, 7, 1), (1, 7, 5)],
        vec![(9, 0, 3), (9, 3, 5), (5, 3, 7)],
        vec![(9, 8, 7), (5, 9, 7)],
        vec![(5, 8, 4), (5, 10, 8), (10, 11, 8)],
        vec![(5, 0, 4), (5, 11, 0), (5, 10, 11), (11, 3, 0)],
        vec![(0, 1, 9), (8, 4, 10), (8, 10, 11), (10, 4, 5)],
        vec![(10, 11, 4), (10, 4, 5), (11, 3, 4), (9, 4, 1), (3, 1, 4)],
        vec![(2, 5, 1), (2, 8, 5), (2, 11, 8), (4, 5, 8)],
        vec![(0, 4, 11), (0, 11, 3), (4, 5, 11), (2, 11, 1), (5, 1, 11)],
        vec![(0, 2, 5), (0, 5, 9), (2, 11, 5), (4, 5, 8), (11, 8, 5)],
        vec![(9, 4, 5), (2, 11, 3)],
        vec![(2, 5, 10), (3, 5, 2), (3, 4, 5), (3, 8, 4)],
        vec![(5, 10, 2), (5, 2, 4), (4, 2, 0)],
        vec![(3, 10, 2), (3, 5, 10), (3, 8, 5), (4, 5, 8), (0, 1, 9)],
        vec![(5, 10, 2), (5, 2, 4), (1, 9, 2), (9, 4, 2)],
        vec![(8, 4, 5), (8, 5, 3), (3, 5, 1)],
        vec![(0, 4, 5), (1, 0, 5)],
        vec![(8, 4, 5), (8, 5, 3), (9, 0, 5), (0, 3, 5)],
        vec![(9, 4, 5)],
        vec![(4, 11, 7), (4, 9, 11), (9, 10, 11)],
        vec![(0, 8, 3), (4, 9, 7), (9, 11, 7), (9, 10, 11)],
        vec![(1, 10, 11), (1, 11, 4), (1, 4, 0), (7, 4, 11)],
        vec![(3, 1, 4), (3, 4, 8), (1, 10, 4), (7, 4, 11), (10, 11, 4)],
        vec![(4, 11, 7), (9, 11, 4), (9, 2, 11), (9, 1, 2)],
        vec![(9, 7, 4), (9, 11, 7), (9, 1, 11), (2, 11, 1), (0, 8, 3)],
        vec![(11, 7, 4), (11, 4, 2), (2, 4, 0)],
        vec![(11, 7, 4), (11, 4, 2), (8, 3, 4), (3, 2, 4)],
        vec![(2, 9, 10), (2, 7, 9), (2, 3, 7), (7, 4, 9)],
        vec![(9, 10, 7), (9, 7, 4), (10, 2, 7), (8, 7, 0), (2, 0, 7)],
        vec![(3, 7, 10), (3, 10, 2), (7, 4, 10), (1, 10, 0), (4, 0, 10)],
        vec![(1, 10, 2), (8, 7, 4)],
        vec![(4, 9, 1), (4, 1, 7), (7, 1, 3)],
        vec![(4, 9, 1), (4, 1, 7), (0, 8, 1), (8, 7, 1)],
        vec![(4, 0, 3), (7, 4, 3)],
        vec![(4, 8, 7)],
        vec![(9, 10, 8), (10, 11, 8)],
        vec![(3, 0, 9), (3, 9, 11), (11, 9, 10)],
        vec![(0, 1, 10), (0, 10, 8), (8, 10, 11)],
        vec![(3, 1, 10), (11, 3, 10)],
        vec![(1, 2, 11), (1, 11, 9), (9, 11, 8)],
        vec![(3, 0, 9), (3, 9, 11), (1, 2, 9), (2, 11, 9)],
        vec![(0, 2, 11), (8, 0, 11)],
        vec![(3, 2, 11)],
        vec![(2, 3, 8), (2, 8, 10), (10, 8, 9)],
        vec![(9, 10, 2), (0, 9, 2)],
        vec![(2, 3, 8), (2, 8, 10), (0, 1, 8), (1, 10, 8)],
        vec![(1, 10, 2)],
        vec![(1, 3, 8), (9, 1, 8)],
        vec![(0, 9, 1)],
        vec![(0, 3, 8)],
        vec![],
    ]
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

/// Computes the center of a voxel in worlds space cartesian units from absolute
/// voxel coordinates (relative to the voxel space origin start).
///
/// # Panics
///
/// Panics if any of the voxel dimensions is equal or below zero.
fn absolute_voxel_to_cartesian_coordinate(
    absolute_coordinate: &Point3<i32>,
    voxel_dimensions: &Vector3<f32>,
) -> Point3<f32> {
    assert!(
        voxel_dimensions.x > 0.0 && voxel_dimensions.y > 0.0 && voxel_dimensions.z > 0.0,
        "Voxel dimensions can't be below or equal to zero"
    );
    Point3::new(
        absolute_coordinate.x as f32 * voxel_dimensions.x,
        absolute_coordinate.y as f32 * voxel_dimensions.y,
        absolute_coordinate.z as f32 * voxel_dimensions.z,
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

    #[test]
    fn test_scalar_field_boolean_intersection_all_volume() {
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

        sf_a.fill_with(Some(0.0));
        sf_b.fill_with(Some(0.0));
        sf_correct.fill_with(Some(0.0));

        sf_a.boolean_intersection(&(0.0..=0.0), &sf_b, &(0.0..=0.0));

        assert_eq!(sf_a, sf_correct);
    }

    #[test]
    fn test_scalar_field_boolean_intersection_one_void() {
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

        sf_a.fill_with(Some(0.0));
        sf_b.fill_with(Some(0.0));
        sf_b.set_value_at_absolute_voxel_coordinate(&Point3::new(2, 2, 2), None);
        sf_correct.fill_with(Some(0.0));
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(2, 2, 2), None);

        sf_a.boolean_intersection(&(0.0..=0.0), &sf_b, &(0.0..=0.0));

        assert_eq!(sf_a, sf_correct);
    }

    #[test]
    fn test_scalar_field_boolean_union_shifted() {
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

        sf_a.fill_with(Some(0.0));
        sf_b.fill_with(Some(0.0));
        sf_b.set_value_at_absolute_voxel_coordinate(&Point3::new(2, 2, 2), None);
        sf_correct.fill_with(Some(0.0));
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(3, 0, 0), None);
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(3, 1, 0), None);
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(3, 2, 0), None);
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(3, 3, 0), None);
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(3, 0, 1), None);
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(3, 0, 2), None);
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(3, 0, 3), None);
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(2, 0, 3), None);
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(1, 0, 3), None);
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(0, 0, 3), None);
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(0, 1, 3), None);
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(0, 2, 3), None);
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(0, 3, 3), None);
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(0, 3, 2), None);
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(0, 3, 1), None);
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(0, 3, 0), None);
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(1, 3, 0), None);
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(2, 3, 0), None);

        sf_a.boolean_union(&(0.0..=0.0), &sf_b, &(0.0..=0.0));

        assert_eq!(sf_a, sf_correct);
    }

    #[test]
    fn test_scalar_field_resize_zero_to_nonzero_all_void() {
        let mut scalar_field: ScalarField = ScalarField::new(
            &Point3::origin(),
            &Vector3::zeros(),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        scalar_field.resize(&Point3::origin(), &Vector3::new(1, 1, 1));

        let voxel = scalar_field.value_at_absolute_voxel_coordinate(&Point3::new(0, 0, 0));

        assert_eq!(voxel, None);
    }

    #[test]
    fn test_scalar_field_resize_zero_to_nonzero_correct_start_and_dimensions() {
        let mut scalar_field: ScalarField = ScalarField::new(
            &Point3::origin(),
            &Vector3::zeros(),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        let new_origin = Point3::new(1, 2, 3);
        let new_block_dimensions = Vector3::new(4, 5, 6);
        scalar_field.resize(&new_origin, &new_block_dimensions);

        assert_eq!(scalar_field.block_start, new_origin);
        assert_eq!(scalar_field.block_dimensions, new_block_dimensions);
        assert_eq!(scalar_field.voxels.len(), 4 * 5 * 6);
    }

    #[test]
    fn test_scalar_field_resize_nonzero_to_zero_correct_start_and_dimensions() {
        let mut scalar_field: ScalarField = ScalarField::new(
            &Point3::new(1, 2, 3),
            &Vector3::new(4, 5, 6),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        let new_origin = Point3::origin();
        let new_block_dimensions = Vector3::zeros();
        scalar_field.resize(&new_origin, &new_block_dimensions);

        assert_eq!(scalar_field.block_start, new_origin);
        assert_eq!(scalar_field.block_dimensions, new_block_dimensions);
        assert_eq!(scalar_field.voxels.len(), 0);
    }

    #[test]
    fn test_scalar_field_resize_nonzero_to_smaller_nonzero_correct_start_and_dimensions() {
        let mut scalar_field: ScalarField = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(4, 5, 6),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        let new_origin = Point3::new(1, 2, 3);
        let new_block_dimensions = Vector3::new(1, 2, 3);
        scalar_field.resize(&new_origin, &new_block_dimensions);

        assert_eq!(scalar_field.block_start, new_origin);
        assert_eq!(scalar_field.block_dimensions, new_block_dimensions);
        assert_eq!(scalar_field.voxels.len(), 1 * 2 * 3);
    }

    #[test]
    fn test_scalar_field_resize_nonzero_to_larger_nonzero_correct_start_and_dimensions() {
        let mut scalar_field: ScalarField = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(1, 2, 3),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        let new_origin = Point3::new(1, 2, 3);
        let new_block_dimensions = Vector3::new(4, 5, 6);
        scalar_field.resize(&new_origin, &new_block_dimensions);

        assert_eq!(scalar_field.block_start, new_origin);
        assert_eq!(scalar_field.block_dimensions, new_block_dimensions);
        assert_eq!(scalar_field.voxels.len(), 4 * 5 * 6);
    }

    #[test]
    fn test_scalar_field_resize_nonzero_to_larger_nonzero_grown_contains_none_rest_original() {
        let original_origin = Point3::new(0i32, 0i32, 0i32);
        let original_block_dimensions = Vector3::new(1u32, 10u32, 3u32);
        let mut scalar_field: ScalarField = ScalarField::new(
            &original_origin,
            &original_block_dimensions,
            &Vector3::new(1.0, 1.0, 1.0),
        );
        let original_block_end = scalar_field.block_end();

        scalar_field.fill_with(Some(0.0));

        let new_origin = Point3::new(-1, 2, 3);
        let new_block_dimensions = Vector3::new(4, 5, 6);
        scalar_field.resize(&new_origin, &new_block_dimensions);

        for (i, v) in scalar_field.voxels.iter().enumerate() {
            let coordinate = one_dimensional_to_absolute_voxel_coordinate(
                i,
                &scalar_field.block_start,
                &scalar_field.block_dimensions,
            );

            if coordinate.x < original_origin.x
                || coordinate.y < original_origin.y
                || coordinate.z < original_origin.z
                || coordinate.x > original_block_end.x
                || coordinate.y > original_block_end.y
                || coordinate.z > original_block_end.z
            {
                assert_eq!(*v, None);
            } else {
                let value = v.unwrap();
                assert_eq!(value, 0.0);
            }
        }
    }

    #[test]
    fn test_scalar_field_shrink_to_volume() {
        let mut scalar_field: ScalarField = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(4, 5, 6),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        scalar_field.set_value_at_absolute_voxel_coordinate(&Point3::new(1, 1, 1), Some(0.0));
        scalar_field.shrink_to_fit(&(0.0..=0.0));

        assert_eq!(scalar_field.block_start, Point3::new(1, 1, 1));
        assert_eq!(scalar_field.block_dimensions, Vector3::new(1, 1, 1));
        assert_eq!(scalar_field.voxels.len(), 1);
        assert_eq!(
            scalar_field
                .value_at_absolute_voxel_coordinate(&Point3::new(1, 1, 1))
                .unwrap(),
            0.0
        );
    }

    #[test]
    fn test_scalar_field_shrink_to_empty() {
        let mut scalar_field: ScalarField = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(4, 5, 6),
            &Vector3::new(1.0, 1.0, 1.0),
        );
        scalar_field.shrink_to_fit(&(0.0..=0.0));

        assert_eq!(scalar_field.block_start, Point3::origin());
        assert_eq!(scalar_field.block_dimensions, Vector3::new(0, 0, 0));
        assert_eq!(scalar_field.voxels.len(), 0);
    }

    #[test]
    fn test_scalar_field_boolean_difference_no_change() {
        let mut sf_a = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(3, 3, 3),
            &Vector3::new(0.5, 0.5, 0.5),
        );
        let mut sf_b = ScalarField::new(
            &Point3::new(4, 4, 4),
            &Vector3::new(1, 1, 1),
            &Vector3::new(0.5, 0.5, 0.5),
        );
        let mut sf_correct = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(3, 3, 3),
            &Vector3::new(0.5, 0.5, 0.5),
        );

        sf_a.fill_with(Some(0.0));
        sf_b.fill_with(Some(0.0));
        sf_correct.fill_with(Some(0.0));

        sf_a.boolean_difference(&(0.0..=0.0), &sf_b, &(0.0..=0.0));

        assert_eq!(sf_a, sf_correct);
    }

    #[test]
    fn test_scalar_field_boolean_difference_empty() {
        let mut sf_a = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(3, 3, 3),
            &Vector3::new(0.5, 0.5, 0.5),
        );
        let mut sf_b = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(4, 5, 6),
            &Vector3::new(0.5, 0.5, 0.5),
        );
        let sf_correct = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(0, 0, 0),
            &Vector3::new(0.5, 0.5, 0.5),
        );

        sf_a.fill_with(Some(0.0));
        sf_b.fill_with(Some(0.0));

        sf_a.boolean_difference(&(0.0..=0.0), &sf_b, &(0.0..=0.0));

        assert_eq!(sf_a, sf_correct);
    }

    #[test]
    fn test_scalar_field_boolean_difference_chop_off_corner() {
        let mut sf_a = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(3, 3, 3),
            &Vector3::new(0.5, 0.5, 0.5),
        );
        let mut sf_b = ScalarField::new(
            &Point3::new(2, 2, 2),
            &Vector3::new(2, 3, 4),
            &Vector3::new(0.5, 0.5, 0.5),
        );
        let mut sf_correct = ScalarField::new(
            &Point3::origin(),
            &Vector3::new(3, 3, 3),
            &Vector3::new(0.5, 0.5, 0.5),
        );

        sf_a.fill_with(Some(0.0));
        sf_b.fill_with(Some(0.0));
        sf_correct.fill_with(Some(0.0));
        sf_correct.set_value_at_absolute_voxel_coordinate(&Point3::new(2, 2, 2), None);

        sf_a.boolean_difference(&(0.0..=0.0), &sf_b, &(0.0..=0.0));

        assert_eq!(sf_a, sf_correct);
    }
}
