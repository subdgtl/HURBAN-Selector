use std::iter;

use nalgebra::{Point3, Vector2, Vector3};

use crate::convert::{cast_i32, cast_u32, cast_usize, clamp_cast_i32_to_u32};
use crate::geometry;
use crate::mesh::analysis::BoundingBox;
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
#[derive(Debug, Clone, serde::Serialize)]
pub struct VoxelCloud {
    block_start: Point3<i32>,
    block_dimensions: Vector3<u32>,
    voxel_dimensions: Vector3<f32>,
    // FIXME: @Optimization Change this to a bit vector
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

    /// Define a new empty voxel space from a bounding box.
    pub fn from_bounding_box(bounding_box: &BoundingBox, voxel_dimensions: &Vector3<f32>) -> Self {
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

    /// Calculate a voxel cloud from an existing mesh.
    pub fn from_mesh(mesh: &Mesh, voxel_dimensions: &Vector3<f32>) -> Self {
        // Determine the needed block of voxel space.
        let b_box = BoundingBox::from_meshes(iter::once(mesh));

        let mut voxel_cloud = VoxelCloud::from_bounding_box(&b_box, voxel_dimensions);

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
                    // Calculate the density of points on the respective face
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
                                // Calculate point position in model space
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

    /// Returns voxel cloud block start in absolute voxel coordinates
    #[allow(dead_code)]
    pub fn block_start(&self) -> Point3<i32> {
        self.block_start
    }

    /// Returns voxel cloud block end in absolute voxel coordinates
    pub fn block_end(&self) -> Point3<i32> {
        calculate_block_end(&self.block_start, &self.block_dimensions)
    }

    /// Returns voxel cloud block dimensions in voxel units
    #[allow(dead_code)]
    pub fn block_dimensions(&self) -> &Vector3<u32> {
        &self.block_dimensions
    }

    /// Returns single voxel dimensions in model space units
    pub fn voxel_dimensions(&self) -> &Vector3<f32> {
        &self.voxel_dimensions
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
        for v in self.voxel_map.iter_mut() {
            *v = false;
        }
        self.voxel_map.resize(grown_voxel_map_len, false);

        for grown_index in 0..self.voxel_map.len() {
            let absolute_coords = one_dimensional_to_absolute_three_dimensional_coordinate(
                grown_index,
                &grown_block_start,
                &grown_block_dimensions,
            )
            .expect("Index out of bounds");

            if let Some(original_index) = absolute_three_dimensional_coordinate_to_one_dimensional(
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

    /// Gets the state of a voxel defined in voxel coordinates relative to the
    /// voxel block start.
    pub fn voxel_at_relative_coords(&self, relative_coords: &Point3<i32>) -> Option<bool> {
        relative_three_dimensional_coordinate_to_one_dimensional(
            relative_coords,
            &self.block_dimensions,
        )
        .map(|index| self.voxel_map[index])
    }

    /// Gets the state of a voxel defined in absolute voxel coordinates
    /// (relative to the voxel space origin).
    pub fn voxel_at_absolute_coords(&self, absolute_coords: &Point3<i32>) -> Option<bool> {
        absolute_three_dimensional_coordinate_to_one_dimensional(
            absolute_coords,
            &self.block_start,
            &self.block_dimensions,
        )
        .map(|index| self.voxel_map[index])
    }

    /// Gets the state of a voxel containing the input point defined in model
    /// space coordinates.
    #[allow(dead_code)]
    pub fn voxel_at_cartesian_coords(&self, point: &Point3<f32>) -> Option<bool> {
        let voxel_coords = cartesian_to_absolute_voxel_coords(point, &self.voxel_dimensions);
        self.voxel_at_absolute_coords(&voxel_coords)
    }

    /// Sets the state of a voxel defined in voxel coordinates relative to the
    /// voxel block start.
    #[allow(dead_code)]
    pub fn set_voxel_at_relative_coords(&mut self, relative_coords: &Point3<i32>, state: bool) {
        let index = relative_three_dimensional_coordinate_to_one_dimensional(
            relative_coords,
            &self.block_dimensions,
        )
        .expect("Coordinates out of bounds");
        self.voxel_map[cast_usize(index)] = state;
    }

    /// Sets the state of a voxel defined in absolute voxel coordinates
    /// (relative to the voxel space origin).
    pub fn set_voxel_at_absolute_coords(&mut self, absolute_coords: &Point3<i32>, state: bool) {
        let index = absolute_three_dimensional_coordinate_to_one_dimensional(
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
        let voxel_coords = cartesian_to_absolute_voxel_coords(point, &self.voxel_dimensions);
        self.set_voxel_at_absolute_coords(&voxel_coords, state);
    }

    /// Resize the voxel cloud block to match new block start and block dimensions.
    ///
    /// This clips the outstanding parts of the original voxel cloud.

    // FIXME: Add tests
    // resize from zero to nonzero dimensions
    //  - should initially be all false
    //  - should have correct block dimensions and underlying voxel count
    // resize from nonzero to zero dimensions
    //  - should have correct block dimensions and underlying voxel count
    // resize from between two nonzero dimensions
    //  - should have correct block dimensions and underlying voxel count
    //  - newly grown area should contain false voxels (if applicable)
    //  - old voxels should still be present (if applicable)
    //  - outstanding voxels should be clipped (if applicable)
    pub fn resize(
        &mut self,
        resized_block_start: &Point3<i32>,
        resized_block_dimensions: &Vector3<u32>,
    ) {
        let original_voxel_map = self.voxel_map.clone();
        let original_block_start = self.block_start;
        let original_block_dimensions = self.block_dimensions;

        self.block_start = *resized_block_start;
        self.block_dimensions = *resized_block_dimensions;

        let resized_voxel_map_len = cast_usize(
            resized_block_dimensions.x * resized_block_dimensions.y * resized_block_dimensions.z,
        );

        self.voxel_map.resize(resized_voxel_map_len, false);

        for resized_index in 0..self.voxel_map.len() {
            let absolute_coords = one_dimensional_to_absolute_three_dimensional_coordinate(
                resized_index,
                resized_block_start,
                resized_block_dimensions,
            )
            .expect("Index out of bounds");

            self.voxel_map[resized_index] =
                match absolute_three_dimensional_coordinate_to_one_dimensional(
                    &absolute_coords,
                    &original_block_start,
                    &original_block_dimensions,
                ) {
                    Some(original_index) => original_voxel_map[original_index],
                    _ => false,
                }
        }
    }

    /// Computes boundaries of volumes contained in voxel cloud. Returns tuple
    /// (block_start, block_dimensions). For empty voxel clouds returns the
    /// original block start and zero block dimensions.
    pub fn compute_volume_boundaries(&self) -> (Point3<i32>, Vector3<u32>) {
        let mut min: Vector3<i32> =
            Vector3::new(i32::max_value(), i32::max_value(), i32::max_value());
        let mut max: Vector3<i32> =
            Vector3::new(i32::min_value(), i32::min_value(), i32::min_value());
        for i in 0..self.voxel_map.len() {
            if self.voxel_map[i] {
                let relative_coords = one_dimensional_to_relative_three_dimensional_coordinate(
                    i,
                    &self.block_dimensions,
                )
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
        let (block_start, block_dimensions) =
        // It's enough to check one of the values because if anything is found,
        // all the values would change.
        if min.x == i32::max_value() {
            (self.block_start, Vector3::zeros())
        } else {
            (
                self.block_start + min,
                Vector3::new(
                    clamp_cast_i32_to_u32(max.x - min.x),
                    clamp_cast_i32_to_u32(max.y - min.y),
                    clamp_cast_i32_to_u32(max.z - min.z),
                )
            )
        };
        (block_start, block_dimensions)
    }

    /// Resize the voxel cloud block to exactly fit the volumetric geometry.
    /// Returns None for empty the voxel cloud.
    #[allow(dead_code)]
    pub fn shrink_to_fit(&mut self) {
        let (shrunk_block_start, shrunk_block_dimensions) = self.compute_volume_boundaries();
        self.resize(&shrunk_block_start, &shrunk_block_dimensions);
    }

    /// Calculates a simple triangulated welded mesh from the current state of
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

        // Pre-calculated geometry helpers
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
                let voxel_coords = one_dimensional_to_relative_three_dimensional_coordinate(
                    one_dimensional_coord,
                    &self.block_dimensions,
                )
                .expect("Out of bounds");
                // calculate the position of its center in model space coordinates
                let voxel_center = relative_voxel_to_cartesian_coords(
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

    #[allow(dead_code)]
    pub fn calculate_bounding_box(&self) -> BoundingBox {
        let voxel_dimensions = self.voxel_dimensions();
        let block_start = self.block_start;
        let block_end = self.block_end();
        BoundingBox::new(
            Point3::new(
                block_start.x as f32 * voxel_dimensions.x,
                block_start.y as f32 * voxel_dimensions.y,
                block_start.z as f32 * voxel_dimensions.z,
            ),
            Point3::new(
                block_end.x as f32 * voxel_dimensions.x,
                block_end.y as f32 * voxel_dimensions.y,
                block_end.z as f32 * voxel_dimensions.z,
            ),
        )
    }

    #[allow(dead_code)]
    pub fn calculate_bounding_box_of_contained_volume(&self) -> BoundingBox {
        let voxel_dimensions = self.voxel_dimensions();
        let (volume_start, volume_dimensions) = self.compute_volume_boundaries();
        BoundingBox::new(
            Point3::new(
                (volume_start.x as f32 - 0.5) * voxel_dimensions.x,
                (volume_start.y as f32 - 0.5) * voxel_dimensions.y,
                (volume_start.z as f32 - 0.5) * voxel_dimensions.z,
            ),
            Point3::new(
                (volume_start.x as f32 + volume_dimensions.x as f32 + 0.5) * voxel_dimensions.x,
                (volume_start.y as f32 + volume_dimensions.x as f32 + 0.5) * voxel_dimensions.y,
                (volume_start.z as f32 + volume_dimensions.x as f32 + 0.5) * voxel_dimensions.z,
            ),
        )
    }
}

/// Computes an index to the linear representation of the voxel block from
/// voxel coordinates relative to the voxel space block start.
///
/// Returns None if out of bounds.
fn relative_three_dimensional_coordinate_to_one_dimensional(
    relative_coords: &Point3<i32>,
    block_dimensions: &Vector3<u32>,
) -> Option<usize> {
    if relative_coords
        .iter()
        .enumerate()
        .all(|(i, coord)| *coord >= 0 && *coord < cast_i32(block_dimensions[i]))
    {
        let index = relative_coords.z * cast_i32(block_dimensions.x) * cast_i32(block_dimensions.y)
            + relative_coords.y * cast_i32(block_dimensions.x)
            + relative_coords.x;
        Some(cast_usize(index))
    } else {
        None
    }
}

/// Computes a voxel position relative to the block start (relative
/// coordinate) from an index to the linear representation of the voxel
/// block.
///
/// Returns None if out of bounds.
fn one_dimensional_to_relative_three_dimensional_coordinate(
    one_dimensional: usize,
    block_dimensions: &Vector3<u32>,
) -> Option<Point3<i32>> {
    let voxel_map_len = cast_usize(block_dimensions.x * block_dimensions.y * block_dimensions.z);
    if one_dimensional < voxel_map_len {
        let one_dimensional_i32 = cast_i32(one_dimensional);
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

/// Gets the index to the voxel map from absolute voxel coordinates
/// (relative to the voxel space origin).
///
/// Returns None if out of bounds.
fn absolute_three_dimensional_coordinate_to_one_dimensional(
    absolute_coords: &Point3<i32>,
    block_start: &Point3<i32>,
    block_dimensions: &Vector3<u32>,
) -> Option<usize> {
    let relative_coords = absolute_coords - block_start.coords;
    relative_three_dimensional_coordinate_to_one_dimensional(&relative_coords, block_dimensions)
}

/// Computes a voxel position relative to the model space origin (absolute
/// coordinate) from an index to the linear representation of the voxel
/// block.
///
/// Returns None if out of bounds.
fn one_dimensional_to_absolute_three_dimensional_coordinate(
    one_dimensional: usize,
    block_start: &Point3<i32>,
    block_dimensions: &Vector3<u32>,
) -> Option<Point3<i32>> {
    one_dimensional_to_relative_three_dimensional_coordinate(one_dimensional, block_dimensions)
        .map(|relative| relative + block_start.coords)
}

/// Calculates the voxel-space coordinates of a voxel containing the input
/// point.
fn cartesian_to_absolute_voxel_coords(
    point: &Point3<f32>,
    voxel_dimensions: &Vector3<f32>,
) -> Point3<i32> {
    assert!(
        !approx::relative_eq!(voxel_dimensions.x, 0.0)
            && !approx::relative_eq!(voxel_dimensions.y, 0.0)
            && !approx::relative_eq!(voxel_dimensions.z, 0.0),
        "Voxel dimensions can't be 0.0"
    );
    Point3::new(
        (point.x / voxel_dimensions.x).round() as i32,
        (point.y / voxel_dimensions.y).round() as i32,
        (point.z / voxel_dimensions.z).round() as i32,
    )
}

/// Calculates the voxel-space coordinates of a voxel containing the input
/// point.
#[allow(dead_code)]
fn cartesian_to_relative_voxel_coords(
    point: &Point3<f32>,
    block_start: &Point3<i32>,
    voxel_dimensions: Vector3<f32>,
) -> Point3<i32> {
    assert!(
        !approx::relative_eq!(voxel_dimensions.x, 0.0)
            && !approx::relative_eq!(voxel_dimensions.y, 0.0)
            && !approx::relative_eq!(voxel_dimensions.z, 0.0),
        "Voxel dimensions can't be 0.0"
    );
    Point3::new(
        (point.x / voxel_dimensions.x).round() as i32 - block_start.x,
        (point.y / voxel_dimensions.y).round() as i32 - block_start.y,
        (point.z / voxel_dimensions.z).round() as i32 - block_start.z,
    )
}

/// Calculates the center of a voxel in model-space coordinates from
/// absolute voxel coordinates (relative to the voxel space origin).
#[allow(dead_code)]
fn absolute_voxel_to_cartesian_coords(
    absolute_coords: &Point3<i32>,
    voxel_dimensions: Vector3<f32>,
) -> Point3<f32> {
    assert!(
        !approx::relative_eq!(voxel_dimensions.x, 0.0)
            && !approx::relative_eq!(voxel_dimensions.y, 0.0)
            && !approx::relative_eq!(voxel_dimensions.z, 0.0),
        "Voxel dimensions can't be 0.0"
    );
    Point3::new(
        absolute_coords.x as f32 * voxel_dimensions.x,
        absolute_coords.y as f32 * voxel_dimensions.y,
        absolute_coords.z as f32 * voxel_dimensions.z,
    )
}

/// Calculates the center of a voxel in model-space coordinates from voxel
/// coordinates relative to the voxel block start.
fn relative_voxel_to_cartesian_coords(
    relative_coords: &Point3<i32>,
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
        (relative_coords.x + block_start.x) as f32 * voxel_dimensions.x,
        (relative_coords.y + block_start.y) as f32 * voxel_dimensions.y,
        (relative_coords.z + block_start.z) as f32 * voxel_dimensions.z,
    )
}

/// Converts relative voxel coordinates into relative voxel coordinates
#[allow(dead_code)]
fn absolute_to_relative_voxel_coords(
    absolute_coords: &Point3<i32>,
    block_start: &Point3<i32>,
) -> Point3<i32> {
    absolute_coords - block_start.coords
}

/// Converts absolute voxel coordinates into relative voxel coordinates
#[allow(dead_code)]
fn relative_to_absolute_voxel_coords(
    relative_coords: &Point3<i32>,
    block_start: &Point3<i32>,
) -> Point3<i32> {
    relative_coords + block_start.coords
}

/// Calculates the absolute coordinates of the block end
fn calculate_block_end(block_start: &Point3<i32>, block_dimensions: &Vector3<u32>) -> Point3<i32> {
    Point3::new(
        block_start.x + cast_i32(block_dimensions.x) - 1,
        block_start.y + cast_i32(block_dimensions.y) - 1,
        block_start.z + cast_i32(block_dimensions.z) - 1,
    )
}

#[cfg(test)]
mod tests {
    use nalgebra::Rotation3;

    use crate::mesh::{analysis, topology, NormalStrategy};

    use super::*;

    // FIXME: Snapshot testing
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
                    let one_dimensional = relative_three_dimensional_coordinate_to_one_dimensional(
                        &relative_position,
                        &voxel_cloud.block_dimensions(),
                    )
                    .unwrap();
                    let three_dimensional =
                        one_dimensional_to_relative_three_dimensional_coordinate(
                            one_dimensional,
                            &voxel_cloud.block_dimensions(),
                        )
                        .unwrap();
                    assert_eq!(relative_position, three_dimensional);
                }
            }
        }
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
}
