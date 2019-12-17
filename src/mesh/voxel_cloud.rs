use std::f32;
use std::iter;

use nalgebra::{Point3, Vector2, Vector3};

use crate::convert::{cast_i32, cast_usize};
use crate::geometry;
use crate::plane::Plane;

use super::{analysis::BoundingBox, primitive, tools, Face, Mesh};

/// Option<bool> cloud is an abstract representation of points in a block of
/// space. The block is delimited by its beginning and its dimensions, both in
/// the units of the voxels. All voxels have the same dimensions, which can be
/// different in each direction.
///
/// The voxel space is discrete and you can't start it half way in a voxel,
/// therefore its beginning as well as the voxel positions are defined in the
/// voxel-space coordinates. The voxel space starts at the cartesian space
/// origin with voxel coordinates 0, 0, 0. Voxel clouds with the same voxel size
/// are compatible and collateral operations be performed on them.
#[derive(Debug, Clone)]
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
            block_dimensions.x > 0 && block_dimensions.y > 0 && block_dimensions.z > 0,
            "One or more block dimensions are 0"
        );
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

    /// Calculate a voxel cloud from an existing mesh.
    pub fn from_mesh(mesh: &Mesh, voxel_dimensions: &Vector3<f32>) -> Self {
        // Determine the needed block of voxel space.
        let b_box = BoundingBox::from_meshes(iter::once(mesh));

        let min_point = &b_box.minimum_point();
        let max_point = &b_box.maximum_point();
        let min_x_index = (min_point.x.min(max_point.x) / voxel_dimensions.x).floor();
        let max_x_index = (min_point.x.max(max_point.x) / voxel_dimensions.x).ceil();
        let min_y_index = (min_point.y.min(max_point.y) / voxel_dimensions.y).floor();
        let max_y_index = (min_point.y.max(max_point.y) / voxel_dimensions.y).ceil();
        let min_z_index = (min_point.z.min(max_point.z) / voxel_dimensions.z).floor();
        let max_z_index = (min_point.z.max(max_point.z) / voxel_dimensions.z).ceil();

        let block_start = Point3::new(
            min_x_index as i32 - 1,
            min_y_index as i32 - 1,
            min_z_index as i32 - 1,
        );
        let block_dimensions = Vector3::new(
            (max_x_index - min_x_index + 2.0) as u32,
            (max_y_index - min_y_index + 2.0) as u32,
            (max_z_index - min_z_index + 2.0) as u32,
        );

        // Going to populate the mesh with points as dense as the smallest voxel dimension.
        let shortest_voxel_dimension = voxel_dimensions
            .x
            .min(voxel_dimensions.y.min(voxel_dimensions.z));

        let mut voxel_cloud = VoxelCloud::new(&block_start, &block_dimensions, voxel_dimensions);

        // Iterate through mesh faces.
        for face in mesh.faces() {
            match face {
                Face::Triangle(f) => {
                    let a = &mesh.vertices()[cast_usize(f.vertices.0)];
                    let b = &mesh.vertices()[cast_usize(f.vertices.1)];
                    let c = &mesh.vertices()[cast_usize(f.vertices.2)];
                    // Calculate the density of points on the respective face
                    let ab_distance_sq = nalgebra::distance_squared(a, b);
                    let bc_distance_sq = nalgebra::distance_squared(b, c);
                    let ca_distance_sq = nalgebra::distance_squared(c, a);
                    let longest_edge_len = ab_distance_sq
                        .max(bc_distance_sq.max(ca_distance_sq))
                        .sqrt();
                    // Number of face divisions (points) in each direction
                    let divisions = (longest_edge_len / shortest_voxel_dimension).ceil() as usize;
                    let divisions_f32 = divisions as f32;

                    for ui in 0..=divisions {
                        for wi in 0..=divisions {
                            let u = ui as f32 / divisions_f32;
                            let w = wi as f32 / divisions_f32;
                            let v = 1.0 - u - w;
                            if v >= 0.0 {
                                let barycentric = Point3::new(u, v, w);
                                // Calculate point position in model space
                                let cartesian =
                                    geometry::barycentric_to_cartesian(&barycentric, &a, &b, &c);
                                // and set_voxel_at_absolute_coords a voxel containing the point to be on
                                voxel_cloud.set_voxel_at_cartesian_coords(true, &cartesian);
                            }
                        }
                    }
                }
            }
        }

        voxel_cloud
    }

    /// For each existing voxel turn on all neighbor voxels to grow (offset) the
    /// volumes stored in the voxel cloud.
    pub fn grow(&self) -> Self {
        // If the voxels in the existing voxel cloud reach the boundaries of the
        // block, it's needed to grow the block by 1 in each direction.
        let new_block_start = self.block_start - Vector3::new(-1, -1, -1);

        let new_block_dimensions = self.block_dimensions + Vector3::new(2, 2, 2);
        let mut voxel_cloud = VoxelCloud::new(
            &new_block_start,
            &new_block_dimensions,
            &self.voxel_dimensions,
        );

        // Iterate through the existing voxel cloud.
        for z in 0..self.block_dimensions.z {
            for y in 0..self.block_dimensions.y {
                for x in 0..self.block_dimensions.x {
                    let voxel_coords = Point3::new(cast_i32(x), cast_i32(y), cast_i32(z));
                    let voxel_state = self.voxel_at_relative_coords(&voxel_coords);
                    // if the voxel is on
                    if let Some(true) = voxel_state {
                        // set_voxel_at_absolute_coords it to be on also in the new voxel cloud
                        // (everything is shifted by 1, 1, 1 because the start
                        // is shifted -1, -1, -1)
                        voxel_cloud.set_voxel_at_relative_coords(
                            true,
                            &(voxel_coords + Vector3::new(1, 1, 1)),
                        );
                        // as well as all its neighbors
                        voxel_cloud.set_voxel_at_relative_coords(
                            true,
                            &(voxel_coords + Vector3::new(0, 1, 1)),
                        );
                        voxel_cloud.set_voxel_at_relative_coords(
                            true,
                            &(voxel_coords + Vector3::new(2, 1, 1)),
                        );
                        voxel_cloud.set_voxel_at_relative_coords(
                            true,
                            &(voxel_coords + Vector3::new(1, 0, 1)),
                        );
                        voxel_cloud.set_voxel_at_relative_coords(
                            true,
                            &(voxel_coords + Vector3::new(1, 2, 1)),
                        );
                        voxel_cloud.set_voxel_at_relative_coords(
                            true,
                            &(voxel_coords + Vector3::new(1, 1, 0)),
                        );
                        voxel_cloud.set_voxel_at_relative_coords(
                            true,
                            &(voxel_coords + Vector3::new(1, 1, 2)),
                        );
                    }
                }
            }
        }

        voxel_cloud
    }

    /// Gets the state of a voxel defined in voxel coordinates relative to the
    /// voxel block start.
    pub fn voxel_at_relative_coords(&self, relative_coords: &Point3<i32>) -> Option<bool> {
        if relative_coords.x < 0
            || relative_coords.y < 0
            || relative_coords.z < 0
            || relative_coords.x >= cast_i32(self.block_dimensions.x)
            || relative_coords.y >= cast_i32(self.block_dimensions.y)
            || relative_coords.z >= cast_i32(self.block_dimensions.z)
        {
            None
        } else {
            Some(
                self.voxel_map[cast_usize(
                    self.relative_three_dimensional_coordinate_to_one_dimensional(relative_coords),
                )],
            )
        }
    }

    /// Gets the state of a voxel defined in absolute voxel coordinates
    /// (relative to the voxel space origin).
    pub fn voxel_at_absolute_coords(&self, absolute_coords: &Point3<i32>) -> Option<bool> {
        if absolute_coords.x >= cast_i32(self.block_dimensions.x) + self.block_start.x
            || absolute_coords.y >= cast_i32(self.block_dimensions.y) + self.block_start.y
            || absolute_coords.z >= cast_i32(self.block_dimensions.z) + self.block_start.z
            || absolute_coords.x < self.block_start.x
            || absolute_coords.y < self.block_start.y
            || absolute_coords.z < self.block_start.z
        {
            None
        } else {
            Some(
                self.voxel_map[cast_usize(
                    self.absolute_three_dimensional_coordinate_to_one_dimensional(absolute_coords),
                )],
            )
        }
    }

    /// Gets the state of a voxel containing the input point defined in model
    /// space coordinates.
    #[allow(dead_code)]
    pub fn get_cartesian(&self, point: &Point3<f32>) -> Option<bool> {
        let voxel_coords = self.cartesian_to_absolute_voxel_coords(point);
        self.voxel_at_absolute_coords(&voxel_coords)
    }

    /// Sets the state of a voxel defined in voxel coordinates relative to the
    /// voxel block start.
    pub fn set_voxel_at_relative_coords(&mut self, state: bool, relative_coords: &Point3<i32>) {
        let index = self.relative_three_dimensional_coordinate_to_one_dimensional(relative_coords);
        assert!(
            index >= 0 && index < cast_i32(self.voxel_map.len()),
            "Coordinates out of bounds"
        );
        self.voxel_map[cast_usize(index)] = state;
    }

    /// Sets the state of a voxel defined in absolute voxel coordinates
    /// (relative to the voxel space origin).
    pub fn set_voxel_at_absolute_coords(&mut self, state: bool, absolute_coords: &Point3<i32>) {
        let index = self.absolute_three_dimensional_coordinate_to_one_dimensional(absolute_coords);
        assert!(
            index >= 0 && index < cast_i32(self.voxel_map.len()),
            "Coordinates out of bounds"
        );
        self.voxel_map[cast_usize(index)] = state;
    }

    /// Sets the state of a voxel containing the input point defined in model
    /// space coordinates.
    pub fn set_voxel_at_cartesian_coords(&mut self, state: bool, point: &Point3<f32>) {
        let voxel_coords = self.cartesian_to_absolute_voxel_coords(point);
        self.set_voxel_at_absolute_coords(state, &voxel_coords)
    }

    /// Calculates a simple triangulated welded mesh from the current state of
    /// the voxel cloud.
    ///
    /// For watertight meshes this creates both, outer and inner boundary mesh.
    /// There is also a high risk of generating a non-manifold mesh if some
    /// voxels touch only diagonally.
    pub fn to_mesh(&self) -> Option<Mesh> {
        // A collection of rectangular meshes (two triangles) defining an outer
        // envelope of volumes stored in the voxel cloud
        let mut plane_meshes: Vec<Mesh> = Vec::new();

        // Pre-calculated geometry helpers
        let top_plane = Plane::new(
            &Point3::origin(),
            &Vector3::new(1.0, 0.0, 0.0),
            &Vector3::new(0.0, 1.0, 0.0),
        );
        let bottom_plane = Plane::new(
            &Point3::origin(),
            &Vector3::new(1.0, 0.0, 0.0),
            &Vector3::new(0.0, -1.0, 0.0),
        );
        let right_plane = Plane::new(
            &Point3::origin(),
            &Vector3::new(0.0, 1.0, 0.0),
            &Vector3::new(0.0, 0.0, 1.0),
        );
        let left_plane = Plane::new(
            &Point3::origin(),
            &Vector3::new(0.0, 1.0, 0.0),
            &Vector3::new(0.0, 0.0, -1.0),
        );
        let front_plane = Plane::new(
            &Point3::origin(),
            &Vector3::new(1.0, 0.0, 0.0),
            &Vector3::new(0.0, 0.0, 1.0),
        );
        let rear_plane = Plane::new(
            &Point3::origin(),
            &Vector3::new(1.0, 0.0, 0.0),
            &Vector3::new(0.0, 0.0, -1.0),
        );
        let right_voxel_direction = Vector3::new(self.voxel_dimensions.x / 2.0, 0.0, 0.0);
        let front_voxel_direction = Vector3::new(0.0, self.voxel_dimensions.y / 2.0, 0.0);
        let upwards_voxel_direction = Vector3::new(0.0, 0.0, self.voxel_dimensions.z / 2.0);
        let top_bottom_voxel_dimensions =
            Vector2::new(self.voxel_dimensions.x, self.voxel_dimensions.y);
        let right_left_voxel_dimensions =
            Vector2::new(self.voxel_dimensions.y, self.voxel_dimensions.z);
        let front_rear_voxel_dimensions =
            Vector2::new(self.voxel_dimensions.x, self.voxel_dimensions.z);

        // Iterate through the voxel cloud
        for z in 0..self.block_dimensions.z {
            for y in 0..self.block_dimensions.y {
                for x in 0..self.block_dimensions.x {
                    // If the current voxel is on
                    let voxel_coords = Point3::new(cast_i32(x), cast_i32(y), cast_i32(z));
                    if let Some(state) = self.voxel_at_relative_coords(&voxel_coords) {
                        if state {
                            // calculate the position of its center in model space coordinates
                            let voxel_center =
                                self.relative_voxel_to_cartesian_coords(&voxel_coords);
                            // top
                            // and check if there is any voxel above it
                            match self
                                .voxel_at_relative_coords(&(voxel_coords + Vector3::new(0, 0, 1)))
                            {
                                // if there isn't or if the current voxel is on the
                                // boundary of the voxel space block
                                Some(false) | None => {
                                    // add a horizontal rectangle
                                    plane_meshes.push(primitive::create_mesh_plane(
                                        Plane::from_origin_and_plane(
                                            // above the voxel center half way the height of the voxel
                                            &(voxel_center + upwards_voxel_direction),
                                            // align it properly
                                            &top_plane,
                                        ),
                                        // and set_voxel_at_absolute_coords its size to match the dimensions
                                        // of the top side of a voxel
                                        top_bottom_voxel_dimensions,
                                    ));
                                }
                                // if there is a neighbor above the current voxel,
                                // it means no boundary side of the voxel box should
                                // be materialized
                                _ => {}
                            }
                            // bottom
                            match self
                                .voxel_at_relative_coords(&(voxel_coords + Vector3::new(0, 0, -1)))
                            {
                                Some(false) | None => {
                                    plane_meshes.push(primitive::create_mesh_plane(
                                        Plane::from_origin_and_plane(
                                            &(voxel_center - upwards_voxel_direction),
                                            &bottom_plane,
                                        ),
                                        top_bottom_voxel_dimensions,
                                    ));
                                }
                                _ => {}
                            }
                            // right
                            match self
                                .voxel_at_relative_coords(&(voxel_coords + Vector3::new(1, 0, 0)))
                            {
                                Some(false) | None => {
                                    plane_meshes.push(primitive::create_mesh_plane(
                                        Plane::from_origin_and_plane(
                                            &(voxel_center + right_voxel_direction),
                                            &right_plane,
                                        ),
                                        right_left_voxel_dimensions,
                                    ));
                                }
                                _ => {}
                            }
                            // left
                            match self
                                .voxel_at_relative_coords(&(voxel_coords + Vector3::new(-1, 0, 0)))
                            {
                                Some(false) | None => {
                                    plane_meshes.push(primitive::create_mesh_plane(
                                        Plane::from_origin_and_plane(
                                            &(voxel_center - right_voxel_direction),
                                            &left_plane,
                                        ),
                                        right_left_voxel_dimensions,
                                    ));
                                }
                                _ => {}
                            }
                            // front
                            match self
                                .voxel_at_relative_coords(&(voxel_coords + Vector3::new(0, 1, 0)))
                            {
                                Some(false) | None => {
                                    plane_meshes.push(primitive::create_mesh_plane(
                                        Plane::from_origin_and_plane(
                                            &(voxel_center + front_voxel_direction),
                                            &front_plane,
                                        ),
                                        front_rear_voxel_dimensions,
                                    ));
                                }
                                _ => {}
                            }
                            // rear
                            match self
                                .voxel_at_relative_coords(&(voxel_coords + Vector3::new(0, -1, 0)))
                            {
                                Some(false) | None => {
                                    plane_meshes.push(primitive::create_mesh_plane(
                                        Plane::from_origin_and_plane(
                                            &(voxel_center - front_voxel_direction),
                                            &rear_plane,
                                        ),
                                        front_rear_voxel_dimensions,
                                    ));
                                }
                                _ => {}
                            }
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
        // and weld naked edges
        tools::weld(&joined_voxel_mesh, (min_voxel_dimension as f32) / 4.0)
    }

    /// Computes a position in the linear representation of the voxel block from
    /// voxel coordinates relative to the voxel space block start.
    ///
    /// The function doesn't clamp the result because it may be used also to
    /// check if the coordinates are out of bounds.
    fn relative_three_dimensional_coordinate_to_one_dimensional(
        &self,
        relative_coords: &Point3<i32>,
    ) -> i32 {
        relative_coords.z * cast_i32(self.block_dimensions.x) * cast_i32(self.block_dimensions.y)
            + relative_coords.y * cast_i32(self.block_dimensions.x)
            + relative_coords.x
    }

    /// Gets the index to the voxel map from absolute voxel coordinates
    /// (relative to the voxel space origin).
    ///
    /// The function doesn't clamp the result because it may be used also to
    /// check if the coordinates are out of bounds.
    fn absolute_three_dimensional_coordinate_to_one_dimensional(
        &self,
        absolute_coords: &Point3<i32>,
    ) -> i32 {
        let relative_coords = absolute_coords - self.block_start.coords;
        self.relative_three_dimensional_coordinate_to_one_dimensional(&relative_coords)
    }

    /// Calculates the voxel-space coordinates of a voxel containing the input
    /// point.
    fn cartesian_to_absolute_voxel_coords(&self, point: &Point3<f32>) -> Point3<i32> {
        Point3::new(
            ((point.x / self.voxel_dimensions.x).round()) as i32,
            ((point.y / self.voxel_dimensions.y).round()) as i32,
            ((point.z / self.voxel_dimensions.z).round()) as i32,
        )
    }

    /// Calculates the voxel-space coordinates of a voxel containing the input
    /// point.
    #[allow(dead_code)]
    fn cartesian_to_relative_voxel_coords(&self, point: &Point3<f32>) -> Point3<i32> {
        Point3::new(
            (point.x / self.voxel_dimensions.x).round() as i32 - self.block_start.x,
            (point.y / self.voxel_dimensions.y).round() as i32 - self.block_start.y,
            (point.z / self.voxel_dimensions.z).round() as i32 - self.block_start.z,
        )
    }

    /// Calculates the center of a voxel in model-space coordinates from
    /// absolute voxel coordinates (relative to the voxel space origin).
    #[allow(dead_code)]
    fn absolute_voxel_to_cartesian_coords(&self, absolute_coords: &Point3<i32>) -> Point3<f32> {
        Point3::new(
            absolute_coords.x as f32 * self.voxel_dimensions.x,
            absolute_coords.y as f32 * self.voxel_dimensions.y,
            absolute_coords.z as f32 * self.voxel_dimensions.z,
        )
    }

    /// Calculates the center of a voxel in model-space coordinates from voxel
    /// coordinates relative to the voxel block start.
    fn relative_voxel_to_cartesian_coords(&self, relative_coords: &Point3<i32>) -> Point3<f32> {
        Point3::new(
            (cast_i32(relative_coords.x) + self.block_start.x) as f32 * self.voxel_dimensions.x,
            (cast_i32(relative_coords.y) + self.block_start.y) as f32 * self.voxel_dimensions.y,
            (cast_i32(relative_coords.z) + self.block_start.z) as f32 * self.voxel_dimensions.z,
        )
    }
}
