use std::collections::HashSet;

use arrayvec::ArrayVec;
use nalgebra as na;
use nalgebra::base::Vector3;
use nalgebra::geometry::Point3;

use crate::convert::{cast_u32, cast_usize};

#[derive(Debug, Clone, Copy)]
pub enum NormalStrategy {
    Sharp,
    // FIXME: add `Smooth`
}

/// Geometric data containing multiple possibly _variable-length_
/// lists of geometric data, such as vertices and normals, and faces -
/// a single list containing the index topology that describes the
/// structure of data in those lists.
///
/// Currently only `Face::Triangle` is supported. It binds vertices
/// and normals in triangular faces. `Face::Triangle` is always
/// ensured to have counter-clockwise winding. Quad or polygonal faces
/// are not supported currently, but might be in the future.
///
/// The geometry data lives in right-handed coordinate space with the
/// XY plane being the ground and Z axis growing upwards.
#[derive(Debug, Clone, PartialEq)]
pub struct Geometry {
    faces: Vec<Face>,
    vertices: Vec<Point3<f32>>,
    normals: Vec<Vector3<f32>>,
}

impl Geometry {
    /// Create new triangle face geometry from provided faces and
    /// vertices, and compute normals based on `normal_strategy`.
    ///
    /// # Panics
    /// Panics if faces refer to out-of-bounds vertices.
    pub fn from_triangle_faces_with_vertices_and_computed_normals(
        faces: Vec<(u32, u32, u32)>,
        vertices: Vec<Point3<f32>>,
        normal_strategy: NormalStrategy,
    ) -> Self {
        // FIXME: orphan removal

        let mut normals = Vec::with_capacity(faces.len());
        let vertices_range = 0..cast_u32(vertices.len());
        for &(v1, v2, v3) in &faces {
            assert!(
                vertices_range.contains(&v1),
                "Faces reference out of bounds position data"
            );
            assert!(
                vertices_range.contains(&v2),
                "Faces reference out of bounds position data"
            );
            assert!(
                vertices_range.contains(&v3),
                "Faces reference out of bounds position data"
            );

            // FIXME: computing smooth normals in the future won't be
            // so simple as just computing a normal per face, we will
            // need to analyze larger parts of the geometry
            let face_normal = match normal_strategy {
                NormalStrategy::Sharp => compute_triangle_normal(
                    &vertices[cast_usize(v1)],
                    &vertices[cast_usize(v2)],
                    &vertices[cast_usize(v3)],
                ),
            };

            normals.push(face_normal);
        }

        assert_eq!(normals.len(), faces.len());
        assert_eq!(normals.capacity(), faces.len());

        Self {
            faces: faces
                .into_iter()
                .enumerate()
                .map(|(i, (i1, i2, i3))| {
                    let normal_index = cast_u32(i);
                    TriangleFace::new_separate(i1, i2, i3, normal_index, normal_index, normal_index)
                })
                .map(Face::from)
                .collect(),
            vertices,
            normals,
        }
    }

    /// Create new triangle face geometry from provided faces,
    /// vertices, and normals.
    ///
    /// # Panics
    /// Panics if faces refer to out-of-bounds vertices or normals.
    pub fn from_triangle_faces_with_vertices_and_normals(
        faces: Vec<TriangleFace>,
        vertices: Vec<Point3<f32>>,
        normals: Vec<Vector3<f32>>,
    ) -> Self {
        // FIXME: orphan removal

        let vertices_range = 0..cast_u32(vertices.len());
        let normals_range = 0..cast_u32(normals.len());
        for face in &faces {
            let v = face.vertices;
            let n = face.normals;
            assert!(
                vertices_range.contains(&v.0),
                "Faces reference out of bounds position data"
            );
            assert!(
                vertices_range.contains(&v.1),
                "Faces reference out of bounds position data"
            );
            assert!(
                vertices_range.contains(&v.2),
                "Faces reference out of bounds position data"
            );
            assert!(
                normals_range.contains(&n.0),
                "Faces reference out of bounds normal data"
            );
            assert!(
                normals_range.contains(&n.1),
                "Faces reference out of bounds normal data"
            );
            assert!(
                normals_range.contains(&n.2),
                "Faces reference out of bounds normal data"
            );
        }

        Self {
            faces: faces.into_iter().map(Face::Triangle).collect(),
            vertices,
            normals,
        }
    }

    /// Return a view of all triangle faces in this geometry. Skip all
    /// other types of faces.
    pub fn triangle_faces_iter<'a>(&'a self) -> impl Iterator<Item = TriangleFace> + 'a {
        self.faces.iter().copied().map(|index| match index {
            Face::Triangle(f) => f,
        })
    }

    /// Return count of all triangle faces in this geometry. Skip all
    /// other types of faces.
    pub fn triangle_faces_len(&self) -> usize {
        self.faces
            .iter()
            .filter(|index| match index {
                Face::Triangle(_) => true,
            })
            .count()
    }

    pub fn vertices(&self) -> &[Point3<f32>] {
        &self.vertices
    }

    pub fn vertices_mut(&mut self) -> &mut [Point3<f32>] {
        &mut self.vertices
    }

    pub fn normals(&self) -> &[Vector3<f32>] {
        &self.normals
    }

    pub fn edges_iter<'a>(&'a self) -> impl Iterator<Item = HalfEdge> + 'a {
        self.triangle_faces_iter()
            .flat_map(|face| ArrayVec::from(face.to_edges()).into_iter())
    }

    pub fn has_no_orphan_vertices(&self) -> bool {
        let mut used_vertices = HashSet::new();
        for face in self.triangle_faces_iter() {
            used_vertices.insert(face.vertices.0);
            used_vertices.insert(face.vertices.1);
            used_vertices.insert(face.vertices.2);
        }
        used_vertices.len() == self.vertices().len()
    }

    pub fn has_no_orphan_normals(&self) -> bool {
        let mut used_normals = HashSet::new();
        for face in self.triangle_faces_iter() {
            used_normals.insert(face.normals.0);
            used_normals.insert(face.normals.1);
            used_normals.insert(face.normals.2);
        }
        used_normals.len() == self.normals().len()
    }
}

/// A geometry index. Describes topology of geometry data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Face {
    Triangle(TriangleFace),
}

impl From<TriangleFace> for Face {
    fn from(triangle_face: TriangleFace) -> Face {
        Face::Triangle(triangle_face)
    }
}

/// A triangular face. Contains indices to other geometry data, such
/// as vertices and normals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TriangleFace {
    pub vertices: (u32, u32, u32),
    pub normals: (u32, u32, u32),
}

impl TriangleFace {
    pub fn new(i1: u32, i2: u32, i3: u32) -> TriangleFace {
        assert!(
            i1 != i2 && i1 != i3 && i2 != i3,
            "One or more face edges consists of the same vertex"
        );
        TriangleFace {
            vertices: (i1, i2, i3),
            normals: (i1, i2, i3),
        }
    }

    pub fn new_separate(
        vi1: u32,
        vi2: u32,
        vi3: u32,
        ni1: u32,
        ni2: u32,
        ni3: u32,
    ) -> TriangleFace {
        assert!(
            vi1 != vi2 && vi1 != vi3 && vi2 != vi3,
            "One or more face edges consists of the same vertex"
        );
        TriangleFace {
            vertices: (vi1, vi2, vi3),
            normals: (ni1, ni2, ni3),
        }
    }

    /// Generates 3 edges from the respective triangular face
    pub fn to_edges(&self) -> [HalfEdge; 3] {
        [
            HalfEdge::new(self.vertices.0, self.vertices.1),
            HalfEdge::new(self.vertices.1, self.vertices.2),
            HalfEdge::new(self.vertices.2, self.vertices.0),
        ]
    }
}

impl From<(u32, u32, u32)> for TriangleFace {
    fn from((i1, i2, i3): (u32, u32, u32)) -> TriangleFace {
        TriangleFace::new(i1, i2, i3)
    }
}

/// A face edge. Contains indices to other geometry data - vertices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HalfEdge {
    pub vertices: (u32, u32),
}

impl HalfEdge {
    pub fn new(i1: u32, i2: u32) -> Self {
        HalfEdge { vertices: (i1, i2) }
    }
    pub fn equal_bidi(self, other: HalfEdge) -> bool {
        (self.vertices.0 == other.vertices.0 && self.vertices.1 == other.vertices.1)
            || (self.vertices.0 == other.vertices.1 && self.vertices.1 == other.vertices.0)
    }
}

impl From<(u32, u32)> for HalfEdge {
    fn from((i1, i2): (u32, u32)) -> HalfEdge {
        HalfEdge::new(i1, i2)
    }
}

pub fn plane_same_len(position: [f32; 3], scale: f32) -> Geometry {
    #[rustfmt::skip]
    let vertex_positions = vec![
        v(-1.0, -1.0,  0.0, position, scale),
        v( 1.0, -1.0,  0.0, position, scale),
        v( 1.0,  1.0,  0.0, position, scale),
        v( 1.0,  1.0,  0.0, position, scale),
        v(-1.0,  1.0,  0.0, position, scale),
        v(-1.0, -1.0,  0.0, position, scale),
    ];

    #[rustfmt::skip]
    let vertex_normals = vec![
        n( 0.0,  0.0,  1.0),
        n( 0.0,  0.0,  1.0),
        n( 0.0,  0.0,  1.0),
        n( 0.0,  0.0,  1.0),
        n( 0.0,  0.0,  1.0),
        n( 0.0,  0.0,  1.0),
    ];

    #[rustfmt::skip]
    let faces = vec![
        TriangleFace::new(0, 1, 2),
        TriangleFace::new(3, 4, 5),
    ];

    Geometry::from_triangle_faces_with_vertices_and_normals(faces, vertex_positions, vertex_normals)
}

pub fn plane_var_len(position: [f32; 3], scale: f32) -> Geometry {
    #[rustfmt::skip]
    let vertex_positions = vec![
        v(-1.0, -1.0,  0.0, position, scale),
        v( 1.0, -1.0,  0.0, position, scale),
        v( 1.0,  1.0,  0.0, position, scale),
        v( 1.0,  1.0,  0.0, position, scale),
        v(-1.0,  1.0,  0.0, position, scale),
        v(-1.0, -1.0,  0.0, position, scale),
    ];

    #[rustfmt::skip]
    let vertex_normals = vec![
        n( 0.0,  0.0,  1.0),
    ];

    #[rustfmt::skip]
    let faces = vec![
        TriangleFace::new_separate(0, 1, 2, 0, 0, 0),
        TriangleFace::new_separate(3, 4, 5, 0, 0, 0),
    ];

    Geometry::from_triangle_faces_with_vertices_and_normals(faces, vertex_positions, vertex_normals)
}

pub fn cube_smooth_same_len(position: [f32; 3], scale: f32) -> Geometry {
    #[rustfmt::skip]
    let vertex_positions = vec![
        // back
        v(-1.0,  1.0, -1.0, position, scale),
        v(-1.0,  1.0,  1.0, position, scale),
        v( 1.0,  1.0,  1.0, position, scale),
        v( 1.0,  1.0, -1.0, position, scale),
        // front
        v(-1.0, -1.0, -1.0, position, scale),
        v( 1.0, -1.0, -1.0, position, scale),
        v( 1.0, -1.0,  1.0, position, scale),
        v(-1.0, -1.0,  1.0, position, scale),
    ];

    // FIXME: make const once float arithmetic is stabilized in const fns
    // let sqrt_3 = 3.0f32.sqrt();
    let frac_1_sqrt_3 = 1.0 / 3.0_f32.sqrt();

    #[rustfmt::skip]
    let vertex_normals = vec![
        // back
        n(-frac_1_sqrt_3,  frac_1_sqrt_3, -frac_1_sqrt_3),
        n(-frac_1_sqrt_3,  frac_1_sqrt_3,  frac_1_sqrt_3),
        n( frac_1_sqrt_3,  frac_1_sqrt_3,  frac_1_sqrt_3),
        n( frac_1_sqrt_3,  frac_1_sqrt_3, -frac_1_sqrt_3),
        // front
        n(-frac_1_sqrt_3, -frac_1_sqrt_3, -frac_1_sqrt_3),
        n( frac_1_sqrt_3, -frac_1_sqrt_3, -frac_1_sqrt_3),
        n( frac_1_sqrt_3, -frac_1_sqrt_3,  frac_1_sqrt_3),
        n(-frac_1_sqrt_3, -frac_1_sqrt_3,  frac_1_sqrt_3),
    ];

    #[rustfmt::skip]
    let faces = vec![
        // back
        TriangleFace::new(0, 1, 2),
        TriangleFace::new(2, 3, 0),
        // front
        TriangleFace::new(4, 5, 6),
        TriangleFace::new(6, 7, 4),
        // top
        TriangleFace::new(7, 6, 2),
        TriangleFace::new(2, 1, 7),
        // bottom
        TriangleFace::new(4, 0, 3),
        TriangleFace::new(3, 5, 4),
        // right
        TriangleFace::new(5, 3, 2),
        TriangleFace::new(2, 6, 5),
        // left
        TriangleFace::new(4, 7, 1),
        TriangleFace::new(1, 0, 4),
    ];

    Geometry::from_triangle_faces_with_vertices_and_normals(faces, vertex_positions, vertex_normals)
}

pub fn cube_sharp_same_len(position: [f32; 3], scale: f32) -> Geometry {
    #[rustfmt::skip]
    let vertex_positions = vec![
        // back
        v(-1.0,  1.0, -1.0, position, scale),
        v(-1.0,  1.0,  1.0, position, scale),
        v( 1.0,  1.0,  1.0, position, scale),
        v( 1.0,  1.0, -1.0, position, scale),
        // front
        v(-1.0, -1.0, -1.0, position, scale),
        v( 1.0, -1.0, -1.0, position, scale),
        v( 1.0, -1.0,  1.0, position, scale),
        v(-1.0, -1.0,  1.0, position, scale),
        // top
        v(-1.0,  1.0,  1.0, position, scale),
        v(-1.0, -1.0,  1.0, position, scale),
        v( 1.0, -1.0,  1.0, position, scale),
        v( 1.0,  1.0,  1.0, position, scale),
        // bottom
        v(-1.0,  1.0, -1.0, position, scale),
        v( 1.0,  1.0, -1.0, position, scale),
        v( 1.0, -1.0, -1.0, position, scale),
        v(-1.0, -1.0, -1.0, position, scale),
        // right
        v( 1.0,  1.0, -1.0, position, scale),
        v( 1.0,  1.0,  1.0, position, scale),
        v( 1.0, -1.0,  1.0, position, scale),
        v( 1.0, -1.0, -1.0, position, scale),
        // left
        v(-1.0,  1.0, -1.0, position, scale),
        v(-1.0, -1.0, -1.0, position, scale),
        v(-1.0, -1.0,  1.0, position, scale),
        v(-1.0,  1.0,  1.0, position, scale),
    ];

    #[rustfmt::skip]
    let vertex_normals = vec![
        // back
        n( 0.0,  1.0,  0.0),
        n( 0.0,  1.0,  0.0),
        n( 0.0,  1.0,  0.0),
        n( 0.0,  1.0,  0.0),
        // front
        n( 0.0, -1.0,  0.0),
        n( 0.0, -1.0,  0.0),
        n( 0.0, -1.0,  0.0),
        n( 0.0, -1.0,  0.0),
        // top
        n( 0.0,  0.0,  1.0),
        n( 0.0,  0.0,  1.0),
        n( 0.0,  0.0,  1.0),
        n( 0.0,  0.0,  1.0),
        // bottom
        n( 0.0,  0.0, -1.0),
        n( 0.0,  0.0, -1.0),
        n( 0.0,  0.0, -1.0),
        n( 0.0,  0.0, -1.0),
        // right
        n( 1.0,  0.0,  0.0),
        n( 1.0,  0.0,  0.0),
        n( 1.0,  0.0,  0.0),
        n( 1.0,  0.0,  0.0),
        // left
        n(-1.0,  0.0,  0.0),
        n(-1.0,  0.0,  0.0),
        n(-1.0,  0.0,  0.0),
        n(-1.0,  0.0,  0.0),
    ];

    #[rustfmt::skip]
    let faces = vec![
        // back
        TriangleFace::new(0, 1, 2),
        TriangleFace::new(2, 3, 0),
        // front
        TriangleFace::new(4, 5, 6),
        TriangleFace::new(6, 7, 4),
        // top
        TriangleFace::new(8, 9, 10),
        TriangleFace::new(10, 11, 8),
        // bottom
        TriangleFace::new(12, 13, 14),
        TriangleFace::new(14, 15, 12),
        // right
        TriangleFace::new(16, 17, 18),
        TriangleFace::new(18, 19, 16),
        // left
        TriangleFace::new(20, 21, 22),
        TriangleFace::new(22, 23, 20),
    ];

    Geometry::from_triangle_faces_with_vertices_and_normals(faces, vertex_positions, vertex_normals)
}

pub fn cube_sharp_var_len(position: [f32; 3], scale: f32) -> Geometry {
    #[rustfmt::skip]
    let vertex_positions = vec![
        // back
        v(-1.0,  1.0, -1.0, position, scale),
        v(-1.0,  1.0,  1.0, position, scale),
        v( 1.0,  1.0,  1.0, position, scale),
        v( 1.0,  1.0, -1.0, position, scale),
        // front
        v(-1.0, -1.0, -1.0, position, scale),
        v( 1.0, -1.0, -1.0, position, scale),
        v( 1.0, -1.0,  1.0, position, scale),
        v(-1.0, -1.0,  1.0, position, scale),
    ];

    #[rustfmt::skip]
    let vertex_normals = vec![
        // back
        n( 0.0,  1.0,  0.0),
        // front
        n( 0.0, -1.0,  0.0),
        // top
        n( 0.0,  0.0,  1.0),
        // bottom
        n( 0.0,  0.0, -1.0),
        // right
        n( 1.0,  0.0,  0.0),
        // left
        n(-1.0,  0.0,  0.0),
    ];

    #[rustfmt::skip]
    let faces = vec![
        // back
        TriangleFace::new_separate(0, 1, 2, 0, 0, 0),
        TriangleFace::new_separate(2, 3, 0, 0, 0, 0),
        // front
        TriangleFace::new_separate(4, 5, 6, 1, 1, 1),
        TriangleFace::new_separate(6, 7, 4, 1, 1, 1),
        // top
        TriangleFace::new_separate(7, 6, 2, 2, 2, 2),
        TriangleFace::new_separate(2, 1, 7, 2, 2, 2),
        // bottom
        TriangleFace::new_separate(4, 0, 3, 3, 3, 3),
        TriangleFace::new_separate(3, 5, 4, 3, 3, 3),
        // right
        TriangleFace::new_separate(5, 3, 2, 4, 4, 4),
        TriangleFace::new_separate(2, 6, 5, 4, 4, 4),
        // left
        TriangleFace::new_separate(4, 7, 1, 5, 5, 5),
        TriangleFace::new_separate(1, 0, 4, 5, 5, 5),
    ];

    Geometry::from_triangle_faces_with_vertices_and_normals(faces, vertex_positions, vertex_normals)
}

/// Create UV Sphere primitive at `position` with `scale`,
/// `n_parallels` and `n_meridians`.
///
/// # Panics
/// Panics if number of parallels is less than 2 or number of
/// meridians is less than 3.
pub fn uv_sphere(position: [f32; 3], scale: f32, n_parallels: u32, n_meridians: u32) -> Geometry {
    assert!(n_parallels >= 2, "Need at least 2 paralells");
    assert!(n_meridians >= 3, "Need at least 3 meridians");

    // Add the poles
    let lat_line_max = n_parallels + 2;
    // Add the last, wrapping meridian
    let lng_line_max = n_meridians + 1;

    use std::f32::consts::PI;
    const TWO_PI: f32 = 2.0 * PI;

    // 1 North pole + 1 South pole + `n_parallels` * `n_meridians`
    let vertex_data_count = cast_usize(2 + n_parallels * n_meridians);
    let mut vertex_positions = Vec::with_capacity(vertex_data_count);

    // Produce vertex data for bands in between parallels

    for lat_line in 0..n_parallels {
        for lng_line in 0..n_meridians {
            let polar_t = (lat_line + 1) as f32 / (lat_line_max - 1) as f32;
            let azimuthal_t = lng_line as f32 / (lng_line_max - 1) as f32;

            let x = (PI * polar_t).sin() * (TWO_PI * azimuthal_t).cos();
            let y = (PI * polar_t).sin() * (TWO_PI * azimuthal_t).sin();
            let z = (PI * polar_t).cos();

            vertex_positions.push(v(x, y, z, position, scale));
        }
    }

    // Triangles from North and South poles to the nearest band + 2 * quads in bands
    let faces_count = cast_usize(2 * n_meridians + 2 * n_meridians * (n_parallels - 1));
    let mut faces = Vec::with_capacity(faces_count);

    // Produce faces for bands in-between parallels

    for i in 1..n_parallels {
        for j in 0..n_meridians {
            // Produces 2 CCW wound triangles: (p1, p2, p3) and (p3, p4, p1)

            let p1 = i * n_meridians + j;
            let p2 = i * n_meridians + ((j + 1) % n_meridians);

            let p4 = (i - 1) * n_meridians + j;
            let p3 = (i - 1) * n_meridians + ((j + 1) % n_meridians);

            faces.push((p1, p2, p3));
            faces.push((p3, p4, p1));
        }
    }

    // Add vertex data and band-connecting faces for North and South poles

    let north_pole = cast_u32(vertex_positions.len());
    vertex_positions.push(v(0.0, 0.0, 1.0, position, scale));

    let south_pole = cast_u32(vertex_positions.len());
    vertex_positions.push(v(0.0, 0.0, -1.0, position, scale));

    for i in 0..n_meridians {
        let north_p1 = i;
        let north_p2 = (i + 1) % n_meridians;

        let south_p1 = (n_parallels - 1) * n_meridians + i;
        let south_p2 = (n_parallels - 1) * n_meridians + ((i + 1) % n_meridians);

        faces.push((north_p1, north_p2, north_pole));
        faces.push((south_p2, south_p1, south_pole));
    }

    assert_eq!(vertex_positions.len(), vertex_data_count);
    assert_eq!(vertex_positions.capacity(), vertex_data_count);
    assert_eq!(faces.len(), faces_count);
    assert_eq!(faces.capacity(), faces_count);

    Geometry::from_triangle_faces_with_vertices_and_computed_normals(
        faces,
        vertex_positions,
        NormalStrategy::Sharp,
    )
}

pub fn compute_bounding_sphere(geometries: &[Geometry]) -> (Point3<f32>, f32) {
    let centroid = compute_centroid(geometries);
    let mut max_distance = 0.0;

    for geometry in geometries {
        for vertex in &geometry.vertices {
            // Can't use `distance_squared` for values 0..1

            // FIXME: @Optimization Benchmark this against a 0..1 vs
            // 1..inf branching version using distance_squared for 1..inf
            let distance = na::distance(&centroid, vertex);
            if distance > max_distance {
                max_distance = distance;
            }
        }
    }

    (centroid, max_distance)
}

pub fn compute_centroid(geometries: &[Geometry]) -> Point3<f32> {
    let mut vertex_count = 0;
    let mut centroid = Point3::origin();
    for geometry in geometries {
        vertex_count += geometry.vertices.len();
        for vertex in &geometry.vertices {
            let v = vertex - Point3::origin();
            centroid += v;
        }
    }

    centroid / (vertex_count as f32)
}

pub fn find_closest_point(position: &Point3<f32>, geometry: &Geometry) -> Option<Point3<f32>> {
    let vertices = geometry.vertices();
    if vertices.is_empty() {
        return None;
    }

    let mut closest = vertices[0];
    // FIXME: @Optimization benchmark `distance` vs `distance_squared`
    // with branching (0..1, 1..inf)
    let mut closest_distance = na::distance(position, &closest);
    for point in &vertices[1..] {
        let distance = na::distance(position, &point);
        if distance < closest_distance {
            closest = *point;
            closest_distance = distance;
        }
    }

    Some(closest)
}

fn v(x: f32, y: f32, z: f32, translation: [f32; 3], scale: f32) -> Point3<f32> {
    Point3::new(
        scale * x + translation[0],
        scale * y + translation[1],
        scale * z + translation[2],
    )
}

fn n(x: f32, y: f32, z: f32) -> Vector3<f32> {
    Vector3::new(x, y, z)
}

fn compute_triangle_normal(p1: &Point3<f32>, p2: &Point3<f32>, p3: &Point3<f32>) -> Vector3<f32> {
    let u = p2 - p1;
    let v = p3 - p1;

    Vector3::cross(&u, &v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> (Vec<(u32, u32, u32)>, Vec<Point3<f32>>) {
        #[rustfmt::skip]
        let vertices = vec![
            v(-1.0, -1.0,  0.0, [0.0, 0.0, 0.0], 1.0),
            v( 1.0, -1.0,  0.0, [0.0, 0.0, 0.0], 1.0),
            v( 1.0,  1.0,  0.0, [0.0, 0.0, 0.0], 1.0),
            v(-1.0,  1.0,  0.0, [0.0, 0.0, 0.0], 1.0),
        ];

        #[rustfmt::skip]
        let faces = vec![
            (0, 1, 2),
            (2, 3, 0),
        ];

        (faces, vertices)
    }

    fn quad_with_normals() -> (Vec<TriangleFace>, Vec<Point3<f32>>, Vec<Vector3<f32>>) {
        #[rustfmt::skip]
        let vertices = vec![
            v(-1.0, -1.0,  0.0, [0.0, 0.0, 0.0], 1.0),
            v( 1.0, -1.0,  0.0, [0.0, 0.0, 0.0], 1.0),
            v( 1.0,  1.0,  0.0, [0.0, 0.0, 0.0], 1.0),
            v(-1.0,  1.0,  0.0, [0.0, 0.0, 0.0], 1.0),
        ];

        #[rustfmt::skip]
        let normals = vec![
            n( 0.0,  0.0,  1.0),
            n( 0.0,  0.0,  1.0),
            n( 0.0,  0.0,  1.0),
            n( 0.0,  0.0,  1.0),
        ];

        #[rustfmt::skip]
        let faces = vec![
            TriangleFace::new(0, 1, 2),
            TriangleFace::new(2, 3, 0),
        ];

        (faces, vertices, normals)
    }

    #[test]
    fn test_geometry_from_triangle_faces_with_vertices_and_computed_normals() {
        let (faces, vertices) = quad();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let geometry_faces: Vec<_> = geometry.triangle_faces_iter().collect();

        assert_eq!(vertices.as_slice(), geometry.vertices());
        assert_eq!(
            faces,
            geometry_faces
                .into_iter()
                .map(|triangle_face| triangle_face.vertices)
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    #[should_panic(expected = "Faces reference out of bounds position data")]
    fn test_geometry_from_triangle_faces_with_vertices_and_computed_normals_bounds_check() {
        let (_, vertices) = quad();
        #[rustfmt::skip]
        let faces = vec![
            (0, 1, 2),
            (2, 3, 4),
        ];

        let _geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );
    }

    #[test]
    fn test_geometry_from_triangle_faces_with_vertices_and_normals() {
        let (faces, vertices, normals) = quad_with_normals();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_normals(
            faces.clone(),
            vertices.clone(),
            normals.clone(),
        );
        let geometry_faces: Vec<_> = geometry.triangle_faces_iter().collect();

        assert_eq!(vertices.as_slice(), geometry.vertices());
        assert_eq!(normals.as_slice(), geometry.normals());
        assert_eq!(faces.as_slice(), geometry_faces.as_slice());
    }

    #[test]
    #[should_panic(expected = "Faces reference out of bounds position data")]
    fn test_geometry_from_triangle_faces_with_vertices_and_normals_bounds_check() {
        let (_, vertices, normals) = quad_with_normals();
        #[rustfmt::skip]
        let faces = vec![
            TriangleFace::new(0, 1, 2),
            TriangleFace::new(2, 3, 4),
        ];

        let _geometry = Geometry::from_triangle_faces_with_vertices_and_normals(
            faces.clone(),
            vertices.clone(),
            normals.clone(),
        );
    }

    #[test]
    #[should_panic(expected = "One or more face edges consists of the same vertex")]
    fn test_triangle_face_new_with_invalid_vertex_indices_0_1_should_panic() {
        let _invalid_face_1 = TriangleFace::new(0, 0, 2);
    }

    #[test]
    #[should_panic(expected = "One or more face edges consists of the same vertex")]
    fn test_triangle_face_new_with_invalid_vertex_indices_1_2_should_panic() {
        let _invalid_face_2 = TriangleFace::new(0, 2, 2);
    }

    #[test]
    #[should_panic(expected = "One or more face edges consists of the same vertex")]
    fn test_triangle_face_new_with_invalid_vertex_indices_0_2_should_panic() {
        let _invalid_face_3 = TriangleFace::new(0, 2, 0);
    }

    #[test]
    #[should_panic(expected = "One or more face edges consists of the same vertex")]
    fn test_triangle_face_new_separate_with_invalid_vertex_indices_0_1_should_panic() {
        let _invalid_face_1 = TriangleFace::new_separate(0, 0, 2, 0, 0, 0);
    }

    #[test]
    #[should_panic(expected = "One or more face edges consists of the same vertex")]
    fn test_triangle_face_new_separate_with_invalid_vertex_indices_1_2_should_panic() {
        let _invalid_face_2 = TriangleFace::new_separate(0, 2, 2, 0, 0, 0);
    }

    #[test]
    #[should_panic(expected = "One or more face edges consists of the same vertex")]
    fn test_triangle_face_new_separate_with_invalid_vertex_indices_0_2_should_panic() {
        let _invalid_face_3 = TriangleFace::new_separate(0, 2, 0, 0, 0, 0);
    }
}
