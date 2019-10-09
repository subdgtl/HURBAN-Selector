use std::collections::{HashMap, HashSet};

use arrayvec::ArrayVec;
use std::cmp;

use nalgebra as na;
use nalgebra::base::Vector3;
use nalgebra::geometry::Point3;

use crate::convert::{cast_i32, cast_u32, cast_usize};
use std::hash::{Hash, Hasher};

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

    /// Extracts oriented edges from all mesh faces
    pub fn oriented_edges_iter<'a>(&'a self) -> impl Iterator<Item = OrientedEdge> + 'a {
        self.triangle_faces_iter()
            .flat_map(|face| ArrayVec::from(face.to_oriented_edges()).into_iter())
    }

    /// Extracts unoriented edges from all mesh faces
    pub fn unoriented_edges_iter<'a>(&'a self) -> impl Iterator<Item = UnorientedEdge> + 'a {
        self.triangle_faces_iter()
            .flat_map(|face| ArrayVec::from(face.to_unoriented_edges()).into_iter())
    }

    /// Genus of a mesh is the number of holes in topology / conectivity
    /// The mesh must be triangular and watertight
    /// V - E + F = 2 (1 - G)
    pub fn mesh_genus(&self, edges: &HashSet<UnorientedEdge>) -> i32 {
        let vertex_count = cast_i32(self.vertices.len());
        let edge_count = cast_i32(edges.len());
        let face_count = cast_i32(self.faces.len());

        1 - (vertex_count - edge_count + face_count) / 2
    }

    /// Does the mesh contain unused (not referenced in faces) vertices
    pub fn has_no_orphan_vertices(&self) -> bool {
        let mut used_vertices = HashSet::new();
        for face in self.triangle_faces_iter() {
            used_vertices.insert(face.vertices.0);
            used_vertices.insert(face.vertices.1);
            used_vertices.insert(face.vertices.2);
        }
        used_vertices.len() == self.vertices().len()
    }

    /// Does the mesh contain unused (not referenced in faces) normals
    pub fn has_no_orphan_normals(&self) -> bool {
        let mut used_normals = HashSet::new();
        for face in self.triangle_faces_iter() {
            used_normals.insert(face.normals.0);
            used_normals.insert(face.normals.1);
            used_normals.insert(face.normals.2);
        }
        used_normals.len() == self.normals().len()
    }

    /// Calculates topological relations (neighborhood) of mesh faces.
    /// Returns a Map (key: face index, value: list of its neighboring faces indices)
    pub fn face_to_face_topology(&self) -> HashMap<usize, Vec<usize>> {
        let mut f2f: HashMap<usize, Vec<usize>> = HashMap::new();
        for (from_counter, f) in self.triangle_faces_iter().enumerate() {
            let [f_e_0, f_e_1, f_e_2] = f.to_unoriented_edges();
            for (to_counter, t_f) in self.triangle_faces_iter().enumerate() {
                if from_counter != to_counter && t_f.contains_unoriented_edge(f_e_0)
                    || t_f.contains_unoriented_edge(f_e_1)
                    || t_f.contains_unoriented_edge(f_e_2)
                {
                    let neighbors = f2f.entry(from_counter).or_insert_with(|| vec![]);
                    neighbors.push(to_counter);
                }
            }
        }
        f2f
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
        TriangleFace {
            vertices: (vi1, vi2, vi3),
            normals: (ni1, ni2, ni3),
        }
    }

    /// Generates 3 oriented edges from the respective triangular face
    pub fn to_oriented_edges(&self) -> [OrientedEdge; 3] {
        [
            OrientedEdge::new(self.vertices.0, self.vertices.1),
            OrientedEdge::new(self.vertices.1, self.vertices.2),
            OrientedEdge::new(self.vertices.2, self.vertices.0),
        ]
    }

    /// Generates 3 unoriented edges from the respective triangular face
    pub fn to_unoriented_edges(&self) -> [UnorientedEdge; 3] {
        [
            UnorientedEdge(OrientedEdge::new(self.vertices.0, self.vertices.1)),
            UnorientedEdge(OrientedEdge::new(self.vertices.1, self.vertices.2)),
            UnorientedEdge(OrientedEdge::new(self.vertices.2, self.vertices.0)),
        ]
    }

    /// Does the face contain the specific unoriented edge
    pub fn contains_unoriented_edge(&self, unoriented_edge: UnorientedEdge) -> bool {
        let [o_e_0, o_e_1, o_e_2] = &self.to_unoriented_edges();
        o_e_0 == &unoriented_edge || o_e_1 == &unoriented_edge || o_e_2 == &unoriented_edge
    }
}

impl From<(u32, u32, u32)> for TriangleFace {
    fn from((i1, i2, i3): (u32, u32, u32)) -> TriangleFace {
        TriangleFace::new(i1, i2, i3)
    }
}

/// Oriented face edge. Contains indices to other geometry data - vertices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OrientedEdge {
    pub vertices: (u32, u32),
}

impl OrientedEdge {
    pub fn new(i1: u32, i2: u32) -> Self {
        assert!(
            i1 != i2,
            "The oriented edge is constituted of the same vertex"
        );
        OrientedEdge { vertices: (i1, i2) }
    }

    pub fn is_reverted(self, other: OrientedEdge) -> bool {
        self.vertices.0 == other.vertices.1 && self.vertices.1 == other.vertices.0
    }
}

impl From<(u32, u32)> for OrientedEdge {
    fn from((i1, i2): (u32, u32)) -> OrientedEdge {
        OrientedEdge::new(i1, i2)
    }
}

/// Implements orientation indifferent hash and equal methods
#[derive(Debug, Clone, Copy, Eq)]
pub struct UnorientedEdge(pub OrientedEdge);

impl PartialEq for UnorientedEdge {
    fn eq(&self, other: &Self) -> bool {
        (self.0.vertices.0 == other.0.vertices.0 && self.0.vertices.1 == other.0.vertices.1)
            || (self.0.vertices.0 == other.0.vertices.1 && self.0.vertices.1 == other.0.vertices.0)
    }
}

impl Hash for UnorientedEdge {
    fn hash<H: Hasher>(&self, state: &mut H) {
        cmp::min(self.0.vertices.0, self.0.vertices.1).hash(state);
        cmp::max(self.0.vertices.0, self.0.vertices.1).hash(state);
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

#[deprecated(note = "Don't use, generates open geometry")]
// FIXME: Remove eventually
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
    use std::collections::hash_map::DefaultHasher;

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

    fn torus() -> (Vec<(u32, u32, u32)>, Vec<Point3<f32>>) {
        #[rustfmt::skip]
            let vertices = vec![
            Point3::new(1.0, 0.0, 0.25),
            Point3::new(0.821486, 0.0, 0.175022),
            Point3::new(0.75, 0.0, 0.0),
            Point3::new(0.824978, 0.0, -0.178514),
            Point3::new(1.0, 0.0, -0.25),
            Point3::new(0.891778, 0.76512, 0.178514),
            Point3::new(0.758946, 0.651154, 0.25),
            Point3::new(0.623464, 0.534914, 0.175022),
            Point3::new(0.569209, 0.488365, 0.0),
            Point3::new(0.626113, 0.537187, -0.178514),
            Point3::new(0.758946, 0.651154, -0.25),
            Point3::new(0.894428, 0.767394, -0.175022),
            Point3::new(0.948682, 0.813942, 0.0),
            Point3::new(0.201501, 1.157616, 0.178514),
            Point3::new(0.171487, 0.985186, 0.25),
            Point3::new(0.140874, 0.809317, 0.175022),
            Point3::new(0.128615, 0.73889, 0.0),
            Point3::new(0.141473, 0.812757, -0.178514),
            Point3::new(0.171487, 0.985186, -0.25),
            Point3::new(0.202099, 1.161056, -0.175022),
            Point3::new(0.214358, 1.231483, 0.0),
            Point3::new(-0.600858, 1.009776, 0.178514),
            Point3::new(-0.511359, 0.859367, 0.25),
            Point3::new(-0.420074, 0.705959, 0.175022),
            Point3::new(-0.383519, 0.644526, 0.0),
            Point3::new(-0.421859, 0.708959, -0.178514),
            Point3::new(-0.511359, 0.859367, -0.25),
            Point3::new(-0.602643, 1.012776, -0.175022),
            Point3::new(-0.639198, 1.074209, 0.0),
            Point3::new(-1.105913, 0.397032, 0.178514),
            Point3::new(-0.941185, 0.337893, 0.25),
            Point3::new(-0.77317, 0.277574, 0.175022),
            Point3::new(-0.705888, 0.25342, 0.0),
            Point3::new(-0.776456, 0.278754, -0.178514),
            Point3::new(-0.941185, 0.337893, -0.25),
            Point3::new(-1.109199, 0.398211, -0.175022),
            Point3::new(-1.176481, 0.422366, 0.0),
            Point3::new(-1.099292, -0.415012, 0.178514),
            Point3::new(-0.93555, -0.353195, 0.25),
            Point3::new(-0.768541, -0.290145, 0.175022),
            Point3::new(-0.701662, -0.264896, 0.0),
            Point3::new(-0.771807, -0.291378, -0.178514),
            Point3::new(-0.93555, -0.353195, -0.25),
            Point3::new(-1.102558, -0.416245, -0.175022),
            Point3::new(-1.169437, -0.441494, 0.0),
            Point3::new(-0.5808, -1.021445, 0.178514),
            Point3::new(-0.494288, -0.869298, 0.25),
            Point3::new(-0.406051, -0.714117, 0.175022),
            Point3::new(-0.370716, -0.651974, 0.0),
            Point3::new(-0.407777, -0.717151, -0.178514),
            Point3::new(-0.494288, -0.869298, -0.25),
            Point3::new(-0.582525, -1.02448, -0.175022),
            Point3::new(-0.61786, -1.086623, 0.0),
            Point3::new(0.21476, -1.15523, 0.178514),
            Point3::new(0.182771, -0.983156, 0.25),
            Point3::new(0.150144, -0.807649, 0.175022),
            Point3::new(0.137078, -0.737367, 0.0),
            Point3::new(0.150782, -0.811081, -0.178514),
            Point3::new(0.182771, -0.983156, -0.25),
            Point3::new(0.215398, -1.158662, -0.175022),
            Point3::new(0.228463, -1.228944, 0.0),
            Point3::new(0.964571, -0.795049, 0.0),
            Point3::new(0.906714, -0.74736, 0.178514),
            Point3::new(0.771657, -0.636039, 0.25),
            Point3::new(0.633906, -0.522497, 0.175022),
            Point3::new(0.578743, -0.477029, 0.0),
            Point3::new(0.6366, -0.524718, -0.178514),
            Point3::new(0.771657, -0.636039, -0.25),
            Point3::new(0.909408, -0.749581, -0.175022),
            Point3::new(1.25, 0.0, 0.0),
            Point3::new(1.175022, 0.0, 0.178514),
            Point3::new(1.178514, 0.0, -0.175022),
        ];

        #[rustfmt::skip]
            let faces = vec![
            (5, 70, 69),
            (6, 0, 5),
            (7, 1, 6),
            (8, 2, 1),
            (9, 3, 8),
            (10, 4, 3),
            (11, 71, 10),
            (12, 69, 11),
            (13, 5, 20),
            (14, 6, 5),
            (15, 7, 14),
            (16, 8, 7),
            (17, 9, 16),
            (18, 10, 9),
            (19, 11, 10),
            (20, 12, 11),
            (21, 13, 20),
            (22, 14, 21),
            (23, 15, 22),
            (24, 16, 15),
            (25, 17, 24),
            (26, 18, 17),
            (27, 19, 18),
            (28, 20, 27),
            (29, 21, 28),
            (30, 22, 29),
            (31, 23, 22),
            (32, 24, 23),
            (33, 25, 32),
            (34, 26, 25),
            (35, 27, 26),
            (36, 28, 35),
            (37, 29, 44),
            (38, 30, 37),
            (39, 31, 30),
            (40, 32, 39),
            (41, 33, 32),
            (42, 34, 33),
            (43, 35, 34),
            (44, 36, 35),
            (45, 37, 52),
            (46, 38, 37),
            (47, 39, 38),
            (48, 40, 47),
            (49, 41, 40),
            (50, 42, 49),
            (51, 43, 42),
            (52, 44, 43),
            (53, 45, 60),
            (54, 46, 45),
            (55, 47, 54),
            (56, 48, 55),
            (57, 49, 48),
            (58, 50, 49),
            (59, 51, 58),
            (60, 52, 59),
            (62, 53, 60),
            (63, 54, 62),
            (64, 55, 63),
            (65, 56, 64),
            (66, 57, 65),
            (67, 58, 57),
            (68, 59, 67),
            (61, 60, 68),
            (70, 62, 61),
            (0, 63, 62),
            (1, 64, 0),
            (2, 65, 1),
            (3, 66, 2),
            (4, 67, 66),
            (71, 68, 4),
            (69, 61, 71),
            (5, 69, 12),
            (0, 70, 5),
            (1, 0, 6),
            (8, 1, 7),
            (3, 2, 8),
            (10, 3, 9),
            (71, 4, 10),
            (69, 71, 11),
            (5, 12, 20),
            (14, 5, 13),
            (7, 6, 14),
            (16, 7, 15),
            (9, 8, 16),
            (18, 9, 17),
            (19, 10, 18),
            (20, 11, 19),
            (21, 20, 28),
            (14, 13, 21),
            (15, 14, 22),
            (24, 15, 23),
            (17, 16, 24),
            (26, 17, 25),
            (27, 18, 26),
            (20, 19, 27),
            (29, 28, 36),
            (22, 21, 29),
            (31, 22, 30),
            (32, 23, 31),
            (25, 24, 32),
            (34, 25, 33),
            (35, 26, 34),
            (28, 27, 35),
            (29, 36, 44),
            (30, 29, 37),
            (39, 30, 38),
            (32, 31, 39),
            (41, 32, 40),
            (42, 33, 41),
            (43, 34, 42),
            (44, 35, 43),
            (37, 44, 52),
            (46, 37, 45),
            (47, 38, 46),
            (40, 39, 47),
            (49, 40, 48),
            (42, 41, 49),
            (51, 42, 50),
            (52, 43, 51),
            (45, 52, 60),
            (54, 45, 53),
            (47, 46, 54),
            (48, 47, 55),
            (57, 48, 56),
            (58, 49, 57),
            (51, 50, 58),
            (52, 51, 59),
            (62, 60, 61),
            (54, 53, 62),
            (55, 54, 63),
            (56, 55, 64),
            (57, 56, 65),
            (67, 57, 66),
            (59, 58, 67),
            (60, 59, 68),
            (70, 61, 69),
            (0, 62, 70),
            (64, 63, 0),
            (65, 64, 1),
            (66, 65, 2),
            (4, 66, 3),
            (68, 67, 4),
            (61, 68, 71),
        ];

        (faces, vertices)
    }

    fn double_torus() -> (Vec<(u32, u32, u32)>, Vec<Point3<f32>>) {
        #[rustfmt::skip]
            let vertices = vec![
            Point3::new(-1.375, 0.0, 0.0),
            Point3::new(-1.300022, 0.0, -0.178514),
            Point3::new(-1.125, 0.0, -0.25),
            Point3::new(-1.302379, 0.83903, 0.178514),
            Point3::new(-1.42491, 0.714055, 0.25),
            Point3::new(-1.549886, 0.586586, 0.175022),
            Point3::new(-1.599933, 0.535541, 0.0),
            Point3::new(-1.547442, 0.589079, -0.178514),
            Point3::new(-1.42491, 0.714055, -0.25),
            Point3::new(-1.299935, 0.841523, -0.175022),
            Point3::new(-1.249888, 0.892568, 0.0),
            Point3::new(-2.125, 1.175022, 0.178514),
            Point3::new(-2.125, 1.0, 0.25),
            Point3::new(-2.125, 0.821486, 0.175022),
            Point3::new(-2.125, 0.75, 0.0),
            Point3::new(-2.125, 0.824978, -0.178514),
            Point3::new(-2.125, 1.0, -0.25),
            Point3::new(-2.125, 1.178514, -0.175022),
            Point3::new(-2.125, 1.25, 0.0),
            Point3::new(-2.96403, 0.822621, 0.178514),
            Point3::new(-2.839055, 0.70009, 0.25),
            Point3::new(-2.711586, 0.575114, 0.175022),
            Point3::new(-2.660541, 0.525067, 0.0),
            Point3::new(-2.714079, 0.577558, -0.178514),
            Point3::new(-2.839055, 0.70009, -0.25),
            Point3::new(-2.966523, 0.825066, -0.175022),
            Point3::new(-3.017568, 0.875112, 0.0),
            Point3::new(-3.300022, 0.0, 0.178514),
            Point3::new(-3.125, 0.0, 0.25),
            Point3::new(-2.946486, 0.0, 0.175022),
            Point3::new(-2.875, 0.0, 0.0),
            Point3::new(-2.949978, 0.0, -0.178514),
            Point3::new(-3.125, 0.0, -0.25),
            Point3::new(-3.303514, 0.0, -0.175022),
            Point3::new(-3.375, 0.0, 0.0),
            Point3::new(-2.947621, -0.83903, 0.178514),
            Point3::new(-2.82509, -0.714055, 0.25),
            Point3::new(-2.700114, -0.586586, 0.175022),
            Point3::new(-2.650067, -0.535541, 0.0),
            Point3::new(-2.702559, -0.589079, -0.178514),
            Point3::new(-2.82509, -0.714055, -0.25),
            Point3::new(-2.950066, -0.841523, -0.175022),
            Point3::new(-3.000112, -0.892568, 0.0),
            Point3::new(-2.125, -1.175022, 0.178514),
            Point3::new(-2.125, -1.0, 0.25),
            Point3::new(-2.125, -0.821486, 0.175022),
            Point3::new(-2.125, -0.75, 0.0),
            Point3::new(-2.125, -0.824978, -0.178514),
            Point3::new(-2.125, -1.0, -0.25),
            Point3::new(-2.125, -1.178514, -0.175022),
            Point3::new(-2.125, -1.25, 0.0),
            Point3::new(-1.28597, -0.822621, 0.178514),
            Point3::new(-1.410945, -0.70009, 0.25),
            Point3::new(-1.538414, -0.575114, 0.175022),
            Point3::new(-1.589459, -0.525067, 0.0),
            Point3::new(-1.535921, -0.577558, -0.178514),
            Point3::new(-1.410945, -0.70009, -0.25),
            Point3::new(-1.283477, -0.825066, -0.175022),
            Point3::new(-1.232432, -0.875112, 0.0),
            Point3::new(-1.125, 0.0, 0.25),
            Point3::new(-1.303514, 0.0, 0.175022),
            Point3::new(-1.0625, 0.658478, 0.0),
            Point3::new(-1.0625, 0.65752, -0.015931),
            Point3::new(-1.0625, 0.585174, -0.130911),
            Point3::new(-1.0625, 0.505863, -0.176777),
            Point3::new(-1.0625, 0.446047, -0.198228),
            Point3::new(-1.0625, 0.382955, -0.213901),
            Point3::new(-1.0625, 0.25586, -0.232109),
            Point3::new(-1.0625, 0.0, -0.242061),
            Point3::new(-1.262882, 0.879315, -0.094529),
            Point3::new(-1.253167, 0.889224, -0.048167),
            Point3::new(-2.125, 1.231439, -0.094529),
            Point3::new(-1.652353, 1.157197, 0.0),
            Point3::new(-1.679384, 1.091018, -0.175022),
            Point3::new(-1.659371, 1.140015, -0.094529),
            Point3::new(-1.263543, 0.878641, 0.09681),
            Point3::new(-1.0625, 0.25586, 0.232109),
            Point3::new(-1.0625, 0.382955, 0.213901),
            Point3::new(-1.0625, 0.446047, 0.198228),
            Point3::new(-1.0625, 0.505863, 0.176777),
            Point3::new(-1.0625, 0.560057, 0.148562),
            Point3::new(-1.0625, 0.585174, 0.130911),
            Point3::new(-1.0625, 0.628344, 0.086934),
            Point3::new(-1.0625, 0.636947, 0.073995),
            Point3::new(-1.0625, 0.650292, 0.04616),
            Point3::new(-1.0625, 0.65752, 0.015931),
            Point3::new(-2.125, 1.230495, 0.09681),
            Point3::new(-1.680704, 1.087786, 0.178514),
            Point3::new(-1.659729, 1.13914, 0.09681),
            Point3::new(-1.0625, -0.25586, -0.232109),
            Point3::new(-1.0625, -0.446047, -0.198228),
            Point3::new(-1.0625, -0.505863, -0.176777),
            Point3::new(-1.0625, -0.650292, -0.04616),
            Point3::new(-1.245685, -0.862118, -0.094529),
            Point3::new(-1.246359, -0.861457, 0.09681),
            Point3::new(-1.64095, -1.152474, 0.0),
            Point3::new(-2.125, -1.231439, -0.094529),
            Point3::new(-1.668632, -1.086565, -0.175022),
            Point3::new(-1.648137, -1.135361, -0.094529),
            Point3::new(-1.0625, -0.65752, 0.015931),
            Point3::new(-1.0625, -0.654743, 0.031328),
            Point3::new(-1.0625, -0.636947, 0.073995),
            Point3::new(-1.0625, -0.628344, 0.086934),
            Point3::new(-1.0625, -0.608, 0.110688),
            Point3::new(-1.0625, -0.585174, 0.130911),
            Point3::new(-1.0625, -0.560057, 0.148562),
            Point3::new(-1.0625, -0.505863, 0.176777),
            Point3::new(-1.0625, -0.25586, 0.232109),
            Point3::new(-1.0625, 0.0, 0.242061),
            Point3::new(-1.235948, -0.871665, 0.049378),
            Point3::new(-1.669984, -1.083346, 0.178514),
            Point3::new(-2.125, -1.230495, 0.09681),
            Point3::new(-1.648503, -1.13449, 0.09681),
            Point3::new(1.0, 0.0, 0.25),
            Point3::new(0.821486, 0.0, 0.175022),
            Point3::new(0.75, 0.0, 0.0),
            Point3::new(0.824978, 0.0, -0.178514),
            Point3::new(1.0, 0.0, -0.25),
            Point3::new(0.891778, 0.76512, 0.178514),
            Point3::new(0.758946, 0.651154, 0.25),
            Point3::new(0.623464, 0.534914, 0.175022),
            Point3::new(0.569209, 0.488365, 0.0),
            Point3::new(0.626113, 0.537187, -0.178514),
            Point3::new(0.758946, 0.651154, -0.25),
            Point3::new(0.894428, 0.767394, -0.175022),
            Point3::new(0.948682, 0.813942, 0.0),
            Point3::new(0.201501, 1.157616, 0.178514),
            Point3::new(0.171487, 0.985186, 0.25),
            Point3::new(0.140874, 0.809317, 0.175022),
            Point3::new(0.128615, 0.73889, 0.0),
            Point3::new(0.141473, 0.812757, -0.178514),
            Point3::new(0.171487, 0.985186, -0.25),
            Point3::new(0.202099, 1.161056, -0.175022),
            Point3::new(0.214358, 1.231483, 0.0),
            Point3::new(-0.600858, 1.009776, 0.178514),
            Point3::new(-0.511359, 0.859367, 0.25),
            Point3::new(-0.420074, 0.705959, 0.175022),
            Point3::new(-0.383519, 0.644526, 0.0),
            Point3::new(-0.421859, 0.708959, -0.178514),
            Point3::new(-0.511359, 0.859367, -0.25),
            Point3::new(-0.602643, 1.012776, -0.175022),
            Point3::new(-0.639198, 1.074209, 0.0),
            Point3::new(-0.941185, 0.337893, 0.25),
            Point3::new(-0.77317, 0.277574, 0.175022),
            Point3::new(-0.705888, 0.25342, 0.0),
            Point3::new(-0.776456, 0.278754, -0.178514),
            Point3::new(-0.941185, 0.337893, -0.25),
            Point3::new(-0.93555, -0.353195, 0.25),
            Point3::new(-0.768541, -0.290145, 0.175022),
            Point3::new(-0.701662, -0.264896, 0.0),
            Point3::new(-0.771807, -0.291378, -0.178514),
            Point3::new(-0.93555, -0.353195, -0.25),
            Point3::new(-0.5808, -1.021445, 0.178514),
            Point3::new(-0.494288, -0.869298, 0.25),
            Point3::new(-0.406051, -0.714117, 0.175022),
            Point3::new(-0.370716, -0.651974, 0.0),
            Point3::new(-0.407777, -0.717151, -0.178514),
            Point3::new(-0.494288, -0.869298, -0.25),
            Point3::new(-0.582525, -1.02448, -0.175022),
            Point3::new(-0.61786, -1.086623, 0.0),
            Point3::new(0.21476, -1.15523, 0.178514),
            Point3::new(0.182771, -0.983156, 0.25),
            Point3::new(0.150144, -0.807649, 0.175022),
            Point3::new(0.137078, -0.737367, 0.0),
            Point3::new(0.150782, -0.811081, -0.178514),
            Point3::new(0.182771, -0.983156, -0.25),
            Point3::new(0.215398, -1.158662, -0.175022),
            Point3::new(0.228463, -1.228944, 0.0),
            Point3::new(0.964571, -0.795049, 0.0),
            Point3::new(0.906714, -0.74736, 0.178514),
            Point3::new(0.771657, -0.636039, 0.25),
            Point3::new(0.633906, -0.522497, 0.175022),
            Point3::new(0.578743, -0.477029, 0.0),
            Point3::new(0.6366, -0.524718, -0.178514),
            Point3::new(0.771657, -0.636039, -0.25),
            Point3::new(0.909408, -0.749581, -0.175022),
            Point3::new(1.25, 0.0, 0.0),
            Point3::new(1.175022, 0.0, 0.178514),
            Point3::new(1.178514, 0.0, -0.175022),
            Point3::new(-1.0625, 0.560057, -0.148562),
            Point3::new(-1.0625, 0.608, -0.110688),
            Point3::new(-1.0625, 0.628344, -0.086934),
            Point3::new(-1.0625, 0.636947, -0.073995),
            Point3::new(-1.0625, 0.644312, -0.060393),
            Point3::new(-1.0625, 0.650292, -0.04616),
            Point3::new(-1.0625, 0.654743, -0.031328),
            Point3::new(-0.629707, 1.058259, -0.094529),
            Point3::new(-0.909408, 0.749581, -0.175022),
            Point3::new(-0.950249, 0.783243, -0.094529),
            Point3::new(-1.0625, 0.654743, 0.031328),
            Point3::new(-1.0625, 0.644312, 0.060393),
            Point3::new(-1.0625, 0.608, 0.110688),
            Point3::new(-0.906714, 0.74736, 0.178514),
            Point3::new(-0.629224, 1.057447, 0.09681),
            Point3::new(-0.964571, 0.795049, 0.0),
            Point3::new(-0.94952, 0.782643, 0.09681),
            Point3::new(-0.960771, 0.791916, 0.049378),
            Point3::new(-1.0625, -0.658478, 0.0),
            Point3::new(-1.0625, -0.65752, -0.015931),
            Point3::new(-1.0625, -0.654743, -0.031328),
            Point3::new(-1.0625, -0.644312, -0.060393),
            Point3::new(-1.0625, -0.636947, -0.073995),
            Point3::new(-1.0625, -0.628344, -0.086934),
            Point3::new(-1.0625, -0.608, -0.110688),
            Point3::new(-1.0625, -0.585174, -0.130911),
            Point3::new(-1.0625, -0.560057, -0.148562),
            Point3::new(-1.0625, -0.382955, -0.213901),
            Point3::new(-0.608686, -1.070488, -0.094529),
            Point3::new(-0.950082, -0.812308, 0.0),
            Point3::new(-0.895747, -0.765853, -0.175022),
            Point3::new(-0.935975, -0.800247, -0.094529),
            Point3::new(-0.608219, -1.069667, 0.09681),
            Point3::new(-0.893094, -0.763584, 0.178514),
            Point3::new(-0.935256, -0.799633, 0.09681),
            Point3::new(-1.0625, -0.382955, 0.213901),
            Point3::new(-1.0625, -0.446047, 0.198228),
            Point3::new(-1.0625, -0.644312, 0.060393),
            Point3::new(-1.0625, -0.650292, 0.04616),
            Point3::new(-0.946339, -0.809108, 0.049378),
            Point3::new(-0.960957, 0.792069, -0.048167),
        ];

        #[rustfmt::skip]
            let faces = vec![
            (5, 60, 4),
            (6, 0, 60),
            (7, 1, 6),
            (8, 2, 1),
            (11, 87, 88),
            (87, 3, 88),
            (86, 88, 18),
            (87, 11, 12),
            (4, 3, 87),
            (12, 4, 87),
            (13, 5, 12),
            (14, 6, 5),
            (15, 7, 14),
            (16, 8, 7),
            (73, 9, 8),
            (16, 17, 73),
            (8, 16, 73),
            (18, 72, 74),
            (70, 69, 74),
            (72, 10, 70),
            (74, 72, 70),
            (74, 69, 9),
            (71, 74, 17),
            (86, 18, 26),
            (19, 11, 86),
            (26, 19, 86),
            (20, 12, 11),
            (21, 13, 20),
            (22, 14, 13),
            (23, 15, 22),
            (24, 16, 15),
            (25, 17, 24),
            (71, 17, 25),
            (26, 18, 71),
            (25, 26, 71),
            (27, 19, 26),
            (28, 20, 27),
            (29, 21, 28),
            (30, 22, 29),
            (31, 23, 30),
            (32, 24, 23),
            (33, 25, 32),
            (34, 26, 25),
            (35, 27, 42),
            (36, 28, 27),
            (37, 29, 36),
            (38, 30, 29),
            (39, 31, 30),
            (40, 32, 39),
            (41, 33, 32),
            (42, 34, 41),
            (111, 43, 35),
            (42, 50, 111),
            (35, 42, 111),
            (44, 36, 35),
            (45, 37, 44),
            (46, 38, 45),
            (47, 39, 38),
            (48, 40, 47),
            (49, 41, 40),
            (96, 50, 42),
            (41, 49, 96),
            (42, 41, 96),
            (51, 110, 112),
            (110, 43, 111),
            (112, 111, 95),
            (109, 94, 112),
            (95, 58, 109),
            (112, 95, 109),
            (110, 51, 52),
            (44, 43, 110),
            (52, 44, 110),
            (53, 45, 44),
            (54, 46, 53),
            (55, 47, 46),
            (56, 48, 55),
            (97, 49, 48),
            (56, 57, 97),
            (48, 56, 97),
            (95, 50, 98),
            (98, 96, 49),
            (93, 98, 57),
            (60, 53, 52),
            (0, 54, 60),
            (1, 55, 54),
            (2, 56, 1),
            (3, 79, 80),
            (75, 191, 82),
            (75, 81, 191),
            (3, 80, 81),
            (76, 59, 108),
            (3, 77, 78),
            (76, 4, 59),
            (76, 77, 4),
            (78, 79, 3),
            (10, 189, 85),
            (85, 61, 10),
            (84, 189, 10),
            (72, 88, 75),
            (75, 190, 84),
            (75, 83, 190),
            (10, 75, 84),
            (75, 3, 81),
            (83, 75, 82),
            (77, 3, 4),
            (67, 68, 2),
            (67, 8, 66),
            (9, 65, 66),
            (8, 67, 2),
            (9, 179, 64),
            (9, 63, 179),
            (69, 181, 180),
            (69, 182, 181),
            (180, 63, 69),
            (64, 65, 9),
            (66, 8, 9),
            (9, 69, 63),
            (69, 70, 182),
            (183, 182, 70),
            (70, 185, 184),
            (70, 62, 185),
            (61, 62, 10),
            (70, 10, 62),
            (70, 184, 183),
            (99, 58, 197),
            (109, 58, 100),
            (58, 99, 100),
            (217, 109, 100),
            (109, 101, 94),
            (214, 52, 51),
            (51, 94, 104),
            (101, 109, 216),
            (216, 109, 217),
            (94, 101, 102),
            (51, 104, 105),
            (94, 103, 104),
            (94, 102, 103),
            (51, 215, 214),
            (51, 106, 215),
            (59, 107, 108),
            (107, 59, 52),
            (214, 107, 52),
            (51, 105, 106),
            (200, 93, 201),
            (93, 57, 204),
            (93, 58, 95),
            (206, 57, 56),
            (200, 92, 93),
            (198, 197, 58),
            (58, 199, 198),
            (58, 92, 199),
            (58, 93, 92),
            (57, 206, 90),
            (68, 89, 2),
            (56, 89, 206),
            (89, 56, 2),
            (93, 202, 201),
            (93, 203, 202),
            (57, 205, 204),
            (57, 91, 205),
            (204, 203, 93),
            (57, 90, 91),
            (118, 177, 176),
            (119, 113, 118),
            (120, 114, 119),
            (121, 115, 114),
            (122, 116, 121),
            (123, 117, 116),
            (124, 178, 123),
            (125, 176, 124),
            (126, 118, 133),
            (127, 119, 118),
            (128, 120, 127),
            (129, 121, 120),
            (130, 122, 129),
            (131, 123, 122),
            (132, 124, 123),
            (133, 125, 124),
            (193, 134, 126),
            (133, 141, 193),
            (126, 133, 193),
            (135, 127, 134),
            (136, 128, 135),
            (137, 129, 128),
            (138, 130, 137),
            (139, 131, 130),
            (140, 132, 131),
            (186, 141, 133),
            (132, 140, 186),
            (133, 132, 186),
            (192, 134, 193),
            (196, 195, 193),
            (141, 194, 196),
            (193, 141, 196),
            (143, 136, 135),
            (144, 137, 136),
            (145, 138, 144),
            (146, 139, 138),
            (219, 194, 141),
            (186, 188, 219),
            (141, 186, 219),
            (188, 186, 187),
            (148, 143, 142),
            (149, 144, 148),
            (150, 145, 144),
            (151, 146, 145),
            (152, 212, 213),
            (218, 208, 159),
            (211, 213, 218),
            (159, 211, 218),
            (154, 148, 147),
            (155, 149, 154),
            (156, 150, 149),
            (157, 151, 156),
            (207, 210, 158),
            (211, 159, 167),
            (160, 152, 211),
            (167, 160, 211),
            (161, 153, 152),
            (162, 154, 161),
            (163, 155, 162),
            (164, 156, 155),
            (165, 157, 156),
            (166, 158, 165),
            (207, 158, 166),
            (167, 159, 207),
            (166, 167, 207),
            (169, 160, 167),
            (170, 161, 169),
            (171, 162, 170),
            (172, 163, 171),
            (173, 164, 172),
            (174, 165, 164),
            (175, 166, 174),
            (168, 167, 175),
            (177, 169, 168),
            (113, 170, 169),
            (114, 171, 113),
            (115, 172, 114),
            (116, 173, 115),
            (117, 174, 173),
            (178, 175, 117),
            (176, 168, 178),
            (85, 194, 61),
            (196, 194, 189),
            (194, 85, 189),
            (84, 196, 189),
            (192, 135, 134),
            (196, 83, 195),
            (195, 81, 192),
            (77, 135, 192),
            (84, 190, 196),
            (196, 190, 83),
            (195, 191, 81),
            (195, 82, 191),
            (195, 83, 82),
            (192, 79, 78),
            (192, 80, 79),
            (76, 142, 77),
            (142, 135, 77),
            (192, 78, 77),
            (192, 81, 80),
            (108, 142, 76),
            (187, 140, 139),
            (182, 219, 188),
            (187, 63, 188),
            (183, 219, 182),
            (187, 139, 66),
            (61, 194, 62),
            (219, 62, 194),
            (219, 185, 62),
            (219, 184, 185),
            (184, 219, 183),
            (66, 65, 187),
            (68, 67, 146),
            (66, 146, 67),
            (139, 146, 66),
            (187, 65, 64),
            (181, 182, 188),
            (180, 181, 188),
            (179, 63, 187),
            (64, 179, 187),
            (180, 188, 63),
            (108, 147, 142),
            (105, 212, 106),
            (102, 213, 103),
            (103, 213, 104),
            (105, 104, 212),
            (147, 107, 214),
            (214, 153, 147),
            (106, 212, 215),
            (212, 214, 215),
            (68, 146, 151),
            (213, 102, 101),
            (100, 218, 217),
            (208, 99, 197),
            (100, 99, 208),
            (208, 218, 100),
            (101, 218, 213),
            (217, 218, 216),
            (104, 213, 212),
            (153, 212, 152),
            (214, 212, 153),
            (218, 101, 216),
            (89, 68, 151),
            (209, 90, 206),
            (209, 91, 90),
            (89, 151, 206),
            (209, 204, 205),
            (210, 203, 204),
            (210, 201, 202),
            (210, 200, 201),
            (202, 203, 210),
            (205, 91, 209),
            (200, 210, 92),
            (209, 157, 158),
            (157, 209, 206),
            (207, 159, 210),
            (208, 199, 92),
            (208, 198, 199),
            (198, 208, 197),
            (210, 208, 92),
            (209, 210, 204),
            (151, 157, 206),
            (107, 147, 108),
            (60, 59, 4),
            (6, 60, 5),
            (1, 0, 6),
            (8, 1, 7),
            (11, 88, 86),
            (3, 75, 88),
            (88, 72, 18),
            (5, 4, 12),
            (14, 5, 13),
            (7, 6, 14),
            (16, 7, 15),
            (18, 74, 71),
            (74, 9, 73),
            (74, 73, 17),
            (20, 11, 19),
            (13, 12, 20),
            (22, 13, 21),
            (15, 14, 22),
            (24, 15, 23),
            (17, 16, 24),
            (27, 26, 34),
            (20, 19, 27),
            (21, 20, 28),
            (22, 21, 29),
            (23, 22, 30),
            (32, 23, 31),
            (25, 24, 32),
            (34, 25, 33),
            (27, 34, 42),
            (36, 27, 35),
            (29, 28, 36),
            (38, 29, 37),
            (39, 30, 38),
            (32, 31, 39),
            (41, 32, 40),
            (34, 33, 41),
            (44, 35, 43),
            (37, 36, 44),
            (38, 37, 45),
            (47, 38, 46),
            (40, 39, 47),
            (49, 40, 48),
            (51, 112, 94),
            (110, 111, 112),
            (111, 50, 95),
            (53, 44, 52),
            (46, 45, 53),
            (55, 46, 54),
            (48, 47, 55),
            (50, 96, 98),
            (98, 49, 97),
            (98, 97, 57),
            (60, 52, 59),
            (54, 53, 60),
            (1, 54, 0),
            (56, 55, 1),
            (72, 75, 10),
            (93, 95, 98),
            (118, 176, 125),
            (113, 177, 118),
            (114, 113, 119),
            (121, 114, 120),
            (116, 115, 121),
            (123, 116, 122),
            (178, 117, 123),
            (176, 178, 124),
            (118, 125, 133),
            (127, 118, 126),
            (120, 119, 127),
            (129, 120, 128),
            (122, 121, 129),
            (131, 122, 130),
            (132, 123, 131),
            (133, 124, 132),
            (127, 126, 134),
            (128, 127, 135),
            (137, 128, 136),
            (130, 129, 137),
            (139, 130, 138),
            (140, 131, 139),
            (192, 193, 195),
            (143, 135, 142),
            (144, 136, 143),
            (138, 137, 144),
            (146, 138, 145),
            (186, 140, 187),
            (148, 142, 147),
            (144, 143, 148),
            (150, 144, 149),
            (151, 145, 150),
            (152, 213, 211),
            (154, 147, 153),
            (149, 148, 154),
            (156, 149, 155),
            (151, 150, 156),
            (210, 209, 158),
            (161, 152, 160),
            (154, 153, 161),
            (155, 154, 162),
            (164, 155, 163),
            (165, 156, 164),
            (158, 157, 165),
            (169, 167, 168),
            (161, 160, 169),
            (162, 161, 170),
            (163, 162, 171),
            (164, 163, 172),
            (174, 164, 173),
            (166, 165, 174),
            (167, 166, 175),
            (177, 168, 176),
            (113, 169, 177),
            (171, 170, 113),
            (172, 171, 114),
            (173, 172, 115),
            (117, 173, 116),
            (175, 174, 117),
            (168, 175, 178),
            (159, 208, 210),
        ];

        (faces, vertices)
    }

    fn triple_torus() -> (Vec<(u32, u32, u32)>, Vec<Point3<f32>>) {
        #[rustfmt::skip]
            let vertices = vec![
            Point3::new(-3.5, 0.0, 0.0),
            Point3::new(-3.425022, 0.0, -0.178514),
            Point3::new(-3.25, 0.0, -0.25),
            Point3::new(-3.374888, 0.892568, 0.0),
            Point3::new(-3.427379, 0.83903, 0.178514),
            Point3::new(-3.54991, 0.714055, 0.25),
            Point3::new(-3.674886, 0.586586, 0.175022),
            Point3::new(-3.724933, 0.535541, 0.0),
            Point3::new(-3.672441, 0.589079, -0.178514),
            Point3::new(-3.54991, 0.714055, -0.25),
            Point3::new(-3.424934, 0.841523, -0.175022),
            Point3::new(-4.25, 1.175022, 0.178514),
            Point3::new(-4.25, 1.0, 0.25),
            Point3::new(-4.25, 0.821486, 0.175022),
            Point3::new(-4.25, 0.75, 0.0),
            Point3::new(-4.25, 0.824978, -0.178514),
            Point3::new(-4.25, 1.0, -0.25),
            Point3::new(-4.25, 1.178514, -0.175022),
            Point3::new(-4.25, 1.25, 0.0),
            Point3::new(-5.08903, 0.822621, 0.178514),
            Point3::new(-4.964055, 0.70009, 0.25),
            Point3::new(-4.836586, 0.575114, 0.175022),
            Point3::new(-4.785541, 0.525067, 0.0),
            Point3::new(-4.839079, 0.577558, -0.178514),
            Point3::new(-4.964055, 0.70009, -0.25),
            Point3::new(-5.091523, 0.825066, -0.175022),
            Point3::new(-5.142569, 0.875112, 0.0),
            Point3::new(-5.425023, 0.0, 0.178514),
            Point3::new(-5.25, 0.0, 0.25),
            Point3::new(-5.071486, 0.0, 0.175022),
            Point3::new(-5.0, 0.0, 0.0),
            Point3::new(-5.074977, 0.0, -0.178514),
            Point3::new(-5.25, 0.0, -0.25),
            Point3::new(-5.428514, 0.0, -0.175022),
            Point3::new(-5.5, 0.0, 0.0),
            Point3::new(-5.072621, -0.83903, 0.178514),
            Point3::new(-4.95009, -0.714055, 0.25),
            Point3::new(-4.825114, -0.586586, 0.175022),
            Point3::new(-4.775067, -0.535541, 0.0),
            Point3::new(-4.827559, -0.589079, -0.178514),
            Point3::new(-4.95009, -0.714055, -0.25),
            Point3::new(-5.075066, -0.841523, -0.175022),
            Point3::new(-5.125113, -0.892568, 0.0),
            Point3::new(-4.25, -1.175022, 0.178514),
            Point3::new(-4.25, -1.0, 0.25),
            Point3::new(-4.25, -0.821486, 0.175022),
            Point3::new(-4.25, -0.75, 0.0),
            Point3::new(-4.25, -0.824978, -0.178514),
            Point3::new(-4.25, -1.0, -0.25),
            Point3::new(-4.25, -1.178514, -0.175022),
            Point3::new(-4.25, -1.25, 0.0),
            Point3::new(-3.41097, -0.822621, 0.178514),
            Point3::new(-3.535945, -0.70009, 0.25),
            Point3::new(-3.663414, -0.575114, 0.175022),
            Point3::new(-3.714459, -0.525067, 0.0),
            Point3::new(-3.660921, -0.577558, -0.178514),
            Point3::new(-3.535945, -0.70009, -0.25),
            Point3::new(-3.408477, -0.825066, -0.175022),
            Point3::new(-3.357432, -0.875112, 0.0),
            Point3::new(-3.25, 0.0, 0.25),
            Point3::new(-3.428514, 0.0, 0.175022),
            Point3::new(-3.1875, 0.25586, 0.232109),
            Point3::new(-3.1875, 0.382955, 0.213901),
            Point3::new(-3.1875, 0.446047, 0.198228),
            Point3::new(-3.1875, 0.505863, 0.176777),
            Point3::new(-3.1875, 0.560057, 0.148562),
            Point3::new(-3.1875, 0.585174, 0.130911),
            Point3::new(-3.1875, 0.650292, 0.04616),
            Point3::new(-3.1875, 0.65752, 0.015931),
            Point3::new(-3.1875, 0.658478, 0.0),
            Point3::new(-3.388543, 0.878641, 0.09681),
            Point3::new(-3.387882, 0.879315, -0.094529),
            Point3::new(-4.25, 1.230495, 0.09681),
            Point3::new(-3.805704, 1.087786, 0.178514),
            Point3::new(-3.784729, 1.13914, 0.09681),
            Point3::new(-3.1875, 0.65752, -0.015931),
            Point3::new(-3.1875, 0.636947, -0.073995),
            Point3::new(-3.1875, 0.608, -0.110688),
            Point3::new(-3.1875, 0.585174, -0.130911),
            Point3::new(-3.1875, 0.505863, -0.176777),
            Point3::new(-3.1875, 0.446047, -0.198228),
            Point3::new(-3.1875, 0.382955, -0.213901),
            Point3::new(-3.1875, 0.25586, -0.232109),
            Point3::new(-3.1875, 0.0, -0.242061),
            Point3::new(-3.378167, 0.889224, -0.048167),
            Point3::new(-4.25, 1.231439, -0.094529),
            Point3::new(-3.777354, 1.157197, 0.0),
            Point3::new(-3.804384, 1.091018, -0.175022),
            Point3::new(-3.784372, 1.140015, -0.094529),
            Point3::new(-3.1875, -0.65752, 0.015931),
            Point3::new(-3.1875, -0.654743, 0.031328),
            Point3::new(-3.1875, -0.585174, 0.130911),
            Point3::new(-3.1875, -0.560057, 0.148562),
            Point3::new(-3.1875, -0.505863, 0.176777),
            Point3::new(-3.1875, -0.25586, 0.232109),
            Point3::new(-3.1875, 0.0, 0.242061),
            Point3::new(-3.37136, -0.861457, 0.09681),
            Point3::new(-3.360948, -0.871665, 0.049378),
            Point3::new(-3.794984, -1.083346, 0.178514),
            Point3::new(-4.25, -1.230495, 0.09681),
            Point3::new(-3.773503, -1.13449, 0.09681),
            Point3::new(-3.370685, -0.862118, -0.094529),
            Point3::new(-3.1875, -0.25586, -0.232109),
            Point3::new(-3.1875, -0.446047, -0.198228),
            Point3::new(-3.1875, -0.505863, -0.176777),
            Point3::new(-3.1875, -0.644312, -0.060393),
            Point3::new(-3.1875, -0.650292, -0.04616),
            Point3::new(-3.1875, -0.658478, 0.0),
            Point3::new(-3.76595, -1.152474, 0.0),
            Point3::new(-4.25, -1.231439, -0.094529),
            Point3::new(-3.793632, -1.086565, -0.175022),
            Point3::new(-3.773137, -1.135361, -0.094529),
            Point3::new(1.0, 0.0, 0.25),
            Point3::new(0.821486, 0.0, 0.175022),
            Point3::new(0.75, 0.0, 0.0),
            Point3::new(0.824978, 0.0, -0.178514),
            Point3::new(1.0, 0.0, -0.25),
            Point3::new(0.891778, 0.76512, 0.178514),
            Point3::new(0.758946, 0.651154, 0.25),
            Point3::new(0.623464, 0.534914, 0.175022),
            Point3::new(0.569209, 0.488365, 0.0),
            Point3::new(0.626113, 0.537187, -0.178514),
            Point3::new(0.758946, 0.651154, -0.25),
            Point3::new(0.894428, 0.767394, -0.175022),
            Point3::new(0.948682, 0.813942, 0.0),
            Point3::new(0.201501, 1.157616, 0.178514),
            Point3::new(0.171487, 0.985186, 0.25),
            Point3::new(0.140874, 0.809317, 0.175022),
            Point3::new(0.128615, 0.73889, 0.0),
            Point3::new(0.141473, 0.812757, -0.178514),
            Point3::new(0.171487, 0.985186, -0.25),
            Point3::new(0.202099, 1.161056, -0.175022),
            Point3::new(0.214358, 1.231483, 0.0),
            Point3::new(-0.600858, 1.009776, 0.178514),
            Point3::new(-0.511359, 0.859367, 0.25),
            Point3::new(-0.420074, 0.705959, 0.175022),
            Point3::new(-0.383519, 0.644526, 0.0),
            Point3::new(-0.421859, 0.708959, -0.178514),
            Point3::new(-0.511359, 0.859367, -0.25),
            Point3::new(-0.602643, 1.012776, -0.175022),
            Point3::new(-0.639198, 1.074209, 0.0),
            Point3::new(-0.941185, 0.337893, 0.25),
            Point3::new(-0.77317, 0.277574, 0.175022),
            Point3::new(-0.705888, 0.25342, 0.0),
            Point3::new(-0.776456, 0.278754, -0.178514),
            Point3::new(-0.941185, 0.337893, -0.25),
            Point3::new(-0.93555, -0.353195, 0.25),
            Point3::new(-0.768541, -0.290145, 0.175022),
            Point3::new(-0.701662, -0.264896, 0.0),
            Point3::new(-0.771807, -0.291378, -0.178514),
            Point3::new(-0.93555, -0.353195, -0.25),
            Point3::new(-0.5808, -1.021445, 0.178514),
            Point3::new(-0.494288, -0.869298, 0.25),
            Point3::new(-0.406051, -0.714117, 0.175022),
            Point3::new(-0.370716, -0.651974, 0.0),
            Point3::new(-0.407777, -0.717151, -0.178514),
            Point3::new(-0.494288, -0.869298, -0.25),
            Point3::new(-0.582525, -1.02448, -0.175022),
            Point3::new(-0.61786, -1.086623, 0.0),
            Point3::new(0.21476, -1.15523, 0.178514),
            Point3::new(0.182771, -0.983156, 0.25),
            Point3::new(0.150144, -0.807649, 0.175022),
            Point3::new(0.137078, -0.737367, 0.0),
            Point3::new(0.150782, -0.811081, -0.178514),
            Point3::new(0.182771, -0.983156, -0.25),
            Point3::new(0.215398, -1.158662, -0.175022),
            Point3::new(0.228463, -1.228944, 0.0),
            Point3::new(0.964571, -0.795049, 0.0),
            Point3::new(0.906714, -0.74736, 0.178514),
            Point3::new(0.771657, -0.636039, 0.25),
            Point3::new(0.633906, -0.522497, 0.175022),
            Point3::new(0.578743, -0.477029, 0.0),
            Point3::new(0.6366, -0.524718, -0.178514),
            Point3::new(0.771657, -0.636039, -0.25),
            Point3::new(0.909408, -0.749581, -0.175022),
            Point3::new(1.25, 0.0, 0.0),
            Point3::new(1.175022, 0.0, 0.178514),
            Point3::new(1.178514, 0.0, -0.175022),
            Point3::new(-1.0625, 0.382955, 0.213901),
            Point3::new(-1.0625, 0.25586, 0.232109),
            Point3::new(-1.0625, 0.0, 0.242061),
            Point3::new(-0.906714, 0.74736, 0.178514),
            Point3::new(-0.629224, 1.057447, 0.09681),
            Point3::new(-0.94952, 0.782643, 0.09681),
            Point3::new(-0.960771, 0.791916, 0.049378),
            Point3::new(-0.964571, 0.795049, 0.0),
            Point3::new(-0.629707, 1.058259, -0.094529),
            Point3::new(-0.909408, 0.749581, -0.175022),
            Point3::new(-0.950249, 0.783243, -0.094529),
            Point3::new(-1.0625, 0.0, -0.242061),
            Point3::new(-1.0625, 0.25586, -0.232109),
            Point3::new(-1.0625, 0.382955, -0.213901),
            Point3::new(-1.0625, 0.446047, -0.198228),
            Point3::new(-1.0625, 0.505863, -0.176777),
            Point3::new(-1.0625, 0.560057, -0.148562),
            Point3::new(-1.0625, 0.65752, -0.015931),
            Point3::new(-1.0625, -0.382955, 0.213901),
            Point3::new(-0.608219, -1.069667, 0.09681),
            Point3::new(-0.893094, -0.763584, 0.178514),
            Point3::new(-0.950082, -0.812308, 0.0),
            Point3::new(-0.935256, -0.799633, 0.09681),
            Point3::new(-0.946339, -0.809108, 0.049378),
            Point3::new(-0.608686, -1.070488, -0.094529),
            Point3::new(-0.895747, -0.765853, -0.175022),
            Point3::new(-0.935975, -0.800247, -0.094529),
            Point3::new(-1.0625, -0.650292, -0.04616),
            Point3::new(-1.0625, -0.628344, -0.086934),
            Point3::new(-1.0625, -0.608, -0.110688),
            Point3::new(-1.0625, -0.585174, -0.130911),
            Point3::new(-1.0625, -0.560057, -0.148562),
            Point3::new(-1.0625, -0.505863, -0.176777),
            Point3::new(-1.0625, -0.382955, -0.213901),
            Point3::new(-0.960957, 0.792069, -0.048167),
            Point3::new(-1.125, 0.0, 0.25),
            Point3::new(-1.303514, 0.0, 0.175022),
            Point3::new(-1.375, 0.0, 0.0),
            Point3::new(-1.300022, 0.0, -0.178514),
            Point3::new(-1.249888, 0.892568, 0.0),
            Point3::new(-1.302379, 0.83903, 0.178514),
            Point3::new(-1.42491, 0.714055, 0.25),
            Point3::new(-1.549886, 0.586586, 0.175022),
            Point3::new(-1.599933, 0.535541, 0.0),
            Point3::new(-1.547442, 0.589079, -0.178514),
            Point3::new(-1.42491, 0.714055, -0.25),
            Point3::new(-1.299935, 0.841523, -0.175022),
            Point3::new(-2.125, 1.25, 0.0),
            Point3::new(-2.125, 1.175022, 0.178514),
            Point3::new(-2.125, 1.0, 0.25),
            Point3::new(-2.125, 0.821486, 0.175022),
            Point3::new(-2.125, 0.75, 0.0),
            Point3::new(-2.125, 0.824978, -0.178514),
            Point3::new(-2.125, 1.0, -0.25),
            Point3::new(-2.125, 1.178514, -0.175022),
            Point3::new(-3.017568, 0.875112, 0.0),
            Point3::new(-2.96403, 0.822621, 0.178514),
            Point3::new(-2.839055, 0.70009, 0.25),
            Point3::new(-2.711586, 0.575114, 0.175022),
            Point3::new(-2.660541, 0.525067, 0.0),
            Point3::new(-2.714079, 0.577558, -0.178514),
            Point3::new(-2.839055, 0.70009, -0.25),
            Point3::new(-2.966523, 0.825066, -0.175022),
            Point3::new(-3.125, 0.0, 0.25),
            Point3::new(-2.946486, 0.0, 0.175022),
            Point3::new(-2.875, 0.0, 0.0),
            Point3::new(-2.949978, 0.0, -0.178514),
            Point3::new(-3.125, 0.0, -0.25),
            Point3::new(-2.947621, -0.83903, 0.178514),
            Point3::new(-2.82509, -0.714055, 0.25),
            Point3::new(-2.700114, -0.586586, 0.175022),
            Point3::new(-2.650067, -0.535541, 0.0),
            Point3::new(-2.702559, -0.589079, -0.178514),
            Point3::new(-2.82509, -0.714055, -0.25),
            Point3::new(-2.950066, -0.841523, -0.175022),
            Point3::new(-3.000112, -0.892568, 0.0),
            Point3::new(-2.125, -1.25, 0.0),
            Point3::new(-2.125, -1.175022, 0.178514),
            Point3::new(-2.125, -1.0, 0.25),
            Point3::new(-2.125, -0.821486, 0.175022),
            Point3::new(-2.125, -0.75, 0.0),
            Point3::new(-2.125, -0.824978, -0.178514),
            Point3::new(-2.125, -1.0, -0.25),
            Point3::new(-2.125, -1.178514, -0.175022),
            Point3::new(-1.232432, -0.875112, 0.0),
            Point3::new(-1.28597, -0.822621, 0.178514),
            Point3::new(-1.410945, -0.70009, 0.25),
            Point3::new(-1.538414, -0.575114, 0.175022),
            Point3::new(-1.589459, -0.525067, 0.0),
            Point3::new(-1.535921, -0.577558, -0.178514),
            Point3::new(-1.410945, -0.70009, -0.25),
            Point3::new(-1.283477, -0.825066, -0.175022),
            Point3::new(-1.125, 0.0, -0.25),
            Point3::new(-1.0625, 0.446047, 0.198228),
            Point3::new(-1.0625, 0.505863, 0.176777),
            Point3::new(-1.0625, 0.560057, 0.148562),
            Point3::new(-1.0625, 0.585174, 0.130911),
            Point3::new(-1.0625, 0.608, 0.110688),
            Point3::new(-1.0625, 0.628344, 0.086934),
            Point3::new(-1.0625, 0.636947, 0.073995),
            Point3::new(-1.0625, 0.644312, 0.060393),
            Point3::new(-1.0625, 0.650292, 0.04616),
            Point3::new(-1.0625, 0.654743, 0.031328),
            Point3::new(-1.0625, 0.65752, 0.015931),
            Point3::new(-1.0625, 0.658478, 0.0),
            Point3::new(-1.263543, 0.878641, 0.09681),
            Point3::new(-1.262882, 0.879315, -0.094529),
            Point3::new(-2.125, 1.230495, 0.09681),
            Point3::new(-1.680704, 1.087786, 0.178514),
            Point3::new(-1.659729, 1.13914, 0.09681),
            Point3::new(-1.0625, 0.654743, -0.031328),
            Point3::new(-1.0625, 0.650292, -0.04616),
            Point3::new(-1.0625, 0.644312, -0.060393),
            Point3::new(-1.0625, 0.636947, -0.073995),
            Point3::new(-1.0625, 0.628344, -0.086934),
            Point3::new(-1.0625, 0.608, -0.110688),
            Point3::new(-1.0625, 0.585174, -0.130911),
            Point3::new(-1.253167, 0.889224, -0.048167),
            Point3::new(-2.125, 1.231439, -0.094529),
            Point3::new(-1.652353, 1.157197, 0.0),
            Point3::new(-1.679384, 1.091018, -0.175022),
            Point3::new(-1.659371, 1.140015, -0.094529),
            Point3::new(-1.0625, -0.658478, 0.0),
            Point3::new(-1.0625, -0.65752, 0.015931),
            Point3::new(-1.0625, -0.654743, 0.031328),
            Point3::new(-1.0625, -0.650292, 0.04616),
            Point3::new(-1.0625, -0.644312, 0.060393),
            Point3::new(-1.0625, -0.636947, 0.073995),
            Point3::new(-1.0625, -0.628344, 0.086934),
            Point3::new(-1.0625, -0.608, 0.110688),
            Point3::new(-1.0625, -0.585174, 0.130911),
            Point3::new(-1.0625, -0.560057, 0.148562),
            Point3::new(-1.0625, -0.505863, 0.176777),
            Point3::new(-1.0625, -0.446047, 0.198228),
            Point3::new(-1.0625, -0.25586, 0.232109),
            Point3::new(-1.246359, -0.861457, 0.09681),
            Point3::new(-1.235948, -0.871665, 0.049378),
            Point3::new(-1.669984, -1.083346, 0.178514),
            Point3::new(-2.125, -1.230495, 0.09681),
            Point3::new(-1.64095, -1.152474, 0.0),
            Point3::new(-1.648503, -1.13449, 0.09681),
            Point3::new(-1.245685, -0.862118, -0.094529),
            Point3::new(-1.0625, -0.25586, -0.232109),
            Point3::new(-1.0625, -0.446047, -0.198228),
            Point3::new(-1.0625, -0.636947, -0.073995),
            Point3::new(-1.0625, -0.644312, -0.060393),
            Point3::new(-1.0625, -0.654743, -0.031328),
            Point3::new(-1.0625, -0.65752, -0.015931),
            Point3::new(-2.125, -1.231439, -0.094529),
            Point3::new(-1.668632, -1.086565, -0.175022),
            Point3::new(-1.648137, -1.135361, -0.094529),
            Point3::new(-3.1875, 0.654743, 0.031328),
            Point3::new(-3.1875, 0.644312, 0.060393),
            Point3::new(-3.1875, 0.636947, 0.073995),
            Point3::new(-3.1875, 0.628344, 0.086934),
            Point3::new(-3.1875, 0.608, 0.110688),
            Point3::new(-3.00364, 0.861457, 0.09681),
            Point3::new(-3.014052, 0.871665, 0.049378),
            Point3::new(-2.580016, 1.083346, 0.178514),
            Point3::new(-2.60905, 1.152474, 0.0),
            Point3::new(-2.601497, 1.13449, 0.09681),
            Point3::new(-3.004315, 0.862118, -0.094529),
            Point3::new(-3.1875, 0.560057, -0.148562),
            Point3::new(-3.1875, 0.628344, -0.086934),
            Point3::new(-3.1875, 0.644312, -0.060393),
            Point3::new(-3.1875, 0.650292, -0.04616),
            Point3::new(-3.1875, 0.654743, -0.031328),
            Point3::new(-2.581368, 1.086565, -0.175022),
            Point3::new(-2.601863, 1.135361, -0.094529),
            Point3::new(-3.1875, -0.382955, 0.213901),
            Point3::new(-3.1875, -0.446047, 0.198228),
            Point3::new(-3.1875, -0.608, 0.110688),
            Point3::new(-3.1875, -0.628344, 0.086934),
            Point3::new(-3.1875, -0.636947, 0.073995),
            Point3::new(-3.1875, -0.644312, 0.060393),
            Point3::new(-3.1875, -0.650292, 0.04616),
            Point3::new(-2.986457, -0.878641, 0.09681),
            Point3::new(-2.987118, -0.879315, -0.094529),
            Point3::new(-2.569296, -1.087786, 0.178514),
            Point3::new(-2.590271, -1.13914, 0.09681),
            Point3::new(-3.1875, -0.65752, -0.015931),
            Point3::new(-3.1875, -0.654743, -0.031328),
            Point3::new(-3.1875, -0.636947, -0.073995),
            Point3::new(-3.1875, -0.628344, -0.086934),
            Point3::new(-3.1875, -0.608, -0.110688),
            Point3::new(-3.1875, -0.585174, -0.130911),
            Point3::new(-3.1875, -0.560057, -0.148562),
            Point3::new(-3.1875, -0.382955, -0.213901),
            Point3::new(-2.597646, -1.157197, 0.0),
            Point3::new(-2.570616, -1.091018, -0.175022),
            Point3::new(-2.590628, -1.140015, -0.094529),
        ];

        #[rustfmt::skip]
            let faces = vec![
            (6, 60, 5),
            (7, 0, 60),
            (8, 1, 0),
            (9, 2, 8),
            (11, 73, 74),
            (73, 4, 70),
            (72, 74, 86),
            (73, 11, 12),
            (5, 4, 73),
            (12, 5, 73),
            (13, 6, 12),
            (14, 7, 13),
            (15, 8, 7),
            (16, 9, 15),
            (87, 10, 9),
            (16, 17, 87),
            (9, 16, 87),
            (18, 86, 88),
            (84, 71, 88),
            (86, 3, 84),
            (88, 86, 84),
            (88, 71, 87),
            (85, 88, 17),
            (72, 18, 26),
            (19, 11, 72),
            (26, 19, 72),
            (20, 12, 11),
            (21, 13, 12),
            (22, 14, 21),
            (23, 15, 14),
            (24, 16, 23),
            (25, 17, 24),
            (85, 17, 25),
            (26, 18, 85),
            (25, 26, 85),
            (27, 19, 26),
            (28, 20, 19),
            (29, 21, 20),
            (30, 22, 21),
            (31, 23, 22),
            (32, 24, 31),
            (33, 25, 24),
            (34, 26, 33),
            (35, 27, 42),
            (36, 28, 35),
            (37, 29, 36),
            (38, 30, 37),
            (39, 31, 38),
            (40, 32, 39),
            (41, 33, 40),
            (42, 34, 33),
            (99, 43, 35),
            (42, 50, 99),
            (35, 42, 99),
            (44, 36, 35),
            (45, 37, 44),
            (46, 38, 45),
            (47, 39, 38),
            (48, 40, 47),
            (49, 41, 40),
            (109, 50, 42),
            (41, 49, 109),
            (42, 41, 109),
            (51, 98, 96),
            (98, 43, 100),
            (100, 99, 108),
            (97, 96, 100),
            (108, 58, 97),
            (100, 108, 97),
            (98, 51, 52),
            (44, 43, 98),
            (52, 44, 98),
            (53, 45, 52),
            (54, 46, 45),
            (55, 47, 54),
            (56, 48, 47),
            (110, 49, 48),
            (56, 57, 110),
            (48, 56, 110),
            (108, 50, 109),
            (111, 109, 49),
            (101, 111, 57),
            (60, 53, 59),
            (0, 54, 60),
            (1, 55, 0),
            (2, 56, 55),
            (4, 64, 65),
            (70, 333, 332),
            (70, 66, 333),
            (4, 65, 66),
            (61, 59, 95),
            (4, 62, 63),
            (61, 5, 59),
            (61, 62, 5),
            (63, 64, 4),
            (3, 329, 68),
            (68, 69, 3),
            (67, 329, 3),
            (86, 74, 3),
            (70, 330, 67),
            (70, 331, 330),
            (3, 70, 67),
            (70, 4, 66),
            (331, 70, 332),
            (62, 4, 5),
            (82, 83, 2),
            (82, 9, 81),
            (10, 80, 81),
            (9, 82, 2),
            (10, 340, 79),
            (10, 78, 340),
            (71, 341, 77),
            (71, 76, 341),
            (77, 78, 71),
            (79, 80, 10),
            (81, 9, 10),
            (10, 71, 78),
            (71, 84, 76),
            (342, 76, 84),
            (84, 344, 343),
            (84, 75, 344),
            (69, 75, 3),
            (84, 3, 75),
            (84, 343, 342),
            (89, 58, 107),
            (97, 58, 90),
            (58, 89, 90),
            (353, 97, 90),
            (97, 351, 96),
            (347, 52, 51),
            (51, 96, 91),
            (351, 97, 352),
            (352, 97, 353),
            (96, 351, 350),
            (51, 91, 92),
            (96, 349, 91),
            (96, 350, 349),
            (51, 348, 347),
            (51, 93, 348),
            (59, 94, 95),
            (94, 59, 52),
            (347, 94, 52),
            (51, 92, 93),
            (105, 101, 360),
            (101, 57, 363),
            (101, 58, 111),
            (365, 57, 56),
            (105, 106, 101),
            (358, 107, 58),
            (58, 359, 358),
            (58, 106, 359),
            (58, 101, 106),
            (57, 365, 103),
            (83, 102, 2),
            (56, 102, 365),
            (102, 56, 2),
            (101, 361, 360),
            (101, 362, 361),
            (57, 364, 363),
            (57, 104, 364),
            (363, 362, 101),
            (57, 103, 104),
            (117, 176, 175),
            (118, 112, 117),
            (119, 113, 118),
            (120, 114, 113),
            (121, 115, 120),
            (122, 116, 115),
            (123, 177, 122),
            (124, 175, 123),
            (125, 117, 132),
            (126, 118, 117),
            (127, 119, 126),
            (128, 120, 119),
            (129, 121, 128),
            (130, 122, 121),
            (131, 123, 122),
            (132, 124, 123),
            (182, 133, 125),
            (132, 140, 182),
            (125, 132, 182),
            (134, 126, 133),
            (135, 127, 134),
            (136, 128, 127),
            (137, 129, 136),
            (138, 130, 129),
            (139, 131, 130),
            (186, 140, 132),
            (131, 139, 186),
            (132, 131, 186),
            (181, 133, 182),
            (184, 183, 182),
            (140, 185, 184),
            (182, 140, 184),
            (142, 135, 134),
            (143, 136, 135),
            (144, 137, 143),
            (145, 138, 137),
            (212, 185, 140),
            (186, 188, 212),
            (140, 186, 212),
            (188, 186, 187),
            (147, 142, 141),
            (148, 143, 147),
            (149, 144, 143),
            (150, 145, 144),
            (151, 198, 200),
            (201, 199, 158),
            (197, 200, 201),
            (158, 197, 201),
            (153, 147, 146),
            (154, 148, 153),
            (155, 149, 148),
            (156, 150, 155),
            (202, 204, 157),
            (197, 158, 166),
            (159, 151, 197),
            (166, 159, 197),
            (160, 152, 151),
            (161, 153, 160),
            (162, 154, 161),
            (163, 155, 154),
            (164, 156, 155),
            (165, 157, 164),
            (202, 157, 165),
            (166, 158, 202),
            (165, 166, 202),
            (168, 159, 166),
            (169, 160, 168),
            (170, 161, 169),
            (171, 162, 170),
            (172, 163, 171),
            (173, 164, 163),
            (174, 165, 173),
            (167, 166, 174),
            (176, 168, 167),
            (112, 169, 168),
            (113, 170, 112),
            (114, 171, 113),
            (115, 172, 114),
            (116, 173, 172),
            (177, 174, 116),
            (175, 167, 177),
            (281, 185, 282),
            (184, 185, 280),
            (185, 281, 280),
            (279, 184, 280),
            (181, 134, 133),
            (184, 277, 183),
            (183, 274, 181),
            (178, 134, 181),
            (279, 278, 184),
            (184, 278, 277),
            (183, 275, 274),
            (183, 276, 275),
            (183, 277, 276),
            (181, 272, 271),
            (181, 273, 272),
            (179, 141, 178),
            (141, 134, 178),
            (181, 271, 178),
            (181, 274, 273),
            (180, 141, 179),
            (187, 139, 138),
            (291, 212, 188),
            (187, 294, 188),
            (290, 212, 291),
            (187, 138, 191),
            (282, 185, 195),
            (212, 195, 185),
            (212, 288, 195),
            (212, 289, 288),
            (289, 212, 290),
            (191, 192, 187),
            (189, 190, 145),
            (191, 145, 190),
            (138, 145, 191),
            (187, 192, 193),
            (292, 291, 188),
            (293, 292, 188),
            (194, 294, 187),
            (193, 194, 187),
            (293, 188, 294),
            (180, 146, 141),
            (309, 198, 310),
            (306, 200, 307),
            (307, 200, 308),
            (309, 308, 198),
            (146, 312, 196),
            (196, 152, 146),
            (310, 198, 311),
            (198, 196, 311),
            (189, 145, 150),
            (200, 306, 305),
            (302, 201, 303),
            (199, 301, 300),
            (302, 301, 199),
            (199, 201, 302),
            (305, 201, 200),
            (303, 201, 304),
            (308, 200, 198),
            (152, 198, 151),
            (196, 198, 152),
            (201, 305, 304),
            (320, 189, 150),
            (203, 321, 211),
            (203, 210, 321),
            (320, 150, 211),
            (203, 208, 209),
            (204, 207, 208),
            (204, 322, 206),
            (204, 323, 322),
            (206, 207, 204),
            (209, 210, 203),
            (323, 204, 205),
            (203, 156, 157),
            (156, 203, 211),
            (202, 158, 204),
            (199, 324, 205),
            (199, 325, 324),
            (325, 199, 300),
            (204, 199, 205),
            (203, 204, 208),
            (150, 156, 211),
            (312, 146, 180),
            (220, 214, 219),
            (221, 215, 214),
            (222, 216, 221),
            (223, 270, 216),
            (226, 286, 287),
            (286, 218, 287),
            (285, 287, 225),
            (286, 226, 227),
            (219, 218, 286),
            (227, 219, 286),
            (228, 220, 227),
            (229, 221, 220),
            (230, 222, 229),
            (231, 223, 222),
            (298, 224, 223),
            (231, 232, 298),
            (223, 231, 298),
            (225, 297, 299),
            (295, 284, 299),
            (297, 217, 295),
            (299, 297, 295),
            (299, 284, 224),
            (296, 299, 232),
            (234, 336, 334),
            (336, 226, 338),
            (338, 285, 337),
            (335, 334, 338),
            (337, 233, 335),
            (338, 337, 335),
            (336, 234, 235),
            (227, 226, 336),
            (235, 227, 336),
            (236, 228, 235),
            (237, 229, 228),
            (238, 230, 237),
            (239, 231, 230),
            (345, 232, 231),
            (239, 240, 345),
            (231, 239, 345),
            (337, 225, 296),
            (346, 296, 232),
            (339, 346, 240),
            (242, 236, 241),
            (243, 237, 242),
            (244, 238, 243),
            (245, 239, 238),
            (248, 242, 247),
            (249, 243, 242),
            (250, 244, 243),
            (251, 245, 250),
            (255, 356, 357),
            (356, 246, 354),
            (316, 357, 366),
            (356, 255, 256),
            (247, 246, 356),
            (256, 247, 356),
            (257, 248, 256),
            (258, 249, 257),
            (259, 250, 249),
            (260, 251, 259),
            (367, 252, 251),
            (260, 261, 367),
            (251, 260, 367),
            (254, 366, 368),
            (368, 355, 367),
            (326, 368, 261),
            (263, 315, 318),
            (315, 255, 316),
            (318, 316, 317),
            (314, 313, 318),
            (317, 262, 314),
            (318, 317, 314),
            (315, 263, 264),
            (256, 255, 315),
            (264, 256, 315),
            (265, 257, 256),
            (266, 258, 265),
            (267, 259, 258),
            (268, 260, 267),
            (327, 261, 260),
            (268, 269, 327),
            (260, 268, 327),
            (317, 254, 328),
            (328, 326, 261),
            (319, 328, 269),
            (214, 265, 264),
            (215, 266, 214),
            (216, 267, 266),
            (270, 268, 216),
            (217, 279, 280),
            (283, 277, 278),
            (283, 276, 277),
            (278, 279, 283),
            (281, 282, 217),
            (217, 283, 279),
            (283, 217, 297),
            (281, 217, 280),
            (275, 276, 283),
            (218, 178, 271),
            (213, 180, 179),
            (179, 219, 213),
            (219, 179, 178),
            (274, 275, 283),
            (218, 273, 274),
            (178, 218, 219),
            (273, 218, 272),
            (272, 218, 271),
            (274, 283, 218),
            (68, 233, 69),
            (233, 329, 335),
            (68, 329, 233),
            (335, 329, 67),
            (334, 335, 331),
            (335, 330, 331),
            (335, 67, 330),
            (331, 332, 334),
            (66, 334, 333),
            (332, 333, 334),
            (62, 235, 234),
            (334, 66, 234),
            (66, 65, 234),
            (234, 64, 63),
            (234, 65, 64),
            (241, 235, 61),
            (235, 62, 61),
            (234, 63, 62),
            (95, 241, 61),
            (190, 189, 270),
            (190, 223, 191),
            (224, 192, 191),
            (223, 190, 270),
            (193, 192, 224),
            (223, 224, 191),
            (294, 194, 224),
            (193, 224, 194),
            (284, 294, 224),
            (284, 292, 293),
            (295, 290, 291),
            (295, 289, 290),
            (291, 292, 284),
            (288, 289, 295),
            (284, 295, 291),
            (195, 288, 295),
            (282, 195, 217),
            (295, 217, 195),
            (284, 293, 294),
            (79, 340, 240),
            (339, 240, 78),
            (240, 340, 78),
            (81, 240, 239),
            (77, 339, 78),
            (239, 82, 81),
            (80, 79, 240),
            (81, 80, 240),
            (82, 239, 245),
            (339, 233, 346),
            (233, 75, 69),
            (343, 233, 339),
            (339, 76, 342),
            (339, 341, 76),
            (233, 344, 75),
            (233, 343, 344),
            (339, 342, 343),
            (341, 339, 77),
            (241, 95, 94),
            (253, 353, 90),
            (354, 351, 352),
            (354, 350, 351),
            (352, 353, 354),
            (89, 107, 253),
            (253, 354, 353),
            (354, 253, 357),
            (89, 253, 90),
            (349, 350, 354),
            (348, 246, 347),
            (94, 247, 241),
            (94, 347, 247),
            (246, 348, 93),
            (354, 91, 349),
            (91, 354, 246),
            (347, 246, 247),
            (91, 246, 92),
            (92, 246, 93),
            (301, 262, 300),
            (305, 313, 314),
            (314, 262, 302),
            (262, 301, 302),
            (314, 302, 303),
            (313, 305, 306),
            (314, 304, 305),
            (314, 303, 304),
            (307, 313, 306),
            (313, 308, 263),
            (196, 264, 263),
            (308, 309, 263),
            (309, 310, 263),
            (307, 308, 313),
            (263, 311, 196),
            (263, 310, 311),
            (213, 312, 180),
            (312, 213, 264),
            (196, 312, 264),
            (102, 83, 245),
            (252, 103, 365),
            (252, 104, 103),
            (365, 102, 251),
            (104, 252, 364),
            (252, 363, 364),
            (363, 355, 362),
            (355, 363, 252),
            (251, 252, 365),
            (102, 245, 251),
            (355, 360, 361),
            (355, 106, 105),
            (253, 359, 106),
            (105, 360, 355),
            (253, 358, 359),
            (355, 253, 106),
            (368, 366, 253),
            (358, 253, 107),
            (355, 361, 362),
            (210, 209, 269),
            (319, 269, 208),
            (269, 209, 208),
            (211, 269, 268),
            (207, 319, 208),
            (189, 320, 270),
            (268, 320, 211),
            (269, 321, 210),
            (269, 211, 321),
            (270, 320, 268),
            (319, 262, 317),
            (262, 325, 300),
            (205, 262, 319),
            (319, 322, 323),
            (319, 206, 322),
            (262, 324, 325),
            (262, 205, 324),
            (319, 323, 205),
            (206, 319, 207),
            (82, 245, 83),
            (60, 59, 5),
            (7, 60, 6),
            (8, 0, 7),
            (2, 1, 8),
            (11, 74, 72),
            (73, 70, 74),
            (72, 86, 18),
            (6, 5, 12),
            (7, 6, 13),
            (15, 7, 14),
            (9, 8, 15),
            (18, 88, 85),
            (71, 10, 87),
            (88, 87, 17),
            (20, 11, 19),
            (21, 12, 20),
            (14, 13, 21),
            (23, 14, 22),
            (16, 15, 23),
            (17, 16, 24),
            (27, 26, 34),
            (28, 19, 27),
            (29, 20, 28),
            (30, 21, 29),
            (31, 22, 30),
            (24, 23, 31),
            (33, 24, 32),
            (26, 25, 33),
            (27, 34, 42),
            (28, 27, 35),
            (29, 28, 36),
            (30, 29, 37),
            (31, 30, 38),
            (32, 31, 39),
            (33, 32, 40),
            (42, 33, 41),
            (44, 35, 43),
            (37, 36, 44),
            (38, 37, 45),
            (47, 38, 46),
            (40, 39, 47),
            (49, 40, 48),
            (98, 100, 96),
            (43, 99, 100),
            (99, 50, 108),
            (45, 44, 52),
            (54, 45, 53),
            (47, 46, 54),
            (56, 47, 55),
            (108, 109, 111),
            (111, 49, 110),
            (111, 110, 57),
            (53, 52, 59),
            (54, 53, 60),
            (55, 54, 0),
            (2, 55, 1),
            (74, 70, 3),
            (58, 108, 111),
            (117, 175, 124),
            (112, 176, 117),
            (113, 112, 118),
            (120, 113, 119),
            (115, 114, 120),
            (122, 115, 121),
            (177, 116, 122),
            (175, 177, 123),
            (117, 124, 132),
            (126, 117, 125),
            (119, 118, 126),
            (128, 119, 127),
            (121, 120, 128),
            (130, 121, 129),
            (131, 122, 130),
            (132, 123, 131),
            (126, 125, 133),
            (127, 126, 134),
            (136, 127, 135),
            (129, 128, 136),
            (138, 129, 137),
            (139, 130, 138),
            (181, 182, 183),
            (142, 134, 141),
            (143, 135, 142),
            (137, 136, 143),
            (145, 137, 144),
            (186, 139, 187),
            (147, 141, 146),
            (143, 142, 147),
            (149, 143, 148),
            (150, 144, 149),
            (151, 200, 197),
            (153, 146, 152),
            (148, 147, 153),
            (155, 148, 154),
            (150, 149, 155),
            (204, 203, 157),
            (160, 151, 159),
            (153, 152, 160),
            (154, 153, 161),
            (163, 154, 162),
            (164, 155, 163),
            (157, 156, 164),
            (168, 166, 167),
            (160, 159, 168),
            (161, 160, 169),
            (162, 161, 170),
            (163, 162, 171),
            (173, 163, 172),
            (165, 164, 173),
            (166, 165, 174),
            (176, 167, 175),
            (112, 168, 176),
            (170, 169, 112),
            (171, 170, 113),
            (172, 171, 114),
            (116, 172, 115),
            (174, 173, 116),
            (167, 174, 177),
            (158, 199, 204),
            (214, 213, 219),
            (221, 214, 220),
            (216, 215, 221),
            (223, 216, 222),
            (226, 287, 285),
            (218, 283, 287),
            (287, 297, 225),
            (220, 219, 227),
            (229, 220, 228),
            (222, 221, 229),
            (231, 222, 230),
            (225, 299, 296),
            (299, 224, 298),
            (299, 298, 232),
            (336, 338, 334),
            (226, 285, 338),
            (285, 225, 337),
            (228, 227, 235),
            (237, 228, 236),
            (230, 229, 237),
            (239, 230, 238),
            (337, 296, 346),
            (346, 232, 345),
            (346, 345, 240),
            (236, 235, 241),
            (237, 236, 242),
            (238, 237, 243),
            (245, 238, 244),
            (242, 241, 247),
            (249, 242, 248),
            (250, 243, 249),
            (245, 244, 250),
            (255, 357, 316),
            (356, 354, 357),
            (316, 366, 254),
            (248, 247, 256),
            (249, 248, 257),
            (259, 249, 258),
            (251, 250, 259),
            (254, 368, 326),
            (355, 252, 367),
            (368, 367, 261),
            (263, 318, 313),
            (315, 316, 318),
            (316, 254, 317),
            (265, 256, 264),
            (258, 257, 265),
            (267, 258, 266),
            (260, 259, 267),
            (254, 326, 328),
            (328, 261, 327),
            (328, 327, 269),
            (214, 264, 213),
            (266, 265, 214),
            (216, 266, 215),
            (268, 267, 216),
            (283, 297, 287),
            (233, 337, 346),
            (253, 366, 357),
            (368, 253, 355),
            (319, 317, 328),
        ];

        (faces, vertices)
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

        Geometry::from_triangle_faces_with_vertices_and_computed_normals(
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

        Geometry::from_triangle_faces_with_vertices_and_normals(
            faces.clone(),
            vertices.clone(),
            normals.clone(),
        );
    }

    #[test]
    fn test_oriented_edge_eq_returns_true() {
        let oriented_edge_one_way = OrientedEdge::new(0, 1);
        let oriented_edge_other_way = OrientedEdge::new(0, 1);
        assert_eq!(oriented_edge_one_way, oriented_edge_other_way);
    }

    #[test]
    fn test_oriented_edge_eq_returns_false_because_different() {
        let oriented_edge_one_way = OrientedEdge::new(0, 1);
        let oriented_edge_other_way = OrientedEdge::new(2, 1);
        assert_ne!(oriented_edge_one_way, oriented_edge_other_way);
    }

    #[test]
    fn test_oriented_edge_eq_returns_false_because_reverted() {
        let oriented_edge_one_way = OrientedEdge::new(0, 1);
        let oriented_edge_other_way = OrientedEdge::new(1, 0);
        assert_ne!(oriented_edge_one_way, oriented_edge_other_way);
    }

    #[test]
    #[should_panic(expected = "The oriented edge is constituted of the same vertex")]
    fn test_oriented_edge_constructor_consists_of_the_same_vertex_should_panic() {
        OrientedEdge::new(0, 0);
    }

    #[test]
    fn test_oriented_edge_constructor_doesnnt_consist_of_the_same_vertex_should_pass() {
        OrientedEdge::new(0, 1);
    }

    #[test]
    fn test_oriented_edge_is_reverted_returns_true_because_same() {
        let oriented_edge_one_way = OrientedEdge::new(0, 1);
        let oriented_edge_other_way = OrientedEdge::new(1, 0);
        assert!(oriented_edge_one_way.is_reverted(oriented_edge_other_way));
    }

    #[test]
    fn test_oriented_edge_is_reverted_returns_false_because_is_same() {
        let oriented_edge_one_way = OrientedEdge::new(0, 1);
        let oriented_edge_other_way = OrientedEdge::new(0, 1);
        assert!(!oriented_edge_one_way.is_reverted(oriented_edge_other_way));
    }

    #[test]
    fn test_oriented_edge_is_reverted_returns_false_because_is_different() {
        let oriented_edge_one_way = OrientedEdge::new(0, 1);
        let oriented_edge_other_way = OrientedEdge::new(2, 1);
        assert!(!oriented_edge_one_way.is_reverted(oriented_edge_other_way));
    }

    #[test]
    fn test_unoriented_edge_eq_returns_true_because_same() {
        let unoriented_edge_one_way = UnorientedEdge(OrientedEdge::new(0, 1));
        let unoriented_edge_other_way = UnorientedEdge(OrientedEdge::new(0, 1));
        assert_eq!(unoriented_edge_one_way, unoriented_edge_other_way);
    }

    #[test]
    fn test_unoriented_edge_eq_returns_true_because_reverted() {
        let unoriented_edge_one_way = UnorientedEdge(OrientedEdge::new(0, 1));
        let unoriented_edge_other_way = UnorientedEdge(OrientedEdge::new(1, 0));
        assert_eq!(unoriented_edge_one_way, unoriented_edge_other_way);
    }

    #[test]
    fn test_unoriented_edge_eq_returns_false_because_different() {
        let unoriented_edge_one_way = UnorientedEdge(OrientedEdge::new(0, 1));
        let unoriented_edge_other_way = UnorientedEdge(OrientedEdge::new(2, 1));
        assert_ne!(unoriented_edge_one_way, unoriented_edge_other_way);
    }

    #[test]
    fn test_unoriented_edge_hash_returns_true_because_same() {
        let unoriented_edge_one_way = UnorientedEdge(OrientedEdge::new(0, 1));
        let unoriented_edge_other_way = UnorientedEdge(OrientedEdge::new(0, 1));
        let mut hasher_1 = DefaultHasher::new();
        let mut hasher_2 = DefaultHasher::new();
        unoriented_edge_one_way.hash(&mut hasher_1);
        unoriented_edge_other_way.hash(&mut hasher_2);
        assert_eq!(hasher_1.finish(), hasher_2.finish());
    }

    #[test]
    fn test_unoriented_edge_hash_returns_true_because_reverted() {
        let unoriented_edge_one_way = UnorientedEdge(OrientedEdge::new(0, 1));
        let unoriented_edge_other_way = UnorientedEdge(OrientedEdge::new(1, 0));
        let mut hasher_1 = DefaultHasher::new();
        let mut hasher_2 = DefaultHasher::new();
        unoriented_edge_one_way.hash(&mut hasher_1);
        unoriented_edge_other_way.hash(&mut hasher_2);
        assert_eq!(hasher_1.finish(), hasher_2.finish());
    }

    #[test]
    fn test_unoriented_edge_hash_returns_false_because_different() {
        let unoriented_edge_one_way = UnorientedEdge(OrientedEdge::new(0, 1));
        let unoriented_edge_other_way = UnorientedEdge(OrientedEdge::new(2, 1));
        let mut hasher_1 = DefaultHasher::new();
        let mut hasher_2 = DefaultHasher::new();
        unoriented_edge_one_way.hash(&mut hasher_1);
        unoriented_edge_other_way.hash(&mut hasher_2);
        assert_ne!(hasher_1.finish(), hasher_2.finish());
    }

    #[test]
    fn test_triangle_face_to_oriented_edges() {
        let face = TriangleFace::new(0, 1, 2);

        let oriented_edges_correct: [OrientedEdge; 3] = [
            OrientedEdge::new(0, 1),
            OrientedEdge::new(1, 2),
            OrientedEdge::new(2, 0),
        ];

        let oriented_edges_to_check: [OrientedEdge; 3] = face.to_oriented_edges();

        assert_eq!(oriented_edges_to_check[0], oriented_edges_correct[0]);
        assert_eq!(oriented_edges_to_check[1], oriented_edges_correct[1]);
        assert_eq!(oriented_edges_to_check[2], oriented_edges_correct[2]);
    }

    #[test]
    fn test_triangle_face_to_unoriented_edges() {
        let face = TriangleFace::new(0, 1, 2);

        let unoriented_edges_correct: [UnorientedEdge; 3] = [
            UnorientedEdge(OrientedEdge::new(0, 1)),
            UnorientedEdge(OrientedEdge::new(1, 2)),
            UnorientedEdge(OrientedEdge::new(2, 0)),
        ];

        let unoriented_edges_to_check: [UnorientedEdge; 3] = face.to_unoriented_edges();

        assert_eq!(unoriented_edges_to_check[0], unoriented_edges_correct[0]);
        assert_eq!(unoriented_edges_to_check[1], unoriented_edges_correct[1]);
        assert_eq!(unoriented_edges_to_check[2], unoriented_edges_correct[2]);
    }

    #[test]
    fn test_has_no_orphan_vertices_returns_true_if_there_are_some() {
        let (faces, vertices, normals) = quad_with_normals();

        let geometry_without_orphans = Geometry::from_triangle_faces_with_vertices_and_normals(
            faces.clone(),
            vertices.clone(),
            normals.clone(),
        );

        assert!(geometry_without_orphans.has_no_orphan_vertices());
    }

    #[test]
    fn test_has_no_orphan_vertices_returns_false_if_there_are_none() {
        let (faces, vertices, normals) = quad_with_normals();
        let extra_vertex = vec![v(0.0, 0.0, 0.0, [0.0, 0.0, 0.0], 1.0)];
        let vertices_extended = [&vertices[..], &extra_vertex[..]].concat();

        let geometry_with_orphans = Geometry::from_triangle_faces_with_vertices_and_normals(
            faces.clone(),
            vertices_extended.clone(),
            normals.clone(),
        );

        assert!(!geometry_with_orphans.has_no_orphan_vertices());
    }

    #[test]
    fn test_has_no_orphan_normals_returns_true_if_there_are_some() {
        let (faces, vertices, normals) = quad_with_normals();

        let geometry_without_orphans = Geometry::from_triangle_faces_with_vertices_and_normals(
            faces.clone(),
            vertices.clone(),
            normals.clone(),
        );

        assert!(geometry_without_orphans.has_no_orphan_normals());
    }

    #[test]
    fn test_geometry_unoriented_edges_iter() {
        let (faces, vertices, normals) = quad_with_normals();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_normals(
            faces.clone(),
            vertices.clone(),
            normals.clone(),
        );
        let unoriented_edges_correct = vec![
            UnorientedEdge(OrientedEdge::new(0, 1)),
            UnorientedEdge(OrientedEdge::new(1, 2)),
            UnorientedEdge(OrientedEdge::new(2, 0)),
            UnorientedEdge(OrientedEdge::new(2, 3)),
            UnorientedEdge(OrientedEdge::new(3, 0)),
            UnorientedEdge(OrientedEdge::new(0, 2)),
        ];
        let unoriented_edges_to_check: Vec<UnorientedEdge> =
            geometry.unoriented_edges_iter().collect();

        assert!(unoriented_edges_to_check
            .iter()
            .all(|u_e| unoriented_edges_correct.iter().any(|e| e == u_e)));

        let len_1 = unoriented_edges_to_check.len();
        let len_2 = unoriented_edges_correct.len();
        assert_eq!(
            len_1, len_2,
            "unoriented_edges_to_check.len() = {}, unoriented_edges_correct.len() = {}",
            len_1, len_2
        );
    }

    #[test]
    fn test_geometry_oriented_edges_iter() {
        let (faces, vertices, normals) = quad_with_normals();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_normals(
            faces.clone(),
            vertices.clone(),
            normals.clone(),
        );

        let oriented_edges_correct = vec![
            OrientedEdge::new(0, 1),
            OrientedEdge::new(1, 2),
            OrientedEdge::new(2, 0),
            OrientedEdge::new(2, 3),
            OrientedEdge::new(3, 0),
            OrientedEdge::new(0, 2),
        ];
        let oriented_edges_to_check: Vec<OrientedEdge> = geometry.oriented_edges_iter().collect();

        assert!(oriented_edges_to_check
            .iter()
            .all(|o_e| oriented_edges_correct.iter().any(|e| e == o_e)));

        let len_1 = oriented_edges_to_check.len();
        let len_2 = oriented_edges_correct.len();

        assert_eq!(
            len_1, len_2,
            "oriented_edges_to_check.len() = {}, oriented_edges_correct.len() = {}",
            len_1, len_2
        );
    }

    #[test]
    fn test_has_no_orphan_normals_returns_false_if_there_are_none() {
        let (faces, vertices, normals) = quad_with_normals();
        let extra_normal = vec![n(0.0, 0.0, 0.0)];
        let normals_extended = [&normals[..], &extra_normal[..]].concat();

        let geometry_with_orphans = Geometry::from_triangle_faces_with_vertices_and_normals(
            faces.clone(),
            vertices.clone(),
            normals_extended.clone(),
        );

        assert!(!geometry_with_orphans.has_no_orphan_normals());
    }

    #[test]
    fn test_geometry_mesh_genus_box_should_be_0() {
        let geometry = cube_sharp_var_len([0.0, 0.0, 0.0], 1.0);
        let edges: HashSet<UnorientedEdge> = geometry.unoriented_edges_iter().collect();

        let genus = geometry.mesh_genus(&edges);
        assert_eq!(genus, 0);
    }

    #[test]
    fn test_geometry_mesh_genus_torus_should_be_1() {
        let (faces, vertices) = torus();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let edges: HashSet<UnorientedEdge> = geometry.unoriented_edges_iter().collect();

        let genus = geometry.mesh_genus(&edges);
        assert_eq!(genus, 1);
    }

    #[test]
    fn test_geometry_mesh_genus_double_torus_should_be_2() {
        let (faces, vertices) = double_torus();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let edges: HashSet<UnorientedEdge> = geometry.unoriented_edges_iter().collect();

        let genus = geometry.mesh_genus(&edges);
        assert_eq!(genus, 2);
    }

    #[test]
    fn test_geometry_mesh_genus_triple_torus_should_be_3() {
        let (faces, vertices) = triple_torus();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let edges: HashSet<UnorientedEdge> = geometry.unoriented_edges_iter().collect();

        let genus = geometry.mesh_genus(&edges);
        assert_eq!(genus, 3);
    }
}
