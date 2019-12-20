use std::cmp;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter::IntoIterator;

use arrayvec::ArrayVec;
use nalgebra::{Point3, Vector3};

use crate::convert::{cast_u32, cast_usize};
use crate::geometry;

pub mod analysis;
pub mod primitive;
pub mod smoothing;
pub mod tools;
pub mod topology;

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
/// The mesh data lives in right-handed coordinate space with the
/// XY plane being the ground and Z axis growing upwards.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Mesh {
    faces: Vec<Face>,
    vertices: Vec<Point3<f32>>,
    normals: Vec<Vector3<f32>>,
}

impl Mesh {
    /// Creates new triangulated mesh geometry from provided triangle
    /// faces and vertices, and computes normals based on
    /// `normal_strategy`.
    ///
    /// # Panics
    /// Panics if faces refer to out-of-bounds vertices.
    pub fn from_triangle_faces_with_vertices_and_computed_normals<F, V>(
        faces: F,
        vertices: V,
        normal_strategy: NormalStrategy,
    ) -> Self
    where
        F: IntoIterator<Item = (u32, u32, u32)>,
        V: IntoIterator<Item = Point3<f32>>,
    {
        match normal_strategy {
            NormalStrategy::Sharp => {
                // To avoid one additional cloning of this collection, we
                // first materialize it into its final structure (Vec<Face>),
                // and later assert the only kind of faces there are
                // triangles.
                let faces_collection: Vec<_> = faces
                    .into_iter()
                    .enumerate()
                    .map(|(i, (i1, i2, i3))| {
                        let normal_index = cast_u32(i);
                        TriangleFace::new(i1, i2, i3, normal_index, normal_index, normal_index)
                    })
                    .map(Face::from)
                    .collect();

                assert!(
                    !faces_collection.is_empty(),
                    "Empty (faceless) meshes are not supported",
                );

                let vertices_collection: Vec<_> = vertices.into_iter().collect();
                let mut normals_collection = Vec::with_capacity(faces_collection.len());

                let vertices_range = 0..cast_u32(vertices_collection.len());
                for face in &faces_collection {
                    match face {
                        Face::Triangle(triangle_face) => {
                            let (v1, v2, v3) = triangle_face.vertices;

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

                            let face_normal = geometry::compute_triangle_normal(
                                &vertices_collection[cast_usize(v1)],
                                &vertices_collection[cast_usize(v2)],
                                &vertices_collection[cast_usize(v3)],
                            );

                            normals_collection.push(face_normal);

                        }
                        // FIXME: once we add other kinds of faces, they must panic here
                        // _ => panic!("Face must be a triangle, we just created it"),
                    }
                }

                assert_eq!(normals_collection.len(), faces_collection.len());
                assert_eq!(normals_collection.capacity(), faces_collection.len());

                Self {
                    faces: faces_collection,
                    vertices: vertices_collection,
                    normals: normals_collection,
                }
            }
        }
    }

    /// Creates new triangulated mesh geometry from provided triangle
    /// faces and vertices, removes orphan vertices, and computes
    /// normals based on `normal_strategy`.
    ///
    /// # Panics
    /// Panics if faces refer to out-of-bounds vertices.
    pub fn from_triangle_faces_with_vertices_and_computed_normals_remove_orphans<F, V>(
        faces: F,
        vertices: V,
        normal_strategy: NormalStrategy,
    ) -> Self
    where
        F: IntoIterator<Item = (u32, u32, u32)>,
        V: IntoIterator<Item = Point3<f32>>,
    {
        let (faces_purged, vertices_purged) =
            remove_orphan_vertices(faces.into_iter().collect(), vertices.into_iter().collect());
        Self::from_triangle_faces_with_vertices_and_computed_normals(
            faces_purged,
            vertices_purged,
            normal_strategy,
        )
    }

    /// Creates new triangulated mesh geometry from provided triangle
    /// faces, vertices and normals.
    ///
    /// # Panics
    /// Panics if faces refer to out-of-bounds vertices or normals.
    pub fn from_triangle_faces_with_vertices_and_normals<F, V, N>(
        faces: F,
        vertices: V,
        normals: N,
    ) -> Self
    where
        F: IntoIterator<Item = TriangleFace>,
        V: IntoIterator<Item = Point3<f32>>,
        N: IntoIterator<Item = Vector3<f32>>,
    {
        Self::from_faces_with_vertices_and_normals(
            faces.into_iter().map(Face::Triangle),
            vertices,
            normals,
        )
    }

    /// Creates new triangulated mesh geometry from provided triangle
    /// faces, vertices and normals, and removes orphan vertices and
    /// normals.
    ///
    /// # Panics
    /// Panics if faces refer to out-of-bounds vertices or normals.
    pub fn from_triangle_faces_with_vertices_and_normals_remove_orphans<F, V, N>(
        faces: F,
        vertices: V,
        normals: N,
    ) -> Self
    where
        F: IntoIterator<Item = TriangleFace>,
        V: IntoIterator<Item = Point3<f32>>,
        N: IntoIterator<Item = Vector3<f32>>,
    {
        let (faces_purged, vertices_purged, normals_purged) = remove_orphan_vertices_and_normals(
            faces.into_iter().map(Face::Triangle).collect(),
            vertices.into_iter().collect(),
            normals.into_iter().collect(),
        );

        Self::from_faces_with_vertices_and_normals(faces_purged, vertices_purged, normals_purged)
    }

    /// Creates new mesh of any face kind from provided faces,
    /// vertices and normals.
    ///
    /// # Panics
    /// Panics if faces refer to out-of-bounds vertices or normals.
    pub fn from_faces_with_vertices_and_normals<F, V, N>(faces: F, vertices: V, normals: N) -> Self
    where
        F: IntoIterator<Item = Face>,
        V: IntoIterator<Item = Point3<f32>>,
        N: IntoIterator<Item = Vector3<f32>>,
    {
        let faces_collection: Vec<_> = faces.into_iter().collect();
        assert!(
            !faces_collection.is_empty(),
            "Empty (faceless) meshes are not supported.",
        );

        let vertices_collection: Vec<_> = vertices.into_iter().collect();
        let normals_collection: Vec<_> = normals.into_iter().collect();

        let vertices_range = 0..cast_u32(vertices_collection.len());
        let normals_range = 0..cast_u32(normals_collection.len());

        for face in &faces_collection {
            match face {
                Face::Triangle(triangle_face) => {
                    let v = triangle_face.vertices;
                    let n = triangle_face.normals;
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
            }
        }

        Self {
            faces: faces_collection,
            vertices: vertices_collection,
            normals: normals_collection,
        }
    }

    /// Creates new triangulated mesh from provided triangle faces,
    /// vertices, and normals and removes orphan vertices and normals.
    ///
    /// # Panics
    /// Panics if faces refer to out-of-bounds vertices or normals.
    pub fn from_faces_with_vertices_and_normals_remove_orphans<F, V, N>(
        faces: F,
        vertices: V,
        normals: N,
    ) -> Self
    where
        F: IntoIterator<Item = Face>,
        V: IntoIterator<Item = Point3<f32>>,
        N: IntoIterator<Item = Vector3<f32>>,
    {
        let (faces_purged, vertices_purged, normals_purged) = remove_orphan_vertices_and_normals(
            faces.into_iter().collect(),
            vertices.into_iter().collect(),
            normals.into_iter().collect(),
        );

        Self::from_faces_with_vertices_and_normals(faces_purged, vertices_purged, normals_purged)
    }

    pub fn faces(&self) -> &[Face] {
        &self.faces
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

    /// Extracts oriented edges from all mesh faces.
    pub fn oriented_edges_iter<'a>(&'a self) -> impl Iterator<Item = OrientedEdge> + 'a {
        self.faces.iter().flat_map(|face| match face {
            Face::Triangle(triangle_face) => {
                ArrayVec::from(triangle_face.to_oriented_edges()).into_iter()
            }
        })
    }

    /// Extracts unoriented edges from all mesh faces.
    pub fn unoriented_edges_iter<'a>(&'a self) -> impl Iterator<Item = UnorientedEdge> + 'a {
        self.faces.iter().flat_map(|face| match face {
            Face::Triangle(triangle_face) => {
                ArrayVec::from(triangle_face.to_unoriented_edges()).into_iter()
            }
        })
    }

    /// Returns whether the mesh is triangulated - contains
    /// exclusively triangle faces.
    pub fn is_triangulated(&self) -> bool {
        self.faces().iter().all(|face| match face {
            Face::Triangle(_) => true,
        })
    }

    /// Returns whether the mesh contains unused (not referenced in
    /// faces) vertices.
    pub fn has_no_orphan_vertices(&self) -> bool {
        let mut used_vertices = HashSet::new();

        for face in self.faces() {
            match face {
                Face::Triangle(triangle_face) => {
                    used_vertices.insert(triangle_face.vertices.0);
                    used_vertices.insert(triangle_face.vertices.1);
                    used_vertices.insert(triangle_face.vertices.2);
                }
            }
        }

        used_vertices.len() == self.vertices().len()
    }

    /// Returns whether the mesh contains unused (not referenced in
    /// faces) normals.
    pub fn has_no_orphan_normals(&self) -> bool {
        let mut used_normals = HashSet::new();

        for face in self.faces() {
            match face {
                Face::Triangle(triangle_face) => {
                    used_normals.insert(triangle_face.normals.0);
                    used_normals.insert(triangle_face.normals.1);
                    used_normals.insert(triangle_face.normals.2);
                }
            }
        }

        used_normals.len() == self.normals().len()
    }
}

impl fmt::Display for Mesh {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let vertices: Vec<_> = self
            .vertices
            .iter()
            .enumerate()
            .map(|(i, v)| format!("{}: {}", i, v))
            .collect();
        let faces: Vec<_> = self
            .faces
            .iter()
            .enumerate()
            .map(|(i, f)| format!("{}: {}", i, f))
            .collect();
        let normals: Vec<_> = self
            .normals
            .iter()
            .enumerate()
            .map(|(i, n)| format!("{}: ({}, {}, {})", i, n.x, n.y, n.z))
            .collect();

        write!(
            f,
            "G( V({}): {:?}, N({}): {:?}, F({}): {:?} )",
            self.vertices.len(),
            vertices,
            self.normals.len(),
            normals,
            self.faces.len(),
            faces,
        )
    }
}

/// A mesh face. Contains indices to other mesh data, such as vertices
/// and normals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Face {
    Triangle(TriangleFace),
}

impl Face {
    pub fn contains_vertex(&self, vertex_index: u32) -> bool {
        match self {
            Face::Triangle(triangle_face) => triangle_face.contains_vertex(vertex_index),
        }
    }
}

impl From<TriangleFace> for Face {
    fn from(triangle_face: TriangleFace) -> Face {
        Face::Triangle(triangle_face)
    }
}

impl fmt::Display for Face {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Face::Triangle(face) => write!(f, "{}", face),
        }
    }
}

/// A triangular mesh face. Contains indices to other mesh data, such
/// as vertices and normals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub struct TriangleFace {
    pub vertices: (u32, u32, u32),
    pub normals: (u32, u32, u32),
}

impl TriangleFace {
    pub fn new(vi1: u32, vi2: u32, vi3: u32, ni1: u32, ni2: u32, ni3: u32) -> TriangleFace {
        assert!(
            vi1 != vi2 && vi1 != vi3 && vi2 != vi3,
            "One or more face edges consists of the same vertex"
        );

        if vi1 < vi2 && vi1 < vi3 {
            TriangleFace {
                vertices: (vi1, vi2, vi3),
                normals: (ni1, ni2, ni3),
            }
        } else if vi2 < vi1 && vi2 < vi3 {
            TriangleFace {
                vertices: (vi2, vi3, vi1),
                normals: (ni2, ni3, ni1),
            }
        } else {
            TriangleFace {
                vertices: (vi3, vi1, vi2),
                normals: (ni3, ni1, ni2),
            }
        }
    }

    pub fn from_same_vertex_and_normal_index(i1: u32, i2: u32, i3: u32) -> TriangleFace {
        TriangleFace::new(i1, i2, i3, i1, i2, i3)
    }

    /// Generates 3 oriented edges from the respective triangular face.
    pub fn to_oriented_edges(&self) -> [OrientedEdge; 3] {
        [
            OrientedEdge::new(self.vertices.0, self.vertices.1),
            OrientedEdge::new(self.vertices.1, self.vertices.2),
            OrientedEdge::new(self.vertices.2, self.vertices.0),
        ]
    }

    /// Generates 3 unoriented edges from the respective triangular face.
    pub fn to_unoriented_edges(&self) -> [UnorientedEdge; 3] {
        [
            UnorientedEdge(OrientedEdge::new(self.vertices.0, self.vertices.1)),
            UnorientedEdge(OrientedEdge::new(self.vertices.1, self.vertices.2)),
            UnorientedEdge(OrientedEdge::new(self.vertices.2, self.vertices.0)),
        ]
    }

    /// Returns whether the face contains the vertex index.
    pub fn contains_vertex(&self, vertex_index: u32) -> bool {
        self.vertices.0 == vertex_index
            || self.vertices.1 == vertex_index
            || self.vertices.2 == vertex_index
    }

    /// Returns whether the face contains the oriented edge.
    pub fn contains_oriented_edge(&self, oriented_edge: OrientedEdge) -> bool {
        let [oe0, oe1, oe2] = self.to_oriented_edges();
        oe0 == oriented_edge || oe1 == oriented_edge || oe2 == oriented_edge
    }

    /// Returns whether the face contains the unoriented edge.
    pub fn contains_unoriented_edge(&self, unoriented_edge: UnorientedEdge) -> bool {
        let [ue0, ue1, ue2] = self.to_unoriented_edges();
        ue0 == unoriented_edge || ue1 == unoriented_edge || ue2 == unoriented_edge
    }

    /// Returns the same face with reverted vertex and normal winding.
    pub fn to_reverted(&self) -> TriangleFace {
        TriangleFace::new(
            self.vertices.2,
            self.vertices.1,
            self.vertices.0,
            self.normals.2,
            self.normals.1,
            self.normals.0,
        )
    }

    /// Checks if the other face references the same vertices and normals in a
    /// reverted order.
    pub fn is_reverted(&self, other: &Self) -> bool {
        self.to_reverted() == *other
    }
}

impl From<(u32, u32, u32)> for TriangleFace {
    fn from((i1, i2, i3): (u32, u32, u32)) -> TriangleFace {
        TriangleFace::from_same_vertex_and_normal_index(i1, i2, i3)
    }
}

impl fmt::Display for TriangleFace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "T(V: ({}, {}, {}); N: ({}, {}, {}))",
            self.vertices.0,
            self.vertices.1,
            self.vertices.2,
            self.normals.0,
            self.normals.1,
            self.normals.2,
        )
    }
}

/// Oriented face edge.
///
/// Contains indices to other mesh data - vertices.
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

    pub fn contains_vertex(self, vertex_index: u32) -> bool {
        self.vertices.0 == vertex_index || self.vertices.1 == vertex_index
    }

    pub fn to_reverted(self) -> Self {
        OrientedEdge::new(self.vertices.1, self.vertices.0)
    }

    pub fn to_unoriented(self) -> UnorientedEdge {
        UnorientedEdge(self)
    }
}

/// Unoriented face edge.
///
/// Contains indices to other mesh data - vertices. Implements
/// orientation indifferent hash and equal methods
#[derive(Debug, Clone, Copy, Eq)]
pub struct UnorientedEdge(pub OrientedEdge);

impl UnorientedEdge {
    pub fn shares_vertex(self, other: UnorientedEdge) -> bool {
        other.0.contains_vertex(self.0.vertices.0) || other.0.contains_vertex(self.0.vertices.1)
    }
}

impl PartialEq for UnorientedEdge {
    fn eq(&self, other: &Self) -> bool {
        (self.0.vertices.0 == other.0.vertices.0 && self.0.vertices.1 == other.0.vertices.1)
            || (self.0.vertices.0 == other.0.vertices.1 && self.0.vertices.1 == other.0.vertices.0)
    }
}

// FIXME: test
impl Hash for UnorientedEdge {
    fn hash<H: Hasher>(&self, state: &mut H) {
        cmp::min(self.0.vertices.0, self.0.vertices.1).hash(state);
        cmp::max(self.0.vertices.0, self.0.vertices.1).hash(state);
    }
}

#[allow(clippy::type_complexity)]
fn remove_orphan_vertices(
    faces: Vec<(u32, u32, u32)>,
    vertices: Vec<Point3<f32>>,
) -> (Vec<(u32, u32, u32)>, Vec<Point3<f32>>) {
    let mut vertices_reduced: Vec<Point3<f32>> = Vec::with_capacity(vertices.len());
    let original_vertex_len = vertices.len();
    let unused_vertex_marker = vertices.len();
    let mut old_new_vertex_map: Vec<usize> = vec![unused_vertex_marker; original_vertex_len];
    let mut faces_renumbered: Vec<(u32, u32, u32)> = Vec::with_capacity(faces.len());

    for face in faces {
        let old_vertex_index_0 = cast_usize(face.0);
        let new_vertex_index_0 = if old_new_vertex_map[old_vertex_index_0] == unused_vertex_marker {
            let new_index = vertices_reduced.len();
            vertices_reduced.push(vertices[old_vertex_index_0]);
            old_new_vertex_map[old_vertex_index_0] = new_index;
            new_index
        } else {
            old_new_vertex_map[old_vertex_index_0]
        };

        let old_vertex_index_1 = cast_usize(face.1);
        let new_vertex_index_1 = if old_new_vertex_map[old_vertex_index_1] == unused_vertex_marker {
            let new_index = vertices_reduced.len();
            vertices_reduced.push(vertices[old_vertex_index_1]);
            old_new_vertex_map[old_vertex_index_1] = new_index;
            new_index
        } else {
            old_new_vertex_map[old_vertex_index_1]
        };

        let old_vertex_index_2 = cast_usize(face.2);
        let new_vertex_index_2 = if old_new_vertex_map[old_vertex_index_2] == unused_vertex_marker {
            let new_index = vertices_reduced.len();
            vertices_reduced.push(vertices[old_vertex_index_2]);
            old_new_vertex_map[old_vertex_index_2] = new_index;
            new_index
        } else {
            old_new_vertex_map[old_vertex_index_2]
        };

        faces_renumbered.push((
            cast_u32(new_vertex_index_0),
            cast_u32(new_vertex_index_1),
            cast_u32(new_vertex_index_2),
        ));
    }

    faces_renumbered.shrink_to_fit();
    vertices_reduced.shrink_to_fit();

    (faces_renumbered, vertices_reduced)
}

fn remove_orphan_vertices_and_normals(
    faces: Vec<Face>,
    vertices: Vec<Point3<f32>>,
    normals: Vec<Vector3<f32>>,
) -> (Vec<Face>, Vec<Point3<f32>>, Vec<Vector3<f32>>) {
    let mut vertices_reduced: Vec<Point3<f32>> = Vec::with_capacity(vertices.len());
    let original_vertex_len = vertices.len();
    let unused_vertex_marker = vertices.len();
    let mut old_new_vertex_map: Vec<usize> = vec![unused_vertex_marker; original_vertex_len];

    let mut normals_reduced: Vec<Vector3<f32>> = Vec::with_capacity(normals.len());
    let original_normal_len = normals.len();
    let unused_normal_marker = normals.len();
    let mut old_new_normal_map: Vec<usize> = vec![unused_normal_marker; original_normal_len];

    let mut faces_renumbered: Vec<Face> = Vec::with_capacity(faces.len());

    for face in faces {
        match face {
            Face::Triangle(triangle_face) => {
                let old_vertex_index_0 = cast_usize(triangle_face.vertices.0);
                let new_vertex_index_0 =
                    if old_new_vertex_map[old_vertex_index_0] == unused_vertex_marker {
                        let new_index = vertices_reduced.len();
                        vertices_reduced.push(vertices[old_vertex_index_0]);
                        old_new_vertex_map[old_vertex_index_0] = new_index;
                        new_index
                    } else {
                        old_new_vertex_map[old_vertex_index_0]
                    };

                let old_vertex_index_1 = cast_usize(triangle_face.vertices.1);
                let new_vertex_index_1 =
                    if old_new_vertex_map[old_vertex_index_1] == unused_vertex_marker {
                        let new_index = vertices_reduced.len();
                        vertices_reduced.push(vertices[old_vertex_index_1]);
                        old_new_vertex_map[old_vertex_index_1] = new_index;
                        new_index
                    } else {
                        old_new_vertex_map[old_vertex_index_1]
                    };

                let old_vertex_index_2 = cast_usize(triangle_face.vertices.2);
                let new_vertex_index_2 =
                    if old_new_vertex_map[old_vertex_index_2] == unused_vertex_marker {
                        let new_index = vertices_reduced.len();
                        vertices_reduced.push(vertices[old_vertex_index_2]);
                        old_new_vertex_map[old_vertex_index_2] = new_index;
                        new_index
                    } else {
                        old_new_vertex_map[old_vertex_index_2]
                    };

                let old_normal_index_0 = cast_usize(triangle_face.normals.0);
                let new_normal_index_0 =
                    if old_new_normal_map[old_normal_index_0] == unused_normal_marker {
                        let new_index = normals_reduced.len();
                        normals_reduced.push(normals[old_normal_index_0]);
                        old_new_normal_map[old_normal_index_0] = new_index;
                        new_index
                    } else {
                        old_new_normal_map[old_normal_index_0]
                    };

                let old_normal_index_1 = cast_usize(triangle_face.normals.1);
                let new_normal_index_1 =
                    if old_new_normal_map[old_normal_index_1] == unused_normal_marker {
                        let new_index = normals_reduced.len();
                        normals_reduced.push(normals[old_normal_index_1]);
                        old_new_normal_map[old_normal_index_1] = new_index;
                        new_index
                    } else {
                        old_new_normal_map[old_normal_index_1]
                    };

                let old_normal_index_2 = cast_usize(triangle_face.normals.2);
                let new_normal_index_2 =
                    if old_new_normal_map[old_normal_index_2] == unused_normal_marker {
                        let new_index = normals_reduced.len();
                        normals_reduced.push(normals[old_normal_index_2]);
                        old_new_normal_map[old_normal_index_2] = new_index;
                        new_index
                    } else {
                        old_new_normal_map[old_normal_index_2]
                    };

                faces_renumbered.push(Face::Triangle(TriangleFace::new(
                    cast_u32(new_vertex_index_0),
                    cast_u32(new_vertex_index_1),
                    cast_u32(new_vertex_index_2),
                    cast_u32(new_normal_index_0),
                    cast_u32(new_normal_index_1),
                    cast_u32(new_normal_index_2),
                )));
            }
        }
    }

    faces_renumbered.shrink_to_fit();
    vertices_reduced.shrink_to_fit();
    normals_reduced.shrink_to_fit();

    (faces_renumbered, vertices_reduced, normals_reduced)
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;

    use super::*;

    fn quad() -> (Vec<(u32, u32, u32)>, Vec<Point3<f32>>) {
        let vertices = vec![
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(-1.0, 1.0, 0.0),
        ];

        // When comparing TriangleFaces or Faces from Mesh, make sure the
        // manually defined faces start their winding from the lowest vertex
        // index. See TriangleFace constructors for more info.
        let faces = vec![(0, 1, 2), (0, 2, 3)];

        (faces, vertices)
    }

    fn quad_with_normals() -> (Vec<TriangleFace>, Vec<Point3<f32>>, Vec<Vector3<f32>>) {
        let vertices = vec![
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(-1.0, 1.0, 0.0),
        ];

        let normals = vec![
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];

        // When comparing TriangleFaces or Faces from Mesh, make sure the
        // manually defined faces start their winding from the lowest vertex
        // index. See TriangleFace constructors for more info.
        let faces = vec![
            TriangleFace::from_same_vertex_and_normal_index(0, 1, 2),
            TriangleFace::from_same_vertex_and_normal_index(0, 2, 3),
        ];

        (faces, vertices, normals)
    }

    #[test]
    #[should_panic = "Empty (faceless) meshes are not supported"]
    fn test_mesh_from_triangle_faces_with_vertices_and_computed_normals_empty_mesh() {
        Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            vec![],
            vec![],
            NormalStrategy::Sharp,
        );
    }

    #[test]
    #[should_panic = "Empty (faceless) meshes are not supported"]
    fn test_mesh_from_triangle_faces_with_vertices_and_computed_normals_remove_orphans_empty_mesh()
    {
        Mesh::from_triangle_faces_with_vertices_and_computed_normals_remove_orphans(
            vec![],
            vec![],
            NormalStrategy::Sharp,
        );
    }

    #[test]
    #[should_panic = "Empty (faceless) meshes are not supported"]
    fn test_mesh_from_triangle_faces_with_vertices_and_normals_empty_mesh() {
        Mesh::from_triangle_faces_with_vertices_and_normals(vec![], vec![], vec![]);
    }

    #[test]
    #[should_panic = "Empty (faceless) meshes are not supported"]
    fn test_mesh_from_triangle_faces_with_vertices_and_normals_remove_orphans_empty_mesh() {
        Mesh::from_triangle_faces_with_vertices_and_normals_remove_orphans(vec![], vec![], vec![]);
    }

    #[test]
    #[should_panic = "Empty (faceless) meshes are not supported"]
    fn test_mesh_from_faces_with_vertices_and_normals_empty_mesh() {
        Mesh::from_faces_with_vertices_and_normals(vec![], vec![], vec![]);
    }

    #[test]
    #[should_panic = "Empty (faceless) meshes are not supported"]
    fn test_mesh_from_faces_with_vertices_and_normals_remove_orphans_empty_mesh() {
        Mesh::from_faces_with_vertices_and_normals_remove_orphans(vec![], vec![], vec![]);
    }

    #[test]
    fn test_mesh_from_triangle_faces_with_vertices_and_computed_normals() {
        let (faces, vertices) = quad();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        assert!(mesh.is_triangulated());

        let mesh_faces: Vec<_> = mesh
            .faces()
            .iter()
            .filter_map(|face| match face {
                Face::Triangle(triangle_face) => Some(triangle_face),
            })
            .collect();

        assert_eq!(vertices.as_slice(), mesh.vertices());
        assert_eq!(
            faces,
            mesh_faces
                .into_iter()
                .map(|triangle_face| triangle_face.vertices)
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    #[should_panic(expected = "Faces reference out of bounds position data")]
    fn test_mesh_from_triangle_faces_with_vertices_and_computed_normals_bounds_check() {
        let (_, vertices) = quad();

        // When comparing TriangleFaces or Faces from Mesh, make sure the
        // manually defined faces start their winding from the lowest vertex
        // index. See TriangleFace constructors for more info.
        let faces = vec![(0, 1, 2), (2, 3, 4)];

        Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );
    }

    #[test]
    fn test_mesh_from_triangle_faces_with_vertices_and_normals() {
        let (faces, vertices, normals) = quad_with_normals();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_normals(
            faces.clone(),
            vertices.clone(),
            normals.clone(),
        );
        assert!(mesh.is_triangulated());

        let mesh_faces: Vec<_> = mesh
            .faces()
            .iter()
            .filter_map(|face| match face {
                Face::Triangle(triangle_face) => Some(triangle_face),
            })
            .copied()
            .collect();

        assert_eq!(vertices.as_slice(), mesh.vertices());
        assert_eq!(normals.as_slice(), mesh.normals());
        assert_eq!(faces, mesh_faces);
    }

    #[test]
    #[should_panic(expected = "Faces reference out of bounds position data")]
    fn test_mesh_from_triangle_faces_with_vertices_and_normals_bounds_check() {
        let (_, vertices, normals) = quad_with_normals();

        // When comparing TriangleFaces or Faces from Mesh, make sure the
        // manually defined faces start their winding from the lowest vertex
        // index. See TriangleFace constructors for more info.
        let faces = vec![
            TriangleFace::from_same_vertex_and_normal_index(0, 1, 2),
            TriangleFace::from_same_vertex_and_normal_index(2, 3, 4),
        ];

        Mesh::from_triangle_faces_with_vertices_and_normals(faces, vertices, normals);
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
    fn test_oriented_edge_constructor_does_not_consist_of_the_same_vertex_should_pass() {
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
    fn test_triangle_face_to_oriented_edges() {
        let face = TriangleFace::from_same_vertex_and_normal_index(0, 1, 2);

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
        let face = TriangleFace::from_same_vertex_and_normal_index(0, 1, 2);

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
    #[should_panic(expected = "One or more face edges consists of the same vertex")]
    fn test_triangle_face_new_with_invalid_vertex_indices_0_1_should_panic() {
        TriangleFace::from_same_vertex_and_normal_index(0, 0, 2);
    }

    #[test]
    #[should_panic(expected = "One or more face edges consists of the same vertex")]
    fn test_triangle_face_new_with_invalid_vertex_indices_1_2_should_panic() {
        TriangleFace::from_same_vertex_and_normal_index(0, 2, 2);
    }

    #[test]
    #[should_panic(expected = "One or more face edges consists of the same vertex")]
    fn test_triangle_face_new_with_invalid_vertex_indices_0_2_should_panic() {
        TriangleFace::from_same_vertex_and_normal_index(0, 2, 0);
    }

    #[test]
    #[should_panic(expected = "One or more face edges consists of the same vertex")]
    fn test_triangle_face_new_separate_with_invalid_vertex_indices_0_1_should_panic() {
        TriangleFace::new(0, 0, 2, 0, 0, 0);
    }

    #[test]
    #[should_panic(expected = "One or more face edges consists of the same vertex")]
    fn test_triangle_face_new_separate_with_invalid_vertex_indices_1_2_should_panic() {
        TriangleFace::new(0, 2, 2, 0, 0, 0);
    }

    #[test]
    #[should_panic(expected = "One or more face edges consists of the same vertex")]
    fn test_triangle_face_new_separate_with_invalid_vertex_indices_0_2_should_panic() {
        TriangleFace::new(0, 2, 0, 0, 0, 0);
    }

    #[test]
    fn test_has_no_orphan_vertices_returns_true_if_there_are_some() {
        let (faces, vertices, normals) = quad_with_normals();

        let mesh_without_orphans =
            Mesh::from_triangle_faces_with_vertices_and_normals(faces, vertices, normals);

        assert!(mesh_without_orphans.has_no_orphan_vertices());
    }

    #[test]
    fn test_has_no_orphan_vertices_returns_false_if_there_are_none() {
        let (faces, vertices, normals) = quad_with_normals();
        let extra_vertex = vec![Point3::new(0.0, 0.0, 0.0)];
        let vertices_extended = [&vertices[..], &extra_vertex[..]].concat();

        let mesh_with_orphans =
            Mesh::from_triangle_faces_with_vertices_and_normals(faces, vertices_extended, normals);

        assert!(!mesh_with_orphans.has_no_orphan_vertices());
    }

    #[test]
    fn test_has_no_orphan_normals_returns_true_if_there_are_some() {
        let (faces, vertices, normals) = quad_with_normals();

        let mesh_without_orphans =
            Mesh::from_triangle_faces_with_vertices_and_normals(faces, vertices, normals);

        assert!(mesh_without_orphans.has_no_orphan_normals());
    }

    #[test]
    fn test_mesh_unoriented_edges_iter() {
        let (faces, vertices, normals) = quad_with_normals();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_normals(faces, vertices, normals);
        let unoriented_edges_correct = vec![
            UnorientedEdge(OrientedEdge::new(0, 1)),
            UnorientedEdge(OrientedEdge::new(1, 2)),
            UnorientedEdge(OrientedEdge::new(2, 0)),
            UnorientedEdge(OrientedEdge::new(2, 3)),
            UnorientedEdge(OrientedEdge::new(3, 0)),
            UnorientedEdge(OrientedEdge::new(0, 2)),
        ];
        let unoriented_edges_to_check: Vec<UnorientedEdge> = mesh.unoriented_edges_iter().collect();

        assert!(unoriented_edges_correct
            .iter()
            .all(|u_e| unoriented_edges_to_check.iter().any(|e| e == u_e)));

        let len_1 = unoriented_edges_to_check.len();
        let len_2 = unoriented_edges_correct.len();
        assert_eq!(len_1, len_2);
    }

    #[test]
    fn test_mesh_oriented_edges_iter() {
        let (faces, vertices, normals) = quad_with_normals();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_normals(faces, vertices, normals);

        let oriented_edges_correct = vec![
            OrientedEdge::new(0, 1),
            OrientedEdge::new(1, 2),
            OrientedEdge::new(2, 0),
            OrientedEdge::new(2, 3),
            OrientedEdge::new(3, 0),
            OrientedEdge::new(0, 2),
        ];
        let oriented_edges_to_check: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();

        assert!(oriented_edges_correct
            .iter()
            .all(|o_e| oriented_edges_to_check.iter().any(|e| e == o_e)));

        let len_1 = oriented_edges_to_check.len();
        let len_2 = oriented_edges_correct.len();

        assert_eq!(len_1, len_2);
    }

    #[test]
    fn test_has_no_orphan_normals_returns_false_if_there_are_none() {
        let (faces, vertices, normals) = quad_with_normals();
        let extra_normal = vec![Vector3::new(0.0, 0.0, 0.0)];
        let normals_extended = [&normals[..], &extra_normal[..]].concat();

        let mesh_with_orphans =
            Mesh::from_triangle_faces_with_vertices_and_normals(faces, vertices, normals_extended);

        assert!(!mesh_with_orphans.has_no_orphan_normals());
    }

    #[test]
    fn test_remove_orphan_vertices() {
        let (faces, vertices) = quad();
        let extra_vertex = vec![Point3::new(0.0, 0.0, 0.0)];
        let vertices_extended = [&extra_vertex[..], &vertices[..]].concat();
        let faces_renumbered_to_match_extend_vertices: Vec<_> =
            faces.iter().map(|f| (f.0 + 1, f.1 + 1, f.2 + 1)).collect();

        let faces_length = &faces.len();

        let (faces_purged, vertices_purged) =
            remove_orphan_vertices(faces_renumbered_to_match_extend_vertices, vertices_extended);

        let faces_purged_length = &faces_purged.len();

        assert_eq!(faces_length, faces_purged_length);
        assert_eq!(vertices_purged, vertices);
    }

    #[test]
    fn test_remove_orphan_vertices_and_normals() {
        let (faces, vertices, normals) = quad_with_normals();

        let extra_vertex = vec![Point3::new(0.0, 0.0, 0.0)];
        let vertices_extended = [&extra_vertex[..], &vertices[..]].concat();

        let extra_normal = vec![Vector3::new(0.0, 0.0, 0.0)];
        let normals_extended = [&extra_normal[..], &normals[..]].concat();

        let faces_renumbered_to_match_extend_data: Vec<_> = faces
            .iter()
            .map(|f| {
                TriangleFace::new(
                    f.vertices.0 + 1,
                    f.vertices.1 + 1,
                    f.vertices.2 + 1,
                    f.normals.0 + 1,
                    f.normals.1 + 1,
                    f.normals.2 + 1,
                )
            })
            .map(Face::Triangle)
            .collect();

        let faces_length = faces.len();

        let (faces_purged, vertices_purged, normals_purged) = remove_orphan_vertices_and_normals(
            faces_renumbered_to_match_extend_data,
            vertices_extended,
            normals_extended,
        );

        let faces_purged_length = faces_purged.len();

        assert_eq!(faces_length, faces_purged_length);
        assert_eq!(vertices_purged, vertices);
        assert_eq!(normals_purged, normals);
    }

    #[test]
    fn test_mesh_from_triangle_faces_with_vertices_and_computed_normals_remove_orphans() {
        let (faces, vertices) = quad();
        let extra_vertex = vec![Point3::new(0.0, 0.0, 0.0)];
        let vertices_extended = [&extra_vertex[..], &vertices[..]].concat();
        let faces_renumbered_to_match_extend_vertices: Vec<_> =
            faces.iter().map(|f| (f.0 + 1, f.1 + 1, f.2 + 1)).collect();

        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals_remove_orphans(
            faces_renumbered_to_match_extend_vertices,
            vertices_extended,
            NormalStrategy::Sharp,
        );

        assert!(mesh.has_no_orphan_vertices());
    }

    #[test]
    fn test_mesh_from_triangle_faces_with_vertices_and_normals_remove_orphans() {
        let (faces, vertices, normals) = quad_with_normals();
        let extra_vertex = vec![Point3::new(0.0, 0.0, 0.0)];
        let vertices_extended = [&extra_vertex[..], &vertices[..]].concat();
        let extra_normal = vec![Vector3::new(0.0, 0.0, 0.0)];
        let normals_extended = [&extra_normal[..], &normals[..]].concat();

        let faces_renumbered_to_match_extend_vertices_and_normals: Vec<_> = faces
            .iter()
            .map(|f| {
                TriangleFace::new(
                    f.vertices.0 + 1,
                    f.vertices.1 + 1,
                    f.vertices.2 + 1,
                    f.normals.0 + 1,
                    f.normals.1 + 1,
                    f.normals.2 + 1,
                )
            })
            .collect();

        let mesh = Mesh::from_triangle_faces_with_vertices_and_normals_remove_orphans(
            faces_renumbered_to_match_extend_vertices_and_normals,
            vertices_extended,
            normals_extended,
        );

        assert!(mesh.has_no_orphan_vertices());
    }

    #[test]
    fn test_triangle_face_new_lowest_first() {
        let face = TriangleFace::from_same_vertex_and_normal_index(0, 1, 2);
        assert_eq!(face.vertices, (0, 1, 2));
        assert_eq!(face.normals, (0, 1, 2));
    }

    #[test]
    fn test_triangle_face_new_lowest_second() {
        let face = TriangleFace::from_same_vertex_and_normal_index(2, 0, 1);
        assert_eq!(face.vertices, (0, 1, 2));
        assert_eq!(face.normals, (0, 1, 2));
    }

    #[test]
    fn test_triangle_face_new_lowest_third() {
        let face = TriangleFace::from_same_vertex_and_normal_index(1, 2, 0);
        assert_eq!(face.vertices, (0, 1, 2));
        assert_eq!(face.normals, (0, 1, 2));
    }

    #[test]
    fn test_triangle_face_new_separate_lowest_first() {
        let face = TriangleFace::new(0, 1, 2, 3, 4, 5);
        assert_eq!(face.vertices, (0, 1, 2));
        assert_eq!(face.normals, (3, 4, 5));
    }

    #[test]
    fn test_triangle_face_new_separate_lowest_second() {
        let face = TriangleFace::new(2, 0, 1, 5, 3, 4);
        assert_eq!(face.vertices, (0, 1, 2));
        assert_eq!(face.normals, (3, 4, 5));
    }

    #[test]
    fn test_triangle_face_new_separate_lowest_third() {
        let face = TriangleFace::new(1, 2, 0, 4, 5, 3);
        assert_eq!(face.vertices, (0, 1, 2));
        assert_eq!(face.normals, (3, 4, 5));
    }

    #[test]
    fn test_triangle_face_to_reverted_comparison_to_reverted() {
        let face = TriangleFace::new(1, 2, 3, 4, 5, 6);
        let face_reverted_correct = TriangleFace::new(3, 2, 1, 6, 5, 4);

        let face_reverted_calculated = face.to_reverted();
        assert_eq!(face_reverted_correct, face_reverted_calculated);
    }

    #[test]
    fn test_triangle_face_to_reverted_comparison_to_same() {
        let face = TriangleFace::new(1, 2, 3, 4, 5, 6);

        let face_reverted_calculated = face.to_reverted();
        assert_ne!(face, face_reverted_calculated);
    }

    #[test]
    fn test_triangle_face_to_reverted_comparison_to_reverted_and_shifted() {
        let face = TriangleFace::new(1, 2, 3, 4, 5, 6);
        let face_reverted_correct_shifted = TriangleFace::new(2, 1, 3, 5, 4, 6);

        let face_reverted_calculated = face.to_reverted();
        assert_eq!(face_reverted_correct_shifted, face_reverted_calculated);
    }

    #[test]
    fn test_triangle_face_is_reverted_comparison_to_reverted_and_shifted() {
        let face = TriangleFace::new(1, 2, 3, 4, 5, 6);
        let face_reverted_correct_shifted = TriangleFace::new(2, 1, 3, 5, 4, 6);

        assert!(face_reverted_correct_shifted.is_reverted(&face));
    }

    #[test]
    fn test_triangle_face_is_reverted_comparison_to_self() {
        let face = TriangleFace::new(1, 2, 3, 4, 5, 6);

        assert!(!face.is_reverted(&face));
    }

    #[test]
    fn test_triangle_face_contains_oriented_edge_returns_true_because_contains_all() {
        let face = TriangleFace::new(1, 2, 3, 4, 5, 6);
        let oriented_edge_1 = OrientedEdge::new(1, 2);
        let oriented_edge_2 = OrientedEdge::new(2, 3);
        let oriented_edge_3 = OrientedEdge::new(3, 1);

        assert!(face.contains_oriented_edge(oriented_edge_1));
        assert!(face.contains_oriented_edge(oriented_edge_2));
        assert!(face.contains_oriented_edge(oriented_edge_3));
    }

    #[test]
    fn test_triangle_face_contains_oriented_edge_returns_false_because_contains_all_reverted() {
        let face = TriangleFace::new(1, 2, 3, 4, 5, 6);
        let oriented_edge_1 = OrientedEdge::new(2, 1);
        let oriented_edge_2 = OrientedEdge::new(3, 2);
        let oriented_edge_3 = OrientedEdge::new(1, 3);

        assert!(!face.contains_oriented_edge(oriented_edge_1));
        assert!(!face.contains_oriented_edge(oriented_edge_2));
        assert!(!face.contains_oriented_edge(oriented_edge_3));
    }

    #[test]
    fn test_triangle_face_contains_oriented_edge_returns_false_because_different() {
        let face = TriangleFace::new(1, 2, 3, 4, 5, 6);
        let oriented_edge = OrientedEdge::new(4, 5);

        assert!(!face.contains_oriented_edge(oriented_edge));
    }

    #[test]
    fn test_triangle_face_contains_unoriented_edge_returns_true_because_contains_all() {
        let face = TriangleFace::new(1, 2, 3, 4, 5, 6);
        let unoriented_edge_1 = UnorientedEdge(OrientedEdge::new(1, 2));
        let unoriented_edge_2 = UnorientedEdge(OrientedEdge::new(2, 3));
        let unoriented_edge_3 = UnorientedEdge(OrientedEdge::new(3, 1));

        assert!(face.contains_unoriented_edge(unoriented_edge_1));
        assert!(face.contains_unoriented_edge(unoriented_edge_2));
        assert!(face.contains_unoriented_edge(unoriented_edge_3));
    }

    #[test]
    fn test_triangle_face_contains_unoriented_edge_returns_true_because_contains_all_reverted() {
        let face = TriangleFace::new(1, 2, 3, 4, 5, 6);
        let unoriented_edge_1 = UnorientedEdge(OrientedEdge::new(2, 1));
        let unoriented_edge_2 = UnorientedEdge(OrientedEdge::new(3, 2));
        let unoriented_edge_3 = UnorientedEdge(OrientedEdge::new(1, 3));

        assert!(face.contains_unoriented_edge(unoriented_edge_1));
        assert!(face.contains_unoriented_edge(unoriented_edge_2));
        assert!(face.contains_unoriented_edge(unoriented_edge_3));
    }

    #[test]
    fn test_triangle_face_contains_unoriented_edge_returns_false_because_different() {
        let face = TriangleFace::new(1, 2, 3, 4, 5, 6);
        let unoriented_edge = UnorientedEdge(OrientedEdge::new(4, 5));

        assert!(!face.contains_unoriented_edge(unoriented_edge));
    }
}
