use std::collections::{HashSet, VecDeque};

use nalgebra as na;
use nalgebra::Point3;

use crate::convert::{cast_i32, cast_usize};

use super::{Mesh, OrientedEdge, UnorientedEdge};

// FIXME: Make more generic: take &[Point] or Iterator<Item=&Point>
#[allow(dead_code)]
pub fn find_closest_point(position: &Point3<f32>, mesh: &Mesh) -> Option<Point3<f32>> {
    let vertices = mesh.vertices();
    if vertices.is_empty() {
        return None;
    }

    let mut closest = vertices[0];
    let mut closest_distance_squared = na::distance_squared(position, &closest);
    for point in &vertices[1..] {
        let distance_squared = na::distance_squared(position, &point);
        if distance_squared < closest_distance_squared {
            closest = *point;
            closest_distance_squared = distance_squared;
        }
    }

    Some(closest)
}

/// The edges sharing the same vertex indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedEdges {
    /// The lower vertex index shared by the edges.
    pub index_low: u32,
    /// The higher vertex index shared by the edges.
    pub index_high: u32,
    /// Number of oriented edges from `index_low` to `index_high`.
    pub edge_count_ascending: usize,
    /// Number of oriented edges from `index_high` to `index_low`.
    pub edge_count_descending: usize,
}

impl SharedEdges {
    pub fn oriented_edge_iter(&self) -> SharedEdgesOrientedEdgeIter {
        SharedEdgesOrientedEdgeIter {
            shared_edges: *self,
            next_ascending: 0,
            next_descending: 0,
        }
    }
}

/// Iterator that generates all oriented edges from `SharedEdges`.
pub struct SharedEdgesOrientedEdgeIter {
    shared_edges: SharedEdges,
    next_ascending: usize,
    next_descending: usize,
}

impl Iterator for SharedEdgesOrientedEdgeIter {
    type Item = OrientedEdge;

    fn next(&mut self) -> Option<OrientedEdge> {
        if self.next_ascending < self.shared_edges.edge_count_ascending {
            let edge = OrientedEdge {
                vertices: (self.shared_edges.index_low, self.shared_edges.index_high),
            };
            self.next_ascending += 1;

            return Some(edge);
        }

        if self.next_descending < self.shared_edges.edge_count_descending {
            let edge = OrientedEdge {
                vertices: (self.shared_edges.index_high, self.shared_edges.index_low),
            };
            self.next_descending += 1;

            return Some(edge);
        }

        None
    }
}

pub type EdgeSharingMap = fxhash::FxHashMap<UnorientedEdge, SharedEdges>;

// FIXME: Implement edge_sharing also for UnorientedEdges?

/// Computes edge sharing, a structure containing all edges occupying
/// the same space in the mesh topology.
///
/// Edge sharing can be used to compute edge valency - the number of
/// faces sharing an edge. The valency is 1 for border edges, 2 for
/// manifold edges, and 3 or more for non-manifold edges.  Ideally our
/// meshes would only contain manifold edges, but we can also deal
/// with not watertight meshes (those that contain border edges).
/// Meshes containing non-manifold edges are usually corrupted and
/// little useful work can be done on them.
pub fn edge_sharing<'a, I: IntoIterator<Item = &'a OrientedEdge>>(
    oriented_edges: I,
) -> EdgeSharingMap {
    let mut edge_sharing_map = fxhash::FxHashMap::default();

    for edge in oriented_edges {
        let unoriented_edge = UnorientedEdge(*edge);

        let index_low = edge.vertices.0.min(edge.vertices.1);
        let index_high = edge.vertices.0.max(edge.vertices.1);

        let shared_edges = edge_sharing_map
            .entry(unoriented_edge)
            .or_insert(SharedEdges {
                index_low,
                index_high,
                edge_count_ascending: 0,
                edge_count_descending: 0,
            });

        if edge.vertices.0 < edge.vertices.1 {
            shared_edges.edge_count_ascending += 1;
        } else {
            shared_edges.edge_count_descending += 1;
        }
    }

    edge_sharing_map
}

/// Finds edges with a certain valency in a mesh edge collection.
///
/// An edge valency indicates how many faces share the edge.
fn find_edges_with_valency<'a>(
    edge_sharing: &'a EdgeSharingMap,
    valency: usize,
) -> impl Iterator<Item = OrientedEdge> + 'a {
    edge_sharing
        .values()
        .filter(move |shared_edges| {
            shared_edges.edge_count_ascending + shared_edges.edge_count_descending == valency
        })
        .flat_map(|shared_edges| shared_edges.oriented_edge_iter())
}

/// Finds border edges in a mesh edge collection.
///
/// An edge is border when its valency is 1.
pub fn border_edges<'a>(
    edge_sharing: &'a EdgeSharingMap,
) -> impl Iterator<Item = OrientedEdge> + 'a {
    find_edges_with_valency(edge_sharing, 1)
}

/// Finds manifold (inner) edges in a mesh edge collection.
///
/// An edge is manifold when its valency is 2.
#[allow(dead_code)]
pub fn manifold_edges<'a>(
    edge_sharing: &'a EdgeSharingMap,
) -> impl Iterator<Item = OrientedEdge> + 'a {
    find_edges_with_valency(edge_sharing, 2)
}

/// Finds non-manifold (erroneous) edges in a mesh edge collection.
pub fn non_manifold_edges<'a>(
    edge_sharing: &'a EdgeSharingMap,
) -> impl Iterator<Item = OrientedEdge> + 'a {
    edge_sharing
        .values()
        .filter(|shared_edges| {
            shared_edges.edge_count_ascending + shared_edges.edge_count_descending > 2
        })
        .flat_map(|shared_edges| shared_edges.oriented_edge_iter())
}

/// Checks if mesh contains only manifold or border edges.
pub fn is_mesh_manifold(edge_sharing: &EdgeSharingMap) -> bool {
    non_manifold_edges(edge_sharing).next().is_none()
}

/// Finds border vertex indices in a mesh edge collection.
///
/// A vertex is border when its edge's valency is 1.
#[allow(dead_code)]
pub fn border_vertex_indices(edge_sharing: &EdgeSharingMap) -> HashSet<u32> {
    let mut border_vertices = HashSet::new();

    border_edges(edge_sharing).for_each(|edge| {
        border_vertices.insert(edge.vertices.0);
        border_vertices.insert(edge.vertices.1);
    });
    border_vertices
}

#[derive(Debug, PartialEq)]
pub enum BorderEdgeLoopsResult {
    Found(Vec<Vec<OrientedEdge>>),
    FoundWithNondeterminism(Vec<Vec<OrientedEdge>>),
    Watertight,
}

/// Finds continuous loops of border edges, starting from a random edge.
///
/// The mesh may contain holes or islands, therefore it may have an unknown
/// number of edge loops. If two or more edge loops meet at a single vertex, the
/// result is non-deterministic: a random path of more possibilities is chosen
/// to form an edge loop. The algorithm internally operates on oriented edges,
/// however, it may revert them to become a valid continuation of an edge loop.
/// This is to make sure it operates well also on non-orientable meshes. There
/// is no user interaction with edges of any kind in our current front-end but
/// once there is, it will be similar to other 3D editing software - all edges
/// will seem unoriented, therefore the orientation of oriented edges doesn't
/// matter.
pub fn border_edge_loops(edge_sharing: &EdgeSharingMap) -> BorderEdgeLoopsResult {
    let mut border_edges: Vec<_> = border_edges(edge_sharing).collect();

    // If there are no border edges, the mesh is watertight.
    if border_edges.is_empty() {
        return BorderEdgeLoopsResult::Watertight;
    }

    // Check if the border edges are branching (the loops are touching at a
    // single vertex). If they are, the result is non-deterministic.

    // FIXME: @Optimization: use a Map structure or at least trim the list of
    // vertices also from below. Find the maximum index that occurs in the
    // border edge loop.
    let max_vertex_index = border_edges
        .iter()
        .fold(0, |max, edge| max.max(edge.vertices.0.max(edge.vertices.1)));
    // A helper vector - each index represents a vertex in the mesh. The items
    // represent the number of occurrences of each respective vertex in the
    // border edge list.
    let mut vertex_occurrence_counts: Vec<u32> = vec![0; cast_usize(max_vertex_index) + 1];
    // Count the vertex occurrences
    for edge in &border_edges {
        vertex_occurrence_counts[cast_usize(edge.vertices.0)] += 1;
        vertex_occurrence_counts[cast_usize(edge.vertices.1)] += 1;
    }

    // For a deterministic result, each vertex should occur in the list of
    // boundary edges exactly twice - in two edges, which meet at the vertex.
    // Higher number means that more edge loops meet at one vertex and therefore
    // the edge loop list is non-deterministic and possible wrong. If the vertex
    // appears only once, the mesh may be erroneous. There are holes in the list
    // of vertex occurrence, therefore some items may be zero.

    // FIXME: Examine situation when occurrence is 1 and determine what to do.
    // In a broader scope start asserting all edge sharing data comes from a
    // mesh and is not handcrafted and thus possibly invalid and change API of
    // `edge_sharing` to take `&Mesh` instead of iterator of edges.
    let non_deterministic = vertex_occurrence_counts
        .iter()
        .any(|count| *count != 2 && *count != 0);

    let mut edge_loops: Vec<Vec<OrientedEdge>> = Vec::new();

    while let Some(edge) = border_edges.pop() {
        let mut current_chain: VecDeque<OrientedEdge> = VecDeque::with_capacity(1);
        current_chain.push_back(edge);

        // Try constructing a loop by matching edges with either front or back
        // of the deque:

        let loop_closed = loop {
            let mut found = false;

            for (i, other_edge) in border_edges.iter().enumerate() {
                let other_edge_reverted = other_edge.to_reverted();

                let front = current_chain[0];
                if other_edge.chains_to(front) {
                    current_chain.push_front(*other_edge);
                    border_edges.remove(i);
                    found = true;

                    break;
                }

                if other_edge_reverted.chains_to(front) {
                    current_chain.push_front(other_edge_reverted);
                    border_edges.remove(i);
                    found = true;

                    break;
                }

                let back = current_chain[current_chain.len() - 1];
                if back.chains_to(*other_edge) {
                    current_chain.push_back(*other_edge);
                    border_edges.remove(i);
                    found = true;

                    break;
                }

                if back.chains_to(other_edge_reverted) {
                    current_chain.push_back(other_edge_reverted);
                    border_edges.remove(i);
                    found = true;

                    break;
                }
            }

            // Continue until the loop is closed or until we cannot find a
            // matching puzzle piece
            let front = current_chain[0];
            let back = current_chain[current_chain.len() - 1];
            let loop_closed = back.chains_to(front);

            if loop_closed || !found {
                break loop_closed;
            }
        };

        edge_loops.push(Vec::from(current_chain));

        if !loop_closed {
            panic!("Edge loop not closed!");
        }
    }

    if non_deterministic {
        BorderEdgeLoopsResult::FoundWithNondeterminism(edge_loops)
    } else {
        BorderEdgeLoopsResult::Found(edge_loops)
    }
}

/// Checks if all the face normals point the same way.
///
/// In a proper watertight orientable mesh each oriented edge should
/// have a single counterpart in a reverted oriented edge. In an open
/// orientable mesh each internal edge has its counterpart in a single
/// reverted oriented edge and the border edges don't have any
/// counterpart.
pub fn is_mesh_orientable(edge_sharing: &EdgeSharingMap) -> bool {
    edge_sharing.values().all(|shared_edges| {
        // Ascending_count and descending_count can never be both 0 at the same
        // time because there is never a case that the edge doesn't exist in any
        // direction. Even if this happens, it means that the tested edge is
        // non-existing and therefore doesn't affect edge winding.
        shared_edges.edge_count_ascending <= 1 && shared_edges.edge_count_descending <= 1
    })
}

/// Checks if mesh is watertight.
///
/// The mesh is watertight if there is no border or non-manifold edge,
/// which means all the edge valencies are 2.
pub fn is_mesh_watertight(edge_sharing: &EdgeSharingMap) -> bool {
    edge_sharing.values().all(|shared_edges| {
        shared_edges.edge_count_ascending == 1 && shared_edges.edge_count_descending == 1
    })
}

/// Computes the mesh genus of a triangulated mesh geometry.
///
/// Genus of a mesh is the number of holes in topology / connectivity. The mesh
/// **must** be triangulated and watertight for this to produce usable results.
///
/// The genus (G) is computed as: `V - E + F = 2*(1 - G)`.
pub fn triangulated_mesh_genus(vertex_count: usize, edge_count: usize, face_count: usize) -> i32 {
    1 - (cast_i32(vertex_count) - cast_i32(edge_count) + cast_i32(face_count)) / 2
}

/// Checks if two meshes are similar.
///
/// This function is slow and is therefore enabled only for tests.
///
/// Two mesh geometries are similar when they are visually similar (see the
/// definition of `are_visually_similar`), and they have the same number of
/// vertices and normals. Therefore they are going to be treated the same by all
/// functions of this software and all their transformations result in similar
/// mesh geometries.
#[cfg(test)]
pub fn are_similar(mesh1: &Mesh, mesh2: &Mesh) -> bool {
    mesh1.vertices().len() == mesh2.vertices().len()
        && mesh1.normals().len() == mesh2.normals().len()
        && are_visually_similar(mesh1, mesh2)
}

/// Checks if two meshes are visually similar.
///
/// This function is slow and is therefore enabled only for tests.
///
/// Two mesh geometries are visually similar when the position of each vertex in
/// one mesh geometry matches a position of some vertex in the other mesh
/// geometry, when the direction of each normal in one mesh geometry matches a
/// direction of some normal in the other mesh geometry and each face in one
/// mesh geometry refers vertices with the same position and normals with the
/// same direction, both in the same circular order, as exactly one face in the
/// other mesh geometry.
///
/// The indices (order in which they are stored) of vertices, normals and faces
/// can differ but as long as the previous conditions are met, the mesh
/// geometries are similar. It is not necessary that the count of vertices and
/// normals are identical, because one mesh may reuse (share) vertices and
/// normals in more faces and the other doesn't (applies to all or some faces).
///
/// The mesh geometries are not necessarily identical in memory but they look
/// the same. If the number of vertices differs, the mesh geometries don't share
/// the same qualities (they are not welded in the same places and at least one
/// of them is not watertight). Despite that they are considered visually
/// similar, they are not going to be treated the same by some functions of this
/// software and all their transformations result in different mesh geometries.
#[cfg(test)]
pub fn are_visually_similar(mesh1: &Mesh, mesh2: &Mesh) -> bool {
    use nalgebra::Vector3;

    use crate::mesh::Face;

    struct UnpackedFace {
        vertices: (Point3<f32>, Point3<f32>, Point3<f32>),
        normals: (Vector3<f32>, Vector3<f32>, Vector3<f32>),
    }

    impl PartialEq for UnpackedFace {
        fn eq(&self, other: &Self) -> bool {
            (approx::relative_eq!(self.vertices.0, other.vertices.0)
                && approx::relative_eq!(self.vertices.1, other.vertices.1)
                && approx::relative_eq!(self.vertices.2, other.vertices.2)
                && approx::relative_eq!(self.normals.0, other.normals.0)
                && approx::relative_eq!(self.normals.1, other.normals.1)
                && approx::relative_eq!(self.normals.2, other.normals.2))
                || (approx::relative_eq!(self.vertices.0, other.vertices.1)
                    && approx::relative_eq!(self.vertices.1, other.vertices.2)
                    && approx::relative_eq!(self.vertices.2, other.vertices.0)
                    && approx::relative_eq!(self.normals.0, other.normals.1)
                    && approx::relative_eq!(self.normals.1, other.normals.2)
                    && approx::relative_eq!(self.normals.2, other.normals.0))
                || (approx::relative_eq!(self.vertices.0, other.vertices.2)
                    && approx::relative_eq!(self.vertices.1, other.vertices.0)
                    && approx::relative_eq!(self.vertices.2, other.vertices.1)
                    && approx::relative_eq!(self.normals.0, other.normals.2)
                    && approx::relative_eq!(self.normals.1, other.normals.0)
                    && approx::relative_eq!(self.normals.2, other.normals.1))
        }
    }

    let unpacked_faces1 = mesh1.faces().iter().map(|face| match face {
        Face::Triangle(f) => UnpackedFace {
            vertices: (
                mesh1.vertices()[cast_usize(f.vertices.0)],
                mesh1.vertices()[cast_usize(f.vertices.1)],
                mesh1.vertices()[cast_usize(f.vertices.2)],
            ),
            normals: (
                mesh1.normals()[cast_usize(f.normals.0)],
                mesh1.normals()[cast_usize(f.normals.1)],
                mesh1.normals()[cast_usize(f.normals.2)],
            ),
        },
    });

    let unpacked_faces2 = mesh2.faces().iter().map(|face| match face {
        Face::Triangle(f) => UnpackedFace {
            vertices: (
                mesh2.vertices()[cast_usize(f.vertices.0)],
                mesh2.vertices()[cast_usize(f.vertices.1)],
                mesh2.vertices()[cast_usize(f.vertices.2)],
            ),
            normals: (
                mesh2.normals()[cast_usize(f.normals.0)],
                mesh2.normals()[cast_usize(f.normals.1)],
                mesh2.normals()[cast_usize(f.normals.2)],
            ),
        },
    });

    mesh1.faces().len() == mesh2.faces().len()
        && unpacked_faces1
            .clone()
            .all(|f| unpacked_faces2.clone().any(|g| f == g))
        && unpacked_faces2
            .clone()
            .all(|f| unpacked_faces1.clone().any(|g| f == g))
}

#[cfg(test)]
mod tests {
    use nalgebra::{Rotation3, Vector3};

    use crate::mesh::{primitive, NormalStrategy, TriangleFace};

    use super::*;

    fn quad() -> (Vec<(u32, u32, u32)>, Vec<Point3<f32>>) {
        #[rustfmt::skip]
        let vertices = vec![
            Point3::new(-1.0, -1.0,  0.0),
            Point3::new( 1.0, -1.0,  0.0),
            Point3::new( 1.0,  1.0,  0.0),
            Point3::new(-1.0,  1.0,  0.0),
        ];

        let faces = vec![(0, 1, 2), (2, 3, 0)];

        (faces, vertices)
    }

    pub fn quad_with_normals() -> Mesh {
        #[rustfmt::skip]
        let vertices = vec![
            Point3::new(-1.0, -1.0,  0.0),
            Point3::new( 1.0, -1.0,  0.0),
            Point3::new( 1.0,  1.0,  0.0),
            Point3::new(-1.0,  1.0,  0.0),
        ];

        #[rustfmt::skip]
        let vertex_normals = vec![
            Vector3::new(-1.0, -1.0,  1.0),
            Vector3::new( 1.0, -1.0,  1.0),
            Vector3::new( 1.0,  1.0,  1.0),
            Vector3::new(-1.0,  1.0,  1.0),
        ];

        let faces = vec![
            TriangleFace::from_same_vertex_and_normal_index(0, 1, 2),
            TriangleFace::from_same_vertex_and_normal_index(2, 3, 0),
        ];

        Mesh::from_triangle_faces_with_vertices_and_normals(faces, vertices, vertex_normals)
    }

    pub fn quad_with_extra_vertices_and_normals() -> Mesh {
        let vertices = vec![
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0), // first copy of the same vertex
            Point3::new(1.0, 1.0, 0.0), // second copy of the same vertex
            Point3::new(-1.0, 1.0, 0.0),
        ];

        let vertex_normals = vec![
            Vector3::new(-1.0, -1.0, 1.0), // first copy of the same normal
            Vector3::new(-1.0, -1.0, 1.0), // second copy of the same normal
            Vector3::new(1.0, -1.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(-1.0, 1.0, 1.0),
        ];

        let faces = vec![
            TriangleFace::new(0, 1, 2, 1, 2, 3),
            TriangleFace::new(3, 4, 0, 3, 4, 0),
        ];

        Mesh::from_triangle_faces_with_vertices_and_normals(faces, vertices, vertex_normals)
    }

    pub fn quad_renumbered_with_normals() -> Mesh {
        let vertices = vec![
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(-1.0, 1.0, 0.0),
        ];

        let vertex_normals = vec![
            Vector3::new(1.0, -1.0, 1.0),
            Vector3::new(-1.0, -1.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(-1.0, 1.0, 1.0),
        ];

        let faces = vec![
            TriangleFace::from_same_vertex_and_normal_index(1, 0, 2),
            TriangleFace::from_same_vertex_and_normal_index(3, 1, 2),
        ];

        Mesh::from_triangle_faces_with_vertices_and_normals(faces, vertices, vertex_normals)
    }

    pub fn quad_renumbered_more_with_normals() -> Mesh {
        let vertices = vec![
            Point3::new(1.0, -1.0, 0.0),  //1
            Point3::new(-1.0, 1.0, 0.0),  //3
            Point3::new(-1.0, -1.0, 0.0), //0
            Point3::new(1.0, 1.0, 0.0),   //2
        ];

        let vertex_normals = vec![
            Vector3::new(1.0, -1.0, 1.0),  //1
            Vector3::new(-1.0, 1.0, 1.0),  //3
            Vector3::new(-1.0, -1.0, 1.0), //0
            Vector3::new(1.0, 1.0, 1.0),   //2
        ];

        let faces = vec![
            TriangleFace::from_same_vertex_and_normal_index(2, 0, 3),
            TriangleFace::from_same_vertex_and_normal_index(2, 3, 1),
        ];

        Mesh::from_triangle_faces_with_vertices_and_normals(faces, vertices, vertex_normals)
    }

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

    fn double_torus() -> (Vec<(u32, u32, u32)>, Vec<Point3<f32>>) {
        let vertices = vec![
            Point3::new(5.566988, -1.129e-11, 0.25),
            Point3::new(4.283494, 1.241025, 0.25),
            Point3::new(4.716506, 0.491025, 0.25),
            Point3::new(4.283494, -1.241025, 0.25),
            Point3::new(4.716506, -0.491025, 0.25),
            Point3::new(6.0, 0.75, 0.25),
            Point3::new(5.149519, 1.241025, 0.25),
            Point3::new(6.0, 1.732051, 0.25),
            Point3::new(4.608253, 1.053525, -0.5),
            Point3::new(4.5, -0.866025, -0.5),
            Point3::new(6.108253, 0.1875, -0.5),
            Point3::new(6.433012, -1.129e-11, 0.25),
            Point3::new(6.216506, 2.107051, -0.5),
            Point3::new(6.433012, 2.482051, 0.25),
        ];

        let faces = vec![
            (4, 3, 11),
            (0, 11, 2),
            (2, 1, 3),
            (7, 5, 11),
            (5, 6, 11),
            (6, 7, 13),
            (8, 2, 9),
            (1, 8, 9),
            (8, 12, 6),
            (1, 13, 12),
            (9, 4, 0),
            (3, 9, 11),
            (10, 0, 2),
            (12, 10, 5),
            (13, 11, 12),
            (10, 8, 6),
            (4, 11, 0),
            (11, 1, 2),
            (2, 3, 4),
            (7, 11, 13),
            (6, 1, 11),
            (6, 13, 1),
            (2, 4, 9),
            (1, 9, 3),
            (12, 7, 6),
            (1, 12, 8),
            (9, 0, 10),
            (9, 10, 11),
            (10, 2, 8),
            (12, 5, 7),
            (11, 10, 12),
            (10, 6, 5),
        ];

        (faces, vertices)
    }

    fn triple_torus() -> (Vec<(u32, u32, u32)>, Vec<Point3<f32>>) {
        let vertices = vec![
            Point3::new(15.566987, -1.129e-11, 0.25),
            Point3::new(14.283494, 1.241025, 0.25),
            Point3::new(14.716506, 0.491025, 0.25),
            Point3::new(14.283494, -1.241025, 0.25),
            Point3::new(14.716506, -0.491025, 0.25),
            Point3::new(16.0, 0.75, 0.25),
            Point3::new(15.149519, 1.241025, 0.25),
            Point3::new(16.0, 1.732051, 0.25),
            Point3::new(16.108253, 0.1875, -0.5),
            Point3::new(16.433012, -1.129e-11, 0.25),
            Point3::new(14.716506, 1.991025, 0.25),
            Point3::new(15.566987, 2.482051, 0.25),
            Point3::new(14.283494, 3.723076, 0.25),
            Point3::new(14.716506, 2.973076, 0.25),
            Point3::new(14.554127, 1.334775, -0.5),
            Point3::new(14.5, -0.866025, -0.5),
            Point3::new(14.5, 3.348076, -0.5),
            Point3::new(16.108253, 2.294551, -0.5),
            Point3::new(16.433012, 2.482051, 0.25),
        ];

        let faces = vec![
            (4, 3, 0),
            (0, 9, 1),
            (2, 1, 3),
            (7, 5, 9),
            (5, 6, 9),
            (6, 7, 18),
            (15, 4, 0),
            (3, 15, 9),
            (10, 1, 11),
            (11, 18, 12),
            (13, 12, 1),
            (14, 2, 15),
            (1, 14, 15),
            (8, 0, 2),
            (8, 14, 6),
            (16, 13, 10),
            (12, 16, 1),
            (17, 8, 7),
            (18, 9, 8),
            (14, 17, 6),
            (17, 11, 16),
            (18, 17, 16),
            (14, 10, 17),
            (3, 9, 0),
            (0, 1, 2),
            (2, 3, 4),
            (7, 9, 18),
            (6, 1, 9),
            (6, 18, 1),
            (15, 0, 8),
            (15, 8, 9),
            (1, 18, 11),
            (11, 12, 13),
            (13, 1, 10),
            (2, 4, 15),
            (1, 15, 3),
            (8, 2, 14),
            (8, 6, 5),
            (16, 10, 14),
            (16, 14, 1),
            (8, 5, 7),
            (18, 8, 17),
            (17, 7, 6),
            (11, 13, 16),
            (18, 16, 12),
            (10, 11, 17),
        ];

        (faces, vertices)
    }

    pub fn non_manifold_shape() -> (Vec<(u32, u32, u32)>, Vec<Point3<f32>>) {
        let vertices = vec![
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(-1.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];

        let faces = vec![(0, 1, 2), (2, 3, 0), (2, 4, 0)];

        (faces, vertices)
    }

    pub fn box_sharp_mismatching_winding() -> Mesh {
        let vertex_positions = vec![
            // back
            Point3::new(-1.0, 1.0, -1.0),
            Point3::new(-1.0, 1.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(1.0, 1.0, -1.0),
            // front
            Point3::new(-1.0, -1.0, -1.0),
            Point3::new(-1.0, -1.0, 1.0),
            Point3::new(1.0, -1.0, 1.0),
            Point3::new(1.0, -1.0, -1.0),
        ];

        let vertex_normals = vec![
            // back
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            // front
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            // top
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
            // bottom
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, -1.0),
            // right
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            // left
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
        ];

        let faces = vec![
            // back
            TriangleFace::from_same_vertex_and_normal_index(2, 1, 0),
            TriangleFace::from_same_vertex_and_normal_index(2, 3, 0),
            // top
            TriangleFace::from_same_vertex_and_normal_index(2, 1, 5),
            TriangleFace::from_same_vertex_and_normal_index(2, 5, 6),
            // right
            TriangleFace::from_same_vertex_and_normal_index(2, 6, 7),
            TriangleFace::from_same_vertex_and_normal_index(7, 3, 2),
            // bottom
            TriangleFace::from_same_vertex_and_normal_index(3, 7, 4),
            TriangleFace::from_same_vertex_and_normal_index(4, 0, 3),
            // front
            TriangleFace::from_same_vertex_and_normal_index(6, 4, 7),
            TriangleFace::from_same_vertex_and_normal_index(4, 6, 5),
            // left
            TriangleFace::from_same_vertex_and_normal_index(0, 4, 5),
            TriangleFace::from_same_vertex_and_normal_index(5, 1, 0),
        ];

        Mesh::from_triangle_faces_with_vertices_and_normals(faces, vertex_positions, vertex_normals)
    }

    fn tessellated_triangle() -> (Vec<(u32, u32, u32)>, Vec<Point3<f32>>) {
        let vertices = vec![
            Point3::new(-2.0, -2.0, 0.0),
            Point3::new(0.0, -2.0, 0.0),
            Point3::new(2.0, -2.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
        ];

        let faces = vec![(0, 3, 1), (1, 3, 4), (1, 4, 2), (3, 5, 4)];

        (faces, vertices)
    }

    fn tessellated_triangle_with_island() -> (Vec<(u32, u32, u32)>, Vec<Point3<f32>>) {
        let vertices = vec![
            Point3::new(-2.0, -2.0, 0.0),
            Point3::new(0.0, -2.0, 0.0),
            Point3::new(2.0, -2.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
            Point3::new(-1.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 2.0, 1.0),
        ];

        let faces = vec![(0, 3, 1), (1, 3, 4), (1, 4, 2), (3, 5, 4), (6, 7, 8)];

        (faces, vertices)
    }

    #[test]
    fn test_edge_sharing() {
        let (faces, vertices) = quad();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );

        let oriented_edges: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = edge_sharing(&oriented_edges);

        let unoriented_edges_one_way_correct = vec![
            UnorientedEdge(OrientedEdge::new(0, 1)),
            UnorientedEdge(OrientedEdge::new(1, 2)),
            UnorientedEdge(OrientedEdge::new(2, 3)),
            UnorientedEdge(OrientedEdge::new(3, 0)),
        ];
        let unoriented_edges_two_ways_correct = vec![
            UnorientedEdge(OrientedEdge::new(2, 0)),
            UnorientedEdge(OrientedEdge::new(0, 2)),
        ];

        for ue in unoriented_edges_one_way_correct {
            assert!(edge_sharing_map.contains_key(&ue));
            if ue.0.vertices.0 < ue.0.vertices.1 {
                assert_eq!(edge_sharing_map.get(&ue).unwrap().edge_count_ascending, 1);
                assert_eq!(edge_sharing_map.get(&ue).unwrap().edge_count_descending, 0);
            } else {
                assert_eq!(edge_sharing_map.get(&ue).unwrap().edge_count_ascending, 0);
                assert_eq!(edge_sharing_map.get(&ue).unwrap().edge_count_descending, 1);
            }
        }

        for ue in unoriented_edges_two_ways_correct {
            assert!(edge_sharing_map.contains_key(&ue));
            assert_eq!(edge_sharing_map.get(&ue).unwrap().edge_count_ascending, 1);
            assert_eq!(edge_sharing_map.get(&ue).unwrap().edge_count_descending, 1);
        }
    }

    #[test]
    fn test_find_edge_with_valency() {
        let (faces, vertices) = quad();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );
        let oriented_edges: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = edge_sharing(&oriented_edges);

        let oriented_edges_with_valency_1_correct = vec![
            OrientedEdge::new(0, 1),
            OrientedEdge::new(1, 2),
            OrientedEdge::new(2, 3),
            OrientedEdge::new(3, 0),
        ];
        let oriented_edges_with_valency_2_correct =
            vec![OrientedEdge::new(2, 0), OrientedEdge::new(0, 2)];

        let oriented_edges_with_valency_1: Vec<_> =
            find_edges_with_valency(&edge_sharing_map, 1).collect();
        let oriented_edges_with_valency_2: Vec<_> =
            find_edges_with_valency(&edge_sharing_map, 2).collect();

        assert_eq!(oriented_edges_with_valency_1.len(), 4);
        assert_eq!(oriented_edges_with_valency_2.len(), 2);

        for o_e in oriented_edges_with_valency_1_correct {
            assert!(oriented_edges_with_valency_1.iter().any(|e| *e == o_e));
        }

        for o_e in oriented_edges_with_valency_2_correct {
            assert!(oriented_edges_with_valency_2.iter().any(|e| *e == o_e));
        }
    }

    #[test]
    fn test_border_edges() {
        let (faces, vertices) = quad();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );
        let oriented_edges: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = edge_sharing(&oriented_edges);

        let oriented_edges_border_correct = vec![
            OrientedEdge::new(0, 1),
            OrientedEdge::new(1, 2),
            OrientedEdge::new(2, 3),
            OrientedEdge::new(3, 0),
        ];

        let oriented_edges_border_check: Vec<_> = border_edges(&edge_sharing_map).collect();

        for o_e in &oriented_edges_border_correct {
            assert!(oriented_edges_border_check.iter().any(|e| e == o_e));
        }
    }

    #[test]
    fn test_manifold_edges() {
        let (faces, vertices) = quad();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );
        let oriented_edges: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = edge_sharing(&oriented_edges);

        let oriented_edges_manifold_correct =
            vec![OrientedEdge::new(2, 0), OrientedEdge::new(0, 2)];

        let oriented_edges_manifold_check: Vec<_> = manifold_edges(&edge_sharing_map).collect();

        for o_e in oriented_edges_manifold_correct {
            assert!(oriented_edges_manifold_check.iter().any(|e| *e == o_e));
        }
    }

    #[test]
    fn test_non_manifold_edges() {
        let (faces, vertices) = non_manifold_shape();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );
        let oriented_edges: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = edge_sharing(&oriented_edges);

        let oriented_edges_non_manifold_correct =
            vec![OrientedEdge::new(2, 0), OrientedEdge::new(0, 2)];

        let oriented_edges_non_manifold_check: Vec<_> =
            non_manifold_edges(&edge_sharing_map).collect();

        for o_e in &oriented_edges_non_manifold_correct {
            assert!(oriented_edges_non_manifold_check.iter().any(|e| e == o_e));
        }

        assert_eq!(oriented_edges_non_manifold_check.len(), 3);
    }

    #[test]
    fn test_is_mesh_manifold_returns_false_because_non_manifold() {
        let (faces, vertices) = non_manifold_shape();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );

        let oriented_edges: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = edge_sharing(&oriented_edges);

        assert!(!is_mesh_manifold(&edge_sharing_map));
    }

    #[test]
    fn test_is_mesh_manifold_returns_true_because_manifold() {
        let (faces, vertices) = torus();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );

        let oriented_edges: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = edge_sharing(&oriented_edges);

        assert!(is_mesh_manifold(&edge_sharing_map));
    }

    #[test]
    fn test_border_vertex_indices() {
        let (faces, vertices) = quad();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );
        let oriented_edges: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = edge_sharing(&oriented_edges);

        let border_vertex_indices_correct = vec![0, 1, 2, 3];

        let border_vertex_indices_check = border_vertex_indices(&edge_sharing_map);

        for v_i in border_vertex_indices_correct {
            assert!(border_vertex_indices_check.iter().any(|v| *v == v_i));
        }
    }

    #[test]
    fn test_is_mesh_orientable_returns_true_watertight_mesh() {
        let mesh = primitive::create_box(
            Point3::origin(),
            Rotation3::identity(),
            Vector3::new(1.0, 1.0, 1.0),
        );
        let oriented_edges: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = edge_sharing(&oriented_edges);

        assert!(is_mesh_orientable(&edge_sharing_map));
    }

    #[test]
    fn test_is_mesh_orientable_returns_true_open_mesh() {
        let (faces, vertices) = quad();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );
        let oriented_edges: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = edge_sharing(&oriented_edges);

        assert!(is_mesh_orientable(&edge_sharing_map));
    }

    #[test]
    fn test_is_mesh_orientable_returns_false_for_nonorientable_mesh() {
        let mesh = box_sharp_mismatching_winding();

        let oriented_edges: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = edge_sharing(&oriented_edges);

        assert!(!is_mesh_orientable(&edge_sharing_map));
    }

    #[test]
    fn test_is_mesh_watertight_returns_true_for_watertight_mesh() {
        let mesh = primitive::create_box(
            Point3::origin(),
            Rotation3::identity(),
            Vector3::new(1.0, 1.0, 1.0),
        );
        let oriented_edges: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = edge_sharing(&oriented_edges);

        assert!(is_mesh_watertight(&edge_sharing_map));
    }

    #[test]
    fn test_is_mesh_watertight_returns_false_for_open_mesh() {
        let (faces, vertices) = quad();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );

        let oriented_edges: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = edge_sharing(&oriented_edges);

        assert!(!is_mesh_watertight(&edge_sharing_map));
    }

    #[test]
    fn test_triangulated_mesh_genus_box_should_be_0() {
        let mesh = primitive::create_box(
            Point3::origin(),
            Rotation3::identity(),
            Vector3::new(1.0, 1.0, 1.0),
        );
        assert!(mesh.is_triangulated());

        let edges: HashSet<UnorientedEdge> = mesh.unoriented_edges_iter().collect();

        let genus = triangulated_mesh_genus(mesh.vertices().len(), edges.len(), mesh.faces().len());
        assert_eq!(genus, 0);
    }

    #[test]
    fn test_triangulated_mesh_genus_torus_should_be_1() {
        let (faces, vertices) = torus();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );
        assert!(mesh.is_triangulated());

        let edges: HashSet<UnorientedEdge> = mesh.unoriented_edges_iter().collect();

        let genus = triangulated_mesh_genus(mesh.vertices().len(), edges.len(), mesh.faces().len());
        assert_eq!(genus, 1);
    }

    #[test]
    fn test_triangulated_mesh_genus_double_torus_should_be_2() {
        let (faces, vertices) = double_torus();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );
        assert!(mesh.is_triangulated());

        let edges: HashSet<UnorientedEdge> = mesh.unoriented_edges_iter().collect();

        let genus = triangulated_mesh_genus(mesh.vertices().len(), edges.len(), mesh.faces().len());
        assert_eq!(genus, 2);
    }

    #[test]
    fn test_triangulated_mesh_genus_triple_torus_should_be_3() {
        let (faces, vertices) = triple_torus();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );
        assert!(mesh.is_triangulated());

        let edges: HashSet<UnorientedEdge> = mesh.unoriented_edges_iter().collect();

        let genus = triangulated_mesh_genus(mesh.vertices().len(), edges.len(), mesh.faces().len());
        assert_eq!(genus, 3);
    }

    #[test]
    fn test_border_edge_loops_returns_one_for_tessellated_triangle() {
        let (faces, vertices) = tessellated_triangle();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );

        let oriented_edges: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = edge_sharing(&oriented_edges);

        // Since these edges are oriented, we need to account that the triangle
        // face starts the winding with the smallest index and therefore rotates
        // the faces created in `tessellated_triangle`
        let correct_loop = vec![
            OrientedEdge::new(1, 0),
            OrientedEdge::new(2, 1),
            OrientedEdge::new(4, 2),
            OrientedEdge::new(5, 4),
            OrientedEdge::new(3, 5),
            OrientedEdge::new(0, 3),
        ];

        if let BorderEdgeLoopsResult::Found(computed_loops) = border_edge_loops(&edge_sharing_map) {
            assert_eq!(computed_loops.len(), 1);
            assert_eq!(computed_loops[0].len(), correct_loop.len());
            for edge in correct_loop {
                assert!(computed_loops[0].iter().any(|e| *e == edge));
            }
        } else {
            panic!();
        }
    }

    #[test]
    fn test_border_edge_loops_returns_zero_for_box() {
        let mesh = primitive::create_box(
            Point3::origin(),
            Rotation3::identity(),
            Vector3::new(1.0, 1.0, 1.0),
        );

        let oriented_edges: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = edge_sharing(&oriented_edges);

        let computed_loops = border_edge_loops(&edge_sharing_map);

        assert!(computed_loops == BorderEdgeLoopsResult::Watertight);
    }

    #[test]
    fn test_border_edge_loops_returns_two_for_tessellated_triangle_with_island() {
        let (faces, vertices) = tessellated_triangle_with_island();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );

        let oriented_edges: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = edge_sharing(&oriented_edges);

        // Since these edges are oriented, we need to account that the triangle
        // face starts the winding with the smallest index and therefore rotates
        // the faces created in `tessellated_triangle_with_island`
        let correct_loops: Vec<Vec<OrientedEdge>> = vec![
            vec![
                OrientedEdge::new(1, 0),
                OrientedEdge::new(2, 1),
                OrientedEdge::new(4, 2),
                OrientedEdge::new(5, 4),
                OrientedEdge::new(3, 5),
                OrientedEdge::new(0, 3),
            ],
            vec![
                OrientedEdge::new(6, 7),
                OrientedEdge::new(7, 8),
                OrientedEdge::new(8, 6),
            ],
        ];

        if let BorderEdgeLoopsResult::Found(computed_loops) = border_edge_loops(&edge_sharing_map) {
            assert_eq!(computed_loops.len(), 2);
            assert!(
                computed_loops[0].len() == correct_loops[0].len()
                    || computed_loops[0].len() == correct_loops[1].len()
            );

            let computed_loops_tuple = if computed_loops[0].len() == correct_loops[0].len() {
                assert_eq!(computed_loops[1].len(), correct_loops[1].len());
                (&computed_loops[0], &computed_loops[1])
            } else {
                assert_eq!(computed_loops[1].len(), correct_loops[0].len());
                assert_eq!(computed_loops[0].len(), correct_loops[1].len());
                (&computed_loops[1], &computed_loops[0])
            };

            for edge in &correct_loops[0] {
                assert!(computed_loops_tuple.0.iter().any(|e| e == edge));
            }

            for edge in &correct_loops[1] {
                assert!(computed_loops_tuple.1.iter().any(|e| e == edge));
            }
        } else {
            panic!();
        }
    }

    #[test]
    fn test_border_edge_loops_returns_watertight_for_torus() {
        let (faces, vertices) = torus();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );

        let oriented_edges: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = edge_sharing(&oriented_edges);

        assert_eq!(
            BorderEdgeLoopsResult::Watertight,
            border_edge_loops(&edge_sharing_map),
        );
    }

    #[test]
    fn test_are_similar_returns_true_for_same() {
        let (faces, vertices) = quad();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );

        assert!(are_similar(&mesh, &mesh));
    }

    #[test]
    fn test_are_similar_returns_true_for_renumbered() {
        let mesh = quad_with_normals();
        let mesh_renumbered = quad_renumbered_with_normals();

        assert!(are_similar(&mesh, &mesh_renumbered));
    }

    #[test]
    fn test_are_visually_similar_returns_true_for_not_watertight() {
        let mesh = quad_with_normals();
        let mesh_not_watertight = quad_with_extra_vertices_and_normals();

        assert!(are_visually_similar(&mesh, &mesh_not_watertight));
    }

    #[test]
    fn test_are_similar_returns_false_for_not_watertight() {
        let mesh = quad_with_normals();
        let mesh_not_watertight = quad_with_extra_vertices_and_normals();

        assert!(!are_similar(&mesh, &mesh_not_watertight));
    }

    #[test]
    fn test_are_visually_similar_returns_true_for_renumbered_more() {
        let mesh = quad_with_normals();
        let mesh_renumbered = quad_renumbered_more_with_normals();

        assert!(are_visually_similar(&mesh, &mesh_renumbered));
    }

    #[test]
    fn test_are_similar_returns_false_for_different() {
        let mesh = quad_with_normals();
        let (faces_d, vertices_d) = tessellated_triangle();
        let mesh_d = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces_d,
            vertices_d,
            NormalStrategy::Sharp,
        );

        assert!(!are_similar(&mesh, &mesh_d));
    }
}
