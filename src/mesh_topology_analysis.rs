use std::collections::HashMap;

use smallvec::SmallVec;

use crate::convert::cast_u32;
use crate::geometry::{Face, Geometry};

// FIXME: @Optimization Explore whether the topologies wouldn't be
// better served by being backed by a `Vec` instead of
// `HashMap`. Ideally, we'd also create a wrapper struct that casts
// the indices to/from u32 as necessary.

// FIXME: @Optimization Analyze where this threshold is overly
// benevolent and define different thresholds for different
// topologies.

/// The number of relations/neighbors a `SmallVec` is allowed to
/// contain before it spills into heap. Implementation detail.
const MAX_INLINE_NEIGHBOR_COUNT: usize = 8;

/// A topology containing neighborhood relations between faces. Two
/// faces are neighbors if and only if they they share an unoriented
/// edge.
pub type FaceToFaceTopology = HashMap<u32, SmallVec<[u32; MAX_INLINE_NEIGHBOR_COUNT]>>;

/// Computes face to face topology for geometry.
pub fn face_to_face_topology(geometry: &Geometry) -> FaceToFaceTopology {
    let mut f2f: HashMap<u32, SmallVec<[u32; 8]>> = HashMap::new();

    for (from_face_index, from_face) in geometry.faces().iter().enumerate() {
        let from_face_index_u32 = cast_u32(from_face_index);
        let mut neighbors = SmallVec::new();

        match from_face {
            Face::Triangle(triangle_face) => {
                let [e1, e2, e3] = triangle_face.to_unoriented_edges();
                for (to_face_index, to_face) in geometry.faces().iter().enumerate() {
                    let to_face_index_u32 = cast_u32(to_face_index);

                    match to_face {
                        Face::Triangle(triangle_face) => {
                            let face_contains_edge = triangle_face.contains_unoriented_edge(e1)
                                || triangle_face.contains_unoriented_edge(e2)
                                || triangle_face.contains_unoriented_edge(e3);
                            if face_contains_edge
                                && from_face_index != to_face_index
                                && !neighbors.contains(&to_face_index_u32)
                            {
                                neighbors.push(to_face_index_u32);
                            }
                        }
                    }
                }
            }
        }

        f2f.insert(from_face_index_u32, neighbors);
    }

    f2f
}

/// A topology containing neighborhood relations between vertices. Two
/// vertices are neighbors if and only if they they are end points of
/// an edge.
pub type VertexToVertexTopology = HashMap<u32, SmallVec<[u32; MAX_INLINE_NEIGHBOR_COUNT]>>;

/// Computes vertex to vertex topology for geometry.
pub fn vertex_to_vertex_topology(geometry: &Geometry) -> VertexToVertexTopology {
    let mut v2v: HashMap<u32, SmallVec<[u32; 8]>> = HashMap::new();

    for face in geometry.faces() {
        match face {
            Face::Triangle(f) => {
                let vertex_indices = &[f.vertices.0, f.vertices.1, f.vertices.2];
                for i in 0..vertex_indices.len() {
                    let neighbors = v2v.entry(vertex_indices[i]).or_insert_with(SmallVec::new);

                    let neighbor_candidate1 = vertex_indices[(i + 1) % 3];
                    let neighbor_candidate2 = vertex_indices[(i + 2) % 3];

                    if !neighbors.contains(&neighbor_candidate1) {
                        neighbors.push(neighbor_candidate1)
                    }
                    if !neighbors.contains(&neighbor_candidate2) {
                        neighbors.push(neighbor_candidate2)
                    }
                }
            }
        }
    }

    v2v
}

#[cfg(test)]
mod tests {
    use nalgebra::geometry::Point3;
    use smallvec::smallvec;

    use crate::geometry::NormalStrategy;

    use super::*;

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

    // FIXME: make proptests on all topologies verifying that self is
    // not included in neighbors, similar to what
    // test_geometry_face_to_face_topology_does_not_include_self_in_neighbors
    // does

    #[test]
    fn test_geometry_face_to_face_topology_does_not_include_self_in_neighbors() {
        let (faces, vertices) = tessellated_triangle();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );
        let face_to_face_topology = face_to_face_topology(&geometry);

        for (key, value) in face_to_face_topology {
            assert!(!value.contains(&key));
        }
    }

    #[test]
    fn test_geometry_face_to_face_topology_from_tessellated_triangle() {
        let (faces, vertices) = tessellated_triangle();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let mut face_to_face_topology_correct: HashMap<u32, SmallVec<[u32; 8]>> = HashMap::new();
        face_to_face_topology_correct.insert(0, smallvec![1]);
        face_to_face_topology_correct.insert(1, smallvec![0, 2, 3]);
        face_to_face_topology_correct.insert(2, smallvec![1]);
        face_to_face_topology_correct.insert(3, smallvec![1]);

        let face_to_face_topology_calculated = face_to_face_topology(&geometry);

        assert_eq!(
            face_to_face_topology_calculated,
            face_to_face_topology_correct,
        );
    }

    #[test]
    fn test_geometry_vertex_to_vertex_topology_from_tessellated_triangle() {
        let (faces, vertices) = tessellated_triangle();
        let geometry = Geometry::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );

        let vertex_to_vertex_topology_calculated = vertex_to_vertex_topology(&geometry);

        let two_neighbors_count = vertex_to_vertex_topology_calculated
            .iter()
            .filter(|(_, to)| to.len() == 2)
            .count();
        let four_neighbors_count = vertex_to_vertex_topology_calculated
            .iter()
            .filter(|(_, to)| to.len() == 4)
            .count();

        assert_eq!(two_neighbors_count, 3);
        assert_eq!(four_neighbors_count, 3);
    }
}
