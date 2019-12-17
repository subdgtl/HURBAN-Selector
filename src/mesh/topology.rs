use smallvec::SmallVec;

use crate::convert::{cast_u32, cast_usize};

use super::{Face, Mesh};

// FIXME: Ideally, we'd also create a wrapper struct that casts the indices
// to/from u32 as necessary.
//
// FIXME: @Optimization Analyze where this threshold is overly benevolent and
// define different thresholds for different topologies.

/// The number of relations/neighbors a `SmallVec` is allowed to
/// contain before it spills into heap. Implementation detail.
pub const MAX_INLINE_NEIGHBOR_COUNT: usize = 8;

/// Computes topological relations of mesh vertex -> faces. A vertex is related
/// to a face if and only if the face contains the respective vertex.
///
/// Output: The index represents a vertex index, the value is a list of faces
/// containing the respective vertex.
pub fn compute_vertex_to_face_topology(
    mesh: &Mesh,
) -> Vec<SmallVec<[u32; MAX_INLINE_NEIGHBOR_COUNT]>> {
    compute_vertex_to_face_topology_from_components(mesh.faces(), cast_u32(mesh.vertices().len()))
}

/// Computes topological relations of mesh vertex -> faces from standalone mesh
/// components: faces and vertex count. A vertex is related to a face if and
/// only if the face contains the respective vertex.
///
/// Output: The index represents a vertex index, the value is a list of faces
/// containing the respective vertex.
pub fn compute_vertex_to_face_topology_from_components(
    faces: &[Face],
    vertex_count: u32,
) -> Vec<SmallVec<[u32; MAX_INLINE_NEIGHBOR_COUNT]>> {
    let mut v2f: Vec<SmallVec<[u32; 8]>> = vec![SmallVec::new(); cast_usize(vertex_count)];

    for (face_index, face) in faces.iter().enumerate() {
        let face_index_u32 = cast_u32(face_index);

        match face {
            Face::Triangle(triangle_face) => {
                let vertices = &triangle_face.vertices;

                for from_vertex in &[vertices.0, vertices.1, vertices.2] {
                    if !v2f[cast_usize(*from_vertex)].contains(&face_index_u32) {
                        v2f[cast_usize(*from_vertex)].push(face_index_u32);
                    }
                }
            }
        }
    }

    v2f
}

/// Computes topological relations (neighborhood) of mesh face -> faces. Two
/// faces are neighbors if and only if they they share an unoriented edge.
///
/// Output: The index represents a face index, the value is a list of faces
/// neighboring with the respective face.
pub fn compute_face_to_face_topology(
    mesh: &Mesh,
    v2f: &[SmallVec<[u32; MAX_INLINE_NEIGHBOR_COUNT]>],
) -> Vec<SmallVec<[u32; MAX_INLINE_NEIGHBOR_COUNT]>> {
    let mut f2f: Vec<SmallVec<[u32; 8]>> = vec![SmallVec::new(); mesh.faces().len()];

    for (face_index, face) in mesh.faces().iter().enumerate() {
        match face {
            Face::Triangle(triangle_face) => {
                let vertices = &triangle_face.vertices;
                let faces_containing_1st_vertex = &v2f[cast_usize(vertices.0)];
                let faces_containing_2nd_vertex = &v2f[cast_usize(vertices.1)];
                let faces_containing_3rd_vertex = &v2f[cast_usize(vertices.2)];
                for face_containing_1st_vertex in faces_containing_1st_vertex {
                    if *face_containing_1st_vertex != cast_u32(face_index)
                        && (faces_containing_2nd_vertex.contains(&face_containing_1st_vertex)
                            || faces_containing_3rd_vertex.contains(&face_containing_1st_vertex))
                        && !f2f[face_index].contains(face_containing_1st_vertex)
                    {
                        f2f[face_index].push(*face_containing_1st_vertex);
                    }
                }
                for face_containing_second_vertex in faces_containing_2nd_vertex {
                    if *face_containing_second_vertex != cast_u32(face_index)
                        && (faces_containing_3rd_vertex.contains(&face_containing_second_vertex))
                        && !f2f[face_index].contains(face_containing_second_vertex)
                    {
                        f2f[face_index].push(*face_containing_second_vertex);
                    }
                }
            }
        }
    }

    f2f
}

/// Computes topological relations (connections) of mesh vertex -> vertices. Two
/// vertices are neighbors if and only if they they are end points of an edge.
///
/// Output: The index represents a vertex index, the value is a list of vertices
/// connected to the respective vertex.
pub fn compute_vertex_to_vertex_topology(
    mesh: &Mesh,
) -> Vec<SmallVec<[u32; MAX_INLINE_NEIGHBOR_COUNT]>> {
    let mut v2v = vec![SmallVec::new(); mesh.vertices().len()];

    for face in mesh.faces() {
        match face {
            Face::Triangle(f) => {
                let vertex_indices = &[f.vertices.0, f.vertices.1, f.vertices.2];
                for i in 0..vertex_indices.len() {
                    let neighbor_candidate1 = vertex_indices[(i + 1) % 3];
                    let neighbor_candidate2 = vertex_indices[(i + 2) % 3];

                    let neighbor_vertices = &mut v2v[cast_usize(vertex_indices[i])];
                    if !neighbor_vertices.contains(&neighbor_candidate1) {
                        neighbor_vertices.push(neighbor_candidate1)
                    }
                    if !neighbor_vertices.contains(&neighbor_candidate2) {
                        neighbor_vertices.push(neighbor_candidate2)
                    }
                }
            }
        }
    }

    v2v
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;
    use smallvec::smallvec;

    use crate::mesh::NormalStrategy;

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
    // test_face_to_face_topology_does_not_include_self_in_neighbors
    // does

    #[test]
    fn test_compute_face_to_face_topology_does_not_include_self_in_neighbors() {
        let (faces, vertices) = tessellated_triangle();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );
        let vertex_to_face_topology = compute_vertex_to_face_topology(&mesh);
        let face_to_face_topology_calculated =
            compute_face_to_face_topology(&mesh, &vertex_to_face_topology);

        for (self_index, neighbor_indices) in face_to_face_topology_calculated.iter().enumerate() {
            assert!(!neighbor_indices.contains(&cast_u32(self_index)));
        }
    }

    #[test]
    fn test_compute_face_to_face_topology_from_tessellated_triangle() {
        let (faces, vertices) = tessellated_triangle();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );
        let face_to_face_topology_correct: Vec<SmallVec<[u32; MAX_INLINE_NEIGHBOR_COUNT]>> =
            vec![smallvec![1], smallvec![0, 2, 3], smallvec![1], smallvec![1]];

        let vertex_to_face_topology = compute_vertex_to_face_topology(&mesh);
        let face_to_face_topology_calculated =
            compute_face_to_face_topology(&mesh, &vertex_to_face_topology);

        assert_eq!(
            face_to_face_topology_calculated,
            face_to_face_topology_correct,
        );
    }

    #[test]
    fn test_compute_vertex_to_vertex_topology_from_tessellated_triangle() {
        let (faces, vertices) = tessellated_triangle();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );

        let vertex_to_vertex_topology_calculated = compute_vertex_to_vertex_topology(&mesh);

        let two_neighbors_count = vertex_to_vertex_topology_calculated
            .iter()
            .filter(|to| to.len() == 2)
            .count();
        let four_neighbors_count = vertex_to_vertex_topology_calculated
            .iter()
            .filter(|to| to.len() == 4)
            .count();

        assert_eq!(two_neighbors_count, 3);
        assert_eq!(four_neighbors_count, 3);
    }
}
