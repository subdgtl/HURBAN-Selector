use std::cmp;
use std::collections::hash_map::{Entry, HashMap};
use std::hash::{Hash, Hasher};

use smallvec::SmallVec;

use nalgebra as na;
use nalgebra::Point3;

use crate::convert::{cast_u32, cast_usize};
use crate::mesh::{topology, Face, Mesh, NormalStrategy};

/// Relaxes angles between mesh edges, resulting in a smoother
/// mesh, optionally keeping some vertices anchored, resulting in
/// an evenly distributed geometry optionally stretched between the
/// anchor points
///
/// The number of vertices, faces and the overall topology remains unchanged.
/// The more iterations, the smoother result. Too many iterations may cause slow
/// calculation time. In case the `stop_when_stable` flag is set on, the smoothing
/// stops when the mesh stops transforming between iterations, or when it
/// reaches the maximum number of iterations.
///
/// The algorithm is based on replacing each vertex position with an average
/// position of its immediate neighbors.
///
/// Returns `(smooth_mesh: Mesh, executed_iterations: u32, stable: bool)`.
pub fn laplacian_smoothing(
    mesh: &Mesh,
    vertex_to_vertex_topology: &[SmallVec<[u32; topology::MAX_INLINE_NEIGHBOR_COUNT]>],
    max_iterations: u32,
    fixed_vertex_indices: &[u32],
    stop_when_stable: bool,
) -> (Mesh, u32, bool) {
    if max_iterations == 0 {
        return (mesh.clone(), 0, false);
    }

    let mut vertices: Vec<Point3<f32>> = Vec::from(mesh.vertices());
    let mut mesh_vertices: Vec<Point3<f32>>;

    let mut iteration: u32 = 0;

    // Only relevant when fixed vertices are specified
    let mut stable = !fixed_vertex_indices.is_empty();
    while iteration < max_iterations {
        stable = !fixed_vertex_indices.is_empty();
        mesh_vertices = vertices.clone();

        for (current_vertex_index, neighbors_indices) in
            vertex_to_vertex_topology.iter().enumerate()
        {
            if fixed_vertex_indices
                .iter()
                .all(|i| *i != cast_u32(current_vertex_index))
                && !neighbors_indices.is_empty()
            {
                let mut average_position: Point3<f32> = Point3::origin();
                for neighbor_index in neighbors_indices {
                    average_position += mesh_vertices[cast_usize(*neighbor_index)].coords;
                }
                average_position /= neighbors_indices.len() as f32;
                stable &= approx::relative_eq!(
                    &average_position.coords,
                    &vertices[current_vertex_index].coords,
                );
                vertices[current_vertex_index] = average_position;
            }
        }
        iteration += 1;

        if stop_when_stable && stable {
            break;
        }
    }

    (
        Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            mesh.faces().iter().map(|face| match face {
                Face::Triangle(t_f) => (t_f.vertices.0, t_f.vertices.1, t_f.vertices.2),
            }),
            vertices,
            NormalStrategy::Smooth,
        ),
        iteration,
        stable,
    )
}

/// Performs one iteration of Loop Subdivision on mesh.
///
/// The subdivision works in two steps:
///
/// 1) Split each triangle into 4 smaller triangles,
/// 2) Update the position of each vertex of the mesh based on
///    weighted averages of its neighboring vertex positions,
///    depending on where the vertex is in the topology and whether
///    the vertex is newly created, or did already exist.
///
/// The mesh **must** be triangulated and manifold.
///
/// Implementation based on [mdfisher]
/// (https://graphics.stanford.edu/~mdfisher/subdivision.html).
pub fn loop_subdivision(
    mesh: &Mesh,
    vertex_to_vertex_topology: &[SmallVec<[u32; topology::MAX_INLINE_NEIGHBOR_COUNT]>],
    face_to_face_topology: &[SmallVec<[u32; topology::MAX_INLINE_NEIGHBOR_COUNT]>],
) -> Option<Mesh> {
    #[derive(Debug, Eq)]
    struct UnorderedPair(u32, u32);

    impl PartialEq for UnorderedPair {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0 && self.1 == other.1 || self.0 == other.1 && self.1 == other.0
        }
    }

    impl Hash for UnorderedPair {
        fn hash<H: Hasher>(&self, state: &mut H) {
            cmp::min(self.0, self.1).hash(state);
            cmp::max(self.0, self.1).hash(state);
        }
    }

    if !mesh.is_triangulated() {
        return None;
    }

    let mut vertices: Vec<Point3<f32>> = mesh.vertices().iter().copied().collect();

    // Relocate existing vertices first
    for (i, vertex) in vertices.iter_mut().enumerate() {
        let neighbors = &vertex_to_vertex_topology[i];

        match neighbors.len() {
            // N == 0 means this is an orphan vertex. N == 1 can't
            // happen in our mesh representation.
            0 | 1 => (),
            2 => {
                // For edge valency N == 2 (a naked edge vertex), use
                // (3/4, 1/8, 1/8) relocation scheme.

                let vi1 = cast_usize(neighbors[0]);
                let vi2 = cast_usize(neighbors[1]);

                let v1 = mesh.vertices()[vi1];
                let v2 = mesh.vertices()[vi2];

                *vertex = Point3::origin()
                    + vertex.coords * 3.0 / 4.0
                    + v1.coords * 1.0 / 8.0
                    + v2.coords * 1.0 / 8.0;
            }
            3 => {
                // For edge valency N == 3, use (1 - N*BETA, BETA,
                // BETA, BETA) relocation scheme, where BETA is 3/16.

                const N: f32 = 3.0;
                const BETA: f32 = 3.0 / 16.0;

                let vi1 = cast_usize(neighbors[0]);
                let vi2 = cast_usize(neighbors[1]);
                let vi3 = cast_usize(neighbors[2]);

                let v1 = mesh.vertices()[vi1];
                let v2 = mesh.vertices()[vi2];
                let v3 = mesh.vertices()[vi3];

                *vertex = Point3::origin()
                    + vertex.coords * (1.0 - N * BETA)
                    + v1.coords * BETA
                    + v2.coords * BETA
                    + v3.coords * BETA;
            }
            n => {
                // For edge valency N >= 3, use (1 - N*BETA, BETA,
                // ...) relocation scheme, where BETA is 3 / (8*N).

                let n_f32 = n as f32;
                let beta = 3.0 / (8.0 * n_f32);

                *vertex = Point3::origin() + vertex.coords * (1.0 - n_f32 * beta);
                for vi in neighbors {
                    let v = mesh.vertices()[cast_usize(*vi)];
                    *vertex += v.coords * beta;
                }
            }
        }
    }

    // Subdivide existing triangle faces and create new vertices

    let faces_len_estimate = mesh.faces().len() * 4;
    let mut faces: Vec<(u32, u32, u32)> = Vec::with_capacity(faces_len_estimate);

    // We will be creating new mid-edge vertices per face soon. Faces
    // will share and re-use these newly created vertices.

    // The key is an unordered pair of faces that share the mid-edge
    // vertex. The value is the index of the vertex they share.
    let mut created_mid_vertex_indices: HashMap<UnorderedPair, u32> = HashMap::new();

    for (face_index, face) in mesh.faces().iter().enumerate() {
        let face_index_u32 = cast_u32(face_index);
        match face {
            Face::Triangle(triangle_face) => {
                let (vi1, vi2, vi3) = triangle_face.vertices;
                let face_neighbors = &face_to_face_topology[face_index];

                // Our current face should have up to 3 neighboring
                // faces. The mid vertices we are going to create need
                // to be shared with those faces if they exist, so
                // that they are only created once. The array below
                // will be filled with either vertices created here,
                // or obtained from `created_mid_vertex_indices`
                // cache.
                let mut mid_vertex_indices: [Option<u32>; 3] = [None, None, None];

                for (edge_index, (vi_from, vi_to)) in
                    [(vi1, vi2), (vi2, vi3), (vi3, vi1)].iter().enumerate()
                {
                    let neighbor_face_index = face_neighbors
                        .iter()
                        .copied()
                        .map(|i| (i, mesh.faces()[cast_usize(i)]))
                        .find_map(|(i, face)| {
                            if face.contains_vertex(*vi_from) && face.contains_vertex(*vi_to) {
                                Some(i)
                            } else {
                                None
                            }
                        });

                    let mid_vertex_index = if let Some(neighbor_face_index) = neighbor_face_index {
                        let pair = UnorderedPair(face_index_u32, neighbor_face_index);

                        match created_mid_vertex_indices.entry(pair) {
                            // The vertex exists and was therefore
                            // already relocated by visiting a
                            // neighboring face in a previous
                            // iteration
                            Entry::Occupied(occupied) => *occupied.get(),
                            Entry::Vacant(vacant) => {
                                // Create and relocate the vertex
                                // using the (1/8, 3/8, 3/8, 1/8)
                                // scheme. Since there is a neighbor
                                // face, we also write the created
                                // vertex to the cache to be picked up
                                // by subsequent iterations.

                                let edge_vertex_from = mesh.vertices()[cast_usize(*vi_from)];
                                let edge_vertex_to = mesh.vertices()[cast_usize(*vi_to)];

                                let face1 = mesh.faces()[face_index];
                                let face2 = mesh.faces()[cast_usize(neighbor_face_index)];

                                // Find the two vertices that are
                                // opposite to the shared edge of the
                                // face pair.
                                let (opposite_vertex_index1, opposite_vertex_index2) =
                                    match (face1, face2) {
                                        (
                                            Face::Triangle(triangle_face1),
                                            Face::Triangle(triangle_face2),
                                        ) => {
                                            let f1vi1 = triangle_face1.vertices.0;
                                            let f1vi2 = triangle_face1.vertices.1;
                                            let f1vi3 = triangle_face1.vertices.2;

                                            let f2vi1 = triangle_face2.vertices.0;
                                            let f2vi2 = triangle_face2.vertices.1;
                                            let f2vi3 = triangle_face2.vertices.2;

                                            let f1v = [f1vi1, f1vi2, f1vi3];
                                            let f2v = [f2vi1, f2vi2, f2vi3];

                                            let f1_opposite_vertex = f1v
                                                .iter()
                                                .copied()
                                                .find(|vi| !f2v.contains(&vi))?;

                                            let f2_opposite_vertex = f2v
                                                .iter()
                                                .copied()
                                                .find(|vi| !f1v.contains(&vi))?;

                                            (f1_opposite_vertex, f2_opposite_vertex)
                                        }
                                    };

                                let opposite_vertex1 =
                                    mesh.vertices()[cast_usize(opposite_vertex_index1)];
                                let opposite_vertex2 =
                                    mesh.vertices()[cast_usize(opposite_vertex_index2)];

                                let new_vertex = Point3::origin()
                                    + opposite_vertex1.coords * 1.0 / 8.0
                                    + opposite_vertex2.coords * 1.0 / 8.0
                                    + edge_vertex_from.coords * 3.0 / 8.0
                                    + edge_vertex_to.coords * 3.0 / 8.0;

                                let index = cast_u32(vertices.len());
                                vacant.insert(index);
                                vertices.push(new_vertex);

                                index
                            }
                        }
                    } else {
                        // Create and relocate the vertex using the (1/2, 1/2) scheme
                        let vertex_from = mesh.vertices()[cast_usize(*vi_from)];
                        let vertex_to = mesh.vertices()[cast_usize(*vi_to)];

                        let new_vertex = na::center(&vertex_from, &vertex_to);

                        let index = cast_u32(vertices.len());
                        vertices.push(new_vertex);

                        index
                    };

                    mid_vertex_indices[edge_index] = Some(mid_vertex_index);
                }

                let mid_v1v2_index =
                    mid_vertex_indices[0].expect("Must have been produced by earlier loop");
                let mid_v2v3_index =
                    mid_vertex_indices[1].expect("Must have been produced by earlier loop");
                let mid_v3v1_index =
                    mid_vertex_indices[2].expect("Must have been produced by earlier loop");

                faces.push((vi1, mid_v1v2_index, mid_v3v1_index));
                faces.push((vi2, mid_v2v3_index, mid_v1v2_index));
                faces.push((vi3, mid_v3v1_index, mid_v2v3_index));
                faces.push((mid_v1v2_index, mid_v2v3_index, mid_v3v1_index));
            }
        }
    }

    assert_eq!(faces.len(), faces_len_estimate);
    assert_eq!(faces.capacity(), faces_len_estimate);

    // FIXME: Calculate better normals here? Maybe use `Smooth` strategy once we have it?
    Some(
        Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        ),
    )
}

#[cfg(test)]
mod tests {
    use std::iter::FromIterator;

    use nalgebra::{Rotation3, Vector3};

    use crate::mesh::{analysis, primitive, topology, NormalStrategy, OrientedEdge};

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

    fn shape_for_smoothing_with_anchors() -> (Vec<(u32, u32, u32)>, Vec<Point3<f32>>) {
        let vertices = vec![
            Point3::new(30.21796, -6.119943, 0.0),
            Point3::new(32.031532, 1.328689, 0.0),
            Point3::new(33.875141, -3.522298, 3.718605),
            Point3::new(34.571838, -2.071111, 2.77835),
            Point3::new(34.778172, -5.285372, 3.718605),
            Point3::new(36.243252, -3.80194, 3.718605),
            Point3::new(36.741604, -10.146505, 0.0),
            Point3::new(39.676025, 1.905633, 0.0),
            Point3::new(42.587009, -5.186427, 0.0),
        ];

        let faces = vec![
            (4, 8, 5),
            (4, 6, 8),
            (5, 8, 7),
            (3, 5, 7),
            (0, 2, 1),
            (1, 2, 3),
            (0, 4, 2),
            (1, 3, 7),
            (0, 6, 4),
            (2, 4, 5),
            (2, 5, 3),
        ];

        (faces, vertices)
    }

    fn shape_for_smoothing_with_anchors_50_iterations() -> (Vec<(u32, u32, u32)>, Vec<Point3<f32>>)
    {
        let vertices = vec![
            Point3::new(30.21796, -6.119943, 0.0),
            Point3::new(32.031532, 1.328689, 0.0),
            Point3::new(34.491065, -2.551039, 0.0),
            Point3::new(36.00632, -0.404003, 0.0),
            Point3::new(36.372859, -5.260642, 0.0),
            Point3::new(37.826656, -2.299296, 0.0),
            Point3::new(36.741604, -10.146505, 0.0),
            Point3::new(39.676025, 1.905633, 0.0),
            Point3::new(42.587009, -5.186427, 0.0),
        ];

        let faces = vec![
            (4, 8, 5),
            (4, 6, 8),
            (5, 8, 7),
            (3, 5, 7),
            (0, 2, 1),
            (1, 2, 3),
            (0, 4, 2),
            (1, 3, 7),
            (0, 6, 4),
            (2, 4, 5),
            (2, 5, 3),
        ];

        (faces, vertices)
    }

    #[test]
    fn test_laplacian_smoothing_vertex_normal_count_equals_vertex_count() {
        let (faces, vertices) = torus();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );

        let vertex_to_vertex_topology = topology::compute_vertex_to_vertex_topology(&mesh);
        let (relaxed_mesh_0, _, _) =
            laplacian_smoothing(&mesh, &vertex_to_vertex_topology, 0, &[], false);
        let (relaxed_mesh_1, _, _) =
            laplacian_smoothing(&mesh, &vertex_to_vertex_topology, 1, &[], false);
        let (relaxed_mesh_10, _, _) =
            laplacian_smoothing(&mesh, &vertex_to_vertex_topology, 10, &[], false);

        assert_eq!(
            relaxed_mesh_0.faces().len(),
            mesh.faces().len(),
            "Faces for 0 iterations"
        );
        assert_eq!(
            relaxed_mesh_1.faces().len(),
            mesh.faces().len(),
            "Faces for 1 iteration"
        );
        assert_eq!(
            relaxed_mesh_10.faces().len(),
            mesh.faces().len(),
            "Faces for 10 iterations"
        );
        assert_eq!(
            relaxed_mesh_0.vertices().len(),
            mesh.vertices().len(),
            "Vertices for 0 iterations"
        );
        assert_eq!(
            relaxed_mesh_1.vertices().len(),
            mesh.vertices().len(),
            "Vertices for 1 iteration"
        );
        assert_eq!(
            relaxed_mesh_10.vertices().len(),
            mesh.vertices().len(),
            "Vertices for 10 iterations"
        );
        // In this specific case nothing changes, therefore the smoothened mesh
        // should be equal to the original mesh.
        assert_eq!(
            relaxed_mesh_0.normals().len(),
            mesh.normals().len(),
            "Normals for 0 iterations"
        );
        assert_eq!(
            relaxed_mesh_1.normals().len(),
            mesh.vertices().len(),
            "Normals for 1 iteration"
        );
        assert_eq!(
            relaxed_mesh_10.normals().len(),
            mesh.vertices().len(),
            "Normals for 10 iterations"
        );
    }

    #[test]
    fn test_laplacian_smoothing_preserves_original_mesh_with_0_iterations() {
        let (faces, vertices) = triple_torus();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Sharp,
        );
        let v2v = topology::compute_vertex_to_vertex_topology(&mesh);

        let (relaxed_mesh, _, _) = laplacian_smoothing(&mesh, &v2v, 0, &[], false);
        assert_eq!(mesh, relaxed_mesh);
    }

    #[test]
    fn test_laplacian_smoothing_snapshot_triple_torus_1_iteration() {
        let (faces, vertices) = triple_torus();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Smooth,
        );
        let v2v = topology::compute_vertex_to_vertex_topology(&mesh);

        let (relaxed_mesh, _, _) = laplacian_smoothing(&mesh, &v2v, 1, &[], false);
        insta::assert_json_snapshot!(
            "triple_torus_after_1_iteration_of_laplacian_smoothing",
            &relaxed_mesh
        );
    }

    #[test]
    fn test_laplacian_smoothing_snapshot_triple_torus_2_iterations() {
        let (faces, vertices) = triple_torus();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Smooth,
        );
        let v2v = topology::compute_vertex_to_vertex_topology(&mesh);

        let (relaxed_mesh, _, _) = laplacian_smoothing(&mesh, &v2v, 2, &[], false);
        insta::assert_json_snapshot!(
            "triple_torus_after_2_iteration2_of_laplacian_smoothing",
            &relaxed_mesh
        );
    }

    #[test]
    fn test_laplacian_smoothing_snapshot_triple_torus_3_iterations() {
        let (faces, vertices) = triple_torus();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces,
            vertices,
            NormalStrategy::Smooth,
        );
        let v2v = topology::compute_vertex_to_vertex_topology(&mesh);

        let (relaxed_mesh, _, _) = laplacian_smoothing(&mesh, &v2v, 3, &[], false);
        insta::assert_json_snapshot!(
            "triple_torus_after_3_iterations_of_laplacian_smoothing",
            &relaxed_mesh
        );
    }

    #[test]
    fn test_laplacian_smoothing_with_anchors() {
        let (faces, vertices) = shape_for_smoothing_with_anchors();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Smooth,
        );

        let fixed_vertex_indices: Vec<u32> = vec![0, 1, 7, 8, 6];

        let (faces_correct, vertices_correct) = shape_for_smoothing_with_anchors_50_iterations();
        let test_mesh_correct = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces_correct.clone(),
            vertices_correct.clone(),
            NormalStrategy::Smooth,
        );

        let v2v = topology::compute_vertex_to_vertex_topology(&mesh);
        let (relaxed_mesh, _, _) =
            laplacian_smoothing(&mesh, &v2v, 50, &fixed_vertex_indices, false);

        let relaxed_mesh_faces = relaxed_mesh.faces();
        let test_mesh_faces = test_mesh_correct.faces();

        assert_eq!(relaxed_mesh_faces, test_mesh_faces);

        const TOLERANCE_SQUARED: f32 = 0.01 * 0.01;

        let relaxed_mesh_vertices = relaxed_mesh.vertices();
        let test_mesh_vertices = test_mesh_correct.vertices();

        for i in 0..test_mesh_vertices.len() {
            assert!(
                nalgebra::distance_squared(&test_mesh_vertices[i], &relaxed_mesh_vertices[i])
                    < TOLERANCE_SQUARED
            );
        }
    }

    #[test]
    fn test_laplacian_smoothing_with_anchors_find_border_vertices() {
        let (faces, vertices) = shape_for_smoothing_with_anchors();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );

        let oriented_edges: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = analysis::edge_sharing(&oriented_edges);
        let fixed_vertex_indices =
            Vec::from_iter(analysis::border_vertex_indices(&edge_sharing_map).into_iter());

        let (faces_correct, vertices_correct) = shape_for_smoothing_with_anchors_50_iterations();
        let test_mesh_correct = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces_correct.clone(),
            vertices_correct.clone(),
            NormalStrategy::Smooth,
        );
        let v2v = topology::compute_vertex_to_vertex_topology(&mesh);
        let (relaxed_mesh, _, _) =
            laplacian_smoothing(&mesh, &v2v, 50, &fixed_vertex_indices, false);

        // The faces should be made of the same vertex indices (and they should
        // remain in the original order) but the normals can be different due to
        // smoothing.
        let relaxed_mesh_faces_vertices: Vec<_> = relaxed_mesh
            .faces()
            .iter()
            .map(|face| match face {
                Face::Triangle(t) => t.vertices,
            })
            .collect();
        let test_mesh_faces_vertices: Vec<_> = test_mesh_correct
            .faces()
            .iter()
            .map(|face| match face {
                Face::Triangle(t) => t.vertices,
            })
            .collect();

        assert_eq!(relaxed_mesh_faces_vertices, test_mesh_faces_vertices);

        let relaxed_mesh_vertices = relaxed_mesh.vertices();
        let test_mesh_vertices = test_mesh_correct.vertices();

        for i in 0..test_mesh_vertices.len() {
            assert!(test_mesh_vertices[i].coords.relative_eq(
                &relaxed_mesh_vertices[i].coords,
                0.001,
                0.001,
            ));
        }
    }

    #[test]
    fn test_laplacian_smoothing_with_anchors_stop_when_stable_find_border_vertices() {
        let (faces, vertices) = shape_for_smoothing_with_anchors();
        let mesh = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces.clone(),
            vertices.clone(),
            NormalStrategy::Sharp,
        );

        let oriented_edges: Vec<OrientedEdge> = mesh.oriented_edges_iter().collect();
        let edge_sharing_map = analysis::edge_sharing(&oriented_edges);
        let fixed_vertex_indices =
            Vec::from_iter(analysis::border_vertex_indices(&edge_sharing_map).into_iter());

        let (faces_correct, vertices_correct) = shape_for_smoothing_with_anchors_50_iterations();
        let test_mesh_correct = Mesh::from_triangle_faces_with_vertices_and_computed_normals(
            faces_correct.clone(),
            vertices_correct.clone(),
            NormalStrategy::Sharp,
        );

        let v2v = topology::compute_vertex_to_vertex_topology(&mesh);
        let (relaxed_mesh, _, _) =
            laplacian_smoothing(&mesh, &v2v, 255, &fixed_vertex_indices, true);

        // The faces should be made of the same vertex indices (and they should
        // remain in the original order) but the normals can be different due to
        // smoothing.
        let relaxed_mesh_faces_vertices: Vec<_> = relaxed_mesh
            .faces()
            .iter()
            .map(|face| match face {
                Face::Triangle(t) => t.vertices,
            })
            .collect();
        let test_mesh_faces_vertices: Vec<_> = test_mesh_correct
            .faces()
            .iter()
            .map(|face| match face {
                Face::Triangle(t) => t.vertices,
            })
            .collect();

        assert_eq!(relaxed_mesh_faces_vertices, test_mesh_faces_vertices);

        let relaxed_mesh_vertices = relaxed_mesh.vertices();
        let test_mesh_vertices = test_mesh_correct.vertices();

        for i in 0..test_mesh_vertices.len() {
            assert!(test_mesh_vertices[i].coords.relative_eq(
                &relaxed_mesh_vertices[i].coords,
                0.001,
                0.001,
            ));
        }
    }

    #[test]
    fn test_loop_subdivision_snapshot_uv_sphere() {
        let mesh = primitive::create_uv_sphere(
            Point3::origin(),
            Rotation3::identity(),
            Vector3::new(2.0, 2.0, 2.0),
            2,
            3,
        );
        let v2v = topology::compute_vertex_to_vertex_topology(&mesh);
        let v2f = topology::compute_vertex_to_face_topology(&mesh);
        let f2f = topology::compute_face_to_face_topology(&mesh, &v2f);

        let subdivided_mesh = loop_subdivision(&mesh, &v2v, &f2f)
            .expect("The mesh doesn't meet the loop subdivision prerequisites");

        insta::assert_json_snapshot!(
            "uv_sphere_2_3_after_1_iteration_of_loop_subdivision",
            &subdivided_mesh
        );
    }

    #[test]
    fn test_loop_subdivision_snapshot_box_sharp() {
        let mesh = primitive::create_box(
            Point3::origin(),
            Rotation3::identity(),
            Vector3::new(1.0, 1.0, 1.0),
        );
        let v2v = topology::compute_vertex_to_vertex_topology(&mesh);
        let v2f = topology::compute_vertex_to_face_topology(&mesh);
        let f2f = topology::compute_face_to_face_topology(&mesh, &v2f);

        let subdivided_mesh = loop_subdivision(&mesh, &v2v, &f2f)
            .expect("The mesh doesn't meet the loop subdivision prerequisites");

        insta::assert_json_snapshot!(
            "box_sharp_after_1_iteration_of_loop_subdivision",
            &subdivided_mesh
        );
    }
}
