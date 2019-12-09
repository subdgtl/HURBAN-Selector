use nalgebra::{Point3, Vector3};

use crate::convert::{cast_u32, cast_usize};

use super::{Mesh, NormalStrategy, TriangleFace};

pub fn create_mesh_plane(position: [f32; 3], scale: f32) -> Mesh {
    #[rustfmt::skip]
    let vertex_positions = vec![
        v(-1.0, -1.0,  0.0, position, scale),
        v( 1.0, -1.0,  0.0, position, scale),
        v( 1.0,  1.0,  0.0, position, scale),
        v(-1.0,  1.0,  0.0, position, scale),
    ];

    #[rustfmt::skip]
    let vertex_normals = vec![
        Vector3::new( 0.0,  0.0,  1.0),
    ];

    let faces = vec![
        TriangleFace::new_separate(0, 1, 2, 0, 0, 0),
        TriangleFace::new_separate(2, 3, 0, 0, 0, 0),
    ];

    Mesh::from_triangle_faces_with_vertices_and_normals(faces, vertex_positions, vertex_normals)
}

#[allow(dead_code)]
pub fn create_cube_smooth(position: [f32; 3], scale: f32) -> Mesh {
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

    // FIXME: make const once sqrt is const stabilized
    let frac_1_sqrt_3 = 1.0f32 / 3.0f32.sqrt();

    #[rustfmt::skip]
    let vertex_normals = vec![
        // back
        Vector3::new(-frac_1_sqrt_3,  frac_1_sqrt_3, -frac_1_sqrt_3),
        Vector3::new(-frac_1_sqrt_3,  frac_1_sqrt_3,  frac_1_sqrt_3),
        Vector3::new( frac_1_sqrt_3,  frac_1_sqrt_3,  frac_1_sqrt_3),
        Vector3::new( frac_1_sqrt_3,  frac_1_sqrt_3, -frac_1_sqrt_3),
        // front
        Vector3::new(-frac_1_sqrt_3, -frac_1_sqrt_3, -frac_1_sqrt_3),
        Vector3::new( frac_1_sqrt_3, -frac_1_sqrt_3, -frac_1_sqrt_3),
        Vector3::new( frac_1_sqrt_3, -frac_1_sqrt_3,  frac_1_sqrt_3),
        Vector3::new(-frac_1_sqrt_3, -frac_1_sqrt_3,  frac_1_sqrt_3),
    ];

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

    Mesh::from_triangle_faces_with_vertices_and_normals(faces, vertex_positions, vertex_normals)
}

#[allow(dead_code)]
pub fn create_cube_sharp(position: [f32; 3], scale: f32) -> Mesh {
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
        Vector3::new( 0.0,  1.0,  0.0),
        // front
        Vector3::new( 0.0, -1.0,  0.0),
        // top
        Vector3::new( 0.0,  0.0,  1.0),
        // bottom
        Vector3::new( 0.0,  0.0, -1.0),
        // right
        Vector3::new( 1.0,  0.0,  0.0),
        // left
        Vector3::new(-1.0,  0.0,  0.0),
    ];

    let faces = vec![
        // back
        TriangleFace::new_separate(0, 1, 2, 0, 0, 0),
        TriangleFace::new_separate(2, 3, 0, 0, 0, 0),
        // front
        TriangleFace::new_separate(4, 5, 6, 1, 1, 1),
        TriangleFace::new_separate(6, 7, 4, 1, 1, 1),
        // top
        TriangleFace::new_separate(7, 6, 1, 2, 2, 2),
        TriangleFace::new_separate(2, 1, 6, 2, 2, 2),
        // bottom
        TriangleFace::new_separate(5, 0, 3, 3, 3, 3),
        TriangleFace::new_separate(0, 5, 4, 3, 3, 3),
        // right
        TriangleFace::new_separate(6, 3, 2, 4, 4, 4),
        TriangleFace::new_separate(3, 6, 5, 4, 4, 4),
        // left
        TriangleFace::new_separate(4, 7, 0, 5, 5, 5),
        TriangleFace::new_separate(1, 0, 7, 5, 5, 5),
    ];

    Mesh::from_triangle_faces_with_vertices_and_normals(faces, vertex_positions, vertex_normals)
}

#[allow(dead_code)]
pub fn create_box_sharp(center: Point3<f32>, scale: [f32; 3]) -> Mesh {
    #[rustfmt::skip]

    let vertex_positions = vec![
        // back
        Point3::new(-0.5 * scale[0] + center.x,  0.5 * scale[1] + center.y, -0.5 * scale[2] + center.z),
        Point3::new(-0.5 * scale[0] + center.x,  0.5 * scale[1] + center.y,  0.5 * scale[2] + center.z),
        Point3::new( 0.5 * scale[0] + center.x,  0.5 * scale[1] + center.y,  0.5 * scale[2] + center.z),
        Point3::new( 0.5 * scale[0] + center.x,  0.5 * scale[1] + center.y, -0.5 * scale[2] + center.z),
        // front
        Point3::new(-0.5 * scale[0] + center.x, -0.5 * scale[1] + center.y, -0.5 * scale[2] + center.z),
        Point3::new( 0.5 * scale[0] + center.x, -0.5 * scale[1] + center.y, -0.5 * scale[2] + center.z),
        Point3::new( 0.5 * scale[0] + center.x, -0.5 * scale[1] + center.y,  0.5 * scale[2] + center.z),
        Point3::new(-0.5 * scale[0] + center.x, -0.5 * scale[1] + center.y,  0.5 * scale[2] + center.z),
    ];

    #[rustfmt::skip]
    let vertex_normals = vec![
        // back
        Vector3::new( 0.0,  1.0,  0.0),
        // front
        Vector3::new( 0.0, -1.0,  0.0),
        // top
        Vector3::new( 0.0,  0.0,  1.0),
        // bottom
        Vector3::new( 0.0,  0.0, -1.0),
        // right
        Vector3::new( 1.0,  0.0,  0.0),
        // left
        Vector3::new(-1.0,  0.0,  0.0),
    ];

    let faces = vec![
        // back
        TriangleFace::new_separate(0, 1, 2, 0, 0, 0),
        TriangleFace::new_separate(2, 3, 0, 0, 0, 0),
        // front
        TriangleFace::new_separate(4, 5, 6, 1, 1, 1),
        TriangleFace::new_separate(6, 7, 4, 1, 1, 1),
        // top
        TriangleFace::new_separate(7, 6, 1, 2, 2, 2),
        TriangleFace::new_separate(2, 1, 6, 2, 2, 2),
        // bottom
        TriangleFace::new_separate(5, 0, 3, 3, 3, 3),
        TriangleFace::new_separate(0, 5, 4, 3, 3, 3),
        // right
        TriangleFace::new_separate(6, 3, 2, 4, 4, 4),
        TriangleFace::new_separate(3, 6, 5, 4, 4, 4),
        // left
        TriangleFace::new_separate(4, 7, 0, 5, 5, 5),
        TriangleFace::new_separate(1, 0, 7, 5, 5, 5),
    ];

    Mesh::from_triangle_faces_with_vertices_and_normals(faces, vertex_positions, vertex_normals)
}

/// Create UV Sphere primitive at `position` with `scale`,
/// `n_parallels` and `n_meridians`.
///
/// # Panics
/// Panics if number of parallels is less than 2 or number of
/// meridians is less than 3.
pub fn create_uv_sphere(
    position: [f32; 3],
    scale: f32,
    n_parallels: u32,
    n_meridians: u32,
) -> Mesh {
    assert!(n_parallels >= 2, "Need at least 2 parallels");
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
            // Produce 2 CCW wound triangles: (p1, p2, p3) and (p3, p4, p1)

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

    Mesh::from_triangle_faces_with_vertices_and_computed_normals(
        faces,
        vertex_positions,
        NormalStrategy::Sharp,
    )
}

fn v(x: f32, y: f32, z: f32, translation: [f32; 3], scale: f32) -> Point3<f32> {
    Point3::new(
        scale * x + translation[0],
        scale * y + translation[1],
        scale * z + translation[2],
    )
}
