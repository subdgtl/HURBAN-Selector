//! The `primitives` module contains parametric primitives for testing
//! renderer development. Each primitive contains both vertex
//! positions and normals.
//!
//! The primitives live in a right-handed coordinate space with the XY
//! plane being the ground and and Z axis growing up.

use crate::viewport_renderer::{Geometry, Index};

pub fn plane(position: [f32; 3], scale: f32) -> Geometry {
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
        [  0.0,  0.0,  1.0 ],
        [  0.0,  0.0,  1.0 ],
        [  0.0,  0.0,  1.0 ],
        [  0.0,  0.0,  1.0 ],
        [  0.0,  0.0,  1.0 ],
        [  0.0,  0.0,  1.0 ],
    ];

    Geometry::from_positions_and_normals(vertex_positions, vertex_normals)
}

pub fn cube(position: [f32; 3], scale: f32) -> Geometry {
    #[rustfmt::skip]
    let vertex_positions = vec![
        // front
        v(-1.0, -1.0, -1.0, position, scale),
        v( 1.0, -1.0, -1.0, position, scale),
        v( 1.0, -1.0,  1.0, position, scale),
        v(-1.0, -1.0,  1.0, position, scale),
        // back
        v(-1.0,  1.0, -1.0, position, scale),
        v( 1.0,  1.0, -1.0, position, scale),
        v( 1.0,  1.0,  1.0, position, scale),
        v(-1.0,  1.0,  1.0, position, scale),
    ];

    // FIXME: make const once float arithmetic is stabilized in const fns
    // let sqrt_3 = 3.0f32.sqrt();
    let frac_1_sqrt_3 = 1.0 / 3.0_f32.sqrt();

    #[rustfmt::skip]
    let vertex_normals = vec![
        // front
        [ -frac_1_sqrt_3, -frac_1_sqrt_3, -frac_1_sqrt_3 ],
        [  frac_1_sqrt_3, -frac_1_sqrt_3, -frac_1_sqrt_3 ],
        [  frac_1_sqrt_3, -frac_1_sqrt_3,  frac_1_sqrt_3 ],
        [ -frac_1_sqrt_3, -frac_1_sqrt_3,  frac_1_sqrt_3 ],
        // back
        [ -frac_1_sqrt_3,  frac_1_sqrt_3, -frac_1_sqrt_3 ],
        [  frac_1_sqrt_3,  frac_1_sqrt_3, -frac_1_sqrt_3 ],
        [  frac_1_sqrt_3,  frac_1_sqrt_3,  frac_1_sqrt_3 ],
        [ -frac_1_sqrt_3,  frac_1_sqrt_3,  frac_1_sqrt_3 ],

    ];

    #[rustfmt::skip]
    let indices: Vec<Index> = vec![
        // front
        0, 1, 2, 2, 3, 0,
        // right
        1, 5, 6, 6, 2, 1,
        // back
        7, 6, 5, 5, 4, 7,
        // left
        4, 0, 3, 3, 7, 4,
        // bottom
        4, 5, 1, 1, 0, 4,
        // top
        3, 2, 6, 6, 7, 3,
    ];

    Geometry::from_positions_and_normals_indexed(indices, vertex_positions, vertex_normals)
}

pub fn uv_cube(position: [f32; 3], scale: f32) -> Geometry {
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
        [  0.0,  1.0,  0.0 ],
        [  0.0,  1.0,  0.0 ],
        [  0.0,  1.0,  0.0 ],
        [  0.0,  1.0,  0.0 ],
        // front
        [  0.0, -1.0,  0.0 ],
        [  0.0, -1.0,  0.0 ],
        [  0.0, -1.0,  0.0 ],
        [  0.0, -1.0,  0.0 ],
        // top
        [  0.0,  0.0,  1.0 ],
        [  0.0,  0.0,  1.0 ],
        [  0.0,  0.0,  1.0 ],
        [  0.0,  0.0,  1.0 ],
        // bottom
        [  0.0,  0.0, -1.0 ],
        [  0.0,  0.0, -1.0 ],
        [  0.0,  0.0, -1.0 ],
        [  0.0,  0.0, -1.0 ],
        // right
        [  1.0,  0.0,  0.0 ],
        [  1.0,  0.0,  0.0 ],
        [  1.0,  0.0,  0.0 ],
        [  1.0,  0.0,  0.0 ],
        // left
        [ -1.0,  0.0,  0.0 ],
        [ -1.0,  0.0,  0.0 ],
        [ -1.0,  0.0,  0.0 ],
        [ -1.0,  0.0,  0.0 ],
    ];

    #[rustfmt::skip]
    let indices: Vec<Index> = vec![
        // back
        0, 1, 2, 2, 3, 0,
        // front
        4, 5, 6, 6, 7, 4,
        // top
        8, 9, 10, 10, 11, 8,
        // bottom
        12, 13, 14, 14, 15, 12,
        // right
        16, 17, 18, 18, 19, 16,
        // left
        20, 21, 22, 22, 23, 20,
    ];

    Geometry::from_positions_and_normals_indexed(indices, vertex_positions, vertex_normals)
}

fn v(x: f32, y: f32, z: f32, translation: [f32; 3], scale: f32) -> [f32; 3] {
    [
        scale * x + translation[0],
        scale * y + translation[1],
        scale * z + translation[2],
    ]
}
