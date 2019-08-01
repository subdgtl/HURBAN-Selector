use crate::viewport_renderer::{Index, Vertex};

pub fn plane(position: [f32; 3], scale: f32) -> Vec<Vertex> {
    vec![
        v(-1.0, -1.0, 0.0, position, scale),
        v(1.0, -1.0, 0.0, position, scale),
        v(1.0, 1.0, 0.0, position, scale),
        v(1.0, 1.0, 0.0, position, scale),
        v(-1.0, 1.0, 0.0, position, scale),
        v(-1.0, -1.0, 0.0, position, scale),
    ]
}

pub fn cube(position: [f32; 3], scale: f32) -> (Vec<Vertex>, Vec<Index>) {
    let vertex_data = vec![
        // front
        v(-1.0, -1.0, 1.0, position, scale),
        v(1.0, -1.0, 1.0, position, scale),
        v(1.0, 1.0, 1.0, position, scale),
        v(-1.0, 1.0, 1.0, position, scale),
        // back
        v(-1.0, -1.0, -1.0, position, scale),
        v(1.0, -1.0, -1.0, position, scale),
        v(1.0, 1.0, -1.0, position, scale),
        v(-1.0, 1.0, -1.0, position, scale),
    ];
    let index_data: Vec<Index> = vec![
        0, 1, 2, 2, 3, 0, // front
        1, 5, 6, 6, 2, 1, // right
        7, 6, 5, 5, 4, 7, // back
        4, 0, 3, 3, 7, 4, // left
        4, 5, 1, 1, 0, 4, // bottom
        3, 2, 6, 6, 7, 3, // top
    ];

    (vertex_data, index_data)
}

fn v(x: f32, y: f32, z: f32, offset: [f32; 3], scale: f32) -> Vertex {
    Vertex {
        position: [
            scale * x + offset[0],
            scale * y + offset[1],
            scale * z + offset[2],
        ],
    }
}
