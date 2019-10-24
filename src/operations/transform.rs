use nalgebra::base::{Matrix4, Vector3};
use nalgebra::geometry::{Rotation, Translation};

use crate::geometry::Geometry;

pub struct TransformOptions {
    /// How much to translate the geometry.
    pub translate: Option<Vector3<f32>>,

    /// How much to rotate the geometry, in euler angle radians.
    pub rotate: Option<[f32; 3]>,

    /// How much to scale the geometry in various axis.
    pub scale: Option<Vector3<f32>>,
}

/// Transforms the geometry.
///
/// Applies the transformations in the order: scale, rotate, translate.
pub fn transform(geometry: &Geometry, options: TransformOptions) -> Geometry {
    let translate = options.translate.unwrap_or_else(Vector3::zeros);
    let scale = options.scale.unwrap_or_else(|| Vector3::new(1.0, 1.0, 1.0));
    let rotate = options.rotate.unwrap_or([0.0; 3]);

    let translation = Translation::from(translate);
    let rotation = Rotation::from_euler_angles(rotate[0], rotate[1], rotate[2]);
    let scaling = Matrix4::new_nonuniform_scaling(&scale);

    let t = Matrix4::from(translation) * Matrix4::from(rotation) * scaling;

    let vertices_iter = geometry.vertices().iter().map(|v| t.transform_point(v));
    let normals_iter = geometry.normals().iter().map(|n| t.transform_vector(n));

    Geometry::from_faces_with_vertices_and_normals(
        geometry.faces().iter().copied(),
        vertices_iter,
        normals_iter,
    )
}
