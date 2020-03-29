use std::borrow::Borrow;
use std::convert::TryFrom;
use std::io::{self, Write};

use crate::convert::cast_u32;
use crate::mesh::{Face, Mesh};

// FIXME: Mesh arrays are currently exported as objects (o). Export them as
// groups (g).

/// Write mesh models serialized in OBJ format to provided output writer.
///
/// Flushes `writer` at least once - after all data has been written. Formats
/// each floating point number `decimal_precision` digits.
pub fn export_obj<'a, I, N, W>(
    writer: &mut W,
    models: I,
    decimal_precision: u32,
) -> Result<(), io::Error>
where
    I: IntoIterator<Item = (N, &'a Mesh)>,
    N: Borrow<str>,
    W: Write,
{
    // usize to u32 conversion can fail on 16-bit systems, in which case we'll
    // use the max value of usize. This probably won't ever happen.
    let decimal_precision = usize::try_from(decimal_precision).unwrap_or(usize::max_value());

    // OBJ indices are global, meaning a face from object "o2" can reference a
    // vertex from "o1". We maintain proper namespacing between the exported
    // objects by tracking the index offsets for data we export (vertices and
    // normals). Offsets start with 1, because OBJ indices do.
    let mut vertex_index_offset = 1;
    let mut normal_index_offset = 1;

    writeln!(writer, "# Exported by H.U.R.B.A.N selector")?;
    writeln!(writer)?;

    for (name, mesh) in models {
        writeln!(writer, "o {}", name.borrow())?;
        writeln!(writer)?;

        for vertex in mesh.vertices() {
            writeln!(
                writer,
                "v {1:.0$} {2:.0$} {3:.0$}",
                decimal_precision, vertex.x, vertex.y, vertex.z,
            )?;
        }
        writeln!(writer)?;

        for normal in mesh.normals() {
            writeln!(
                writer,
                "vn {1:.0$} {2:.0$} {3:.0$}",
                decimal_precision, normal.x, normal.y, normal.z,
            )?;
        }
        writeln!(writer)?;

        for face in mesh.faces() {
            let Face::Triangle(triangle_face) = face;
            let vertices = triangle_face.vertices;
            let normals = triangle_face.normals;

            writeln!(
                writer,
                "f {1:.0$}//{4:.0$} {2:.0$}//{5:.0$} {3:.0$}//{6:.0$}",
                decimal_precision,
                vertices.0 + vertex_index_offset,
                vertices.1 + vertex_index_offset,
                vertices.2 + vertex_index_offset,
                normals.0 + normal_index_offset,
                normals.1 + normal_index_offset,
                normals.2 + normal_index_offset,
            )?;
        }
        writeln!(writer)?;

        vertex_index_offset += cast_u32(mesh.vertices().len());
        normal_index_offset += cast_u32(mesh.normals().len());
    }

    writer.flush()
}

#[cfg(test)]
mod tests {
    use std::iter;

    use nalgebra::{Point3, Vector3};

    use crate::mesh::TriangleFace;

    use super::*;

    #[test]
    fn test_export_obj_simple() {
        let name = "Our Test-model__";
        let mesh = Mesh::from_triangle_faces_with_vertices_and_normals(
            [TriangleFace::new(0, 1, 2, 0, 0, 0)].iter().copied(),
            [
                Point3::new(-0.3, -0.3, 0.0),
                Point3::new(0.3, -0.3, 0.0),
                Point3::new(0.0, 0.4, 0.0),
            ]
            .iter()
            .copied(),
            [Vector3::new(0.0, 0.0, 1.0)].iter().copied(),
        );

        let expected_output: &[u8] = b"\
            # Exported by H.U.R.B.A.N selector\n\
            \n\
            o Our Test-model__\n\
            \n\
            v -0.30000 -0.30000 0.00000\n\
            v 0.30000 -0.30000 0.00000\n\
            v 0.00000 0.40000 0.00000\n\
            \n\
            vn 0.00000 0.00000 1.00000\n\
            \n\
            f 1//1 2//1 3//1\n\
            \n";

        let mut output = Vec::new();
        export_obj(&mut output, iter::once((name, &mesh)), 5).unwrap();

        assert_eq!(output, Vec::from(expected_output));
    }

    #[test]
    fn test_export_obj_index_namespacing() {
        let name1 = "Our Test-model__";
        let mesh1 = Mesh::from_triangle_faces_with_vertices_and_normals(
            [TriangleFace::new(0, 1, 2, 0, 0, 0)].iter().copied(),
            [
                Point3::new(-0.3, -0.3, 0.0),
                Point3::new(0.3, -0.3, 0.0),
                Point3::new(0.0, 0.4, 0.0),
            ]
            .iter()
            .copied(),
            [Vector3::new(0.0, 0.0, 1.0)].iter().copied(),
        );

        let name2 = "Another model";
        let mesh2 = mesh1.clone();

        let expected_output: &[u8] = b"\
            # Exported by H.U.R.B.A.N selector\n\
            \n\
            o Our Test-model__\n\
            \n\
            v -0.30000 -0.30000 0.00000\n\
            v 0.30000 -0.30000 0.00000\n\
            v 0.00000 0.40000 0.00000\n\
            \n\
            vn 0.00000 0.00000 1.00000\n\
            \n\
            f 1//1 2//1 3//1\n\
            \n\
            o Another model\n\
            \n\
            v -0.30000 -0.30000 0.00000\n\
            v 0.30000 -0.30000 0.00000\n\
            v 0.00000 0.40000 0.00000\n\
            \n\
            vn 0.00000 0.00000 1.00000\n\
            \n\
            f 4//2 5//2 6//2\n\
            \n";

        let mut output = Vec::new();
        export_obj(
            &mut output,
            [(name1, &mesh1), (name2, &mesh2)].iter().copied(),
            5,
        )
        .unwrap();

        assert_eq!(output, Vec::from(expected_output));
    }
}
