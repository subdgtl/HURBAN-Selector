use std::error;
use std::fmt;
use std::sync::Arc;

use nalgebra::{Matrix4, Point3, Vector3};

use crate::analytics;
use crate::bounding_box::BoundingBox;
use crate::importer::{Importer, ImporterError, ObjCache};
use crate::interpreter::{
    BooleanParamRefinement, Func, FuncError, FuncFlags, FuncInfo, LogMessage, MeshArrayValue,
    ParamInfo, ParamRefinement, StringParamRefinement, Ty, Value,
};
use crate::mesh::Mesh;

#[derive(Debug, PartialEq)]
pub enum FuncImportObjMeshError {
    Empty,
    Importer(ImporterError),
}

impl fmt::Display for FuncImportObjMeshError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "No mesh geometry contained in OBJ"),
            Self::Importer(importer_error) => f.write_str(&importer_error.to_string()),
        }
    }
}

impl error::Error for FuncImportObjMeshError {}

pub struct FuncImportObjMesh<C: ObjCache> {
    importer: Importer<C>,
}

impl<C: ObjCache> FuncImportObjMesh<C> {
    pub fn new(importer: Importer<C>) -> Self {
        Self { importer }
    }
}

impl<C: ObjCache> Func for FuncImportObjMesh<C> {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Import OBJ as Group",
            description:
                "IMPORT OBJ FILE AS A MESH GROUP\n\
                 \n\
                 Loads the content of an OBJ file as a mesh group.\n\
                 \n\
                 Mesh group is displayed in the viewport as geometry but is \
                 a distinct data type. Only some operations can use mesh groups \
                 and most of them are intended to generate a proper mesh from the mesh group. \
                 To use the content of the group it is necessary to Extract specific \
                 mesh from group, Extract largest mesh from group or Join mesh group \
                 into a single mesh.\n\
                 \n\
                 The resulting mesh group will be named 'Imported Group'.",
            return_value_name: "Imported Group",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::empty()
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[
            ParamInfo {
                name: "Path",
                description: "Path to the OBJ file.",
                refinement: ParamRefinement::String(StringParamRefinement {
                    default_value: "",
                    file_path: true,
                    file_ext_filter: Some((&["*.obj", "*.OBJ"], "Wavefront (.obj)")),
                }),
                optional: false,
            },
            ParamInfo {
                name: "Move to origin",
                description: "Moves the imported mesh group so that its local \
                              center matches the world origin.",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Snap to ground",
                description: "Moves the imported mesh group so that its bottommost \
                              vertexes vertical coordinate is zero (sits on the ground).",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Group Analysis",
                description: "Reports detailed analytic information on the imported mesh group.",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: false,
                }),
                optional: false,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::MeshArray
    }

    fn call(
        &mut self,
        args: &[Value],
        log: &mut dyn FnMut(LogMessage),
    ) -> Result<Value, FuncError> {
        let path = args[0].unwrap_string();
        let move_to_origin = args[1].unwrap_boolean();
        let snap_to_ground = args[2].unwrap_boolean();
        let analyze = args[3].unwrap_boolean();

        let result = self.importer.import_obj(path);
        match result {
            Ok(models) => {
                if models.is_empty() {
                    let error = FuncError::new(FuncImportObjMeshError::Empty);
                    log(LogMessage::error(format!("Error: {}", error)));
                    Err(error)
                } else {
                    let meshes: Vec<_> = if move_to_origin || snap_to_ground {
                        let meshes_iter = models.into_iter().map(|model| model.mesh);

                        let bboxes = meshes_iter.clone().map(|mesh| mesh.bounding_box());
                        let union_box = BoundingBox::union(bboxes).expect("No valid meshes");

                        let translation_vector = match (move_to_origin, snap_to_ground) {
                            (true, true) => {
                                Point3::origin() - union_box.center()
                                    + Vector3::new(0.0, 0.0, union_box.diagonal().z / 2.0)
                            }
                            (true, false) => Point3::origin() - union_box.center(),
                            (false, true) => Vector3::new(
                                0.0,
                                0.0,
                                union_box.diagonal().z / 2.0 - union_box.center().z,
                            ),
                            _ => Vector3::zeros(),
                        };
                        let translation = Matrix4::new_translation(&translation_vector);

                        meshes_iter
                            .map(|mesh| {
                                let vertices_iter = mesh
                                    .vertices()
                                    .iter()
                                    .map(|v| translation.transform_point(v));

                                Arc::new(Mesh::from_faces_with_vertices_and_normals(
                                    mesh.faces().iter().copied(),
                                    vertices_iter,
                                    mesh.normals().iter().copied(),
                                ))
                            })
                            .collect()
                    } else {
                        models
                            .into_iter()
                            .map(|model| Arc::new(model.mesh))
                            .collect()
                    };

                    let value = MeshArrayValue::new(meshes);

                    if analyze {
                        analytics::report_group_analysis(&value, log);
                    }

                    Ok(Value::MeshArray(Arc::new(value)))
                }
            }
            Err(err) => {
                let error = FuncError::new(FuncImportObjMeshError::Importer(err));
                log(LogMessage::error(format!("Error: {}", error)));
                Err(error)
            }
        }
    }
}
