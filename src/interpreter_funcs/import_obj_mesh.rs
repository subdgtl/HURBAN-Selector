use std::error;
use std::fmt;
use std::sync::Arc;

use nalgebra::{Matrix4, Point3, Vector3};

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
                refinement: ParamRefinement::String(StringParamRefinement {
                    default_value: "",
                    file_path: true,
                    file_ext_filter: Some((&["*.obj", "*.OBJ"], "Wavefront (.obj)")),
                }),
                optional: false,
            },
            ParamInfo {
                name: "Move to origin",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Snap to ground",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
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
        values: &[Value],
        _log: &mut dyn FnMut(LogMessage),
    ) -> Result<Value, FuncError> {
        let path = values[0].unwrap_string();
        let move_to_origin = values[1].unwrap_boolean();
        let snap_to_ground = values[2].unwrap_boolean();

        let result = self.importer.import_obj(path);
        match result {
            Ok(models) => {
                if models.is_empty() {
                    Err(FuncError::new(FuncImportObjMeshError::Empty))
                } else {
                    let meshes: Vec<_> = if move_to_origin || snap_to_ground {
                        let meshes_iter = models.into_iter().map(|model| model.mesh);

                        let bboxes = meshes_iter.clone().map(|mesh| mesh.bounding_box());
                        let union_box = BoundingBox::union(bboxes).expect("No valid meshes");

                        let translation_vector = if move_to_origin {
                            Point3::origin() - union_box.center()
                        } else {
                            Vector3::zeros()
                        } + if snap_to_ground {
                            Vector3::new(0.0, 0.0, union_box.diagonal().z / 2.0)
                        } else {
                            Vector3::zeros()
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
                    Ok(Value::MeshArray(Arc::new(value)))
                }
            }
            Err(err) => Err(FuncError::new(FuncImportObjMeshError::Importer(err))),
        }
    }
}
