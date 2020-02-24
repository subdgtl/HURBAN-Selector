use std::error;
use std::fmt;
use std::sync::Arc;

use nalgebra::{Matrix4, Point3, Vector3};

use crate::analytics;
use crate::importer::{Importer, ImporterError, ObjCache};
use crate::interpreter::{
    BooleanParamRefinement, Func, FuncError, FuncFlags, FuncInfo, LogMessage, ParamInfo,
    ParamRefinement, StringParamRefinement, Ty, Value,
};
use crate::mesh::{tools, Mesh};

#[derive(Debug, PartialEq)]
pub enum FuncImportObjJoinError {
    Empty,
    Importer(ImporterError),
}

impl fmt::Display for FuncImportObjJoinError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "No mesh geometry contained in OBJ"),
            Self::Importer(importer_error) => f.write_str(&importer_error.to_string()),
        }
    }
}

impl error::Error for FuncImportObjJoinError {}

pub struct FuncImportObjJoin<C: ObjCache> {
    importer: Importer<C>,
}

impl<C: ObjCache> FuncImportObjJoin<C> {
    pub fn new(importer: Importer<C>) -> Self {
        Self { importer }
    }
}

impl<C: ObjCache> Func for FuncImportObjJoin<C> {
    fn info(&self) -> &FuncInfo {
        &FuncInfo {
            name: "Import OBJ",
            description:
                "IMPORT OBJ FILE AND JOIN ALL ITS CONTAINED GEOMETRIES INTO A SINGLE MESH\n\
                 \n\
                 Loads the content of an OBJ file and joins all ats contained mesh geometries \
                 into a single mesh. \
                 The meshes will not be welded.\n\
                 \n\
                 The resulting mesh geometry will be named 'Imported Mesh'.",
            return_value_name: "Imported Mesh",
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
                description: "Moves the imported mesh geometry so that its local \
                              center matches the world origin.",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Snap to ground",
                description: "Moves the imported mesh geometry so that its bottommost \
                              vertexes vertical coordinate is zero (sits on the ground).",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Bounding Box Analysis",
                description: "Reports basic and quick analytic information on the created mesh.",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: true,
                }),
                optional: false,
            },
            ParamInfo {
                name: "Detailed Mesh Analysis",
                description: "Reports detailed analytic information on the created mesh.\n\
                              The analysis may be slow, therefore it is by default off.",
                refinement: ParamRefinement::Boolean(BooleanParamRefinement {
                    default_value: false,
                }),
                optional: false,
            },
        ]
    }

    fn return_ty(&self) -> Ty {
        Ty::Mesh
    }

    fn call(
        &mut self,
        args: &[Value],
        log: &mut dyn FnMut(LogMessage),
    ) -> Result<Value, FuncError> {
        let path = args[0].unwrap_string();
        let move_to_origin = args[1].unwrap_boolean();
        let snap_to_ground = args[2].unwrap_boolean();
        let analyze_bbox = args[3].unwrap_boolean();
        let analyze_mesh = args[4].unwrap_boolean();

        let result = self.importer.import_obj(path);
        match result {
            Ok(models) => {
                if models.is_empty() {
                    let error = FuncError::new(FuncImportObjJoinError::Empty);
                    log(LogMessage::error(format!("Error: {}", error)));
                    Err(error)
                } else {
                    let imported_meshes: Vec<_> =
                        models.into_iter().map(|model| model.mesh).collect();
                    let single_mesh = tools::join_multiple_meshes(imported_meshes.iter());

                    let value = if move_to_origin || snap_to_ground {
                        let bbox = single_mesh.bounding_box();

                        let translation_vector = match (move_to_origin, snap_to_ground) {
                            (true, true) => {
                                Point3::origin() - bbox.center()
                                    + Vector3::new(0.0, 0.0, bbox.diagonal().z / 2.0)
                            }
                            (true, false) => Point3::origin() - bbox.center(),
                            (false, true) => {
                                Vector3::new(0.0, 0.0, bbox.diagonal().z / 2.0 - bbox.center().z)
                            }
                            _ => Vector3::zeros(),
                        };

                        let translation = Matrix4::new_translation(&translation_vector);

                        let vertices_iter = single_mesh
                            .vertices()
                            .iter()
                            .map(|v| translation.transform_point(v));

                        Mesh::from_faces_with_vertices_and_normals(
                            single_mesh.faces().iter().copied(),
                            vertices_iter,
                            single_mesh.normals().iter().copied(),
                        )
                    } else {
                        single_mesh
                    };

                    if analyze_bbox {
                        analytics::report_bounding_box_analysis(&value, log);
                    }
                    if analyze_mesh {
                        analytics::report_mesh_analysis(&value, log);
                    }

                    Ok(Value::Mesh(Arc::new(value)))
                }
            }
            Err(err) => {
                let error = FuncError::new(FuncImportObjJoinError::Importer(err));
                log(LogMessage::error(format!("Error: {}", error)));
                Err(error)
            }
        }
    }
}
