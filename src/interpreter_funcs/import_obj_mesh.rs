use std::error;
use std::fmt;
use std::sync::Arc;

use crate::importer::{Importer, ImporterError, ObjCache};
use crate::interpreter::{
    Func, FuncError, FuncFlags, FuncInfo, ParamInfo, ParamRefinement, Ty, Value,
};

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
            name: "Import OBJ",
            return_value_name: "Imported Mesh",
        }
    }

    fn flags(&self) -> FuncFlags {
        FuncFlags::empty()
    }

    fn param_info(&self) -> &[ParamInfo] {
        &[ParamInfo {
            name: "Path",
            refinement: ParamRefinement::String,
            optional: false,
        }]
    }

    fn return_ty(&self) -> Ty {
        Ty::Geometry
    }

    fn call(&mut self, values: &[Value]) -> Result<Value, FuncError> {
        let path = values[0].unwrap_string();

        let result = self.importer.import_obj(path);
        match result {
            Ok(models) => {
                // FIXME: @Correctness Join all meshes into one once
                // we have join implemented for more than just 2
                // meshes

                let first_model = models.into_iter().next();
                if let Some(first_model) = first_model {
                    Ok(Value::Geometry(Arc::new(first_model.geometry)))
                } else {
                    Err(FuncError::new(FuncImportObjMeshError::Empty))
                }
            }
            Err(err) => Err(FuncError::new(FuncImportObjMeshError::Importer(err))),
        }
    }
}
