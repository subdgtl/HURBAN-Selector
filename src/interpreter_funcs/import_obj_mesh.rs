use std::error;
use std::fmt;
use std::sync::Arc;

use crate::importer::{Importer, ImporterError, ObjCache};
use crate::interpreter::{
    Func, FuncError, FuncFlags, FuncInfo, GeometryArrayValue, ParamInfo, ParamRefinement, Ty, Value,
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
            return_value_name: "Imported Group",
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
        Ty::GeometryArray
    }

    fn call(&mut self, values: &[Value]) -> Result<Value, FuncError> {
        let path = values[0].unwrap_string();

        let result = self.importer.import_obj(path);
        match result {
            Ok(models) => {
                if models.is_empty() {
                    Err(FuncError::new(FuncImportObjMeshError::Empty))
                } else {
                    let geometries: Vec<_> = models
                        .into_iter()
                        .map(|model| Arc::new(model.geometry))
                        .collect();

                    let value = GeometryArrayValue::new(geometries);
                    Ok(Value::GeometryArray(Arc::new(value)))
                }
            }
            Err(err) => Err(FuncError::new(FuncImportObjMeshError::Importer(err))),
        }
    }
}
