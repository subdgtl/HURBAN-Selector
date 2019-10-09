use std::collections::HashMap;

use crate::geometry::Geometry;
use crate::interpreter::ast;
use crate::interpreter::VarIdent;
use crate::interpreter_funcs;
use crate::interpreter_server::{InterpreterRequest, InterpreterResponse, InterpreterServer};
use crate::renderer::GpuGeometryId;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OpStatus {
    Ready,
    Running,
    Finished,
    #[allow(unused)]
    Error,
}

#[derive(Debug, Clone)]
pub enum OpParamUiRepr {
    IntInput,
    IntSlider,
    FloatInput,
    #[allow(unused)]
    FloatSlider,
    #[allow(unused)]
    Checkbox,
    #[allow(unused)]
    Radio,
    Dropdown(Vec<(u64, String)>),
}

#[derive(Debug, Clone)]
pub struct OpUiParam {
    pub name: String,
    pub repr: OpParamUiRepr,
    pub value: ast::LitExpr,
}

#[derive(Debug, Clone)]
pub struct Op {
    pub name: String,
    pub params: Vec<OpUiParam>,
    pub op: ast::FuncIdent,
}

#[derive(Debug)]
pub struct SelectedOp {
    pub op: Op,
    pub status: OpStatus,
}

#[derive(Debug)]
pub struct GeometryMetadata {
    name: String,
    pub geometry: Geometry,
    var_ident: u64,
    geometry_id: Option<GpuGeometryId>,
    used: bool,
}

/// Manages operations and geometries they produce.
pub struct OperationManager {
    interpreter_server: InterpreterServer,
    selected_ops: Vec<SelectedOp>,

    /// Geometries produced by operations and their metadata. Order matters
    /// for `geometry_stack` to be able to properly work with geometries.
    geometry_metadata: Vec<GeometryMetadata>,

    /// Since operations can produce multiple geometries, this stack tracks
    /// number of geometries returned from each operation. This is useful
    /// for removing operations and tracking what geometries should be
    /// available for which operations in the pipeline.
    geometry_stack: Vec<usize>,
}

impl OperationManager {
    pub fn new() -> Self {
        Self {
            selected_ops: Vec::new(),
            interpreter_server: InterpreterServer::new(),
            geometry_metadata: Vec::new(),
            geometry_stack: Vec::new(),
        }
    }

    /// Adds operation and submits its interpreter statement.
    ///
    /// Submitted statement is non-blocking, but it is processed in intepreter
    /// server thread.
    pub fn add_operation(&mut self, operation: Op) {
        let var_ident = ast::VarIdent(self.selected_ops.len() as u64);

        self.selected_ops.push(SelectedOp {
            op: operation.clone(),
            status: OpStatus::Ready,
        });

        let mut args: Vec<ast::Expr> = vec![];

        for param in &operation.params {
            let expr = match param.repr {
                OpParamUiRepr::Dropdown(_) => match param.value {
                    ast::LitExpr::Uint(uint) => {
                        ast::Expr::Var(ast::VarExpr::new(VarIdent(u64::from(uint))))
                    }
                    _ => ast::Expr::Var(ast::VarExpr::new(VarIdent(0 as u64))),
                },
                _ => ast::Expr::Lit(param.value.clone()),
            };
            args.push(expr);
        }

        self.interpreter_server
            .submit_request(InterpreterRequest::PushProgStmt(ast::Stmt::VarDecl(
                ast::VarDeclStmt::new(var_ident, ast::CallExpr::new(operation.op, args)),
            )));
    }

    /// Removes last operation, cleans up any of its data and returns geometry
    /// IDs to be removed from renderer.
    ///
    /// Removed statement is non-blocking, but it is processed in intepreter
    /// server thread.
    pub fn remove_last_operation(&mut self) -> Option<Vec<GpuGeometryId>> {
        if self.selected_ops.is_empty() {
            return None;
        }

        self.interpreter_server
            .submit_request(InterpreterRequest::PopProgStmt);
        let removed_op = self
            .selected_ops
            .pop()
            .expect("Failed to pop selected operation");

        // Having an operation doesn't mean it also produced any geometries.
        // Those are cleaned up only if this operation finished properly.
        if removed_op.status == OpStatus::Finished {
            if let Some(geometries_len) = self.geometry_stack.pop() {
                let first_index = self.geometry_metadata.len() - geometries_len;
                let geometry_ids = self
                    .geometry_metadata
                    .drain(first_index..)
                    .filter_map(|geometry_metadata| geometry_metadata.geometry_id)
                    .collect();

                return Some(geometry_ids);
            }
        }

        None
    }

    /// Submits program to interpreter and sets status of last operation
    /// to "running".
    pub fn submit_program(&mut self) {
        let last_op = self
            .selected_ops
            .last_mut()
            .expect("Failed to get last selected operation");
        last_op.status = OpStatus::Running;

        self.interpreter_server
            .submit_request(InterpreterRequest::Interpret);
    }

    /// Polls for respone from interpreter server and runs the handler function
    /// for each unused geometry in case a result is returned.
    ///
    /// Since unused geometries are the only ones to be shown in scene, the
    /// handler function runs only for them since the geometry ID needs to be
    /// assigned.
    pub fn poll_interpreter_response<F>(&mut self, mut geometry_handler: F)
    where
        F: FnMut(&mut GeometryMetadata) -> GpuGeometryId,
    {
        if let Ok((request_id, response)) = self.interpreter_server.poll_response() {
            match response {
                InterpreterResponse::Completed => {
                    log::info!("Interpreter completed request {:?}", request_id,);
                }
                InterpreterResponse::CompletedWithResult(result) => {
                    log::info!("Interpreter completed request {:?} with result", request_id,);

                    let value_set =
                        result.expect("Failed to get value set from interpreter response");
                    let op_name = self
                        .selected_ops
                        .last()
                        .expect("Failed to get last selected operation")
                        .op
                        .name
                        .clone();

                    for (_, value) in &value_set.used_values {
                        self.add_unused_geometries(&op_name, &[value.unwrap_geometry().clone()]);
                    }

                    for (i, (var_ident, value)) in value_set.unused_values.iter().enumerate() {
                        let mut geometry_metadata = GeometryMetadata {
                            name: format!("Geometry #{} from {}", i + 1, &op_name),
                            geometry: value.unwrap_geometry().clone(),
                            var_ident: var_ident.0,
                            geometry_id: None,
                            used: false,
                        };

                        let geometry_id = geometry_handler(&mut geometry_metadata);

                        geometry_metadata.geometry_id = Some(geometry_id);

                        self.geometry_metadata.push(geometry_metadata);
                        self.geometry_stack.push(1);
                    }

                    self.selected_ops
                        .last_mut()
                        .expect("Failed to mutate last operation")
                        .status = OpStatus::Finished;
                }
            }
        }
    }

    /// Populates geometry dropdowns with geometries produced in previous
    /// operations.
    pub fn prepare_ops(&mut self) {
        for (i, selected_op) in self.selected_ops.iter_mut().enumerate() {
            if i == 0 {
                continue;
            }

            let num_available_geos: usize = self.geometry_stack[0..i].iter().sum();

            if num_available_geos > 0 {
                let available_geo_metadata =
                    &self.geometry_metadata[0..num_available_geos as usize];

                for param in &mut selected_op.op.params {
                    if let OpParamUiRepr::Dropdown(ref mut choices) = param.repr {
                        choices.clear();
                        choices.extend(
                            available_geo_metadata
                                .iter()
                                .enumerate()
                                .map(|(i, geo_metadata)| (i as u64, geo_metadata.name.clone())),
                        );
                    }
                }
            }
        }
    }

    pub fn last_operation_successful(&self) -> bool {
        if self.selected_ops.is_empty() {
            true
        } else {
            self.selected_ops
                .last()
                .expect("Failed to get last selected operation")
                .status
                == OpStatus::Finished
        }
    }

    pub fn selected_ops_iter_mut(&mut self) -> std::slice::IterMut<SelectedOp> {
        self.selected_ops.iter_mut()
    }

    pub fn runnable(&self) -> bool {
        !self.selected_ops.is_empty()
    }

    fn add_unused_geometries(&mut self, op_name: &str, geometries: &[Geometry]) {
        let geometry_metadata =
            geometries
                .iter()
                .enumerate()
                .map(|(i, geometry)| GeometryMetadata {
                    name: format!("Geometry #{} from {}", i + 1, op_name),
                    geometry: geometry.clone(),
                    var_ident: i as u64,
                    used: false,
                    geometry_id: None,
                });

        self.geometry_metadata.extend(geometry_metadata);
        self.geometry_stack.push(geometries.len());
    }
}

pub fn operations_ui_definitions() -> HashMap<String, Op> {
    let mut ops = HashMap::new();

    ops.insert(
        "Create UV Sphere".to_string(),
        Op {
            name: "Create UV Sphere".to_string(),
            op: interpreter_funcs::FUNC_ID_CREATE_UV_SPHERE,
            params: vec![
                OpUiParam {
                    name: "Scale".to_string(),
                    repr: OpParamUiRepr::FloatInput,
                    value: ast::LitExpr::Float(0.0),
                },
                OpUiParam {
                    name: "Parallels".to_string(),
                    repr: OpParamUiRepr::IntInput,
                    value: ast::LitExpr::Uint(2),
                },
                OpUiParam {
                    name: "Meridians".to_string(),
                    repr: OpParamUiRepr::IntInput,
                    value: ast::LitExpr::Uint(3),
                },
            ],
        },
    );

    ops.insert(
        "Shrink Wrap".to_string(),
        Op {
            name: "Shrink Wrap".to_string(),
            op: interpreter_funcs::FUNC_ID_SHRINK_WRAP,
            params: vec![
                OpUiParam {
                    name: "Geometry".to_string(),
                    repr: OpParamUiRepr::Dropdown(vec![]),
                    value: ast::LitExpr::Uint(0),
                },
                OpUiParam {
                    name: "Sphere density".to_string(),
                    repr: OpParamUiRepr::IntSlider,
                    value: ast::LitExpr::Uint(3),
                },
            ],
        },
    );

    ops
}
