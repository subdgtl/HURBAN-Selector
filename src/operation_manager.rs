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
    GeometryDropdown(Vec<(u64, String)>),
}

/// Value aware of its current and previous state.
///
/// This is useful for determining what changed and if operation containing
/// this value should be marked as dirty for purposes of interpreter program
/// invalidation.
#[derive(Debug, Clone)]
pub struct Value {
    previous_value: ast::LitExpr,
    current_value: ast::LitExpr,
}

impl Value {
    pub fn new(value: ast::LitExpr) -> Self {
        Self {
            current_value: value.clone(),
            previous_value: value.clone(),
        }
    }

    pub fn consume(&mut self) {
        self.previous_value = self.current_value.clone();
    }

    pub fn get(&self) -> &ast::LitExpr {
        &self.current_value
    }

    pub fn get_mut(&mut self) -> &mut ast::LitExpr {
        &mut self.current_value
    }

    pub fn is_dirty(&self) -> bool {
        self.previous_value != self.current_value
    }
}

#[derive(Debug, Clone)]
pub struct OpUiParam {
    pub name: String,
    pub repr: OpParamUiRepr,
    pub value: Value,
}

#[derive(Debug, Clone)]
pub struct Op {
    pub name: String,
    pub params: Vec<OpUiParam>,
    pub op: ast::FuncIdent,
}

#[derive(Debug, Clone)]
pub struct SelectedOp {
    pub op: Op,
    pub status: OpStatus,
}

impl SelectedOp {
    pub fn param_values_dirty(&self) -> bool {
        self.op.params.iter().any(|param| param.value.is_dirty())
    }

    pub fn consume_all_values(&mut self) {
        for param in &mut self.op.params {
            param.value.consume();
        }
    }
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
        self.selected_ops.push(SelectedOp {
            op: operation.clone(),
            status: OpStatus::Ready,
        });

        let var_decl = build_operation_var_ident(&operation, self.selected_ops.len() as u64);

        self.interpreter_server
            .submit_request(InterpreterRequest::PushProgStmt(var_decl));
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
        let mut invalidate_statements = false;

        for (i, selected_op) in self.selected_ops.iter_mut().enumerate() {
            selected_op.status = OpStatus::Running;

            if !invalidate_statements && selected_op.param_values_dirty() {
                invalidate_statements = true;

                log::info!(
                    "Operation \"{}\" is marked as dirty. All subsequent operations will be invalidated.",
                    selected_op.op.name
                );

                selected_op.consume_all_values();
            }

            if invalidate_statements {
                let var_decl = build_operation_var_ident(&selected_op.op, i as u64);

                self.interpreter_server
                    .submit_request(InterpreterRequest::SetProgStmtAt(i, var_decl));
            }
        }

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
                    if let OpParamUiRepr::GeometryDropdown(ref mut choices) = param.repr {
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

fn build_operation_var_ident(operation: &Op, var_ident_id: u64) -> ast::Stmt {
    let var_ident = ast::VarIdent(var_ident_id);
    let mut args = vec![];

    for param in &operation.params {
        let expr = match param.repr {
            OpParamUiRepr::GeometryDropdown(_) => match param.value.get() {
                ast::LitExpr::Uint(uint) => {
                    ast::Expr::Var(ast::VarExpr::new(VarIdent(u64::from(*uint))))
                }
                _ => unreachable!(),
            },
            _ => ast::Expr::Lit(param.value.get().clone()),
        };
        args.push(expr);
    }

    ast::Stmt::VarDecl(ast::VarDeclStmt::new(
        var_ident,
        ast::CallExpr::new(operation.op, args),
    ))
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
                    value: Value::new(ast::LitExpr::Float(0.0)),
                },
                OpUiParam {
                    name: "Parallels".to_string(),
                    repr: OpParamUiRepr::IntInput,
                    value: Value::new(ast::LitExpr::Uint(2)),
                },
                OpUiParam {
                    name: "Meridians".to_string(),
                    repr: OpParamUiRepr::IntInput,
                    value: Value::new(ast::LitExpr::Uint(3)),
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
                    repr: OpParamUiRepr::GeometryDropdown(vec![]),
                    value: Value::new(ast::LitExpr::Uint(0)),
                },
                OpUiParam {
                    name: "Sphere density".to_string(),
                    repr: OpParamUiRepr::IntSlider,
                    value: Value::new(ast::LitExpr::Uint(3)),
                },
            ],
        },
    );

    ops
}
