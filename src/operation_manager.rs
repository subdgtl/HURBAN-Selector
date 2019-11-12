use std::collections::{HashMap, HashSet};

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
    Float3Input,
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
        if self.status == OpStatus::Ready {
            true
        } else {
            self.op.params.iter().any(|param| param.value.is_dirty())
        }
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
    geometry: Geometry,
    var_ident: u64,
    geometry_id: Option<GpuGeometryId>,
    used: bool,
}

impl GeometryMetadata {
    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }
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

    /// Submits program to interpreter, invalidates stale geometries and sets
    /// status of last operation to "running". Returns number of unused
    /// geometries that should be removed from scene.
    ///
    /// Each operation is checked for dirty (changed) values and all subsequent
    /// operations are marked as invalidated. This means that new statements
    /// are submitted to interpreter, invalid geometries popped out of geometry
    /// stacks and number of geometries to be removed from scene returned. In
    /// case nothing changed, interpreter doesn't run.
    pub fn submit_program(&mut self) -> usize {
        let mut invalidate_statements = false;
        let mut results_to_invalidate = 0;
        let mut should_interpret = false;

        // Find out which operations should be invalidated.

        for (var_ident_id, selected_op) in self.selected_ops.iter_mut().enumerate() {
            // First dirty operation marks all subsequent ones as invalidated.
            if !invalidate_statements && selected_op.param_values_dirty() {
                invalidate_statements = true;

                log::info!("Operation \"{}\" is marked as dirty.", selected_op.op.name);

                selected_op.consume_all_values();
            }

            if invalidate_statements {
                let var_decl = build_operation_var_ident(&selected_op.op, var_ident_id as u64);

                self.interpreter_server
                    .submit_request(InterpreterRequest::SetProgStmtAt(var_ident_id, var_decl));

                // If this operation wasn't even run, there is nothing to
                // invalidate.
                if selected_op.status == OpStatus::Finished {
                    results_to_invalidate += 1;
                }

                should_interpret = true;
                selected_op.status = OpStatus::Running;
            }
        }

        log::info!(
            "Geometries from {} operations will be invalidated.",
            results_to_invalidate
        );

        let mut geometries_to_invalidate: usize = 0;

        // Find out which geometries should be invalidated.

        for _ in 0..results_to_invalidate {
            let result_geometries = self
                .geometry_stack
                .pop()
                .expect("Failed to pop value off geometry stack");

            for _ in 0..result_geometries {
                let geometry_metadata = self
                    .geometry_metadata
                    .pop()
                    .expect("Failed to pop value off geometry metadata");

                // Since frontend shows only unused geometries, only those
                // should be counted.
                if !geometry_metadata.used {
                    geometries_to_invalidate += 1;
                }
            }
        }

        if should_interpret {
            self.interpreter_server
                .submit_request(InterpreterRequest::Interpret);
        }

        log::info!(
            "{} geometries can be removed from the scene.",
            geometries_to_invalidate
        );

        geometries_to_invalidate
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

                    // We care only about results with var idents larger than
                    // the geometry stack length. This means that only new
                    // geometries are going to be processed.
                    let last_used_var_ident = self.geometry_stack.len();
                    let mut new_geometry_metadata = HashMap::new();
                    let mut var_idents = HashSet::new();

                    for (var_ident, value) in &value_set.used_values {
                        if var_ident.0 + 1 > last_used_var_ident as u64 {
                            let geometry_metadata = GeometryMetadata {
                                name: format!("Geometry #{} from {}", 1, op_name),
                                geometry: value.unwrap_geometry().clone(),
                                var_ident: var_ident.0,
                                used: true,
                                geometry_id: None,
                            };
                            new_geometry_metadata.insert(var_ident.0, geometry_metadata);
                            var_idents.insert(var_ident.0);
                        }
                    }

                    for (var_ident, value) in value_set.unused_values {
                        if var_ident.0 + 1 > last_used_var_ident as u64 {
                            let geometry_metadata = GeometryMetadata {
                                name: format!("Geometry #{} from {}", 1, op_name),
                                geometry: value.unwrap_geometry().clone(),
                                var_ident: var_ident.0,
                                used: false,
                                geometry_id: None,
                            };
                            new_geometry_metadata.insert(var_ident.0, geometry_metadata);
                            var_idents.insert(var_ident.0);
                        }
                    }

                    // Since ordering matters in case of stacks, var idents need
                    // to be sorted.
                    let mut var_idents: Vec<_> = var_idents.iter().collect();
                    var_idents.sort();

                    // Geometry metadata are pushed onto geometry stacks in order.
                    // Geometry handler runs for each unused geometry so it can
                    // be added to scene.
                    for var_ident in var_idents {
                        let mut geometry_metadata = new_geometry_metadata
                            .remove(&var_ident)
                            .expect("Failed to remove new geometry");

                        if !geometry_metadata.used {
                            let geometry_id = geometry_handler(&mut geometry_metadata);

                            geometry_metadata.geometry_id = Some(geometry_id);
                        }

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

    ops.insert(
        "Transform".to_string(),
        Op {
            name: "Transform".to_string(),
            op: interpreter_funcs::FUNC_ID_TRANSFORM,
            params: vec![
                OpUiParam {
                    name: "Geometry".to_string(),
                    repr: OpParamUiRepr::GeometryDropdown(vec![]),
                    value: Value::new(ast::LitExpr::Uint(0)),
                },
                OpUiParam {
                    name: "Translate".to_string(),
                    repr: OpParamUiRepr::Float3Input,
                    value: Value::new(ast::LitExpr::Float3([0.0, 0.0, 0.0])),
                },
                OpUiParam {
                    name: "Rotate".to_string(),
                    repr: OpParamUiRepr::Float3Input,
                    value: Value::new(ast::LitExpr::Float3([0.0, 0.0, 0.0])),
                },
                OpUiParam {
                    name: "Scale".to_string(),
                    repr: OpParamUiRepr::Float3Input,
                    value: Value::new(ast::LitExpr::Float3([0.0, 0.0, 0.0])),
                },
            ],
        },
    );

    ops
}
