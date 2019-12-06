use std::collections::hash_map::{Entry, HashMap};
use std::collections::{BTreeMap, HashSet};

use crate::interpreter::ast::{FuncIdent, Prog, Stmt, VarIdent};
use crate::interpreter::{Func, Ty, Value};
use crate::interpreter_funcs;
use crate::interpreter_server::{
    InterpreterRequest, InterpreterResponse, InterpreterServer, PollResponseError, RequestId,
};

/// A notification from the session to the surrounding environment
/// about what values have been added since the last poll, and what
/// values have been removed are no longer required.
pub enum PollInterpreterResponseNotification {
    Add(VarIdent, Value),
    Remove(VarIdent, Value),
}

/// An editing session.
///
/// Contains the current definition of the pipeline program and can
/// execute it via the interpreter running in a separate thread.
///
/// All program mutations are non-blocking (the session does not wait
/// for the interpreter to acknowledge them), and polling the session
/// for interpreter results is also non-blocking - if there is no
/// response, nothing happens.
pub struct Session {
    interpreter_server: InterpreterServer,
    interpreter_interpret_request_in_flight: Option<RequestId>,
    interpreter_edit_prog_requests_in_flight: HashSet<RequestId>,

    prog: Prog,

    unused_values: HashMap<VarIdent, Value>,

    // Auxiliary side-arrays for prog. Determine geometry and
    // geometry-array vars visible from a stmt. The value is read by
    // producing a slice from the begining of the array to the current
    // stmt's index (exclusive), and filtering only `Some`
    // values. E.g. 0th stmt can not see any vars, 1st stmt can see
    // vars produced by the 0th stmt (if it is `Some`), etc.
    var_visibility_geometry: Vec<Option<VarIdent>>,
    var_visibility_geometry_array: Vec<Option<VarIdent>>,

    function_table: BTreeMap<FuncIdent, Box<dyn Func>>,
}

impl Session {
    pub fn new() -> Self {
        Self {
            interpreter_server: InterpreterServer::new(),
            interpreter_interpret_request_in_flight: None,
            interpreter_edit_prog_requests_in_flight: HashSet::new(),

            prog: Prog::new(Vec::new()),

            unused_values: HashMap::new(),

            var_visibility_geometry: Vec::new(),
            var_visibility_geometry_array: Vec::new(),

            // FIXME: @Correctness this is a hack that is currently
            // harmless, but should eventually be cleaned up. Some
            // funcs have internal state (at the time of writing this
            // is only the import obj func with its importer
            // cache). If we create multiple instances of the function
            // table as we do here (the interpreter has its own
            // instance), this state will not be shared, which could
            // lead to unexpected behavior, if the state is mutated
            // from multiple places. Fortunately, we currenly don't
            // mutate the state here in the session (nor do we need
            // to), that's why the hack is harmless. The most clean
            // solution would be to split the stateful part of funcs
            // away from the stateless func descriptors, so that the
            // state only exists in the interpreter and this table
            // would just contain the function descriptors, which we
            // wouldn't have to care there are multiple copies of.
            function_table: interpreter_funcs::create_function_table(),
        }
    }

    /// Pushes a new statement onto the program.
    ///
    /// # Panics
    /// Panics if the interpreter is busy.
    pub fn push_prog_stmt(&mut self, stmt: Stmt) {
        // This is because the current session could want to report
        // errors and we would like to show them somewhere
        assert!(
            !self.interpreter_busy(),
            "Can't submit a request while the interpreter is already interpreting",
        );

        self.prog.push_stmt(stmt.clone());

        let request_id = self
            .interpreter_server
            .submit_request(InterpreterRequest::PushProgStmt(stmt));
        let tracked = self
            .interpreter_edit_prog_requests_in_flight
            .insert(request_id);
        assert!(
            tracked,
            "Interpreter server must provide unique request ids"
        );

        self.recompute_var_visibility();
    }

    /// Pops a statement from the program.
    ///
    /// # Panics
    /// Panics if the interpreter is busy.
    pub fn pop_prog_stmt(&mut self) {
        // This is because the current session could want to report
        // errors and we would like to show them somewhere
        assert!(
            !self.interpreter_busy(),
            "Can't submit a request while the interpreter is already interpreting",
        );

        self.prog.pop_stmt();

        let request_id = self
            .interpreter_server
            .submit_request(InterpreterRequest::PopProgStmt);
        let tracked = self
            .interpreter_edit_prog_requests_in_flight
            .insert(request_id);
        assert!(
            tracked,
            "Interpreter server must provide unique request ids"
        );

        self.recompute_var_visibility();
    }

    /// Edits a program statement at the index.
    ///
    /// # Panics
    /// Panics if the interpreter is busy.
    pub fn set_prog_stmt_at(&mut self, index: usize, stmt: Stmt) {
        // This is because the current session could want to report
        // errors and we would like to show them somewhere
        assert!(
            !self.interpreter_busy(),
            "Can't submit a request while the interpreter is already interpreting",
        );

        self.prog.set_stmt_at(index, stmt.clone());

        let request_id = self
            .interpreter_server
            .submit_request(InterpreterRequest::SetProgStmtAt(index, stmt));
        let tracked = self
            .interpreter_edit_prog_requests_in_flight
            .insert(request_id);
        assert!(
            tracked,
            "Interpreter server must provide unique request ids"
        );

        self.recompute_var_visibility();
    }

    /// Returns the statements currently contained in the current pipeline's
    /// program.
    pub fn stmts(&self) -> &[Stmt] {
        self.prog.stmts()
    }

    /// Returns the definitions of all known functions.
    pub fn function_table(&self) -> &BTreeMap<FuncIdent, Box<dyn Func>> {
        &self.function_table
    }

    /// Returns the next free variable identifier for the current
    /// program definition.
    pub fn next_free_var_ident(&self) -> VarIdent {
        VarIdent(self.prog.stmts().len() as u64)
    }

    /// Returns human readable variable name for a variable identifier
    /// or `None` if the variable identifier does not exist in the
    /// current program.
    pub fn var_name_for_ident(&self, var_ident: VarIdent) -> Option<&str> {
        // FIXME: @Optimization Don't iterate all the time
        self.stmts()
            .iter()
            .find_map(|stmt| match stmt {
                Stmt::VarDecl(var_decl) => {
                    if var_decl.ident() == var_ident {
                        Some(var_decl)
                    } else {
                        None
                    }
                }
            })
            .map(|var_decl| {
                self.function_table[&var_decl.init_expr().ident()]
                    .info()
                    .return_value_name
            })
    }

    /// Returns all visible variable identifiers from a position
    /// (index) in the program.
    pub fn visible_vars_at_stmt<'a>(
        &'a self,
        index: usize,
        ty: Ty,
    ) -> impl Iterator<Item = VarIdent> + Clone + 'a {
        static EMPTY: Vec<Option<VarIdent>> = Vec::new();
        let var_visibility = match ty {
            Ty::Geometry => &self.var_visibility_geometry,
            Ty::GeometryArray => &self.var_visibility_geometry_array,
            _ => &EMPTY,
        };

        var_visibility[0..index]
            .iter()
            .filter_map(|var_ident| var_ident.as_ref())
            .copied()
    }

    /// Returns whether the interpreter is currently running. Program
    /// modifications and running the interpreter (again) are
    /// disallowed in this state.
    pub fn interpreter_busy(&self) -> bool {
        self.interpreter_interpret_request_in_flight.is_some()
    }

    /// Starts the interpreter on the current program.
    pub fn interpret(&mut self) {
        // This is because the current session could want to report
        // errors and we would like to show them somewhere
        assert!(
            !self.interpreter_busy(),
            "Can't submit a request while the interpreter is already interpreting",
        );

        let request_id = self
            .interpreter_server
            .submit_request(InterpreterRequest::Interpret);
        self.interpreter_interpret_request_in_flight
            .replace(request_id);
    }

    /// Poll the interpreter for responses and call the callback for
    /// each notification generated this way.
    ///
    /// The notifications can ask the surrounding environment to start
    /// or stop tracking a value with a variable identifier.
    ///
    /// Polls the interpreter until there are no more messages in the
    /// response channel.
    pub fn poll_interpreter_response<C>(&mut self, mut callback: C)
    where
        C: FnMut(PollInterpreterResponseNotification),
    {
        // Loop over all responses

        // This is allowed, because we might add other kinds of errors
        // to `poll_response()` besides `Pending` and satisfying
        // clippy would mean this will still compile, when in fact we
        // don't want it to.
        #[allow(clippy::while_let_loop)]
        loop {
            match self.interpreter_server.poll_response() {
                Ok((request_id, response)) => {
                    match response {
                        InterpreterResponse::Completed => {
                            let tracked = self
                                .interpreter_edit_prog_requests_in_flight
                                .remove(&request_id);
                            assert!(tracked, "Each edit prog request must have been tracked");

                            log::info!("Interpreter completed request {}", request_id);
                        }
                        InterpreterResponse::CompletedWithResult(result) => {
                            let tracked = self
                                .interpreter_interpret_request_in_flight
                                .take()
                                .is_some();
                            assert!(tracked, "The interpret request must have been tracked");

                            log::info!("Interpreter completed request {} with result", request_id);

                            match result {
                                Ok(value_set) => {
                                    // Now we track whether the usage of any value changed. Adding
                                    // an operation to the pipeline can:
                                    // - create a new unused_value
                                    // - create a new used_value
                                    // - change an existing unused_value to used_value
                                    //
                                    // Removing an operation from the pipeline can:
                                    // - remove an existing unused_value
                                    // - remove an existing used_value
                                    // - change an existing used_value to unused_value
                                    //
                                    // We diff the old and new sets and perform the following
                                    // operations:
                                    // - Detect which values should be removed and `Remove` them
                                    // - Detect which values should be retained, but they
                                    //   changed, we both `Remove` and `Add` them to notify
                                    //   the outside of the change
                                    // - Detect which values should be added and `Add` them

                                    // FIXME: Currently, we only call the add/remove
                                    // handlers with when a value enters or leaves the
                                    // unused_values set. This can be easily copy-paste
                                    // extended to handle used_values the same way.

                                    let mut to_remove =
                                        Vec::with_capacity(self.unused_values.len());
                                    let mut to_reinsert =
                                        Vec::with_capacity(self.unused_values.len());
                                    for (current_var_ident, current_value) in &self.unused_values {
                                        if let Some((var_ident, value)) = value_set
                                            .unused_values
                                            .iter()
                                            .find(|(var_ident, _)| var_ident == current_var_ident)
                                        {
                                            // The value is present under the same
                                            // identifier in both new and old value set,
                                            // but it could have changed! If it did, we
                                            // both remove the old value and add the new.
                                            if value != current_value {
                                                to_remove.push(*var_ident);
                                                to_reinsert.push((*var_ident, value.clone()));
                                            }
                                        } else {
                                            // The value is no longer present in the new
                                            // value set, let's remove it!
                                            to_remove.push(*current_var_ident);
                                        }
                                    }

                                    // Process values to remove and values to reinsert
                                    for var_ident in to_remove {
                                        let value = self.unused_values.remove(&var_ident).expect(
                                            "Value must be present if we want to remove it",
                                        );
                                        callback(PollInterpreterResponseNotification::Remove(
                                            var_ident, value,
                                        ));
                                    }
                                    for (var_ident, value) in to_reinsert {
                                        let inserted = self
                                            .unused_values
                                            .insert(var_ident, value.clone())
                                            .is_none();
                                        assert!(
                                            inserted,
                                            "Value must have been removed previously for reinsertion",
                                        );
                                        callback(PollInterpreterResponseNotification::Add(
                                            var_ident, value,
                                        ))
                                    }

                                    for (var_ident, value) in value_set.unused_values {
                                        // Only add and emit events for values that we
                                        // didn't have before. We handled re-insertions
                                        // earlier by comparing the values.
                                        if let Entry::Vacant(vacant) =
                                            self.unused_values.entry(var_ident)
                                        {
                                            vacant.insert(value.clone());
                                            callback(PollInterpreterResponseNotification::Add(
                                                var_ident, value,
                                            ));
                                        }
                                    }
                                }
                                Err(interpret_error) => {
                                    // FIXME: Display error on UI
                                    log::error!(
                                        "Interpreter failed with error {}",
                                        interpret_error
                                    );
                                }
                            }
                        }
                    }

                    self.recompute_var_visibility();
                }
                Err(PollResponseError::Pending) => {
                    // No more responses, break out of the loop
                    break;
                }
            }
        }
    }

    fn recompute_var_visibility(&mut self) {
        // FIXME: Get variable visibility analysis from interpreter

        self.var_visibility_geometry.clear();
        self.var_visibility_geometry_array.clear();

        let mut n_geometries = 0;
        let mut n_geometry_arrays = 0;

        for stmt in self.prog.stmts() {
            let Stmt::VarDecl(var_decl) = stmt;
            let func_ident = var_decl.init_expr().ident();
            let func = &self.function_table[&func_ident];
            match func.return_ty() {
                Ty::Geometry => {
                    self.var_visibility_geometry.push(Some(var_decl.ident()));
                    self.var_visibility_geometry_array.push(None);

                    n_geometries += 1;
                }
                Ty::GeometryArray => {
                    self.var_visibility_geometry.push(None);
                    self.var_visibility_geometry_array
                        .push(Some(var_decl.ident()));

                    n_geometry_arrays += 1;
                }
                _ => panic!("Unsupported variable type"),
            }
        }

        assert_eq!(
            n_geometries + n_geometry_arrays,
            self.prog.stmts().len(),
            "Each stmt is a var decl and must produce a variable",
        );
    }
}
