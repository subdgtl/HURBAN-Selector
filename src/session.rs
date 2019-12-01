use std::collections::hash_map::{Entry, HashMap};
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use crate::geometry::Geometry;
use crate::interpreter::ast::{FuncIdent, Prog, Stmt, VarIdent};
use crate::interpreter::Func;
use crate::interpreter_funcs;
use crate::interpreter_server::{
    InterpreterRequest, InterpreterResponse, InterpreterServer, PollResponseError, RequestId,
};

/// A notification from the session to the surrounding environment
/// about what values have been added since the last poll, and what
/// values have been removed are no longer required.
pub enum PollInterpreterResponseNotification {
    Add(VarIdent, Arc<Geometry>),
    Remove(VarIdent),
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
    interpreter_requests_in_flight: HashSet<RequestId>,

    prog: Prog,

    unused_values: HashMap<VarIdent, Arc<Geometry>>,

    /// Auxiliary side-array for prog. Determines the vars visible
    /// from the statement. The value is read by producing a slice
    /// from the begining of the array to the current stmt's index
    /// (exclusive), e.g. 0th stmt can not see any vars, 1st stmt can
    /// see vars produced by the 0th stmt, etc.
    var_visibility: Vec<VarIdent>,

    function_table: BTreeMap<FuncIdent, Box<dyn Func>>,
}

impl Session {
    pub fn new() -> Self {
        Self {
            interpreter_server: InterpreterServer::new(),
            interpreter_requests_in_flight: HashSet::new(),

            prog: Prog::new(Vec::new()),

            unused_values: HashMap::new(),

            var_visibility: Vec::new(),

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
        self.submit(InterpreterRequest::PushProgStmt(stmt));

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
        self.submit(InterpreterRequest::PopProgStmt);

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

        self.submit(InterpreterRequest::SetProgStmtAt(index, stmt));
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
    pub fn var_visibility_at_stmt(&self, index: usize) -> &[VarIdent] {
        &self.var_visibility[0..index]
    }

    /// Returns whether the interpreter is currently running. Program
    /// modifications and running the interpreter (again) are
    /// disallowed in this state.
    pub fn interpreter_busy(&self) -> bool {
        !self.interpreter_requests_in_flight.is_empty()
    }

    /// Starts the interpreter on the current program.
    pub fn interpret(&mut self) {
        // This is because the current session could want to report
        // errors and we would like to show them somewhere
        assert!(
            !self.interpreter_busy(),
            "Can't submit a request while the interpreter is already interpreting",
        );

        self.submit(InterpreterRequest::Interpret);
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
                    let tracked = self.interpreter_requests_in_flight.remove(&request_id);
                    assert!(tracked, "Each request must have been tracked");

                    match response {
                        InterpreterResponse::Completed => {
                            log::info!("Interpreter completed request {}", request_id);
                        }
                        InterpreterResponse::CompletedWithResult(result) => {
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
                                    for (current_var_ident, current_geometry) in &self.unused_values
                                    {
                                        if let Some((var_ident, value)) = value_set
                                            .unused_values
                                            .iter()
                                            .find(|(var_ident, _)| var_ident == current_var_ident)
                                        {
                                            // The value is present under the same
                                            // identifier in both new and old value set,
                                            // but it could have changed! If it did, we
                                            // both remove the old value and add the new.
                                            let geometry = value.unwrap_refcounted_geometry();

                                            if geometry != *current_geometry {
                                                to_remove.push(*var_ident);
                                                to_reinsert
                                                    .push((*var_ident, Arc::clone(&geometry)));
                                            }
                                        } else {
                                            // The value is no longer present in the new
                                            // value set, let's remove it!
                                            to_remove.push(*current_var_ident);
                                        }
                                    }

                                    // Process values to remove and values to reinsert
                                    for var_ident in to_remove {
                                        self.unused_values.remove(&var_ident).expect(
                                            "Value must be present if we want to remove it",
                                        );
                                        callback(PollInterpreterResponseNotification::Remove(
                                            var_ident,
                                        ));
                                    }
                                    for (var_ident, geometry) in to_reinsert {
                                        let inserted = self
                                            .unused_values
                                            .insert(var_ident, Arc::clone(&geometry))
                                            .is_none();
                                        assert!(
                                        inserted,
                                        "Value must have been removed previously for reinsertion",
                                    );
                                        callback(PollInterpreterResponseNotification::Add(
                                            var_ident, geometry,
                                        ))
                                    }

                                    for (var_ident, value) in value_set.unused_values {
                                        let geometry = value.unwrap_refcounted_geometry();
                                        // Only add and emit events for values that we
                                        // didn't have before. We handled re-insertions
                                        // earlier by comparing the values.
                                        if let Entry::Vacant(vacant) =
                                            self.unused_values.entry(var_ident)
                                        {
                                            vacant.insert(Arc::clone(&geometry));
                                            callback(PollInterpreterResponseNotification::Add(
                                                var_ident, geometry,
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
        self.var_visibility.clear();

        for stmt in self.prog.stmts() {
            match stmt {
                Stmt::VarDecl(var_decl) => {
                    self.var_visibility.push(var_decl.ident());
                }
            }
        }

        assert_eq!(
            self.var_visibility.len(),
            self.prog.stmts().len(),
            "Each stmt is a var decl and must produce a variable",
        );
    }

    fn submit(&mut self, request: InterpreterRequest) {
        let request_id = self.interpreter_server.submit_request(request);
        let tracked = self.interpreter_requests_in_flight.insert(request_id);
        assert!(
            tracked,
            "Interpreter server must provide unique request ids"
        );
    }
}
