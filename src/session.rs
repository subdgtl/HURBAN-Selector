use std::collections::hash_map::{Entry, HashMap};
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

use crate::interpreter::ast::{FuncIdent, Prog, Stmt, VarIdent};
use crate::interpreter::{Func, InterpretError, LogMessage, Ty, Value};
use crate::interpreter_funcs;
use crate::interpreter_server::{
    InterpreterRequest, InterpreterResponse, InterpreterServer, PollResponseError, RequestId,
};

/// A notification from the session to the surrounding environment
/// about what values have been added since the last poll, and what
/// values have been removed are no longer required.
pub enum PollNotification {
    Add(VarIdent, Value),
    Remove(VarIdent, Value),
    FinishedSuccessfully,
    FinishedWithError(String),
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
    autorun_delay: Option<Duration>,
    last_uninterpreted_edit: Option<Instant>,

    interpreter_server: InterpreterServer,
    interpreter_interpret_request_in_flight: Option<RequestId>,
    interpreter_edit_prog_requests_in_flight: HashSet<RequestId>,

    prog: Prog,
    log_messages: Vec<Vec<LogMessage>>,
    error: Option<InterpretError>,

    unused_values: HashMap<VarIdent, Value>,

    // Auxiliary side-arrays for prog. Determine mesh and mesh-array
    // vars visible from a stmt. The value is read by producing a
    // slice from the beginning of the array to the current stmt's
    // index (exclusive), and filtering only `Some` values. E.g. 0th
    // stmt can not see any vars, 1st stmt can see vars produced by
    // the 0th stmt (if it is `Some`), etc.
    var_visibility_mesh: Vec<Option<VarIdent>>,
    var_visibility_mesh_array: Vec<Option<VarIdent>>,

    function_table: BTreeMap<FuncIdent, Box<dyn Func>>,
}

impl Session {
    pub fn new() -> Self {
        Self {
            autorun_delay: None,
            last_uninterpreted_edit: None,

            interpreter_server: InterpreterServer::new(),
            interpreter_interpret_request_in_flight: None,
            interpreter_edit_prog_requests_in_flight: HashSet::new(),

            prog: Prog::new(Vec::new()),
            log_messages: Vec::new(),
            error: None,

            unused_values: HashMap::new(),

            var_visibility_mesh: Vec::new(),
            var_visibility_mesh_array: Vec::new(),

            // FIXME: @Correctness this is a hack that is currently
            // harmless, but should eventually be cleaned up. Some
            // funcs have internal state (at the time of writing this
            // is only the import obj func with its importer
            // cache). If we create multiple instances of the function
            // table as we do here (the interpreter has its own
            // instance), this state will not be shared, which could
            // lead to unexpected behavior, if the state is mutated
            // from multiple places. Fortunately, we currently don't
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

    pub fn autorun_delay(&self) -> Option<Duration> {
        self.autorun_delay
    }

    pub fn set_autorun_delay(&mut self, autorun_delay: Option<Duration>) {
        self.autorun_delay = autorun_delay;
    }

    /// Pushes a new statement onto the program.
    ///
    /// # Panics
    /// Panics if the interpreter is busy.
    pub fn push_prog_stmt(&mut self, current_time: Instant, stmt: Stmt) {
        // This is because the current session could want to report
        // errors and we would like to show them somewhere
        assert!(
            !self.interpreter_busy(),
            "Can't submit a request while the interpreter is already interpreting",
        );

        self.last_uninterpreted_edit = Some(current_time);
        self.prog.push_stmt(stmt.clone());
        self.log_messages.push(Vec::new());
        self.error = None;

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
    pub fn pop_prog_stmt(&mut self, current_time: Instant) {
        // This is because the current session could want to report
        // errors and we would like to show them somewhere
        assert!(
            !self.interpreter_busy(),
            "Can't submit a request while the interpreter is already interpreting",
        );

        self.last_uninterpreted_edit = Some(current_time);
        self.prog.pop_stmt();
        self.log_messages.pop();
        self.error = None;

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
    pub fn set_prog_stmt_at(&mut self, current_time: Instant, stmt_index: usize, stmt: Stmt) {
        // This is because the current session could want to report
        // errors and we would like to show them somewhere
        assert!(
            !self.interpreter_busy(),
            "Can't submit a request while the interpreter is already interpreting",
        );

        // If we are replacing one function call with a completely
        // different function call (as opposed to just updating
        // parameters), we want to clear the logs.
        let current_stmt = &self.prog.stmts()[stmt_index];
        match (current_stmt, &stmt) {
            (Stmt::VarDecl(current_var_decl), Stmt::VarDecl(new_var_decl)) => {
                if current_var_decl.init_expr().ident() != new_var_decl.init_expr().ident() {
                    self.log_messages[stmt_index].clear();
                }
            }
        }

        self.last_uninterpreted_edit = Some(current_time);
        self.prog.set_stmt_at(stmt_index, stmt.clone());
        self.error = None;

        let request_id = self
            .interpreter_server
            .submit_request(InterpreterRequest::SetProgStmtAt(stmt_index, stmt));
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
        stmt_index: usize,
        ty: Ty,
    ) -> impl Iterator<Item = VarIdent> + Clone + 'a {
        static EMPTY: Vec<Option<VarIdent>> = Vec::new();
        let var_visibility = match ty {
            Ty::Mesh => &self.var_visibility_mesh,
            Ty::MeshArray => &self.var_visibility_mesh_array,
            _ => &EMPTY,
        };

        var_visibility[0..stmt_index]
            .iter()
            .filter_map(|var_ident| var_ident.as_ref())
            .copied()
    }

    pub fn log_messages_at_stmt(&self, stmt_index: usize) -> &[LogMessage] {
        &self.log_messages[stmt_index]
    }

    pub fn error_at_stmt(&self, stmt_index: usize) -> Option<&impl fmt::Display> {
        self.error.as_ref().and_then(|err| {
            if stmt_index == err.stmt_index() {
                Some(err)
            } else {
                None
            }
        })
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

    /// Poll the interpreter for responses and call the callback for each
    /// notification generated this way. Polls the interpreter until there are
    /// no more messages in the response channel.
    ///
    /// The notifications can ask the surrounding environment to start or stop
    /// tracking a value with a variable identifier.
    ///
    /// If `autorun_delay` is not `None`, this also tries to run the interpreter
    /// if it is not already busy and sufficient time has passed since the last
    /// program edit.
    pub fn poll<C>(&mut self, current_time: Instant, mut callback: C)
    where
        C: FnMut(PollNotification),
    {
        if let Some(delay) = self.autorun_delay {
            if let Some(last_uninterpreted_edit) = self.last_uninterpreted_edit {
                // Note: used `Instant::saturating_duration_since` so that this
                // doesn't panic when caller passes inconsistent time, e.g. the
                // edit time is later than current time.
                if current_time.saturating_duration_since(last_uninterpreted_edit) > delay
                    && !self.interpreter_busy()
                {
                    self.last_uninterpreted_edit = None;
                    self.interpret();
                }
            }
        }

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
                        InterpreterResponse::CompletedEditProg => {
                            let tracked = self
                                .interpreter_edit_prog_requests_in_flight
                                .remove(&request_id);
                            assert!(tracked, "Each edit prog request must have been tracked");

                            log::info!("Interpreter completed edit program request {}", request_id);
                        }
                        InterpreterResponse::CompletedInterpret(interpret_outcome) => {
                            let tracked = self
                                .interpreter_interpret_request_in_flight
                                .take()
                                .is_some();
                            assert!(tracked, "The interpret request must have been tracked");

                            log::info!("Interpreter completed interpret request {}", request_id);

                            match interpret_outcome.result {
                                Ok(interpret_value) => {
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
                                        if let Some((var_ident, value)) = interpret_value
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
                                        callback(PollNotification::Remove(var_ident, value));
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
                                        callback(PollNotification::Add(var_ident, value))
                                    }

                                    for (var_ident, value) in interpret_value.unused_values {
                                        // Only add and emit events for values that we
                                        // didn't have before. We handled re-insertions
                                        // earlier by comparing the values.
                                        if let Entry::Vacant(vacant) =
                                            self.unused_values.entry(var_ident)
                                        {
                                            vacant.insert(value.clone());
                                            callback(PollNotification::Add(var_ident, value));
                                        }
                                    }

                                    // Impure funcs (think import) can sometimes succeed
                                    // with the same parameters for which they previously
                                    // failed. We want to clear the error in this case.
                                    self.error = None;

                                    callback(PollNotification::FinishedSuccessfully);
                                }
                                Err(interpret_error) => {
                                    let error_message = format!("{}", interpret_error);
                                    callback(PollNotification::FinishedWithError(
                                        error_message.clone(),
                                    ));
                                    log::error!("Interpreter failed with error: {}", error_message);

                                    self.error = Some(interpret_error);
                                }
                            }

                            assert_eq!(
                                self.log_messages.len(),
                                interpret_outcome.log_messages.len(),
                                "Every statement must have its own log messages",
                            );
                            for (i, log_messages_at_stmt) in
                                interpret_outcome.log_messages.into_iter().enumerate()
                            {
                                self.log_messages[i].extend(log_messages_at_stmt);
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

        self.var_visibility_mesh.clear();
        self.var_visibility_mesh_array.clear();

        let mut n_mesh = 0;
        let mut n_mesh_array = 0;

        for stmt in self.prog.stmts() {
            let Stmt::VarDecl(var_decl) = stmt;
            let func_ident = var_decl.init_expr().ident();
            let func = &self.function_table[&func_ident];
            match func.return_ty() {
                Ty::Mesh => {
                    self.var_visibility_mesh.push(Some(var_decl.ident()));
                    self.var_visibility_mesh_array.push(None);

                    n_mesh += 1;
                }
                Ty::MeshArray => {
                    self.var_visibility_mesh.push(None);
                    self.var_visibility_mesh_array.push(Some(var_decl.ident()));

                    n_mesh_array += 1;
                }
                _ => panic!("Unsupported variable type"),
            }
        }

        assert_eq!(
            n_mesh + n_mesh_array,
            self.prog.stmts().len(),
            "Each stmt is a var decl and must produce a variable",
        );
    }
}
