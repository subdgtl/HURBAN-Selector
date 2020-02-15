use std::collections::hash_map::{Entry, HashMap};
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

use crate::interpreter::ast::{FuncIdent, Prog, Stmt, VarIdent};
use crate::interpreter::{Func, InterpretError, InterpretValue, LogMessage, Ty, Value};
use crate::interpreter_funcs;
use crate::interpreter_server::{
    InterpreterRequest, InterpreterResponse, InterpreterServer, PollResponseError, RequestId,
};

/// A notification from the session to the surrounding environment
/// about what values have been added since the last poll, and what
/// values have been removed are no longer required.
pub enum PollNotification {
    UsedValueAdded(VarIdent, Value),
    UsedValueRemoved(VarIdent, Value),
    UnusedValueAdded(VarIdent, Value),
    UnusedValueRemoved(VarIdent, Value),
    FinishedSuccessfully,
    // FIXME: Replace String with InterpretError
    FinishedWithError(String),
}

#[derive(Debug, Clone, PartialEq)]
enum DiffEvent {
    AddUsed(VarIdent, Value),
    RemoveUsed(VarIdent),
    VerifyUsed(VarIdent, Value),
    TransitionUsedToUnused(VarIdent, Value),
    AddUnused(VarIdent, Value),
    RemoveUnused(VarIdent),
    VerifyUnused(VarIdent, Value),
    TransitionUnusedToUsed(VarIdent, Value),
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

    used_values: HashMap<VarIdent, Value>,
    unused_values: HashMap<VarIdent, Value>,

    // Working memory for diffing interpreter responses
    diff_events: Vec<DiffEvent>,
    diff_processed_idents: HashSet<VarIdent>,

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

            diff_events: Vec::with_capacity(64),
            diff_processed_idents: HashSet::with_capacity(64),

            prog: Prog::new(Vec::new()),
            log_messages: Vec::new(),
            error: None,

            used_values: HashMap::new(),
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
                                    self.process_interpret_value(interpret_value, &mut callback);

                                    // Impure funcs (think import) can sometimes succeed
                                    // with the same parameters for which they previously
                                    // failed. We want to clear the error in this case.
                                    self.error = None;

                                    callback(PollNotification::FinishedSuccessfully);
                                }
                                Err(interpret_error) => {
                                    log::error!(
                                        "Interpreter failed with error: {}",
                                        interpret_error,
                                    );

                                    let error_message = format!("{}", interpret_error);

                                    self.error = Some(interpret_error);

                                    callback(PollNotification::FinishedWithError(error_message));
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

    /// Tracks whether the usage of any value changed and calls callback with
    /// corresponding notification.
    ///
    /// Executing a program with new added statements can:
    ///
    /// - Create a new unused value
    /// - Create a new used value
    /// - Transition an existing unused value to used value
    ///
    /// Executing a program with statements removed can:
    ///
    /// - Remove an existing unused value
    /// - Remove an existing used value
    /// - Transition an existing used value to unused value
    ///
    /// We diff the old and new sets and perform the following operations:
    ///
    /// - Detect which values should be added and add them (for both used and unused sets),
    /// - Detect which values should be removed and remove them (for both used
    ///   and unused sets),
    /// - Detect which values should be retained, but they changed, we both
    ///   remove and add them to notify the outside of the change (for both used
    ///   and unused sets),
    /// - Detect which values should be transitioned from the used set to unused
    ///   and miami vice versa.
    fn process_interpret_value<C>(&mut self, interpret_value: InterpretValue, mut callback: C)
    where
        C: FnMut(PollNotification),
    {
        // To correctly diff new state against old, we need to keep a copy of
        // the old state around. Therefore, we perform changes in 2 stages:
        //
        // 1) Generate a list of diff events by comparing new and old sets,
        // 2) Interpret the list of diff events and modify the old set and fire
        //    callbacks with corresponding notifications.

        let InterpretValue {
            used_values,
            unused_values,
            ..
        } = interpret_value;

        let events = &mut self.diff_events;
        let processed_idents = &mut self.diff_processed_idents;

        // Look at new used values to detect addition, verification and transition events.
        for (var_ident, value) in used_values {
            let contained_in_used = self.used_values.contains_key(&var_ident);
            let contained_in_unused = self.unused_values.contains_key(&var_ident);

            match (contained_in_used, contained_in_unused) {
                (true, true) => panic!("Value can't be both used and unused"),
                (true, false) => events.push(DiffEvent::VerifyUsed(var_ident, value)),
                (false, true) => events.push(DiffEvent::TransitionUnusedToUsed(var_ident, value)),
                (false, false) => events.push(DiffEvent::AddUsed(var_ident, value)),
            }

            processed_idents.insert(var_ident);
        }

        // Look at new unused values to detect addition, verification, and transition events.
        for (var_ident, value) in unused_values {
            let contained_in_used = self.used_values.contains_key(&var_ident);
            let contained_in_unused = self.unused_values.contains_key(&var_ident);

            match (contained_in_used, contained_in_unused) {
                (true, true) => panic!("Value can't be both used and unused"),
                (true, false) => events.push(DiffEvent::TransitionUsedToUnused(var_ident, value)),
                (false, true) => events.push(DiffEvent::VerifyUnused(var_ident, value)),
                (false, false) => events.push(DiffEvent::AddUnused(var_ident, value)),
            }

            processed_idents.insert(var_ident);
        }

        // See if there is a value in the old used set we haven't seen yet. If
        // so, it has to be a removal event.
        for var_ident in self.used_values.keys() {
            if !processed_idents.contains(var_ident) {
                events.push(DiffEvent::RemoveUsed(*var_ident));
            }
        }

        // See if there is a value in the old unused set we haven't seen yet. If
        // so, it has to be a removal event.
        for var_ident in self.unused_values.keys() {
            if !processed_idents.contains(var_ident) {
                events.push(DiffEvent::RemoveUnused(*var_ident));
            }
        }

        // Process all the events, update internal state, call callback with
        // change notifications. Make sure working memory is cleared for next
        // iteration.
        processed_idents.clear();
        for event in events.drain(..) {
            match event {
                DiffEvent::AddUsed(var_ident, value) => {
                    self.used_values.insert(var_ident, value.clone());
                    callback(PollNotification::UsedValueAdded(var_ident, value));
                }
                DiffEvent::RemoveUsed(var_ident) => {
                    let value = self
                        .used_values
                        .remove(&var_ident)
                        .expect("Values scheduled for removal must be present");
                    callback(PollNotification::UsedValueRemoved(var_ident, value));
                }
                DiffEvent::VerifyUsed(var_ident, value) => {
                    match self.used_values.entry(var_ident) {
                        Entry::Occupied(mut occupied) => {
                            if occupied.get() != &value {
                                let old_value = occupied.get().clone();
                                occupied.insert(value.clone());

                                callback(PollNotification::UsedValueRemoved(var_ident, old_value));
                                callback(PollNotification::UsedValueAdded(var_ident, value));
                            }
                        }
                        _ => panic!("Values scheduled for verification must be present"),
                    }
                }
                DiffEvent::TransitionUsedToUnused(var_ident, value) => {
                    let old_value = self
                        .used_values
                        .remove(&var_ident)
                        .expect("Values scheduled for transition must be present");
                    self.unused_values.insert(var_ident, value.clone());

                    callback(PollNotification::UsedValueRemoved(var_ident, old_value));
                    callback(PollNotification::UnusedValueAdded(var_ident, value));
                }
                DiffEvent::AddUnused(var_ident, value) => {
                    self.unused_values.insert(var_ident, value.clone());
                    callback(PollNotification::UnusedValueAdded(var_ident, value));
                }
                DiffEvent::RemoveUnused(var_ident) => {
                    let value = self
                        .unused_values
                        .remove(&var_ident)
                        .expect("Values scheduled for removal must be present");
                    callback(PollNotification::UnusedValueRemoved(var_ident, value));
                }
                DiffEvent::VerifyUnused(var_ident, value) => {
                    match self.unused_values.entry(var_ident) {
                        Entry::Occupied(mut occupied) => {
                            if occupied.get() != &value {
                                let old_value = occupied.get().clone();
                                occupied.insert(value.clone());

                                callback(PollNotification::UnusedValueRemoved(
                                    var_ident, old_value,
                                ));
                                callback(PollNotification::UnusedValueAdded(var_ident, value));
                            }
                        }
                        _ => panic!("Values scheduled for verification must be present"),
                    }
                }
                DiffEvent::TransitionUnusedToUsed(var_ident, value) => {
                    let old_value = self
                        .unused_values
                        .remove(&var_ident)
                        .expect("Values scheduled for transition must be present");
                    self.used_values.insert(var_ident, value.clone());

                    callback(PollNotification::UnusedValueRemoved(var_ident, old_value));
                    callback(PollNotification::UsedValueAdded(var_ident, value));
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
