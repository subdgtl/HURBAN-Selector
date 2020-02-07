use std::borrow::Cow;
use std::cmp;
use std::collections::hash_map::{Entry, HashMap};
use std::collections::{BTreeMap, HashSet};
use std::error;
use std::fmt;
use std::ptr;
use std::sync::Arc;
use std::time::Instant;

pub use self::ast::{FuncIdent, VarIdent};
pub use self::func::{
    BooleanParamRefinement, Float2ParamRefinement, Float3ParamRefinement, FloatParamRefinement,
    Func, FuncFlags, FuncInfo, IntParamRefinement, ParamInfo, ParamRefinement,
    StringParamRefinement, UintParamRefinement,
};
pub use self::value::{MeshArrayValue, Ty, Value};

pub mod ast;
pub mod func;
pub mod value;

// FIXME: All of the `Display` impls below for the error types were changed to
// be directly displayable on the UI, e.g. the word "stmt" was changed to
// "input" and `stmt_index` is displayed as `stmt_index + 1`. Revert these impls
// back to being developer centric once we have context-aware error message
// construction mechanism.

/// A name resolution error.
#[derive(Debug, PartialEq)]
pub enum ResolveError {
    VarRedefinition { stmt_index: usize, var: VarIdent },
    UndeclaredVarUse { stmt_index: usize, var: VarIdent },
    UndeclaredFuncUse { stmt_index: usize, func: FuncIdent },
}

impl ResolveError {
    pub fn stmt_index(&self) -> usize {
        match self {
            ResolveError::VarRedefinition { stmt_index, .. } => *stmt_index,
            ResolveError::UndeclaredVarUse { stmt_index, .. } => *stmt_index,
            ResolveError::UndeclaredFuncUse { stmt_index, .. } => *stmt_index,
        }
    }
}

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ResolveError::VarRedefinition { stmt_index, var } => write!(
                f,
                "Re-definition of already declared variable {} on input {}",
                var,
                stmt_index + 1,
            ),
            ResolveError::UndeclaredVarUse { stmt_index, var } => write!(
                f,
                "Use of an undeclared variable {} on input {}",
                var,
                stmt_index + 1,
            ),
            ResolveError::UndeclaredFuncUse { stmt_index, func } => write!(
                f,
                "Use of an undeclared function {} on input {}",
                func,
                stmt_index + 1,
            ),
        }
    }
}

impl error::Error for ResolveError {}

/// A dynamic func error.
#[derive(Debug)]
pub struct FuncError(Box<dyn error::Error + Send>);

impl FuncError {
    pub fn new<E: error::Error + Send + 'static>(error: E) -> Self {
        Self(Box::new(error))
    }
}

impl PartialEq for FuncError {
    /// Compares whether two func errors are exactly the same instance.
    fn eq(&self, other: &FuncError) -> bool {
        // FIXME: @Correctness Can we somehow make this equality deep
        // so we don't have to do downcasting shenanigans when
        // comparing?
        ptr::eq(&self.0, &other.0)
    }
}

impl fmt::Display for FuncError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl error::Error for FuncError {}

/// A runtime error.
#[derive(Debug, PartialEq)]
pub enum RuntimeError {
    ArgCountMismatch {
        stmt_index: usize,
        call: ast::CallExpr,
        args_expected: usize,
        args_provided: usize,
    },
    ArgTyMismatch {
        stmt_index: usize,
        call: ast::CallExpr,
        optional: bool,
        ty_expected: Ty,
        ty_provided: Ty,
    },
    ReturnTyMismatch {
        stmt_index: usize,
        call: ast::CallExpr,
        ty_expected: Ty,
        ty_provided: Ty,
    },
    Func {
        stmt_index: usize,
        call: ast::CallExpr,
        func_error: FuncError,
    },
}

impl RuntimeError {
    pub fn stmt_index(&self) -> usize {
        match self {
            RuntimeError::ArgCountMismatch { stmt_index, .. } => *stmt_index,
            RuntimeError::ArgTyMismatch { stmt_index, .. } => *stmt_index,
            RuntimeError::ReturnTyMismatch { stmt_index, .. } => *stmt_index,
            RuntimeError::Func { stmt_index, .. } => *stmt_index,
        }
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RuntimeError::ArgCountMismatch {
                stmt_index,
                call,
                args_expected,
                args_provided,
            } => write!(
                f,
                "Function {} declared with {} params, but provided with {} args on input {}",
                call.ident(),
                args_expected,
                args_provided,
                stmt_index + 1,
            ),
            RuntimeError::ArgTyMismatch {
                stmt_index,
                call,
                optional,
                ty_expected,
                ty_provided,
            } => write!(
                f,
                "Function {} declared to take param (optional={}) type {}, but given {} on input {}",
                call.ident(),
                optional,
                ty_expected,
                ty_provided,
                stmt_index + 1,
            ),
            RuntimeError::ReturnTyMismatch {
                stmt_index,
                call,
                ty_expected,
                ty_provided,
            } => write!(
                f,
                "Function {} declared to return type {}, but returned {} on input {}",
                call.ident(),
                ty_expected,
                ty_provided,
                stmt_index + 1,
            ),
            RuntimeError::Func {
                stmt_index,
                call,
                func_error,
            } => write!(
                f,
                "Function {} errored with \"{}\" on input {}",
                call.ident(),
                func_error,
                stmt_index + 1,
            ),
        }
    }
}

impl error::Error for RuntimeError {}

/// An interpreter error.
#[derive(Debug, PartialEq)]
pub enum InterpretError {
    Resolve(ResolveError),
    Runtime(RuntimeError),
}

impl InterpretError {
    pub fn stmt_index(&self) -> usize {
        match self {
            InterpretError::Resolve(resolve_error) => resolve_error.stmt_index(),
            InterpretError::Runtime(runtime_error) => runtime_error.stmt_index(),
        }
    }
}

impl fmt::Display for InterpretError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InterpretError::Resolve(resolve_error) => f.write_str(&resolve_error.to_string()),
            InterpretError::Runtime(runtime_error) => f.write_str(&runtime_error.to_string()),
        }
    }
}

impl error::Error for InterpretError {}

impl From<ResolveError> for InterpretError {
    fn from(resolve_error: ResolveError) -> InterpretError {
        InterpretError::Resolve(resolve_error)
    }
}

impl From<RuntimeError> for InterpretError {
    fn from(runtime_error: RuntimeError) -> InterpretError {
        InterpretError::Runtime(runtime_error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogMessageLevel {
    Info,
    #[allow(dead_code)]
    Warn,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogMessage {
    pub level: LogMessageLevel,
    pub message: Cow<'static, str>,
}

impl LogMessage {
    pub fn info<S: Into<Cow<'static, str>>>(message: S) -> Self {
        Self {
            level: LogMessageLevel::Info,
            message: message.into(),
        }
    }

    #[allow(dead_code)]
    pub fn warn<S: Into<Cow<'static, str>>>(message: S) -> Self {
        Self {
            level: LogMessageLevel::Warn,
            message: message.into(),
        }
    }

    pub fn error<S: Into<Cow<'static, str>>>(message: S) -> Self {
        Self {
            level: LogMessageLevel::Error,
            message: message.into(),
        }
    }
}

/// The resulting state of the interpreter after interpreting.
#[derive(Debug, PartialEq)]
pub struct InterpretOutcome {
    /// The result of interpreting. Either a value containing the
    /// state of variables, or the error on which the interpreting
    /// failed.
    pub result: Result<InterpretValue, InterpretError>,

    /// The program counter - index of the statement the interpreter
    /// would execute next.
    pub pc: usize,

    /// The log messages for each statement. The vector has the same
    /// length as the interpreted program.
    pub log_messages: Vec<Vec<LogMessage>>,
}

/// The state of variable values as captured by interpreting up to a
/// certain point in a program.
#[derive(Debug, Clone, PartialEq)]
pub struct InterpretValue {
    /// The value of the last executed statement. `None` if no
    /// statement was executed.
    pub last_value: Option<Value>,

    /// The variable values that were used as parameters to funcs
    /// within the executed part of the program.
    pub used_values: Vec<(VarIdent, Value)>,

    /// The variable values that were not used as parameters to funcs
    /// within the executed part of the program.
    pub unused_values: Vec<(VarIdent, Value)>,
}

#[derive(Debug, Clone)]
struct VarInfo {
    /// The parameters and function call this variable was created
    /// with. Used to verify validity of this variable.
    created_call: ast::CallExpr,

    /// Current value of this variable.
    value: Value,
}

/// Interpreter of a list of statements.
///
/// Allows to edit program and re-execute, trying to preserve computed
/// data where possible. Program changes can be incremental, which
/// allows for context preservation and less needless work. The
/// smaller part of the program is overwritten, the less work needs to
/// be done.
pub struct Interpreter {
    prog: ast::Prog,
    funcs: BTreeMap<FuncIdent, Box<dyn Func>>,

    /// The environment (scope) of the program's memory. Every
    /// variable's value is looked up here.
    env: HashMap<VarIdent, VarInfo>,

    /// The log messages output by functions. The outer vector has the
    /// same length as the program and is indexed by the same
    /// statement index. Log messages are always cleared before
    /// interpreting. This is just to keep the vector warm.
    log_messages: Vec<Vec<LogMessage>>,

    /// The number of changes to the program since the interpreter was
    /// created. Incremented with each program modification.
    epoch: u64,

    /// The last epoch for which name resolution succeeded. Initially
    /// 0, since empty program is by default resolved.
    last_resolve_epoch: u64,
}

impl Interpreter {
    pub fn new(funcs: BTreeMap<FuncIdent, Box<dyn Func>>) -> Self {
        Self {
            prog: ast::Prog::default(),
            funcs,
            env: HashMap::new(),
            log_messages: Vec::new(),
            epoch: 0,
            last_resolve_epoch: 0,
        }
    }

    #[allow(dead_code)]
    pub fn prog(&self) -> &ast::Prog {
        &self.prog
    }

    pub fn set_prog(&mut self, prog: ast::Prog) {
        self.prog = prog;

        self.env.clear();
        self.log_messages
            .resize_with(self.prog.stmts().len(), Vec::new);

        self.epoch += 1;
    }

    pub fn clear_prog(&mut self) {
        self.prog = ast::Prog::default();

        self.env.clear();
        self.log_messages.clear();

        self.epoch += 1;
    }

    pub fn push_prog_stmt(&mut self, stmt: ast::Stmt) {
        self.prog.push_stmt(stmt);
        self.log_messages.push(Vec::new());
        self.epoch += 1;
    }

    pub fn pop_prog_stmt(&mut self) {
        assert!(
            !self.prog.stmts().is_empty(),
            "Program must not be empty when popping"
        );

        self.prog.pop_stmt();
        self.log_messages.pop();
        self.epoch += 1;
    }

    #[allow(dead_code)]
    pub fn prog_stmt_at(&self, index: usize) -> Option<&ast::Stmt> {
        self.prog.stmts().get(index)
    }

    pub fn set_prog_stmt_at(&mut self, index: usize, stmt: ast::Stmt) {
        self.prog.set_stmt_at(index, stmt);
        self.log_messages[index].clear();
        self.epoch += 1;
    }

    /// Runs name resolution on the currently set program.
    ///
    /// More concretely, this statically verifies that:
    /// - Each used function is registered with the interpreter
    /// - Each variable is defined before it is used
    /// - Variables are not re-declared
    pub fn resolve(&mut self) -> Result<(), ResolveError> {
        if self.last_resolve_epoch == self.epoch {
            return Ok(());
        }

        let mut var_scope = HashSet::new();

        for (stmt_index, stmt) in self.prog.stmts().iter().enumerate() {
            match stmt {
                ast::Stmt::VarDecl(var_decl) => {
                    if var_scope.contains(&var_decl.ident()) {
                        return Err(ResolveError::VarRedefinition {
                            stmt_index,
                            var: var_decl.ident(),
                        });
                    }

                    let func = var_decl.init_expr().ident();
                    if self.funcs.get(&func).is_none() {
                        return Err(ResolveError::UndeclaredFuncUse { stmt_index, func });
                    }

                    for arg in var_decl.init_expr().args() {
                        if let ast::Expr::Var(var) = arg {
                            if !var_scope.contains(&var.ident()) {
                                return Err(ResolveError::UndeclaredVarUse {
                                    stmt_index,
                                    var: var.ident(),
                                });
                            }
                        }
                    }

                    var_scope.insert(var_decl.ident());
                }
            }
        }

        // Mark current epoch as name-resolved.
        self.last_resolve_epoch = self.epoch;

        Ok(())
    }

    /// Interprets the whole currently set program and returns the
    /// used/unused values after the last statement.
    pub fn interpret(&mut self) -> InterpretOutcome {
        self.interpret_up_until(self.prog.stmts().len().saturating_sub(1))
    }

    /// Interprets the currently set program up until the `index`-th
    /// statement (inclusive) and returns the used/unused values after
    /// it.
    ///
    /// If the program does not contain enough statements, interprets
    /// the program up until the end.
    pub fn interpret_up_until(&mut self, mut index: usize) -> InterpretOutcome {
        if self.prog.stmts().is_empty() {
            return InterpretOutcome {
                result: Ok(InterpretValue {
                    last_value: None,
                    used_values: Vec::new(),
                    unused_values: Vec::new(),
                }),
                pc: 0,
                log_messages: vec![Vec::new(); self.log_messages.len()],
            };
        }

        if let Err(err) = self.resolve() {
            return InterpretOutcome {
                result: Err(InterpretError::from(err)),
                pc: 0,
                log_messages: vec![Vec::new(); self.log_messages.len()],
            };
        }

        // FIXME: @Diagnostics Implement type-checking

        index = cmp::min(index, self.prog.stmts().len().saturating_sub(1));

        self.invalidate();
        for log_messages in &mut self.log_messages {
            log_messages.clear();
        }

        log::debug!("Starting program evaluation with PC: 0");

        for (stmt_index, stmt) in self.prog.stmts()[0..=index].iter().enumerate() {
            if let Err(err) = eval_stmt(
                stmt_index,
                stmt,
                &mut self.funcs,
                &mut self.env,
                &mut self.log_messages,
            ) {
                return InterpretOutcome {
                    result: Err(InterpretError::from(err)),
                    pc: stmt_index + 1,
                    log_messages: self.log_messages.clone(),
                };
            }
        }

        log::debug!("Ended program evaluation with PC: {}", index + 1);

        let unused_vars = self.compute_unused_vars_up_until(index);
        let last_value = match &self.prog.stmts()[index] {
            ast::Stmt::VarDecl(var_decl) => {
                let var_ident = var_decl.ident();
                let var_info = self
                    .env
                    .get(&var_ident)
                    .expect("Value must have been populated or already cached");

                var_info.value.clone()
            }
        };

        let mut used_values = Vec::with_capacity(index + 1);
        let mut unused_values = Vec::with_capacity(index + 1);

        for stmt in &self.prog.stmts()[0..=index] {
            match stmt {
                ast::Stmt::VarDecl(var_decl) => {
                    let var_ident = var_decl.ident();
                    let var_info = self
                        .env
                        .get(&var_ident)
                        .expect("Value must have been populated or already cached");

                    if unused_vars.contains(&var_ident) {
                        unused_values.push((var_ident, var_info.value.clone()));
                    } else {
                        used_values.push((var_ident, var_info.value.clone()));
                    }
                }
            }
        }

        InterpretOutcome {
            result: Ok(InterpretValue {
                last_value: Some(last_value),
                used_values,
                unused_values,
            }),
            pc: index + 1,
            log_messages: self.log_messages.clone(),
        }
    }

    /// Computes a set of variable identifiers that would be unused,
    /// if the current program were only interpreted up to index-th
    /// statement.
    ///
    /// Iterates over all variable declaration statements and inserts
    /// an unused variable for each. If the variable identifier is
    /// later referenced by a variable expression, it is removed from
    /// the unused set.
    fn compute_unused_vars_up_until(&self, index: usize) -> HashSet<ast::VarIdent> {
        let mut unused_vars = HashSet::new();

        for stmt in &self.prog.stmts()[0..=index] {
            match stmt {
                ast::Stmt::VarDecl(var_decl) => {
                    let init_expr = var_decl.init_expr();
                    for arg in init_expr.args() {
                        if let ast::Expr::Var(var) = arg {
                            unused_vars.remove(&var.ident());
                        }
                    }

                    unused_vars.insert(var_decl.ident());
                }
            }
        }

        unused_vars
    }

    /// Invalidates variables in the environment.
    ///
    /// Verify all variables we have computed already, invalidating
    /// all that could have possibly changed since last execution.
    /// Invalidated variables are simply removed from the environment
    /// and will be re-computed a-fresh during evaluation. We count on
    /// the fact that any dependency must come before its dependents
    /// in the examined statements due to the serialized nature of the
    /// program.
    ///
    /// There are 3 types of variable invalidation:
    ///
    /// 1) Impurity invalidation: the function producing the variable
    ///    is not pure (import, random, etc.)
    /// 2) Definition invalidation: the call expression definition has
    ///    changed (either the function or the parameters),
    /// 3) Dependency invalidation: variables referenced in the
    ///    parameters have have been invalidated.
    fn invalidate(&mut self) {
        // FIXME: We'd like to have this return an execution plan so
        // that we don't necessarily try to execute stmts only to find
        // that we already have the results in cache.

        // FIXME: This is still very pessimistic, we should support an
        // incremental computation model with fact verification a-lá
        // salsa. https://github.com/salsa-rs/salsa

        for stmt in self.prog.stmts() {
            match stmt {
                ast::Stmt::VarDecl(var_decl) => {
                    let var_ident = var_decl.ident();
                    let init_expr = var_decl.init_expr();
                    let func_ident = init_expr.ident();

                    // Perform 1) Impurity invalidation

                    if !self.funcs[&func_ident].flags().contains(FuncFlags::PURE) {
                        log::debug!("Performing impurity invalidation of {}", var_ident);
                        self.env.remove(&var_ident);

                        continue;
                    }

                    // Perform 2) Definition invalidation

                    if let Entry::Occupied(occupied) = self.env.entry(var_ident) {
                        let var_info = occupied.get();
                        let created_call = &var_info.created_call;

                        if created_call != init_expr {
                            log::debug!("Performing definition invalidation of {}", var_ident);
                            occupied.remove_entry();

                            continue;
                        }
                    }

                    // Perform 3) Dependency invalidation

                    for expr in var_decl.init_expr().args() {
                        if let ast::Expr::Var(var) = expr {
                            if !self.env.contains_key(&var.ident()) {
                                log::debug!("Performing dependency invalidation of {}", var_ident);
                                self.env.remove(&var_ident);

                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

fn eval_stmt(
    stmt_index: usize,
    stmt: &ast::Stmt,
    funcs: &mut BTreeMap<FuncIdent, Box<dyn Func>>,
    env: &mut HashMap<VarIdent, VarInfo>,
    log_messages: &mut [Vec<LogMessage>],
) -> Result<(), RuntimeError> {
    let time_start = Instant::now();
    log::debug!("Evaluating stmt {}: {}", stmt_index, stmt);

    let result = match stmt {
        ast::Stmt::VarDecl(var_decl) => {
            eval_var_decl_stmt(stmt_index, var_decl, funcs, env, &mut |message| {
                log_messages[stmt_index].push(message);
            })
        }
    };

    let elapsed_ms = time_start.elapsed().as_secs_f32() * 1000.0;
    log::debug!("Evaluation of stmt {} took {:.2}ms", stmt_index, elapsed_ms);

    match result {
        Ok(cached) => {
            if cached {
                log_messages[stmt_index].push(LogMessage::info(format!(
                    ">>> Taken from cache ({:.2}ms)",
                    elapsed_ms,
                )));
            } else {
                log_messages[stmt_index]
                    .push(LogMessage::info(format!(">>> Took {:.2}ms", elapsed_ms)));
            }

            Ok(())
        }
        Err(err) => {
            log_messages[stmt_index].push(LogMessage::error(format!(
                ">>> Errored after {:.2}ms",
                elapsed_ms,
            )));

            Err(err)
        }
    }
}

fn eval_var_decl_stmt(
    stmt_index: usize,
    var_decl: &ast::VarDeclStmt,
    funcs: &mut BTreeMap<FuncIdent, Box<dyn Func>>,
    env: &mut HashMap<VarIdent, VarInfo>,
    log: &mut dyn FnMut(LogMessage),
) -> Result<bool, RuntimeError> {
    let var_ident = var_decl.ident();

    // This is a false positive. Bad Clippy, bad! Rewriting the code
    // to use the entry API would fail borrowchecking (and cause
    // pointer invalidation if it didn't!). The entry would create a
    // long-lived mutable borrow which simply can't be live when we
    // call into `eval_call_expr`, which borrows again. Note that
    // having `eval_call_expr` guarded by the map access is the whole
    // point here.

    // FIXME: remove allow after
    // https://github.com/rust-lang/rust-clippy/issues/4674 is
    // resolved

    #[allow(clippy::map_entry)]
    {
        if env.contains_key(&var_ident) {
            Ok(true)
        } else {
            let init_expr = var_decl.init_expr();
            let value = eval_call_expr(stmt_index, init_expr, funcs, env, log)?;

            env.insert(
                var_ident,
                VarInfo {
                    created_call: init_expr.clone(),
                    value,
                },
            );

            Ok(false)
        }
    }
}

fn eval_expr(
    expr: &ast::Expr,
    env: &mut HashMap<VarIdent, VarInfo>,
) -> Result<Value, RuntimeError> {
    match expr {
        ast::Expr::Lit(lit) => eval_lit_expr(lit),
        ast::Expr::Var(var) => eval_var_expr(var, env),
    }
}

fn eval_lit_expr(lit: &ast::LitExpr) -> Result<Value, RuntimeError> {
    let value = match lit {
        ast::LitExpr::Boolean(boolean) => Value::Boolean(*boolean),
        ast::LitExpr::Int(int) => Value::Int(*int),
        ast::LitExpr::Uint(uint) => Value::Uint(*uint),
        ast::LitExpr::Float(float) => Value::Float(*float),
        ast::LitExpr::Float2(float2) => Value::Float2(*float2),
        ast::LitExpr::Float3(float3) => Value::Float3(*float3),
        ast::LitExpr::String(string) => Value::String(Arc::clone(&string)),
        ast::LitExpr::Nil => Value::Nil,
    };

    Ok(value)
}

fn eval_var_expr(
    var: &ast::VarExpr,
    env: &mut HashMap<VarIdent, VarInfo>,
) -> Result<Value, RuntimeError> {
    let var_ident = var.ident();
    let var_info = &env[&var_ident];

    Ok(var_info.value.clone())
}

fn eval_call_expr(
    stmt_index: usize,
    call: &ast::CallExpr,
    funcs: &mut BTreeMap<FuncIdent, Box<dyn Func>>,
    env: &mut HashMap<VarIdent, VarInfo>,
    log: &mut dyn FnMut(LogMessage),
) -> Result<Value, RuntimeError> {
    // FIXME: @Diagnostics use the func name and the param names in
    // the reported errors

    let func = funcs.get_mut(&call.ident()).expect("Failed to find func");

    let arg_exprs = call.args();
    if func.param_info().len() != arg_exprs.len() {
        return Err(RuntimeError::ArgCountMismatch {
            stmt_index,
            call: call.clone(),
            args_expected: func.param_info().len(),
            args_provided: arg_exprs.len(),
        });
    }

    let mut args = Vec::with_capacity(arg_exprs.len());
    for arg_expr in arg_exprs {
        let arg = eval_expr(arg_expr, env)?;
        args.push(arg);
    }

    for (info, value) in func.param_info().iter().zip(args.iter()) {
        let param_ty = info.refinement.ty();
        let value_ty = value.ty();

        if param_ty != value_ty {
            // Nil is an acceptable value for parameters marked optional
            if value_ty == Ty::Nil && info.optional {
                continue;
            }

            return Err(RuntimeError::ArgTyMismatch {
                stmt_index,
                call: call.clone(),
                optional: info.optional,
                ty_expected: param_ty,
                ty_provided: value_ty,
            });
        }
    }

    match func.call(&args, log) {
        Ok(value) => {
            let return_ty = func.return_ty();
            let value_ty = value.ty();

            if return_ty != value_ty {
                return Err(RuntimeError::ReturnTyMismatch {
                    stmt_index,
                    call: call.clone(),
                    ty_expected: return_ty,
                    ty_provided: value_ty,
                });
            }

            Ok(value)
        }
        Err(func_error) => Err(RuntimeError::Func {
            stmt_index,
            call: call.clone(),
            func_error,
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use super::*;

    fn param_info(ty: Ty, optional: bool) -> ParamInfo {
        ParamInfo {
            name: "<anonymous>",
            refinement: match ty {
                Ty::Nil => panic!("Yeah, sure I can do that!"),
                Ty::Boolean => ParamRefinement::Boolean(BooleanParamRefinement::default()),
                Ty::Int => ParamRefinement::Int(IntParamRefinement::default()),
                Ty::Uint => ParamRefinement::Uint(UintParamRefinement::default()),
                Ty::Float => ParamRefinement::Float(FloatParamRefinement::default()),
                Ty::Float2 => ParamRefinement::Float2(Float2ParamRefinement::default()),
                Ty::Float3 => ParamRefinement::Float3(Float3ParamRefinement::default()),
                Ty::String => ParamRefinement::String(StringParamRefinement::default()),
                Ty::Mesh => ParamRefinement::Mesh,
                Ty::MeshArray => ParamRefinement::MeshArray,
            },
            optional,
        }
    }

    struct TestFunc<F>
    where
        F: Fn(&[Value]) -> Result<Value, FuncError>,
    {
        func: F,
        flags: FuncFlags,
        param_info: Vec<ParamInfo>,
        return_ty: Ty,
    }

    impl<F> TestFunc<F>
    where
        F: Fn(&[Value]) -> Result<Value, FuncError>,
    {
        pub fn new(func: F, flags: FuncFlags, param_info: Vec<ParamInfo>, return_ty: Ty) -> Self {
            Self {
                flags,
                func,
                param_info,
                return_ty,
            }
        }
    }

    impl<F> Func for TestFunc<F>
    where
        F: Fn(&[Value]) -> Result<Value, FuncError>,
    {
        fn flags(&self) -> FuncFlags {
            self.flags
        }

        fn param_info(&self) -> &[ParamInfo] {
            &self.param_info
        }

        fn return_ty(&self) -> Ty {
            self.return_ty
        }

        fn call(
            &mut self,
            values: &[Value],
            _log: &mut dyn FnMut(LogMessage),
        ) -> Result<Value, FuncError> {
            (self.func)(values)
        }
    }

    struct CallCount {
        value: Cell<u64>,
    }

    impl CallCount {
        pub fn new() -> Self {
            Self {
                value: Cell::new(0),
            }
        }

        pub fn inc(&self) {
            let current = self.value.get();
            self.value.set(current + 1);
        }

        pub fn get(&self) -> u64 {
            self.value.get()
        }
    }

    // Basic tests a.k.a. does it even run?

    #[test]
    fn test_interpreter_interpret_single_func_parameterless_pure() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |_| Ok(Value::Boolean(true)),
                FuncFlags::PURE,
                vec![],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            ast::CallExpr::new(func_id, vec![]),
        ))]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().result.unwrap();
        assert_eq!(value.last_value, Some(Value::Boolean(true)));
    }

    #[test]
    fn test_interpreter_interpret_single_func_parametrized_pure() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |values| Ok(Value::Boolean(values[0].unwrap_boolean())),
                FuncFlags::PURE,
                vec![param_info(Ty::Boolean, false)],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            ast::CallExpr::new(func_id, vec![ast::Expr::Lit(ast::LitExpr::Boolean(true))]),
        ))]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().result.unwrap();
        assert_eq!(value.last_value, Some(Value::Boolean(true)));
    }

    #[test]
    fn test_interpreter_interpret_single_func_parameterless_impure() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |_| Ok(Value::Boolean(true)),
                FuncFlags::empty(),
                vec![],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            ast::CallExpr::new(func_id, vec![]),
        ))]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().result.unwrap();
        assert_eq!(value.last_value, Some(Value::Boolean(true)));
    }

    #[test]
    fn test_interpreter_interpret_single_func_parametrized_impure() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |values| Ok(Value::Boolean(values[0].unwrap_boolean())),
                FuncFlags::empty(),
                vec![param_info(Ty::Boolean, false)],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            ast::CallExpr::new(func_id, vec![ast::Expr::Lit(ast::LitExpr::Boolean(true))]),
        ))]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().result.unwrap();
        assert_eq!(value.last_value, Some(Value::Boolean(true)));
    }

    #[test]
    fn test_interpreter_interpret_func_chain_with_pure_param() {
        let (func_id1, func1) = (
            FuncIdent(0),
            TestFunc::new(
                |_| Ok(Value::Boolean(true)),
                FuncFlags::PURE,
                vec![],
                Ty::Boolean,
            ),
        );
        let (func_id2, func2) = (
            FuncIdent(1),
            TestFunc::new(
                |values| Ok(Value::Boolean(values[0].unwrap_boolean())),
                FuncFlags::PURE,
                vec![param_info(Ty::Boolean, false)],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                VarIdent(0),
                ast::CallExpr::new(func_id1, vec![]),
            )),
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                VarIdent(1),
                ast::CallExpr::new(
                    func_id2,
                    vec![ast::Expr::Var(ast::VarExpr::new(VarIdent(0)))],
                ),
            )),
        ]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id1, Box::new(func1));
        funcs.insert(func_id2, Box::new(func2));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().result.unwrap();
        assert_eq!(value.last_value, Some(Value::Boolean(true)));
    }

    #[test]
    fn test_interpreter_interpret_func_chain_with_impure_param() {
        let (func_id1, func1) = (
            FuncIdent(0),
            TestFunc::new(
                |_| Ok(Value::Boolean(true)),
                FuncFlags::empty(),
                vec![],
                Ty::Boolean,
            ),
        );
        let (func_id2, func2) = (
            FuncIdent(1),
            TestFunc::new(
                |values| Ok(Value::Boolean(values[0].unwrap_boolean())),
                FuncFlags::empty(),
                vec![param_info(Ty::Boolean, false)],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                VarIdent(0),
                ast::CallExpr::new(func_id1, vec![]),
            )),
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                VarIdent(1),
                ast::CallExpr::new(
                    func_id2,
                    vec![ast::Expr::Var(ast::VarExpr::new(VarIdent(0)))],
                ),
            )),
        ]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id1, Box::new(func1));
        funcs.insert(func_id2, Box::new(func2));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().result.unwrap();
        assert_eq!(value.last_value, Some(Value::Boolean(true)));
    }

    // Prog index

    #[test]
    fn test_interpreter_interpret_empty_prog() {
        let prog = ast::Prog::new(vec![]);

        let mut interpreter = Interpreter::new(BTreeMap::new());
        interpreter.set_prog(prog);

        let interpret_outcome = interpreter.interpret();
        assert_eq!(
            interpret_outcome,
            InterpretOutcome {
                result: Ok(InterpretValue {
                    last_value: None,
                    used_values: Vec::new(),
                    unused_values: Vec::new(),
                }),
                pc: 0,
                log_messages: Vec::new(),
            },
        );
    }

    #[test]
    fn test_interpreter_interpret_up_until_empty_prog() {
        let prog = ast::Prog::new(vec![]);

        let mut interpreter = Interpreter::new(BTreeMap::new());
        interpreter.set_prog(prog);

        let interpret_outcome = interpreter.interpret_up_until(0);
        assert_eq!(
            interpret_outcome,
            InterpretOutcome {
                result: Ok(InterpretValue {
                    last_value: None,
                    used_values: Vec::new(),
                    unused_values: Vec::new(),
                }),
                pc: 0,
                log_messages: Vec::new(),
            },
        );
    }

    #[test]
    fn test_interpreter_interpret_up_until_invalid_index() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |_| Ok(Value::Boolean(true)),
                FuncFlags::PURE,
                vec![],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            ast::CallExpr::new(func_id, vec![]),
        ))]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let interpret_outcome = interpreter.interpret_up_until(1);
        let value = interpret_outcome.result.unwrap();
        assert_eq!(value.last_value, Some(Value::Boolean(true)));
        // The PC must have stayed pointed at 1st stmt, even though it
        // would normally point to 2nd, because we executed less
        // statements than what was requested.
        assert_eq!(interpret_outcome.pc, 1);
    }

    #[test]
    fn test_interpreter_interpret_up_until_valid_index() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |_| Ok(Value::Boolean(true)),
                FuncFlags::PURE,
                vec![],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                VarIdent(0),
                ast::CallExpr::new(func_id, vec![]),
            )),
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                VarIdent(1),
                ast::CallExpr::new(func_id, vec![]),
            )),
        ]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret_up_until(0).result.unwrap();
        assert_eq!(value.last_value, Some(Value::Boolean(true)));
    }

    // Var invalidation tests

    #[test]
    fn test_interpreter_interpret_single_func_pure_caches_results() {
        let n_calls = Rc::new(CallCount::new());
        let c = Rc::clone(&n_calls);

        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                move |values| {
                    c.inc();
                    Ok(Value::Boolean(values[0].unwrap_boolean()))
                },
                FuncFlags::PURE,
                vec![param_info(Ty::Boolean, false)],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(1),
            ast::CallExpr::new(func_id, vec![ast::Expr::Lit(ast::LitExpr::Boolean(true))]),
        ))]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().result.unwrap();
        assert_eq!(value.last_value, Some(Value::Boolean(true)));

        let value = interpreter.interpret().result.unwrap();
        assert_eq!(value.last_value, Some(Value::Boolean(true)));
        assert_eq!(n_calls.get(), 1);
    }

    #[test]
    fn test_interpreter_interpret_single_func_impurity_invalidation() {
        let n_calls = Rc::new(CallCount::new());
        let c = Rc::clone(&n_calls);

        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                move |values| {
                    c.inc();
                    Ok(Value::Boolean(values[0].unwrap_boolean()))
                },
                FuncFlags::empty(),
                vec![param_info(Ty::Boolean, false)],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(1),
            ast::CallExpr::new(func_id, vec![ast::Expr::Lit(ast::LitExpr::Boolean(true))]),
        ))]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().result.unwrap();
        assert_eq!(value.last_value, Some(Value::Boolean(true)));

        let value = interpreter.interpret().result.unwrap();
        assert_eq!(value.last_value, Some(Value::Boolean(true)));
        assert_eq!(n_calls.get(), 2);
    }

    #[test]
    fn test_interpreter_interpret_single_func_definition_invalidation_with_changed_args() {
        let n_calls = Rc::new(CallCount::new());
        let c = Rc::clone(&n_calls);

        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                move |values| {
                    c.inc();
                    let value = values[0].unwrap_boolean();
                    let negate = values[1].unwrap_boolean();
                    if negate {
                        Ok(Value::Boolean(!value))
                    } else {
                        Ok(Value::Boolean(value))
                    }
                },
                FuncFlags::PURE,
                vec![
                    param_info(Ty::Boolean, false),
                    param_info(Ty::Boolean, false),
                ],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            ast::CallExpr::new(
                func_id,
                vec![
                    ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                    ast::Expr::Lit(ast::LitExpr::Boolean(false)),
                ],
            ),
        ))]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().result.unwrap();
        assert_eq!(value.last_value, Some(Value::Boolean(true)));

        // Change the args but not the func
        interpreter.set_prog_stmt_at(
            0,
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                VarIdent(0),
                ast::CallExpr::new(
                    func_id,
                    vec![
                        ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                        ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                    ],
                ),
            )),
        );

        let value = interpreter.interpret().result.unwrap();
        assert_eq!(value.last_value, Some(Value::Boolean(false)));
        assert_eq!(n_calls.get(), 2);
    }

    #[test]
    fn test_interpreter_interpret_single_func_definition_invalidation_with_changed_func() {
        let n_calls1 = Rc::new(CallCount::new());
        let n_calls2 = Rc::new(CallCount::new());
        let c1 = Rc::clone(&n_calls1);
        let c2 = Rc::clone(&n_calls2);

        let (func_id1, func1) = (
            FuncIdent(0),
            TestFunc::new(
                move |values| {
                    c1.inc();
                    let value = values[0].unwrap_boolean();
                    let negate = values[1].unwrap_boolean();
                    if negate {
                        Ok(Value::Boolean(!value))
                    } else {
                        Ok(Value::Boolean(value))
                    }
                },
                FuncFlags::PURE,
                vec![
                    param_info(Ty::Boolean, false),
                    param_info(Ty::Boolean, false),
                ],
                Ty::Boolean,
            ),
        );

        let (func_id2, func2) = (
            FuncIdent(1),
            TestFunc::new(
                move |values| {
                    c2.inc();
                    let value = values[0].unwrap_boolean();
                    let negate = values[1].unwrap_boolean();
                    if negate {
                        Ok(Value::Boolean(!value))
                    } else {
                        Ok(Value::Boolean(value))
                    }
                },
                FuncFlags::PURE,
                vec![
                    param_info(Ty::Boolean, false),
                    param_info(Ty::Boolean, false),
                ],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            ast::CallExpr::new(
                func_id1,
                vec![
                    ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                    ast::Expr::Lit(ast::LitExpr::Boolean(false)),
                ],
            ),
        ))]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id1, Box::new(func1));
        funcs.insert(func_id2, Box::new(func2));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().result.unwrap();
        assert_eq!(value.last_value, Some(Value::Boolean(true)));

        // Change the func but not the args
        interpreter.set_prog_stmt_at(
            0,
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                VarIdent(0),
                ast::CallExpr::new(
                    func_id2,
                    vec![
                        ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                        ast::Expr::Lit(ast::LitExpr::Boolean(false)),
                    ],
                ),
            )),
        );

        let value = interpreter.interpret().result.unwrap();
        assert_eq!(value.last_value, Some(Value::Boolean(true)));

        assert_eq!(n_calls1.get(), 1);
        assert_eq!(n_calls2.get(), 1);
    }

    #[test]
    fn test_interpreter_interpret_func_chain_dependency_invalidation() {
        let n_calls1 = Rc::new(CallCount::new());
        let n_calls2 = Rc::new(CallCount::new());
        let c1 = Rc::clone(&n_calls1);
        let c2 = Rc::clone(&n_calls2);

        let (func_id1, func1) = (
            FuncIdent(0),
            TestFunc::new(
                move |_| {
                    c1.inc();
                    Ok(Value::Boolean(true))
                },
                FuncFlags::empty(),
                vec![],
                Ty::Boolean,
            ),
        );

        let (func_id2, func2) = (
            FuncIdent(1),
            TestFunc::new(
                move |values| {
                    c2.inc();
                    Ok(Value::Boolean(values[0].unwrap_boolean()))
                },
                FuncFlags::PURE,
                vec![param_info(Ty::Boolean, false)],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                VarIdent(0),
                ast::CallExpr::new(func_id1, vec![]),
            )),
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                VarIdent(1),
                ast::CallExpr::new(
                    func_id2,
                    vec![ast::Expr::Var(ast::VarExpr::new(VarIdent(0)))],
                ),
            )),
        ]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id1, Box::new(func1));
        funcs.insert(func_id2, Box::new(func2));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().result.unwrap();
        assert_eq!(value.last_value, Some(Value::Boolean(true)));

        let value = interpreter.interpret().result.unwrap();
        assert_eq!(value.last_value, Some(Value::Boolean(true)));

        assert_eq!(n_calls1.get(), 2);
        assert_eq!(n_calls2.get(), 2);
    }

    // FIXME: Prog manipulation tests

    // Name resolution tests

    #[test]
    fn test_interpreter_interpret_var_redefinition() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |_| Ok(Value::Boolean(true)),
                FuncFlags::PURE,
                vec![],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                VarIdent(0),
                ast::CallExpr::new(func_id, vec![]),
            )),
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                VarIdent(0),
                ast::CallExpr::new(func_id, vec![]),
            )),
        ]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().result.unwrap_err();
        assert_eq!(
            err,
            InterpretError::from(ResolveError::VarRedefinition {
                stmt_index: 1,
                var: VarIdent(0),
            }),
        );
    }

    #[test]
    fn test_interpreter_interpret_undeclared_var() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |values| Ok(Value::Boolean(values[0].unwrap_boolean())),
                FuncFlags::PURE,
                vec![param_info(Ty::Boolean, false)],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            ast::CallExpr::new(
                func_id,
                vec![ast::Expr::Var(ast::VarExpr::new(VarIdent(42)))],
            ),
        ))]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().result.unwrap_err();
        assert_eq!(
            err,
            InterpretError::from(ResolveError::UndeclaredVarUse {
                stmt_index: 0,
                var: VarIdent(42),
            }),
        );
    }

    #[test]
    fn test_interpreter_interpret_undeclared_var_defined_by_self() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |values| Ok(Value::Boolean(values[0].unwrap_boolean())),
                FuncFlags::PURE,
                vec![param_info(Ty::Boolean, false)],
                Ty::Boolean,
            ),
        );

        // The stmt uses the same var it would only just define - it
        // is not defined yet though.
        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            ast::CallExpr::new(
                func_id,
                vec![ast::Expr::Var(ast::VarExpr::new(VarIdent(0)))],
            ),
        ))]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().result.unwrap_err();
        assert_eq!(
            err,
            InterpretError::from(ResolveError::UndeclaredVarUse {
                stmt_index: 0,
                var: VarIdent(0),
            }),
        );
    }

    #[test]
    fn test_interpreter_interpret_undeclared_var_defined_in_future() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |values| Ok(Value::Boolean(values[0].unwrap_boolean())),
                FuncFlags::PURE,
                vec![param_info(Ty::Boolean, false)],
                Ty::Boolean,
            ),
        );

        // Even if a var is defined later in the program, it must not
        // be used before it's defining stmt
        let prog = ast::Prog::new(vec![
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                VarIdent(0),
                ast::CallExpr::new(
                    func_id,
                    vec![ast::Expr::Var(ast::VarExpr::new(VarIdent(1)))],
                ),
            )),
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                VarIdent(1),
                ast::CallExpr::new(func_id, vec![ast::Expr::Lit(ast::LitExpr::Boolean(true))]),
            )),
        ]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().result.unwrap_err();
        assert_eq!(
            err,
            InterpretError::from(ResolveError::UndeclaredVarUse {
                stmt_index: 0,
                var: VarIdent(1),
            }),
        );
    }

    #[test]
    fn test_interpreter_interpret_undeclared_func() {
        // This program uses an undeclared function
        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            ast::CallExpr::new(FuncIdent(0), vec![]),
        ))]);

        let mut interpreter = Interpreter::new(BTreeMap::new());
        interpreter.set_prog(prog);

        let err = interpreter.interpret().result.unwrap_err();
        assert_eq!(
            err,
            InterpretError::from(ResolveError::UndeclaredFuncUse {
                stmt_index: 0,
                func: FuncIdent(0),
            }),
        );
    }

    // FIXME: Static typecheck tests

    // Dynamic typechecks tests

    #[test]
    fn test_interpreter_interpret_single_func_dynamic_arg_count_error() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |values| Ok(Value::Float(values[0].unwrap_float() + 1.0)),
                FuncFlags::PURE,
                vec![param_info(Ty::Float, false)],
                Ty::Float,
            ),
        );

        let call = ast::CallExpr::new(
            func_id,
            vec![
                ast::Expr::Lit(ast::LitExpr::Float(1.0)),
                ast::Expr::Lit(ast::LitExpr::Float(0.0)),
            ],
        );
        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            call.clone(),
        ))]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().result.unwrap_err();
        assert_eq!(
            err,
            InterpretError::from(RuntimeError::ArgCountMismatch {
                stmt_index: 0,
                call,
                args_expected: 1,
                args_provided: 2,
            }),
        );
    }

    #[test]
    fn test_interpreter_interpret_single_func_dynamic_arg_ty_error() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |values| Ok(Value::Float(values[0].unwrap_float() + 1.0)),
                FuncFlags::PURE,
                vec![param_info(Ty::Float, false)],
                Ty::Float,
            ),
        );

        let call = ast::CallExpr::new(func_id, vec![ast::Expr::Lit(ast::LitExpr::Int(1))]);
        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            call.clone(),
        ))]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().result.unwrap_err();
        assert_eq!(
            err,
            InterpretError::from(RuntimeError::ArgTyMismatch {
                stmt_index: 0,
                call,
                optional: false,
                ty_expected: Ty::Float,
                ty_provided: Ty::Int,
            }),
        );
    }

    #[test]
    fn test_interpreter_interpret_single_func_dynamic_optional_arg_ty() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |values| match &values[0] {
                    Value::Float(float_value) => Ok(Value::Float(*float_value)),
                    _ => Ok(Value::Float(1.0)),
                },
                FuncFlags::PURE,
                vec![param_info(Ty::Float, true)],
                Ty::Float,
            ),
        );

        let call = ast::CallExpr::new(func_id, vec![ast::Expr::Lit(ast::LitExpr::Nil)]);
        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            call,
        ))]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().result.unwrap();
        assert_eq!(value.last_value, Some(Value::Float(1.0)));
    }

    #[test]
    fn test_interpreter_run_single_func_dynamic_optional_arg_ty_error() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |values| match &values[0] {
                    Value::Float(float_value) => Ok(Value::Float(*float_value)),
                    _ => Ok(Value::Float(1.0)),
                },
                FuncFlags::PURE,
                vec![param_info(Ty::Float, true)],
                Ty::Float,
            ),
        );

        let call = ast::CallExpr::new(func_id, vec![ast::Expr::Lit(ast::LitExpr::Int(1))]);
        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            call.clone(),
        ))]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().result.unwrap_err();
        assert_eq!(
            err,
            InterpretError::from(RuntimeError::ArgTyMismatch {
                stmt_index: 0,
                call,
                optional: true,
                ty_expected: Ty::Float,
                ty_provided: Ty::Int,
            }),
        );
    }

    #[test]
    fn test_interpreter_interpret_single_func_dynamic_return_ty_error() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(|_| Ok(Value::Int(-1)), FuncFlags::PURE, vec![], Ty::Float),
        );

        let call = ast::CallExpr::new(func_id, vec![]);
        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            call.clone(),
        ))]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().result.unwrap_err();
        assert_eq!(
            err,
            InterpretError::from(RuntimeError::ReturnTyMismatch {
                stmt_index: 0,
                call,
                ty_expected: Ty::Float,
                ty_provided: Ty::Int,
            }),
        );
    }

    // Func runtime errors tests

    #[test]
    fn test_interpreter_interpret_single_func_runtime_error() {
        #[derive(Debug, PartialEq)]
        struct ConcreteFuncError(i32);

        impl fmt::Display for ConcreteFuncError {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "Concrete func error with code {}", self.0)
            }
        }

        impl error::Error for ConcreteFuncError {}

        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |_| Err(FuncError::new(ConcreteFuncError(42))),
                FuncFlags::empty(),
                vec![],
                Ty::Boolean,
            ),
        );

        let call = ast::CallExpr::new(func_id, vec![]);
        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            call.clone(),
        ))]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().result.unwrap_err();

        match err {
            InterpretError::Runtime(RuntimeError::Func {
                stmt_index: runtime_error_stmt_index,
                call: runtime_error_call,
                func_error: runtime_error_func_error,
            }) => {
                assert_eq!(runtime_error_stmt_index, 0);
                assert_eq!(runtime_error_call, call);

                let concrete_error = runtime_error_func_error
                    .0
                    .downcast_ref::<ConcreteFuncError>()
                    .unwrap();
                assert_eq!(concrete_error, &ConcreteFuncError(42));
            }
            _ => panic!(),
        }
    }

    // InterpretOutcome tests

    #[test]
    fn test_interpreter_interpret_outcome() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |values| Ok(Value::Float(values[0].unwrap_float() * 2.0)),
                FuncFlags::PURE,
                vec![param_info(Ty::Float, true)],
                Ty::Float,
            ),
        );

        let prog = ast::Prog::new(vec![
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                VarIdent(0),
                ast::CallExpr::new(func_id, vec![ast::Expr::Lit(ast::LitExpr::Float(1.0))]),
            )),
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                VarIdent(1),
                ast::CallExpr::new(
                    func_id,
                    vec![ast::Expr::Var(ast::VarExpr::new(VarIdent(0)))],
                ),
            )),
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                VarIdent(2),
                ast::CallExpr::new(
                    func_id,
                    vec![ast::Expr::Var(ast::VarExpr::new(VarIdent(1)))],
                ),
            )),
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                VarIdent(3),
                ast::CallExpr::new(
                    func_id,
                    vec![ast::Expr::Var(ast::VarExpr::new(VarIdent(1)))],
                ),
            )),
        ]);

        let mut funcs: BTreeMap<FuncIdent, Box<dyn Func>> = BTreeMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let interpret_outcome = interpreter.interpret();
        assert_eq!(
            interpret_outcome.result,
            Ok(InterpretValue {
                last_value: Some(Value::Float(8.0)),
                used_values: vec![
                    (VarIdent(0), Value::Float(2.0)),
                    (VarIdent(1), Value::Float(4.0)),
                ],
                unused_values: vec![
                    (VarIdent(2), Value::Float(8.0)),
                    (VarIdent(3), Value::Float(8.0)),
                ],
            }),
        );
        assert_eq!(interpret_outcome.pc, 4);
        assert_eq!(interpret_outcome.log_messages.len(), 4);
    }
}
