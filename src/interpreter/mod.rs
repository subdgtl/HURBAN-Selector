use std::collections::hash_map::{Entry, HashMap};
use std::collections::HashSet;
use std::error;
use std::fmt;

pub use self::ast::{FuncIdent, VarIdent};
pub use self::func::{Func, FuncFlags, ParamInfo};
pub use self::value::{Ty, Value};

pub mod ast;
pub mod func;
pub mod value;

/// A name resolution error.
#[derive(Debug, PartialEq)]
pub enum ResolveError {
    VarRedefinition { stmt_index: usize, var: VarIdent },
    UndeclaredVarUse { stmt_index: usize, var: VarIdent },
    UndeclaredFuncUse { stmt_index: usize, func: FuncIdent },
}

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ResolveError::VarRedefinition { stmt_index, var } => write!(
                f,
                "Re-definition of already declared variable {} on stmt {}",
                var, stmt_index,
            ),
            ResolveError::UndeclaredVarUse { stmt_index, var } => write!(
                f,
                "Use of an undeclared variable {} on stmt {}",
                var, stmt_index
            ),
            ResolveError::UndeclaredFuncUse { stmt_index, func } => write!(
                f,
                "Use of an undeclared function {} on stmt {}",
                func, stmt_index
            ),
        }
    }
}

impl error::Error for ResolveError {}

/// A type-checking error.
#[derive(Debug, PartialEq)]
pub enum TypecheckError {}

impl fmt::Display for TypecheckError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TypecheckError")
    }
}

impl error::Error for TypecheckError {}

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
                "Function {} declared with {} params, but provided with {} args on stmt {}",
                call.ident(),
                args_expected,
                args_provided,
                stmt_index,
            ),
            RuntimeError::ArgTyMismatch {
                stmt_index,
                call,
                optional,
                ty_expected,
                ty_provided,
            } => write!(
                f,
                "Function {} declared to take param (optional={}) type {}, but given {} on stmt {}",
                call.ident(),
                optional,
                ty_expected,
                ty_provided,
                stmt_index,
            ),
            RuntimeError::ReturnTyMismatch {
                stmt_index,
                call,
                ty_expected,
                ty_provided,
            } => write!(
                f,
                "Function {} declared to return type {}, but returned {} on stmt {}",
                call.ident(),
                ty_expected,
                ty_provided,
                stmt_index,
            ),
        }
    }
}

impl error::Error for RuntimeError {}

/// An interpreter error.
#[derive(Debug, PartialEq)]
pub enum InterpretError {
    Resolve(ResolveError),
    Typecheck(TypecheckError),
    Runtime(RuntimeError),
}

impl fmt::Display for InterpretError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InterpretError::Resolve(resolve_error) => f.write_str(&resolve_error.to_string()),
            InterpretError::Typecheck(typecheck_error) => f.write_str(&typecheck_error.to_string()),
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

impl From<TypecheckError> for InterpretError {
    fn from(typecheck_error: TypecheckError) -> InterpretError {
        InterpretError::Typecheck(typecheck_error)
    }
}

impl From<RuntimeError> for InterpretError {
    fn from(runtime_error: RuntimeError) -> InterpretError {
        InterpretError::Runtime(runtime_error)
    }
}

pub type InterpretResult = Result<ValueSet, InterpretError>;

/// The state of variable values as captured by interpreting up to a
/// certain point in a program.
#[derive(Debug, Clone, PartialEq)]
pub struct ValueSet {
    /// The value of the last executed statement.
    pub last_value: Value,

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
    funcs: HashMap<FuncIdent, Box<dyn Func>>,
    env: HashMap<VarIdent, VarInfo>,

    /// The program counter. Always points to the **next** stmt to execute.
    pc: usize,

    /// The number of changes to the program since the interpreter was
    /// created. Incremented with each program modification.
    epoch: u64,

    /// The last epoch for which name resolution succeeded. Initially
    /// 0, since empty program is by default resolved.
    last_resolve_epoch: u64,
}

impl Interpreter {
    pub fn new(funcs: HashMap<FuncIdent, Box<dyn Func>>) -> Self {
        Self {
            prog: ast::Prog::default(),
            funcs,
            env: HashMap::new(),
            pc: 0,
            epoch: 0,
            last_resolve_epoch: 0,
        }
    }

    #[allow(dead_code)]
    pub fn prog(&self) -> &ast::Prog {
        &self.prog
    }

    pub fn set_prog(&mut self, prog: ast::Prog) {
        self.env.clear();
        self.pc = 0;
        self.epoch += 1;
        self.prog = prog;
    }

    pub fn clear_prog(&mut self) {
        self.env.clear();
        self.pc = 0;
        self.epoch += 1;
        self.prog = ast::Prog::default();
    }

    pub fn push_prog_stmt(&mut self, stmt: ast::Stmt) {
        self.epoch += 1;
        self.prog.push_stmt(stmt);
    }

    pub fn pop_prog_stmt(&mut self) {
        assert!(
            !self.prog.stmts().is_empty(),
            "Program must not be empty when popping"
        );

        // If we already executed everything we had so far, decrement
        // the program counter to mark the stmt for re-execution.
        if self.pc == self.prog.stmts().len() {
            self.pc -= 1;
            assert_eq!(self.pc, self.prog.stmts().len() - 1);
        }

        self.epoch += 1;
        self.prog.pop_stmt();
    }

    #[allow(dead_code)]
    pub fn prog_stmt_at(&self, index: usize) -> Option<&ast::Stmt> {
        self.prog.stmts().get(index)
    }

    pub fn set_prog_stmt_at(&mut self, index: usize, stmt: ast::Stmt) {
        // Mark the stmt and everything following it for re-execution
        self.pc = index;

        let stmt_mut = self
            .prog
            .stmts_mut()
            .get_mut(index)
            .expect("Expected the program to have index-th element");

        self.epoch += 1;
        *stmt_mut = stmt;
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

    pub fn typecheck(&mut self) -> Result<(), TypecheckError> {
        // FIXME: @Diagnostics Implement type-checking

        Ok(())
    }

    /// Interprets the whole currently set program and returns the
    /// used/unused values after the last statment.
    ///
    /// # Panics
    /// Panics if the currently set program is empty.
    pub fn interpret(&mut self) -> InterpretResult {
        assert!(
            !self.prog.stmts().is_empty(),
            "Can not execute empty program",
        );
        self.interpret_up_until(self.prog.stmts().len() - 1)
    }

    /// Interprets the currently set program up until the `index`-th
    /// statement (inclusive) and returns the used/unused values after
    /// it.
    ///
    /// # Panics
    /// Panics if the currently set program is empty or if `index` is
    /// out of bounds.
    pub fn interpret_up_until(&mut self, index: usize) -> InterpretResult {
        assert!(
            !self.prog.stmts().is_empty(),
            "Can not execute empty program",
        );

        let max_index = self.prog.stmts().len() - 1;
        assert!(
            max_index >= index,
            "Can not execute past the program lenght",
        );

        self.resolve()?;
        self.typecheck()?;

        self.invalidate();

        if self.pc <= index {
            // Run remaining operations, write results to vars
            let range = self.pc..=index;
            for (stmt_index, stmt) in self.prog.stmts()[range].iter().enumerate() {
                eval_stmt(stmt_index, stmt, &self.funcs, &mut self.env)?;
                self.pc += 1;
            }

            assert_eq!(self.pc, index + 1);
        }

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

        Ok(ValueSet {
            last_value,
            used_values,
            unused_values,
        })
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

    /// Invalidate variables in the environment.
    ///
    /// Verify all variables we have computed already, invalidating
    /// all that could have possibly changed since last execution.
    /// Invalidated variables are simply removed from the environment
    /// and will be re-computed a-fresh during evaluation. We count on
    /// the fact that any dependency must come before its dependents
    /// in the examined statements due to the serialized nature of the
    /// program.
    ///
    /// If we invalidate any variable, we will have to reset our
    /// program counter to the statement declaring the earliest
    /// variable we cleared, once again taking advantage of the
    /// program's serialized form. Cached variables will be skipped
    /// over during evaluation though, so this does not generate a lot
    /// of extra work.
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
        // FIXME: This is still very pessimistic, we should support an
        // incremental computation model with fact verification a-lá
        // salsa. https://github.com/salsa-rs/salsa

        let mut new_pc = None;

        for (i, stmt) in self.prog.stmts().iter().enumerate() {
            match stmt {
                ast::Stmt::VarDecl(var_decl) => {
                    let var_ident = var_decl.ident();
                    let init_expr = var_decl.init_expr();
                    let func_ident = init_expr.ident();

                    // Perform 1) Impurity invalidation

                    if !self.funcs[&func_ident].flags().contains(FuncFlags::PURE) {
                        log::debug!("Performing impurity invalidation of {}", var_ident);

                        self.env.remove(&var_ident);
                        if new_pc.is_none() {
                            new_pc = Some(i);
                        }

                        continue;
                    }

                    // Perform 2) Definition invalidation

                    if let Entry::Occupied(occupied) = self.env.entry(var_ident) {
                        let var_info = occupied.get();
                        let created_call = &var_info.created_call;

                        if created_call != init_expr {
                            log::debug!("Performing definition invalidation of {}", var_ident);

                            occupied.remove_entry();
                            if new_pc.is_none() {
                                new_pc = Some(i);
                            }

                            continue;
                        }
                    }

                    // Perform 3) Dependency invalidation

                    for expr in var_decl.init_expr().args() {
                        if let ast::Expr::Var(var) = expr {
                            if !self.env.contains_key(&var.ident()) {
                                log::debug!("Performing dependency invalidation of {}", var_ident);

                                self.env.remove(&var_ident);
                                if new_pc.is_none() {
                                    new_pc = Some(i);
                                }

                                break;
                            }
                        }
                    }
                }
            }
        }

        if let Some(pc) = new_pc {
            if pc < self.pc {
                log::debug!(
                    "Resetting PC after invalidation ({} -> {}) to re-compute vars",
                    self.pc,
                    pc,
                );
                self.pc = pc;
            }
        }
    }
}

fn eval_stmt(
    stmt_index: usize,
    stmt: &ast::Stmt,
    funcs: &HashMap<FuncIdent, Box<dyn Func>>,
    env: &mut HashMap<VarIdent, VarInfo>,
) -> Result<(), RuntimeError> {
    match stmt {
        ast::Stmt::VarDecl(var_decl) => eval_var_decl_stmt(stmt_index, var_decl, funcs, env),
    }
}

fn eval_var_decl_stmt(
    stmt_index: usize,
    var_decl: &ast::VarDeclStmt,
    funcs: &HashMap<FuncIdent, Box<dyn Func>>,
    env: &mut HashMap<VarIdent, VarInfo>,
) -> Result<(), RuntimeError> {
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
        if !env.contains_key(&var_ident) {
            let init_expr = var_decl.init_expr();
            let value = eval_call_expr(stmt_index, init_expr, funcs, env)?;

            env.insert(
                var_ident,
                VarInfo {
                    created_call: init_expr.clone(),
                    value,
                },
            );
        }
    }

    Ok(())
}

fn eval_expr(
    expr: &ast::Expr,
    env: &mut HashMap<VarIdent, VarInfo>,
) -> Result<Value, RuntimeError> {
    match expr {
        ast::Expr::Lit(lit) => eval_lit_expr(lit),
        ast::Expr::Var(var) => eval_var_expr(var, env),
        ast::Expr::Index(_) => unimplemented!("We don't support index expressions yet"),
    }
}

fn eval_lit_expr(lit: &ast::LitExpr) -> Result<Value, RuntimeError> {
    let value = match lit {
        ast::LitExpr::Boolean(boolean) => Value::Boolean(*boolean),
        ast::LitExpr::Int(int) => Value::Int(*int),
        ast::LitExpr::Uint(uint) => Value::Uint(*uint),
        ast::LitExpr::Float(float) => Value::Float(*float),
        ast::LitExpr::Float3(float3) => Value::Float3(*float3),
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
    funcs: &HashMap<FuncIdent, Box<dyn Func>>,
    env: &mut HashMap<VarIdent, VarInfo>,
) -> Result<Value, RuntimeError> {
    let func = &funcs[&call.ident()];

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
        let param_ty = info.ty;
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

    let value = func.call(&args);

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

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use super::*;

    struct TestFunc<F: Fn(&[Value]) -> Value> {
        func: F,
        flags: FuncFlags,
        param_info: Vec<ParamInfo>,
        return_ty: Ty,
    }

    impl<F: Fn(&[Value]) -> Value> TestFunc<F> {
        pub fn new(func: F, flags: FuncFlags, param_info: Vec<ParamInfo>, return_ty: Ty) -> Self {
            Self {
                flags,
                func,
                param_info,
                return_ty,
            }
        }
    }

    impl<F: Fn(&[Value]) -> Value> Func for TestFunc<F> {
        fn flags(&self) -> FuncFlags {
            self.flags
        }

        fn param_info(&self) -> &[ParamInfo] {
            &self.param_info
        }

        fn return_ty(&self) -> Ty {
            self.return_ty
        }

        fn call(&self, values: &[Value]) -> Value {
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
                |_| Value::Boolean(true),
                FuncFlags::PURE,
                vec![],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            ast::CallExpr::new(func_id, vec![]),
        ))]);

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value.last_value, Value::Boolean(true));
    }

    #[test]
    fn test_interpreter_interpret_single_func_parametrized_pure() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |values| Value::Boolean(values[0].unwrap_boolean()),
                FuncFlags::PURE,
                vec![ParamInfo {
                    ty: Ty::Boolean,
                    optional: false,
                }],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            ast::CallExpr::new(func_id, vec![ast::Expr::Lit(ast::LitExpr::Boolean(true))]),
        ))]);

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value.last_value, Value::Boolean(true));
    }

    #[test]
    fn test_interpreter_interpret_single_func_parameterless_impure() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |_| Value::Boolean(true),
                FuncFlags::empty(),
                vec![],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            ast::CallExpr::new(func_id, vec![]),
        ))]);

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value.last_value, Value::Boolean(true));
    }

    #[test]
    fn test_interpreter_interpret_single_func_parametrized_impure() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |values| Value::Boolean(values[0].unwrap_boolean()),
                FuncFlags::empty(),
                vec![ParamInfo {
                    ty: Ty::Boolean,
                    optional: false,
                }],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            ast::CallExpr::new(func_id, vec![ast::Expr::Lit(ast::LitExpr::Boolean(true))]),
        ))]);

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value.last_value, Value::Boolean(true));
    }

    #[test]
    fn test_interpreter_interpret_func_chain_with_pure_param() {
        let (func_id1, func1) = (
            FuncIdent(0),
            TestFunc::new(
                |_| Value::Boolean(true),
                FuncFlags::PURE,
                vec![],
                Ty::Boolean,
            ),
        );
        let (func_id2, func2) = (
            FuncIdent(1),
            TestFunc::new(
                |values| Value::Boolean(values[0].unwrap_boolean()),
                FuncFlags::PURE,
                vec![ParamInfo {
                    ty: Ty::Boolean,
                    optional: false,
                }],
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

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id1, Box::new(func1));
        funcs.insert(func_id2, Box::new(func2));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value.last_value, Value::Boolean(true));
    }

    #[test]
    fn test_interpreter_interpret_func_chain_with_impure_param() {
        let (func_id1, func1) = (
            FuncIdent(0),
            TestFunc::new(
                |_| Value::Boolean(true),
                FuncFlags::empty(),
                vec![],
                Ty::Boolean,
            ),
        );
        let (func_id2, func2) = (
            FuncIdent(1),
            TestFunc::new(
                |values| Value::Boolean(values[0].unwrap_boolean()),
                FuncFlags::empty(),
                vec![ParamInfo {
                    ty: Ty::Boolean,
                    optional: false,
                }],
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

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id1, Box::new(func1));
        funcs.insert(func_id2, Box::new(func2));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value.last_value, Value::Boolean(true));
    }

    // Prog index

    #[test]
    #[should_panic(expected = "Can not execute empty program")]
    fn test_interpreter_interpret_empty_prog() {
        let prog = ast::Prog::new(vec![]);

        let mut interpreter = Interpreter::new(HashMap::new());
        interpreter.set_prog(prog);

        let _ = interpreter.interpret();
    }

    #[test]
    #[should_panic(expected = "Can not execute empty program")]
    fn test_interpreter_interpret_up_until_empty_prog() {
        let prog = ast::Prog::new(vec![]);

        let mut interpreter = Interpreter::new(HashMap::new());
        interpreter.set_prog(prog);

        let _ = interpreter.interpret_up_until(1);
    }

    #[test]
    #[should_panic(expected = "Can not execute past the program lenght")]
    fn test_interpreter_interpret_up_until_invalid_index() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |_| Value::Boolean(true),
                FuncFlags::PURE,
                vec![],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            ast::CallExpr::new(func_id, vec![]),
        ))]);

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let _ = interpreter.interpret_up_until(2);
    }

    #[test]
    fn test_interpreter_interpret_up_until_valid_index() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |_| Value::Boolean(true),
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

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret_up_until(0).unwrap();
        assert_eq!(value.last_value, Value::Boolean(true));
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
                    Value::Boolean(values[0].unwrap_boolean())
                },
                FuncFlags::PURE,
                vec![ParamInfo {
                    ty: Ty::Boolean,
                    optional: false,
                }],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(1),
            ast::CallExpr::new(func_id, vec![ast::Expr::Lit(ast::LitExpr::Boolean(true))]),
        ))]);

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value.last_value, Value::Boolean(true));

        let value = interpreter.interpret().unwrap();
        assert_eq!(value.last_value, Value::Boolean(true));
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
                    Value::Boolean(values[0].unwrap_boolean())
                },
                FuncFlags::empty(),
                vec![ParamInfo {
                    ty: Ty::Boolean,
                    optional: false,
                }],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(1),
            ast::CallExpr::new(func_id, vec![ast::Expr::Lit(ast::LitExpr::Boolean(true))]),
        ))]);

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value.last_value, Value::Boolean(true));

        let value = interpreter.interpret().unwrap();
        assert_eq!(value.last_value, Value::Boolean(true));
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
                        Value::Boolean(!value)
                    } else {
                        Value::Boolean(value)
                    }
                },
                FuncFlags::PURE,
                vec![
                    ParamInfo {
                        ty: Ty::Boolean,
                        optional: false,
                    },
                    ParamInfo {
                        ty: Ty::Boolean,
                        optional: false,
                    },
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

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value.last_value, Value::Boolean(true));

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

        let value = interpreter.interpret().unwrap();
        assert_eq!(value.last_value, Value::Boolean(false));
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
                        Value::Boolean(!value)
                    } else {
                        Value::Boolean(value)
                    }
                },
                FuncFlags::PURE,
                vec![
                    ParamInfo {
                        ty: Ty::Boolean,
                        optional: false,
                    },
                    ParamInfo {
                        ty: Ty::Boolean,
                        optional: false,
                    },
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
                        Value::Boolean(!value)
                    } else {
                        Value::Boolean(value)
                    }
                },
                FuncFlags::PURE,
                vec![
                    ParamInfo {
                        ty: Ty::Boolean,
                        optional: false,
                    },
                    ParamInfo {
                        ty: Ty::Boolean,
                        optional: false,
                    },
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

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id1, Box::new(func1));
        funcs.insert(func_id2, Box::new(func2));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value.last_value, Value::Boolean(true));

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

        let value = interpreter.interpret().unwrap();
        assert_eq!(value.last_value, Value::Boolean(true));

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
                    Value::Boolean(true)
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
                    Value::Boolean(values[0].unwrap_boolean())
                },
                FuncFlags::PURE,
                vec![ParamInfo {
                    ty: Ty::Boolean,
                    optional: false,
                }],
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

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id1, Box::new(func1));
        funcs.insert(func_id2, Box::new(func2));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value.last_value, Value::Boolean(true));

        let value = interpreter.interpret().unwrap();
        assert_eq!(value.last_value, Value::Boolean(true));

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
                |_| Value::Boolean(true),
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

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().unwrap_err();
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
                |values| Value::Boolean(values[0].unwrap_boolean()),
                FuncFlags::PURE,
                vec![ParamInfo {
                    ty: Ty::Boolean,
                    optional: false,
                }],
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

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().unwrap_err();
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
                |values| Value::Boolean(values[0].unwrap_boolean()),
                FuncFlags::PURE,
                vec![ParamInfo {
                    ty: Ty::Boolean,
                    optional: false,
                }],
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

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().unwrap_err();
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
                |values| Value::Boolean(values[0].unwrap_boolean()),
                FuncFlags::PURE,
                vec![ParamInfo {
                    ty: Ty::Boolean,
                    optional: false,
                }],
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

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().unwrap_err();
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

        let mut interpreter = Interpreter::new(HashMap::new());
        interpreter.set_prog(prog);

        let err = interpreter.interpret().unwrap_err();
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
                |values| Value::Float(values[0].unwrap_float() + 1.0),
                FuncFlags::PURE,
                vec![ParamInfo {
                    ty: Ty::Float,
                    optional: false,
                }],
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

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().unwrap_err();
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
                |values| Value::Float(values[0].unwrap_float() + 1.0),
                FuncFlags::PURE,
                vec![ParamInfo {
                    ty: Ty::Float,
                    optional: false,
                }],
                Ty::Float,
            ),
        );

        let call = ast::CallExpr::new(func_id, vec![ast::Expr::Lit(ast::LitExpr::Int(1))]);
        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            call.clone(),
        ))]);

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().unwrap_err();
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
                |values| Value::Float(values[0].get_float().unwrap_or(1.0)),
                FuncFlags::PURE,
                vec![ParamInfo {
                    ty: Ty::Float,
                    optional: true,
                }],
                Ty::Float,
            ),
        );

        let call = ast::CallExpr::new(func_id, vec![ast::Expr::Lit(ast::LitExpr::Nil)]);
        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            call,
        ))]);

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value.last_value, Value::Float(1.0));
    }

    #[test]
    fn test_interpreter_run_single_func_dynamic_optional_arg_ty_error() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |values| Value::Float(values[0].get_float().unwrap_or(1.0)),
                FuncFlags::PURE,
                vec![ParamInfo {
                    ty: Ty::Float,
                    optional: true,
                }],
                Ty::Float,
            ),
        );

        let call = ast::CallExpr::new(func_id, vec![ast::Expr::Lit(ast::LitExpr::Int(1))]);
        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            call.clone(),
        ))]);

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().unwrap_err();
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
            TestFunc::new(|_| Value::Int(-1), FuncFlags::PURE, vec![], Ty::Float),
        );

        let call = ast::CallExpr::new(func_id, vec![]);
        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            VarIdent(0),
            call.clone(),
        ))]);

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().unwrap_err();
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

    // ValueSet tests

    #[test]
    fn test_interpreter_interpret_value_set() {
        let (func_id, func) = (
            FuncIdent(0),
            TestFunc::new(
                |values| Value::Float(values[0].unwrap_float() * 2.0),
                FuncFlags::PURE,
                vec![ParamInfo {
                    ty: Ty::Float,
                    optional: true,
                }],
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

        let mut funcs: HashMap<FuncIdent, Box<dyn Func>> = HashMap::new();
        funcs.insert(func_id, Box::new(func));

        let mut interpreter = Interpreter::new(funcs);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(
            value,
            ValueSet {
                last_value: Value::Float(8.0),
                used_values: vec![
                    (VarIdent(0), Value::Float(2.0)),
                    (VarIdent(1), Value::Float(4.0)),
                ],
                unused_values: vec![
                    (VarIdent(2), Value::Float(8.0)),
                    (VarIdent(3), Value::Float(8.0)),
                ],
            }
        );
    }
}
