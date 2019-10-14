use std::collections::hash_map::{Entry, HashMap};
use std::error;
use std::fmt;

use crate::convert::cast_usize;

pub use self::func::{Func, FuncFlags, ParamInfo};
pub use self::value::{Ty, Value};

pub mod ast;
pub mod func;
pub mod value;

/// A name resolution error.
#[derive(Debug, PartialEq)]
pub enum ResolveError {}

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ResolveError")
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
    UndeclaredVarUse(usize, ast::VarIdent),
    UndeclaredFuncUse(usize, ast::FuncIdent),
    ArgCountMismatch(usize, ast::CallExpr, usize, usize),
    ArgTyMismatch(usize, ast::CallExpr, bool, Ty, Ty),
    ReturnTyMismatch(usize, ast::CallExpr, Ty, Ty),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RuntimeError::UndeclaredVarUse(line, undeclared_var_use) => write!(
                f,
                "Use of an undeclared variable {} on line {}",
                undeclared_var_use.0, line,
            ),
            RuntimeError::UndeclaredFuncUse(line, undeclared_func_use) => write!(
                f,
                "Use of an undeclared function {} on line {}",
                undeclared_func_use.0, line,
            ),
            RuntimeError::ArgCountMismatch(line, call, expected_count, actual_count) => write!(
                f,
                "Function {} declared with {} params, but provided with {} args on line {}",
                call.ident(),
                expected_count,
                actual_count,
                line,
            ),
            RuntimeError::ArgTyMismatch(line, call, optional, expected_ty, actual_ty) => write!(
                f,
                "Function {} declared to take param (optional={}) type {}, but given {} on line {}",
                call.ident(),
                optional,
                expected_ty,
                actual_ty,
                line,
            ),
            RuntimeError::ReturnTyMismatch(line, call, expected_ty, actual_ty) => write!(
                f,
                "Function {} declared to return type {}, but returned {} on line {}",
                call.ident(),
                expected_ty,
                actual_ty,
                line,
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
    funcs: Vec<Box<dyn Func>>,
    env: HashMap<ast::VarIdent, VarInfo>,

    /// The program counter. Always points to the **next** stmt to execute.
    pc: usize,
}

impl Interpreter {
    pub fn new(funcs: Vec<Box<dyn Func>>) -> Self {
        Self {
            prog: ast::Prog::default(),
            funcs,
            env: HashMap::new(),
            pc: 0,
        }
    }

    #[allow(dead_code)]
    pub fn prog(&self) -> &ast::Prog {
        &self.prog
    }

    pub fn set_prog(&mut self, prog: ast::Prog) {
        self.env.clear();
        self.pc = 0;
        self.prog = prog;
    }

    pub fn clear_prog(&mut self) {
        self.env.clear();
        self.pc = 0;
        self.prog = ast::Prog::default();
    }

    pub fn push_prog_stmt(&mut self, stmt: ast::Stmt) {
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

        *stmt_mut = stmt;
    }

    pub fn resolve(&mut self) -> Result<(), ResolveError> {
        // FIXME: @Diagnostics Implement name resolution

        Ok(())
    }

    pub fn typecheck(&mut self) -> Result<(), TypecheckError> {
        // FIXME: @Diagnostics Implement type-checking

        Ok(())
    }

    pub fn interpret(&mut self) -> Result<Value, InterpretError> {
        assert!(
            !self.prog.stmts().is_empty(),
            "Can not execute empty program",
        );
        self.interpret_up_until(self.prog.stmts().len() - 1)
    }

    pub fn interpret_up_until(&mut self, index: usize) -> Result<Value, InterpretError> {
        self.resolve()?;
        self.typecheck()?;

        let max_index = self.prog.stmts().len() - 1;
        assert!(
            max_index >= index,
            "Can not execute past the program lenght",
        );

        self.invalidate();

        if self.pc <= index {
            // Run remaining operations, write results to vars
            let range = self.pc..=index;
            for (line, stmt) in self.prog.stmts()[range].iter().enumerate() {
                eval_stmt(line, stmt, &self.funcs, &mut self.env)?;
                self.pc += 1;
            }

            assert_eq!(self.pc, index + 1);
        }

        let last_executed_stmt = self.pc - 1;
        let value = match &self.prog.stmts()[last_executed_stmt] {
            ast::Stmt::VarDecl(var_decl) => {
                let var_ident = var_decl.ident();
                let var_info = self
                    .env
                    .get(&var_ident)
                    .expect("Value must have been populated or already cached");

                var_info.value.clone()
            }
        };

        Ok(value)
    }

    fn invalidate(&mut self) {
        // FIXME: This is still very pessimistic, we should support an
        // incremental computation model with fact verification a-lá
        // salsa. https://github.com/salsa-rs/salsa

        // Verify all variables we have computed already, invalidating
        // all that could have possibly changed since last execution.
        // Invalidated variables are simply removed from the
        // environment and will be re-computed a-fresh during
        // evaluation. We count on the fact that any dependency must
        // come before its dependents in the examined statements due
        // to the serialized nature of the program.

        // If we invalidate any variable, we will have to reset our
        // program counter to the statement declaring the earliest
        // variable we cleared, once again taking advantage of the
        // program's serialized form. Cached variables will be skipped
        // over during evaluation though, so this does not generate a
        // lot of extra work.
        let mut new_pc = None;

        for (i, stmt) in self.prog.stmts().iter().enumerate() {
            match stmt {
                ast::Stmt::VarDecl(var_decl) => {
                    // There are 3 types of variable invalidation:
                    // 1) Impurity invalidation: the function
                    //    producing the variable is not pure (import,
                    //    random, etc.)
                    // 2) Definition invalidation: the call expression
                    //    definition has changed (either the function
                    //    or the parameters)
                    // 3) Dependency invalidation: variables
                    //    referenced in the parameters have have been
                    //    invalidated.

                    let var_ident = var_decl.ident();
                    let init_expr = var_decl.init_expr();
                    let func_ident = init_expr.ident();

                    // Perform 1) Impurity invalidation

                    if !self.funcs[cast_usize(func_ident.0)]
                        .flags()
                        .contains(FuncFlags::PURE)
                    {
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
    line: usize,
    stmt: &ast::Stmt,
    funcs: &[Box<dyn Func>],
    env: &mut HashMap<ast::VarIdent, VarInfo>,
) -> Result<(), RuntimeError> {
    match stmt {
        ast::Stmt::VarDecl(var_decl) => eval_var_decl_stmt(line, var_decl, funcs, env),
    }
}

fn eval_var_decl_stmt(
    line: usize,
    var_decl: &ast::VarDeclStmt,
    funcs: &[Box<dyn Func>],
    env: &mut HashMap<ast::VarIdent, VarInfo>,
) -> Result<(), RuntimeError> {
    let var_ident = var_decl.ident();

    // This is a false positive. Bad Clippy, bad! Rewriting the code
    // to use the entry API would fail borrowchecking (and cause
    // pointer invalidation if it didn't!). The entry would create a
    // long-lived mutable borrow which simply can't be live when we
    // call into `eval_call_expr`, which borrows again. Note that
    // having `eval_call_expr` guarded by the map access is the whole
    // point here.
    #[allow(clippy::map_entry)]
    {
        if !env.contains_key(&var_ident) {
            let init_expr = var_decl.init_expr();
            let value = eval_call_expr(line, init_expr, funcs, env)?;

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
    line: usize,
    expr: &ast::Expr,
    env: &mut HashMap<ast::VarIdent, VarInfo>,
) -> Result<Value, RuntimeError> {
    match expr {
        ast::Expr::Lit(lit) => eval_lit_expr(lit),
        ast::Expr::Var(var) => eval_var_expr(line, var, env),
        ast::Expr::Index(_) => unimplemented!("We don't support index expressions yet"),
    }
}

fn eval_lit_expr(lit: &ast::LitExpr) -> Result<Value, RuntimeError> {
    let value = match lit {
        ast::LitExpr::Boolean(boolean) => Value::Boolean(*boolean),
        ast::LitExpr::Int(int) => Value::Int(*int),
        ast::LitExpr::Uint(uint) => Value::Uint(*uint),
        ast::LitExpr::Float(float) => Value::Float(*float),
        ast::LitExpr::Nil => Value::Nil,
    };

    Ok(value)
}

fn eval_var_expr(
    line: usize,
    var: &ast::VarExpr,
    env: &mut HashMap<ast::VarIdent, VarInfo>,
) -> Result<Value, RuntimeError> {
    let var_ident = var.ident();
    let var_info = env
        .get(&var_ident)
        .ok_or(RuntimeError::UndeclaredVarUse(line, var_ident))?;

    Ok(var_info.value.clone())
}

fn eval_call_expr(
    line: usize,
    call: &ast::CallExpr,
    funcs: &[Box<dyn Func>],
    env: &mut HashMap<ast::VarIdent, VarInfo>,
) -> Result<Value, RuntimeError> {
    let func_ident = call.ident();

    if let Some(func) = funcs.get(cast_usize(func_ident.0)) {
        let arg_exprs = call.args();
        if func.param_info().len() != arg_exprs.len() {
            return Err(RuntimeError::ArgCountMismatch(
                line,
                call.clone(),
                func.param_info().len(),
                arg_exprs.len(),
            ));
        }

        let mut args = Vec::with_capacity(arg_exprs.len());
        for arg_expr in arg_exprs {
            let arg = eval_expr(line, arg_expr, env)?;
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

                return Err(RuntimeError::ArgTyMismatch(
                    line,
                    call.clone(),
                    info.optional,
                    param_ty,
                    value_ty,
                ));
            }
        }

        let value = func.call(&args);

        let return_ty = func.return_ty();
        let value_ty = value.ty();
        if return_ty != value_ty {
            return Err(RuntimeError::ReturnTyMismatch(
                line,
                call.clone(),
                return_ty,
                value_ty,
            ));
        }

        Ok(value)
    } else {
        Err(RuntimeError::UndeclaredFuncUse(line, func_ident))
    }
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
            0,
            TestFunc::new(
                |_| Value::Boolean(true),
                FuncFlags::PURE,
                vec![],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            ast::VarIdent(0),
            ast::CallExpr::new(ast::FuncIdent(func_id), vec![]),
        ))]);

        let mut interpreter = Interpreter::new(vec![Box::new(func)]);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value, Value::Boolean(true));
    }

    #[test]
    fn test_interpreter_interpret_single_func_parametrized_pure() {
        let (func_id, func) = (
            0,
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
            ast::VarIdent(0),
            ast::CallExpr::new(
                ast::FuncIdent(func_id),
                vec![ast::Expr::Lit(ast::LitExpr::Boolean(true))],
            ),
        ))]);

        let mut interpreter = Interpreter::new(vec![Box::new(func)]);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value, Value::Boolean(true));
    }

    #[test]
    fn test_interpreter_interpret_single_func_parameterless_impure() {
        let (func_id, func) = (
            0,
            TestFunc::new(
                |_| Value::Boolean(true),
                FuncFlags::empty(),
                vec![],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            ast::VarIdent(0),
            ast::CallExpr::new(ast::FuncIdent(func_id), vec![]),
        ))]);

        let mut interpreter = Interpreter::new(vec![Box::new(func)]);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value, Value::Boolean(true));
    }

    #[test]
    fn test_interpreter_interpret_single_func_parametrized_impure() {
        let (func_id, func) = (
            0,
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
            ast::VarIdent(0),
            ast::CallExpr::new(
                ast::FuncIdent(func_id),
                vec![ast::Expr::Lit(ast::LitExpr::Boolean(true))],
            ),
        ))]);

        let mut interpreter = Interpreter::new(vec![Box::new(func)]);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value, Value::Boolean(true));
    }

    #[test]
    fn test_interpreter_interpret_func_chain_with_pure_param() {
        let (func_id1, func1) = (
            0,
            TestFunc::new(
                |_| Value::Boolean(true),
                FuncFlags::PURE,
                vec![],
                Ty::Boolean,
            ),
        );
        let (func_id2, func2) = (
            1,
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
                ast::VarIdent(0),
                ast::CallExpr::new(ast::FuncIdent(func_id1), vec![]),
            )),
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                ast::VarIdent(1),
                ast::CallExpr::new(
                    ast::FuncIdent(func_id2),
                    vec![ast::Expr::Var(ast::VarExpr::new(ast::VarIdent(0)))],
                ),
            )),
        ]);

        let mut interpreter = Interpreter::new(vec![Box::new(func1), Box::new(func2)]);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value, Value::Boolean(true));
    }

    #[test]
    fn test_interpreter_interpret_func_chain_with_impure_param() {
        let (func_id1, func1) = (
            0,
            TestFunc::new(
                |_| Value::Boolean(true),
                FuncFlags::empty(),
                vec![],
                Ty::Boolean,
            ),
        );
        let (func_id2, func2) = (
            1,
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
                ast::VarIdent(0),
                ast::CallExpr::new(ast::FuncIdent(func_id1), vec![]),
            )),
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                ast::VarIdent(1),
                ast::CallExpr::new(
                    ast::FuncIdent(func_id2),
                    vec![ast::Expr::Var(ast::VarExpr::new(ast::VarIdent(0)))],
                ),
            )),
        ]);

        let mut interpreter = Interpreter::new(vec![Box::new(func1), Box::new(func2)]);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value, Value::Boolean(true));
    }

    // Prog index

    #[test]
    #[should_panic(expected = "Can not execute past the program lenght")]
    fn test_interpreter_interpret_up_until_invalid_index() {
        let (func_id, func) = (
            0,
            TestFunc::new(
                |_| Value::Boolean(true),
                FuncFlags::PURE,
                vec![],
                Ty::Boolean,
            ),
        );

        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            ast::VarIdent(0),
            ast::CallExpr::new(ast::FuncIdent(func_id), vec![]),
        ))]);

        let mut interpreter = Interpreter::new(vec![Box::new(func)]);
        interpreter.set_prog(prog);

        let _ = interpreter.interpret_up_until(2);
    }

    // Var invalidation tests

    #[test]
    fn test_interpreter_interpret_single_func_pure_caches_results() {
        let n_calls = Rc::new(CallCount::new());
        let c = Rc::clone(&n_calls);

        let (func_id, func) = (
            0,
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
            ast::VarIdent(1),
            ast::CallExpr::new(
                ast::FuncIdent(func_id),
                vec![ast::Expr::Lit(ast::LitExpr::Boolean(true))],
            ),
        ))]);

        let mut interpreter = Interpreter::new(vec![Box::new(func)]);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value, Value::Boolean(true));

        let value = interpreter.interpret().unwrap();
        assert_eq!(value, Value::Boolean(true));

        assert_eq!(n_calls.get(), 1);
    }

    #[test]
    fn test_interpreter_interpret_single_func_impurity_invalidation() {
        let n_calls = Rc::new(CallCount::new());
        let c = Rc::clone(&n_calls);

        let (func_id, func) = (
            0,
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
            ast::VarIdent(1),
            ast::CallExpr::new(
                ast::FuncIdent(func_id),
                vec![ast::Expr::Lit(ast::LitExpr::Boolean(true))],
            ),
        ))]);

        let mut interpreter = Interpreter::new(vec![Box::new(func)]);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value, Value::Boolean(true));

        let value = interpreter.interpret().unwrap();
        assert_eq!(value, Value::Boolean(true));

        assert_eq!(n_calls.get(), 2);
    }

    #[test]
    fn test_interpreter_interpret_single_func_definition_invalidation_with_changed_args() {
        let n_calls = Rc::new(CallCount::new());
        let c = Rc::clone(&n_calls);

        let (func_id, func) = (
            0,
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
            ast::VarIdent(0),
            ast::CallExpr::new(
                ast::FuncIdent(func_id),
                vec![
                    ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                    ast::Expr::Lit(ast::LitExpr::Boolean(false)),
                ],
            ),
        ))]);

        let mut interpreter = Interpreter::new(vec![Box::new(func)]);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value, Value::Boolean(true));

        // Change the args but not the func
        interpreter.set_prog_stmt_at(
            0,
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                ast::VarIdent(0),
                ast::CallExpr::new(
                    ast::FuncIdent(func_id),
                    vec![
                        ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                        ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                    ],
                ),
            )),
        );

        let value = interpreter.interpret().unwrap();
        assert_eq!(value, Value::Boolean(false));

        assert_eq!(n_calls.get(), 2);
    }

    #[test]
    fn test_interpreter_interpret_single_func_definition_invalidation_with_changed_func() {
        let n_calls1 = Rc::new(CallCount::new());
        let n_calls2 = Rc::new(CallCount::new());
        let c1 = Rc::clone(&n_calls1);
        let c2 = Rc::clone(&n_calls2);

        let (func_id1, func1) = (
            0,
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
            1,
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
            ast::VarIdent(0),
            ast::CallExpr::new(
                ast::FuncIdent(func_id1),
                vec![
                    ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                    ast::Expr::Lit(ast::LitExpr::Boolean(false)),
                ],
            ),
        ))]);

        let mut interpreter = Interpreter::new(vec![Box::new(func1), Box::new(func2)]);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value, Value::Boolean(true));

        // Change the func but not the args
        interpreter.set_prog_stmt_at(
            0,
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                ast::VarIdent(0),
                ast::CallExpr::new(
                    ast::FuncIdent(func_id2),
                    vec![
                        ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                        ast::Expr::Lit(ast::LitExpr::Boolean(false)),
                    ],
                ),
            )),
        );

        let value = interpreter.interpret().unwrap();
        assert_eq!(value, Value::Boolean(true));

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
            0,
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
            1,
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
                ast::VarIdent(0),
                ast::CallExpr::new(ast::FuncIdent(func_id1), vec![]),
            )),
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                ast::VarIdent(1),
                ast::CallExpr::new(
                    ast::FuncIdent(func_id2),
                    vec![ast::Expr::Var(ast::VarExpr::new(ast::VarIdent(0)))],
                ),
            )),
        ]);

        let mut interpreter = Interpreter::new(vec![Box::new(func1), Box::new(func2)]);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value, Value::Boolean(true));

        let value = interpreter.interpret().unwrap();
        assert_eq!(value, Value::Boolean(true));

        assert_eq!(n_calls1.get(), 2);
        assert_eq!(n_calls2.get(), 2);
    }

    // FIXME: Prog manipulation tests

    // FIXME: Name resolution tests (invalid names and redeclaration)

    // FIXME: Static typecheck tests

    // Dynamic typechecks tests

    #[test]
    fn test_interpreter_interpret_single_func_dynamic_arg_count_error() {
        let (func_id, func) = (
            0,
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
            ast::FuncIdent(func_id),
            vec![
                ast::Expr::Lit(ast::LitExpr::Float(1.0)),
                ast::Expr::Lit(ast::LitExpr::Float(0.0)),
            ],
        );
        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            ast::VarIdent(0),
            call.clone(),
        ))]);

        let mut interpreter = Interpreter::new(vec![Box::new(func)]);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().unwrap_err();
        assert_eq!(
            err,
            InterpretError::from(RuntimeError::ArgCountMismatch(0, call, 1, 2)),
        );
    }

    #[test]
    fn test_interpreter_interpret_single_func_dynamic_arg_ty_error() {
        let (func_id, func) = (
            0,
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
            ast::FuncIdent(func_id),
            vec![ast::Expr::Lit(ast::LitExpr::Int(1))],
        );
        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            ast::VarIdent(0),
            call.clone(),
        ))]);

        let mut interpreter = Interpreter::new(vec![Box::new(func)]);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().unwrap_err();
        assert_eq!(
            err,
            InterpretError::from(RuntimeError::ArgTyMismatch(
                0,
                call,
                false,
                Ty::Float,
                Ty::Int
            ))
        );
    }

    #[test]
    fn test_interpreter_interpret_single_func_dynamic_optional_arg_ty() {
        let (func_id, func) = (
            0,
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

        let call = ast::CallExpr::new(
            ast::FuncIdent(func_id),
            vec![ast::Expr::Lit(ast::LitExpr::Nil)],
        );
        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            ast::VarIdent(0),
            call,
        ))]);

        let mut interpreter = Interpreter::new(vec![Box::new(func)]);
        interpreter.set_prog(prog);

        let value = interpreter.interpret().unwrap();
        assert_eq!(value, Value::Float(1.0));
    }

    #[test]
    fn test_interpreter_run_single_func_dynamic_optional_arg_ty_error() {
        let (func_id, func) = (
            0,
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

        let call = ast::CallExpr::new(
            ast::FuncIdent(func_id),
            vec![ast::Expr::Lit(ast::LitExpr::Int(1))],
        );
        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            ast::VarIdent(0),
            call.clone(),
        ))]);

        let mut interpreter = Interpreter::new(vec![Box::new(func)]);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().unwrap_err();
        assert_eq!(
            err,
            InterpretError::from(RuntimeError::ArgTyMismatch(
                0,
                call,
                true,
                Ty::Float,
                Ty::Int
            ))
        );
    }

    #[test]
    fn test_interpreter_interpret_single_func_dynamic_return_ty_error() {
        let (func_id, func) = (
            0,
            TestFunc::new(|_| Value::Int(-1), FuncFlags::PURE, vec![], Ty::Float),
        );

        let call = ast::CallExpr::new(ast::FuncIdent(func_id), vec![]);
        let prog = ast::Prog::new(vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            ast::VarIdent(0),
            call.clone(),
        ))]);

        let mut interpreter = Interpreter::new(vec![Box::new(func)]);
        interpreter.set_prog(prog);

        let err = interpreter.interpret().unwrap_err();
        assert_eq!(
            err,
            InterpretError::from(RuntimeError::ReturnTyMismatch(0, call, Ty::Float, Ty::Int))
        );
    }
}
