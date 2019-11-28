use std::fmt;
use std::sync::Arc;

/// A unique function identifier.
///
/// Has to stay stable for the lifetime of the interpreter and program
/// using it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncIdent(pub(crate) u64);

impl fmt::Display for FuncIdent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<func {}>", self.0)
    }
}

/// A unique variable identifier.
///
/// Has to stay stable for the lifetime of the interpreter and program
/// using it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarIdent(pub(crate) u64);

impl fmt::Display for VarIdent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<var {}>", self.0)
    }
}

/// A program consisting of a list of statements.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Prog {
    stmts: Vec<Stmt>,
}

impl Prog {
    pub fn new(stmts: Vec<Stmt>) -> Self {
        Self { stmts }
    }

    pub fn push_stmt(&mut self, stmt: Stmt) {
        self.stmts.push(stmt);
    }

    pub fn pop_stmt(&mut self) {
        self.stmts.pop();
    }

    pub fn stmts(&self) -> &[Stmt] {
        &self.stmts
    }

    pub fn stmts_mut(&mut self) -> &mut [Stmt] {
        &mut self.stmts
    }
}

impl fmt::Display for Prog {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("(prog")?;
        for stmt in &self.stmts {
            write!(f, "\n    {}", stmt)?;
        }
        f.write_str(")")
    }
}

/// A program statement.
///
/// Statements describe things to do, like execute code or declare a
/// variable.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    VarDecl(VarDeclStmt),
}

impl fmt::Display for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Stmt::VarDecl(var_decl) => write!(f, "{}", var_decl),
        }
    }
}

/// A variable declaration statement.
///
/// Declares a variable with a known identifier, and uses provided
/// initializer expression to produce a value for that variable.
#[derive(Debug, Clone, PartialEq)]
pub struct VarDeclStmt {
    // Note that values for variables can only come from calls, so we
    // use `CallExpr` directly.
    ident: VarIdent,
    init_expr: CallExpr,
}

impl VarDeclStmt {
    pub fn new(ident: VarIdent, init_expr: CallExpr) -> Self {
        Self { ident, init_expr }
    }

    pub fn ident(&self) -> VarIdent {
        self.ident
    }

    pub fn init_expr(&self) -> &CallExpr {
        &self.init_expr
    }
}

impl fmt::Display for VarDeclStmt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(var-decl {} {})", self.ident, self.init_expr)
    }
}

/// A program expression.
///
/// Expressions evaluate to values.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // Note that `CallExpr` is missing here. That is because it is
    // impossible for the frontend to produce a program where a func
    // arg would be the direct result of another func call. We use
    // `VarExpr` instead to refer to a previous result.
    Lit(LitExpr),
    #[allow(dead_code)]
    Var(VarExpr),
    #[allow(dead_code)]
    Index(IndexExpr),
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::Lit(lit) => write!(f, "{}", lit),
            Expr::Var(var) => write!(f, "{}", var),
            Expr::Index(index) => write!(f, "{}", index),
        }
    }
}

/// An expression that evaluates to a constant, literal value.
#[derive(Debug, Clone, PartialEq)]
pub enum LitExpr {
    #[allow(dead_code)]
    Nil,
    #[allow(dead_code)]
    Boolean(bool),
    #[allow(dead_code)]
    Int(i32),
    #[allow(dead_code)]
    Uint(u32),
    #[allow(dead_code)]
    Float(f32),
    #[allow(dead_code)]
    Float3([f32; 3]),
    String(Arc<String>),
}

impl fmt::Display for LitExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LitExpr::Nil => write!(f, "<nil>"),
            LitExpr::Boolean(boolean) => write!(f, "<boolean {}>", boolean),
            LitExpr::Int(int) => write!(f, "<int {}>", int),
            LitExpr::Uint(uint) => write!(f, "<uint {}>", uint),
            LitExpr::Float(float) => write!(f, "<float {}>", float),
            LitExpr::Float3(float3) => {
                write!(f, "<float3 [{}, {}, {}]>", float3[0], float3[1], float3[2])
            }
            LitExpr::String(string) => write!(f, "<string {}>", string),
        }
    }
}

/// An expression that evaluates to a value by extracting the value
/// from a variable.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VarExpr {
    ident: VarIdent,
}

impl VarExpr {
    #[allow(dead_code)]
    pub fn new(ident: VarIdent) -> Self {
        Self { ident }
    }

    pub fn ident(&self) -> VarIdent {
        self.ident
    }
}

impl fmt::Display for VarExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(var {})", self.ident)
    }
}

/// An expression that evaluates to a value by extracting the value
/// from an array by using the index value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IndexExpr {
    // Note that we most likely don't need the full dynamic
    // flexibility of `(get_name())[get_index()]`, so objects are only
    // `VarExpr` and indexes just an `usize` instead of `Box<Expr>`.
    object: VarExpr,
    index: usize,
}

impl fmt::Display for IndexExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(index {} {})", self.object, self.index)
    }
}

/// An expression that evaluates to a value by calling a function.
#[derive(Debug, Clone, PartialEq)]
pub struct CallExpr {
    ident: FuncIdent,
    args: Vec<Expr>,
}

impl CallExpr {
    pub fn new(ident: FuncIdent, args: Vec<Expr>) -> Self {
        Self { ident, args }
    }

    pub fn ident(&self) -> FuncIdent {
        self.ident
    }

    pub fn args(&self) -> &[Expr] {
        &self.args
    }
}

impl fmt::Display for CallExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(call {}", self.ident)?;
        for arg in &self.args {
            write!(f, " {}", arg)?;
        }
        f.write_str(")")
    }
}
