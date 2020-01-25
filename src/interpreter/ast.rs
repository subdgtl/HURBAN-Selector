use std::fmt;
use std::sync::Arc;

/// A unique function identifier.
///
/// Has to stay stable for the lifetime of the interpreter and program
/// using it.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct VarIdent(pub(crate) u64);

impl fmt::Display for VarIdent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<var {}>", self.0)
    }
}

/// A program consisting of a list of statements.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
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

    pub fn set_stmt_at(&mut self, index: usize, stmt: Stmt) {
        self.stmts[index] = stmt;
    }

    pub fn stmts(&self) -> &[Stmt] {
        &self.stmts
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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

    pub fn clone_with_init_expr(&self, init_expr: CallExpr) -> Self {
        Self {
            ident: self.ident,
            init_expr,
        }
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Expr {
    // Note that `CallExpr` is missing here. That is because it is
    // impossible for the frontend to produce a program where a func
    // arg would be the direct result of another func call. We use
    // `VarExpr` instead to refer to a previous result.
    Lit(LitExpr),
    #[allow(dead_code)]
    Var(VarExpr),
}

impl Expr {
    pub fn unwrap_literal(&self) -> &LitExpr {
        match self {
            Expr::Lit(lit) => lit,
            _ => panic!("Expression not literal"),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::Lit(lit) => write!(f, "{}", lit),
            Expr::Var(var) => write!(f, "{}", var),
        }
    }
}

/// An expression that evaluates to a constant, literal value.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum LitExpr {
    Nil,
    #[allow(dead_code)]
    Boolean(bool),
    #[allow(dead_code)]
    Int(i32),
    Uint(u32),
    Float(f32),
    Float2([f32; 2]),
    Float3([f32; 3]),
    String(Arc<String>),
}

impl LitExpr {
    /// Get the literal value if boolean, otherwise panic.
    ///
    /// # Panics
    /// This function panics when value is not a boolean.
    pub fn unwrap_boolean(&self) -> bool {
        match self {
            LitExpr::Boolean(boolean) => *boolean,
            _ => panic!("Literal expression not boolean"),
        }
    }

    /// Get the literal value if int, otherwise panic.
    ///
    /// # Panics
    /// This function panics when literal value is not an int.
    pub fn unwrap_int(&self) -> i32 {
        match self {
            LitExpr::Int(int) => *int,
            _ => panic!("Literal expression not int"),
        }
    }

    /// Get the literal value if uint, otherwise panic.
    ///
    /// # Panics
    /// This function panics when literal value is not an uint.
    pub fn unwrap_uint(&self) -> u32 {
        match self {
            LitExpr::Uint(uint) => *uint,
            _ => panic!("Literal expression not uint"),
        }
    }

    /// Get the literal value if float, otherwise panic.
    ///
    /// # Panics
    /// This function panics when literal value is not a float.
    pub fn unwrap_float(&self) -> f32 {
        match self {
            LitExpr::Float(float) => *float,
            _ => panic!("Literal expression not float"),
        }
    }

    /// Get the literal value if float2, otherwise panic.
    ///
    /// # Panics
    /// This function panics when literal value is not a float2.
    pub fn unwrap_float2(&self) -> [f32; 2] {
        match self {
            LitExpr::Float2(float2) => *float2,
            _ => panic!("Literal expression not float2"),
        }
    }

    /// Get the literal value if float3, otherwise panic.
    ///
    /// # Panics
    /// This function panics when literal value is not a float3.
    pub fn unwrap_float3(&self) -> [f32; 3] {
        match self {
            LitExpr::Float3(float3) => *float3,
            _ => panic!("Literal expression not float3"),
        }
    }

    /// Get the literal value if string, otherwise panic.
    ///
    /// # Panics
    /// This function panics when literal value is not a string.
    pub fn unwrap_string(&self) -> &str {
        match self {
            LitExpr::String(string) => string,
            _ => panic!("Literal expression not string"),
        }
    }
}

impl fmt::Display for LitExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LitExpr::Nil => write!(f, "<nil>"),
            LitExpr::Boolean(boolean) => write!(f, "<boolean {}>", boolean),
            LitExpr::Int(int) => write!(f, "<int {}>", int),
            LitExpr::Uint(uint) => write!(f, "<uint {}>", uint),
            LitExpr::Float(float) => write!(f, "<float {}>", float),
            LitExpr::Float2(float2) => write!(f, "<float2 [{}, {}]>", float2[0], float2[1]),
            LitExpr::Float3(float3) => {
                write!(f, "<float3 [{}, {}, {}]>", float3[0], float3[1], float3[2])
            }
            LitExpr::String(string) => write!(f, "<string {}>", string),
        }
    }
}

/// An expression that evaluates to a value by extracting the value
/// from a variable.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

/// An expression that evaluates to a value by calling a function.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CallExpr {
    ident: FuncIdent,
    args: Vec<Expr>,
}

impl CallExpr {
    pub fn new(ident: FuncIdent, args: Vec<Expr>) -> Self {
        Self { ident, args }
    }

    pub fn clone_with_arg_at(&self, index: usize, arg: Expr) -> Self {
        let mut new_args = self.args.clone();
        new_args[index] = arg;
        Self {
            ident: self.ident,
            args: new_args,
        }
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
