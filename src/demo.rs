use crate::interpreter::ast;
use crate::interpreter_funcs::{
    FUNC_ID_ALIGN, FUNC_ID_IMPORT_OBJ_JOIN, FUNC_ID_INTERPOLATED_UNION,
    FUNC_ID_LAPLACIAN_SMOOTHING, FUNC_ID_TRANSFORM,
};
use crate::project::Project;

pub fn default_project() -> Project {
    Project {
        version: 1,
        stmts: vec![
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                ast::VarIdent(0),
                ast::CallExpr::new(
                    FUNC_ID_IMPORT_OBJ_JOIN,
                    vec![
                        ast::Expr::Lit(ast::LitExpr::String("c:\\tatra.obj".to_string())),
                        ast::Expr::Lit(ast::LitExpr::Boolean(false)),
                        ast::Expr::Lit(ast::LitExpr::Boolean(false)),
                        ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                        ast::Expr::Lit(ast::LitExpr::Boolean(false)),
                    ],
                ),
            )),
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                ast::VarIdent(1),
                ast::CallExpr::new(
                    FUNC_ID_IMPORT_OBJ_JOIN,
                    vec![
                        ast::Expr::Lit(ast::LitExpr::String("c:\\auticko.obj".to_string())),
                        ast::Expr::Lit(ast::LitExpr::Boolean(false)),
                        ast::Expr::Lit(ast::LitExpr::Boolean(false)),
                        ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                        ast::Expr::Lit(ast::LitExpr::Boolean(false)),
                    ],
                ),
            )),
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                ast::VarIdent(2),
                ast::CallExpr::new(
                    FUNC_ID_ALIGN,
                    vec![
                        ast::Expr::Var(ast::VarExpr::new(ast::VarIdent(1))),
                        ast::Expr::Var(ast::VarExpr::new(ast::VarIdent(0))),
                        ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                        ast::Expr::Lit(ast::LitExpr::Boolean(false)),
                    ],
                ),
            )),
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                ast::VarIdent(3),
                ast::CallExpr::new(
                    FUNC_ID_TRANSFORM,
                    vec![
                        ast::Expr::Var(ast::VarExpr::new(ast::VarIdent(2))),
                        ast::Expr::Lit(ast::LitExpr::Float3([0.0, -21.0, 0.0])),
                        ast::Expr::Lit(ast::LitExpr::Float3([0.0, 0.0, 0.0])),
                        ast::Expr::Lit(ast::LitExpr::Float3([1.0, 1.0, 1.0])),
                        ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                        ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                        ast::Expr::Lit(ast::LitExpr::Boolean(false)),
                    ],
                ),
            )),
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                ast::VarIdent(4),
                ast::CallExpr::new(
                    FUNC_ID_INTERPOLATED_UNION,
                    vec![
                        ast::Expr::Var(ast::VarExpr::new(ast::VarIdent(0))),
                        ast::Expr::Var(ast::VarExpr::new(ast::VarIdent(3))),
                        ast::Expr::Lit(ast::LitExpr::Float3([10.0, 10.0, 10.0])),
                        ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                        ast::Expr::Lit(ast::LitExpr::Float(0.5)),
                        ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                        ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                        ast::Expr::Lit(ast::LitExpr::Boolean(false)),
                    ],
                ),
            )),
            ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                ast::VarIdent(5),
                ast::CallExpr::new(
                    FUNC_ID_LAPLACIAN_SMOOTHING,
                    vec![
                        ast::Expr::Var(ast::VarExpr::new(ast::VarIdent(4))),
                        ast::Expr::Lit(ast::LitExpr::Uint(4)),
                        ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                        ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                        ast::Expr::Lit(ast::LitExpr::Boolean(false)),
                    ],
                ),
            )),
        ],
    }
}
