use crate::interpreter::ast;
use crate::interpreter_funcs::FUNC_ID_CREATE_BOX;
use crate::project::Project;

pub fn default_project() -> Project {
    Project {
        version: 1,
        stmts: vec![ast::Stmt::VarDecl(ast::VarDeclStmt::new(
            ast::VarIdent(0),
            ast::CallExpr::new(
                FUNC_ID_CREATE_BOX,
                vec![
                    ast::Expr::Lit(ast::LitExpr::Float3([0.0, 0.0, 0.5])),
                    ast::Expr::Lit(ast::LitExpr::Float3([0.0, 0.0, 0.0])),
                    ast::Expr::Lit(ast::LitExpr::Float3([1.0, 1.0, 1.0])),
                    ast::Expr::Lit(ast::LitExpr::Boolean(true)),
                    ast::Expr::Lit(ast::LitExpr::Boolean(false)),
                ],
            ),
        ))],
    }
}
