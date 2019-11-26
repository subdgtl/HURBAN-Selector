use std::f32;
use std::sync::Arc;

use imgui_winit_support::{HiDpiMode, WinitPlatform};

use crate::convert::{clamp_cast_i32_to_u32, clamp_cast_u32_to_i32};
use crate::interpreter::ast;
use crate::interpreter::ParamRefinement;
use crate::renderer::DrawGeometryMode;
use crate::session::Session;

const OPENSANS_REGULAR_BYTES: &[u8] = include_bytes!("../resources/OpenSans-Regular.ttf");
const OPENSANS_BOLD_BYTES: &[u8] = include_bytes!("../resources/OpenSans-Bold.ttf");
const OPENSANS_LIGHT_BYTES: &[u8] = include_bytes!("../resources/OpenSans-Light.ttf");

const MARGIN: f32 = 10.0;

pub struct FontIds {
    _regular: imgui::FontId,
    _light: imgui::FontId,
    _bold: imgui::FontId,
}

/// Thin wrapper around imgui and its winit platform. Its main responsibilty
/// is to create UI frames which draw the UI itself.
pub struct Ui {
    imgui_context: imgui::Context,
    imgui_winit_platform: WinitPlatform,
    font_ids: FontIds,
}

impl Ui {
    /// Initializes imgui with default settings for our application.
    pub fn new(window: &winit::window::Window) -> Self {
        let mut imgui_context = imgui::Context::create();
        let mut style = imgui_context.style_mut();

        style.window_padding = [4.0, 4.0];
        style.frame_padding = [4.0, 2.0];
        style.item_spacing = [2.0, 2.0];
        style.item_inner_spacing = [2.0, 2.0];
        style.indent_spacing = 8.0;

        style.scrollbar_size = 8.0;
        style.grab_min_size = 4.0;

        style.window_rounding = 3.0;
        style.frame_rounding = 3.0;
        style.scrollbar_rounding = 3.0;
        style.grab_rounding = 3.0;

        imgui_context.set_ini_filename(None);

        let mut platform = WinitPlatform::init(&mut imgui_context);

        platform.attach_window(imgui_context.io_mut(), window, HiDpiMode::Default);

        let hidpi_factor = platform.hidpi_factor();
        let font_size = (15.0 * hidpi_factor) as f32;

        let regular_font_id = imgui_context
            .fonts()
            .add_font(&[imgui::FontSource::TtfData {
                data: OPENSANS_REGULAR_BYTES,
                size_pixels: font_size,
                config: None,
            }]);
        let bold_font_id = imgui_context
            .fonts()
            .add_font(&[imgui::FontSource::TtfData {
                data: OPENSANS_BOLD_BYTES,
                size_pixels: font_size,
                config: None,
            }]);
        let light_font_id = imgui_context
            .fonts()
            .add_font(&[imgui::FontSource::TtfData {
                data: OPENSANS_LIGHT_BYTES,
                size_pixels: font_size,
                config: None,
            }]);

        imgui_context.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

        Ui {
            imgui_context,
            imgui_winit_platform: platform,
            font_ids: FontIds {
                _regular: regular_font_id,
                _bold: bold_font_id,
                _light: light_font_id,
            },
        }
    }

    pub fn fonts(&mut self) -> imgui::FontAtlasRefMut {
        self.imgui_context.fonts()
    }

    pub fn handle_event<T>(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::Event<T>,
    ) {
        self.imgui_winit_platform
            .handle_event(self.imgui_context.io_mut(), window, &event);
    }

    pub fn prepare_frame(&mut self, window: &winit::window::Window) -> UiFrame {
        self.imgui_winit_platform
            .prepare_frame(self.imgui_context.io_mut(), window)
            .expect("Failed to start imgui frame");

        UiFrame::new(
            &mut self.imgui_context,
            &self.imgui_winit_platform,
            &self.font_ids,
        )
    }

    pub fn set_delta_time(&mut self, duration_last_frame_s: f32) {
        self.imgui_context.io_mut().delta_time = duration_last_frame_s;
    }
}

/// This structure is responsible for drawing and rendering of a single UI
/// frame.
pub struct UiFrame<'a> {
    imgui_winit_platform: &'a WinitPlatform,
    imgui_ui: imgui::Ui<'a>,
    _font_ids: &'a FontIds,
}

impl<'a> UiFrame<'a> {
    pub fn new(
        imgui_context: &'a mut imgui::Context,
        imgui_winit_platform: &'a WinitPlatform,
        font_ids: &'a FontIds,
    ) -> Self {
        UiFrame {
            imgui_winit_platform,
            imgui_ui: imgui_context.frame(),
            _font_ids: font_ids,
        }
    }

    pub fn want_capture_keyboard(&self) -> bool {
        self.imgui_ui.io().want_capture_keyboard
    }

    pub fn want_capture_mouse(&self) -> bool {
        self.imgui_ui.io().want_capture_mouse
    }

    pub fn render(self, window: &winit::window::Window) -> &'a imgui::DrawData {
        self.imgui_winit_platform
            .prepare_render(&self.imgui_ui, window);
        self.imgui_ui.render()
    }

    pub fn draw_viewport_settings_window(&self, draw_mode: &mut DrawGeometryMode) -> bool {
        let ui = &self.imgui_ui;

        const VIEWPORT_WINDOW_WIDTH: f32 = 150.0;
        const VIEWPORT_WINDOW_HEIGHT: f32 = 150.0;
        let window_logical_size = ui.io().display_size;
        let window_inner_width = window_logical_size[0] - 2.0 * MARGIN;

        let mut reset_viewport_clicked = false;

        imgui::Window::new(imgui::im_str!("Viewport Settings"))
            .movable(false)
            .resizable(false)
            .size(
                [VIEWPORT_WINDOW_WIDTH, VIEWPORT_WINDOW_HEIGHT],
                imgui::Condition::Always,
            )
            .position(
                [window_inner_width + MARGIN - VIEWPORT_WINDOW_WIDTH, MARGIN],
                imgui::Condition::Always,
            )
            .build(ui, || {
                ui.text(imgui::im_str!("{:.3} fps", ui.io().framerate));

                ui.radio_button(
                    imgui::im_str!("Shaded"),
                    draw_mode,
                    DrawGeometryMode::Shaded,
                );
                ui.radio_button(imgui::im_str!("Edges"), draw_mode, DrawGeometryMode::Edges);
                ui.radio_button(
                    imgui::im_str!("Shaded with Edges"),
                    draw_mode,
                    DrawGeometryMode::ShadedEdges,
                );
                ui.radio_button(
                    imgui::im_str!("X-RAY"),
                    draw_mode,
                    DrawGeometryMode::ShadedEdgesXray,
                );

                reset_viewport_clicked = ui.button(imgui::im_str!("Reset Viewport"), [0.0, 0.0]);
            });

        reset_viewport_clicked
    }

    pub fn draw_pipeline_window(&self, session: &mut Session) {
        let ui = &self.imgui_ui;
        let function_table = session.function_table();

        const PIPELINE_WINDOW_WIDTH: f32 = 400.0;
        const PIPELINE_WINDOW_HEIGHT_MULT: f32 = 0.8;

        let window_logical_size = ui.io().display_size;
        let window_inner_height = window_logical_size[1] - 2.0 * MARGIN;

        let pipeline_window_height = window_inner_height * PIPELINE_WINDOW_HEIGHT_MULT;

        let interpreter_busy = session.interpreter_busy();
        let mut change = None;

        imgui::Window::new(imgui::im_str!("Pipeline"))
            .movable(false)
            .resizable(false)
            .size([PIPELINE_WINDOW_WIDTH, pipeline_window_height], imgui::Condition::Always)
            .position([MARGIN, MARGIN], imgui::Condition::Always)
            .build(ui, || {
                for (stmt_index, stmt) in session.stmts().iter().enumerate() {
                    match stmt {
                        ast::Stmt::VarDecl(var_decl) => {
                            let call_expr = var_decl.init_expr();
                            let func_ident = call_expr.ident();
                            let func = &function_table[&func_ident];

                            if ui
                                .collapsing_header(&imgui::im_str!(
                                    "#{} {} ##{}",
                                    stmt_index + 1,
                                    func.info().name,
                                    stmt_index
                                ))
                                .default_open(true)
                                .build()
                            {
                                ui.indent();

                                assert_eq!(
                                    call_expr.args().len(),
                                    func.param_info().len(),
                                    "Function call must be generated with correct number of arguments",
                                );

                                let operation_arg_style_tokens = if interpreter_busy {
                                    Some(push_disabled_style(ui))
                                } else {
                                    None
                                };

                                for (arg_index, (param_info, arg)) in func
                                    .param_info()
                                    .iter()
                                    .zip(call_expr.args().iter())
                                    .enumerate()
                                {
                                    let input_label = imgui::im_str!(
                                        "{}##{}-{}",
                                        &param_info.name,
                                        stmt_index,
                                        arg_index
                                    );

                                    match param_info.refinement {
                                        ParamRefinement::Boolean(_) => {
                                            let mut boolean_lit =
                                                arg.unwrap_literal().unwrap_boolean();

                                            if ui.checkbox(&input_label, &mut boolean_lit) {
                                                change = Some((
                                                    stmt_index,
                                                    arg_index,
                                                    ast::Expr::Lit(ast::LitExpr::Boolean(
                                                        boolean_lit,
                                                    )),
                                                ));
                                            }
                                        }
                                        ParamRefinement::Int(param_refinement_int) => {
                                            let mut int_lit = arg.unwrap_literal().unwrap_int();

                                            if ui.input_int(&input_label, &mut int_lit)
                                                .read_only(interpreter_busy)
                                                .build()
                                            {
                                                int_lit = param_refinement_int.clamp(int_lit);
                                                change = Some((
                                                    stmt_index,
                                                    arg_index,
                                                    ast::Expr::Lit(ast::LitExpr::Int(int_lit)),
                                                ));
                                            }
                                        }
                                        ParamRefinement::Uint(param_refinement_uint) => {
                                            let uint_lit = arg.unwrap_literal().unwrap_uint();
                                            let mut int_value = clamp_cast_u32_to_i32(uint_lit);

                                            if ui.input_int(&input_label, &mut int_value)
                                                .read_only(interpreter_busy)
                                                .build()
                                            {
                                                let uint_value = clamp_cast_i32_to_u32(int_value);
                                                let uint_value = param_refinement_uint.clamp(uint_value);
                                                change = Some((
                                                    stmt_index,
                                                    arg_index,
                                                    ast::Expr::Lit(ast::LitExpr::Uint(uint_value)),
                                                ));
                                            }
                                        }
                                        ParamRefinement::Float(param_refinement_float) => {
                                            let mut float_lit = arg.unwrap_literal().unwrap_float();

                                            if ui.input_float(&input_label, &mut float_lit)
                                                .read_only(interpreter_busy)
                                                .build()
                                            {
                                                float_lit = param_refinement_float.clamp(float_lit);
                                                change = Some((
                                                    stmt_index,
                                                    arg_index,
                                                    ast::Expr::Lit(ast::LitExpr::Float(float_lit)),
                                                ));
                                            }
                                        }
                                        ParamRefinement::Float3(param_refinement_float3) => {
                                            let mut float3_lit =
                                                arg.unwrap_literal().unwrap_float3();

                                            if ui
                                                .input_float3(&input_label, &mut float3_lit)
                                                .read_only(interpreter_busy)
                                                .build()
                                            {
                                                float3_lit = param_refinement_float3.clamp(float3_lit);
                                                change = Some((
                                                    stmt_index,
                                                    arg_index,
                                                    ast::Expr::Lit(ast::LitExpr::Float3(
                                                        float3_lit,
                                                    )),
                                                ));
                                            }
                                        }
                                        ParamRefinement::String => {
                                            let string_lit = arg.unwrap_literal().unwrap_string();
                                            let mut imstring_value = imgui::ImString::new(string_lit);
                                            imstring_value.reserve(128);

                                            if ui
                                                .input_text(&input_label, &mut imstring_value)
                                                .read_only(interpreter_busy)
                                                .build() {
                                                    let string_value = format!("{}", imstring_value);
                                                    let string_value = Arc::new(string_value);
                                                    change = Some((
                                                        stmt_index,
                                                        arg_index,
                                                        ast::Expr::Lit(ast::LitExpr::String(string_value)),
                                                    ));
                                                }
                                        }
                                        ParamRefinement::Geometry => {
                                            let visible_vars =
                                                session.var_visibility_at_stmt(stmt_index);

                                            let mut selected_var_index = match arg {
                                                ast::Expr::Lit(ast::LitExpr::Nil) => None,
                                                ast::Expr::Var(var) => visible_vars
                                                    .iter()
                                                    .position(|var_ident| *var_ident == var.ident())
                                                    .map(Some)
                                                    .unwrap_or(None),
                                                _ => panic!("Arg can either be a variable or nil"),
                                            };

                                            // FIXME: Show used var idents
                                            // differently from unused,
                                            // e.g. grayed-out

                                            let combo_changed = {
                                                // FIXME: find a way to make combo boxes read-only
                                                let mut combo = imgui::ComboBox::new(&input_label);

                                                let mut result = false;
                                                let preview_value = selected_var_index
                                                    .map(|index| visible_vars.get(index).unwrap())
                                                    .map(|var_ident| {
                                                        format_var_name(
                                                            session
                                                                .var_name_for_ident(*var_ident)
                                                                .expect(
                                                                    "Failed to find name to ident",
                                                                ),
                                                            *var_ident,
                                                        )
                                                    })
                                                    .unwrap_or_else(|| imgui::ImString::new("<Nil>"));

                                                combo = combo.preview_value(&preview_value);

                                                if let Some(combo_token) = combo.begin(ui) {
                                                    for (index, var_ident) in
                                                        visible_vars.iter().enumerate()
                                                    {
                                                        let text = format_var_name(
                                                            session
                                                                .var_name_for_ident(*var_ident)
                                                                .expect(
                                                                    "Failed to find name for ident",
                                                                ),
                                                            *var_ident,
                                                        );
                                                        let selected =
                                                            if let Some(selected_var_index) =
                                                                selected_var_index
                                                            {
                                                                index == selected_var_index
                                                            } else {
                                                                false
                                                            };

                                                        if imgui::Selectable::new(&text)
                                                            .selected(selected)
                                                            .build(ui)
                                                        {
                                                            selected_var_index = Some(index);
                                                            result = true;
                                                        }
                                                    }

                                                    if imgui::Selectable::new(imgui::im_str!(
                                                        "<Nil>"
                                                    ))
                                                    .selected(selected_var_index.is_none())
                                                    .build(ui)
                                                    {
                                                        selected_var_index = None;
                                                        result = true;
                                                    }

                                                    combo_token.end(ui);
                                                }

                                                result
                                            };

                                            if combo_changed {
                                                if let Some(selected_var_index) = selected_var_index
                                                {
                                                    change = Some((
                                                        stmt_index,
                                                        arg_index,
                                                        ast::Expr::Var(ast::VarExpr::new(
                                                            visible_vars[selected_var_index],
                                                        )),
                                                    ))
                                                } else {
                                                    change = Some((
                                                        stmt_index,
                                                        arg_index,
                                                        ast::Expr::Lit(ast::LitExpr::Nil),
                                                    ))
                                                }
                                            }
                                        }
                                    }
                                }

                                let token = ui.push_style_color(
                                    imgui::StyleColor::FrameBg,
                                    [0.080, 0.080, 0.080, 0.940],
                                );

                                imgui::InputTextMultiline::new(
                                    ui,
                                    &imgui::im_str!("##console{}", stmt_index),
                                    &mut imgui::ImString::new("Lorem Ipsum Dolor Sit Amet"),
                                    [0.0, 60.0],
                                )
                                    .read_only(true)
                                    .build();

                                token.pop(ui);

                                if let Some((color_token, style_token)) = operation_arg_style_tokens {
                                    color_token.pop(ui);
                                    style_token.pop(ui);
                                }

                                ui.unindent();
                            }
                        }
                    }
                }
            });

        // FIXME: Debounce changes to parameters

        // Only submit the change if interpreter is not busy. Not all
        // imgui components can be made read-only, so we can not trust
        // it.
        if !interpreter_busy {
            if let Some((stmt_index, arg_index, expr)) = change {
                let stmt = &session.stmts()[stmt_index];
                match stmt {
                    ast::Stmt::VarDecl(var_decl) => {
                        let init_expr = var_decl.init_expr();
                        let new_var_decl = var_decl
                            .clone_with_init_expr(init_expr.clone_with_arg_at(arg_index, expr));

                        session.set_prog_stmt_at(stmt_index, ast::Stmt::VarDecl(new_var_decl));
                    }
                }
            }
        }
    }

    pub fn draw_operations_window(&self, session: &mut Session) {
        let ui = &self.imgui_ui;
        let function_table = session.function_table();

        const OPERATIONS_WINDOW_WIDTH: f32 = 400.0;
        const OPERATIONS_WINDOW_HEIGHT_MULT: f32 = 0.2;

        let window_logical_size = ui.io().display_size;
        let window_inner_height = window_logical_size[1] - 2.0 * MARGIN;

        let operations_window_height = window_inner_height * OPERATIONS_WINDOW_HEIGHT_MULT - MARGIN;
        let operations_window_vertical_position =
            MARGIN * 2.0 + (1.0 - OPERATIONS_WINDOW_HEIGHT_MULT) * window_inner_height;

        let running_enabled = !session.interpreter_busy() && !session.stmts().is_empty();
        let popping_enabled = !session.interpreter_busy() && !session.stmts().is_empty();
        let pushing_enabled = !session.interpreter_busy();

        let mut function_clicked = None;
        let mut interpret_clicked = false;
        let mut pop_stmt_clicked = false;

        imgui::Window::new(imgui::im_str!("Operations"))
            .movable(false)
            .resizable(false)
            .size(
                [OPERATIONS_WINDOW_WIDTH, operations_window_height],
                imgui::Condition::Always,
            )
            .position(
                [MARGIN, operations_window_vertical_position],
                imgui::Condition::Always,
            )
            .build(ui, || {
                ui.columns(2, imgui::im_str!("Controls columns"), false);

                let running_tokens = if running_enabled {
                    None
                } else {
                    Some(push_disabled_style(ui))
                };
                if ui.button(imgui::im_str!("Run pipeline"), [-f32::MIN_POSITIVE, 25.0])
                    && running_enabled
                {
                    interpret_clicked = true;
                }
                if let Some((color_token, style_token)) = running_tokens {
                    color_token.pop(ui);
                    style_token.pop(ui);
                }

                ui.next_column();

                let popping_tokens = if popping_enabled {
                    None
                } else {
                    Some(push_disabled_style(ui))
                };
                if ui.button(
                    imgui::im_str!("Remove last operation"),
                    [-f32::MIN_POSITIVE, 25.0],
                ) && popping_enabled
                {
                    pop_stmt_clicked = true;
                }
                if let Some((color_token, style_token)) = popping_tokens {
                    color_token.pop(ui);
                    style_token.pop(ui);
                }

                ui.separator();

                let pushing_tokens = if pushing_enabled {
                    None
                } else {
                    Some(push_disabled_style(ui))
                };
                ui.columns(3, imgui::im_str!("Add operations columns"), false);
                for (func_ident, func) in function_table {
                    if ui.button(
                        &imgui::im_str!("{}", func.info().name),
                        [-f32::MIN_POSITIVE, 20.0],
                    ) && pushing_enabled
                    {
                        function_clicked = Some(func_ident);
                    }
                    ui.next_column();
                }
                if let Some((color_token, style_token)) = pushing_tokens {
                    color_token.pop(ui);
                    style_token.pop(ui);
                }
            });

        if let Some(func_ident) = function_clicked {
            let func = &function_table[&func_ident];
            let mut args = Vec::with_capacity(func.param_info().len());

            for param_info in func.param_info() {
                let expr = match param_info.refinement {
                    ParamRefinement::Boolean(boolean_refinement) => {
                        ast::Expr::Lit(ast::LitExpr::Boolean(boolean_refinement.default_value))
                    }
                    ParamRefinement::Int(int_param_refinement) => ast::Expr::Lit(
                        ast::LitExpr::Int(int_param_refinement.default_value.unwrap_or_default()),
                    ),
                    ParamRefinement::Uint(uint_param_refinement) => ast::Expr::Lit(
                        ast::LitExpr::Uint(uint_param_refinement.default_value.unwrap_or_default()),
                    ),
                    ParamRefinement::Float(float_param_refinement) => {
                        ast::Expr::Lit(ast::LitExpr::Float(
                            float_param_refinement.default_value.unwrap_or_default(),
                        ))
                    }
                    ParamRefinement::Float3(float3_param_refinement) => {
                        ast::Expr::Lit(ast::LitExpr::Float3([
                            float3_param_refinement.default_value_x.unwrap_or_default(),
                            float3_param_refinement.default_value_y.unwrap_or_default(),
                            float3_param_refinement.default_value_z.unwrap_or_default(),
                        ]))
                    }
                    ParamRefinement::String => {
                        ast::Expr::Lit(ast::LitExpr::String(Arc::new(String::new())))
                    }
                    ParamRefinement::Geometry => {
                        let visible_vars = session.var_visibility_at_stmt(session.stmts().len());
                        if visible_vars.is_empty() {
                            ast::Expr::Lit(ast::LitExpr::Nil)
                        } else {
                            ast::Expr::Var(ast::VarExpr::new(visible_vars[0]))
                        }
                    }
                };

                args.push(expr);
            }

            let init_expr = ast::CallExpr::new(*func_ident, args);
            let stmt = ast::Stmt::VarDecl(ast::VarDeclStmt::new(
                session.next_free_var_ident(),
                init_expr,
            ));

            session.push_prog_stmt(stmt);
        }

        if interpret_clicked {
            session.interpret();
        }

        if pop_stmt_clicked {
            session.pop_prog_stmt();
        }
    }
}

fn format_var_name(name: &str, ident: ast::VarIdent) -> imgui::ImString {
    imgui::im_str!("{} #{}", name, ident.0 + 1)
}

fn push_disabled_style(ui: &imgui::Ui) -> (imgui::ColorStackToken, imgui::StyleStackToken) {
    let frame_color = ui.style_color(imgui::StyleColor::Button);

    let color_token = ui.push_style_colors(&[
        (imgui::StyleColor::Button, frame_color),
        (imgui::StyleColor::ButtonHovered, frame_color),
        (imgui::StyleColor::ButtonActive, frame_color),
    ]);
    let style_token = ui.push_style_vars(&[imgui::StyleVar::Alpha(0.5)]);

    (color_token, style_token)
}
