pub fn input_float2(
    ui: &imgui::Ui,
    label_and_id: &imgui::ImStr,
    value: &mut [f32; 2],
) -> bool {
    ui.input_float2(label_and_id, value).build()
}
