use std::env;
use std::path::Path;

// FIXME: @Correctness Replace tinyfiledialogs usage here with our own immediate
// filepicker so that the UI can be pure computation. The filepicker will have
// to communicate with the outside via some kind of Filesystem interface.

pub fn input_file(
    ui: &imgui::Ui,
    label_and_id: &imgui::ImStr,
    file_ext_filter: Option<(&[&str], &str)>,
    buffer: &mut imgui::ImString,
) -> bool {
    let open_button_label = imgui::im_str!("Open##{}", label_and_id);
    let open_button_width = ui.calc_text_size(&open_button_label, true, 50.0)[0] + 8.0;
    let input_position = open_button_width + 2.0; // Padding

    let mut changed = false;

    let group_token = ui.begin_group();

    if ui.button(&open_button_label, [open_button_width, 0.0]) {
        if let Some(absolute_path_string) =
            tinyfiledialogs::open_file_dialog("Open", "", file_ext_filter)
        {
            buffer.clear();

            let current_dir = env::current_dir().expect("Couldn't get current dir");
            let absolute_path = Path::new(&absolute_path_string);

            match absolute_path.strip_prefix(&current_dir) {
                Ok(stripped_path) => {
                    buffer.push_str(&stripped_path.to_string_lossy());
                }
                Err(_) => {
                    buffer.push_str(&absolute_path.to_string_lossy());
                }
            }
        }

        changed = true;
    }

    ui.same_line(input_position);
    ui.set_next_item_width(ui.calc_item_width() - input_position);

    ui.input_text(&label_and_id, buffer).read_only(true).build();

    group_token.end(ui);

    changed
}
