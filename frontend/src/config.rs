use zoon::{println, named_color::*, *};


#[static_ref]
fn config_bool() -> &'static Mutable<bool> {
    println!("[INFO] static_ref `config_bool` initialized");

    Mutable::new(true)
}

pub fn page_content() -> impl Element {
    Column::new()
        .s(Font::new().color(GRAY_0).size(24))
        .item(Text::with_signal(config_bool().signal().map_bool(|| "TRUE", || "FALSE")))
}