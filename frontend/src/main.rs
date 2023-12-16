use zoon::start_app;

mod app;
mod home;
mod config;

fn main() {
    start_app("app", app::main);
}
