use zoon::{println, *};

use crate::home;
use crate::config;

// ------ Page ------

#[derive(Clone, Copy, Debug, PartialEq)]
enum Page {
    Home,
    Config,
    NotFound,
}

#[static_ref]
fn page() -> &'static Mutable<Page> {
    println!("[INFO] static_ref `page` initialized");

    Mutable::new(Page::Home)
}

fn set_page(new_page: Page) {
    println!("[INFO] Page -> {:?}", new_page);
    page().set_neq(new_page);
}

// ------ Router ------

#[route]
#[derive(Clone, Copy)]
enum Route {
    #[route()]
    Home,
    #[route("config")]
    Config
}

#[static_ref]
fn router() -> &'static Router<Route> {
    Router::new(|route: Option<Route>| async move {
        println!("URL: {}",routing::url());
        let Some(route) = route else { return set_page(Page::NotFound) };

        match route {
            Route::Home => set_page(Page::Home),
            Route::Config => set_page(Page::Config),
        }
    })
}

// ------ Main (init) ------

pub fn main() -> impl Element {
    router();

    root()
}

fn root() -> impl Element {
    El::new()
        .child_signal(page().signal().map(|page| match page {
            Page::Home => home::page_content().into_raw(),
            Page::Config => config::page_content().into_raw(),
            Page::NotFound => El::new().child("Not found").into_raw(),
        }))
}
