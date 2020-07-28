// Do not display console on windows in dist builds
#![cfg_attr(feature = "dist", windows_subsystem = "windows")]

use clap::Clap as _;
use hurban_selector as hs;

fn main() {
    let options = hs::Options::parse();
    hs::init_and_run(options);
}
