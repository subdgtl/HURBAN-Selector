# HURBAN Selector

[![Build Status](https://travis-ci.org/sub-digital/HURBAN-Selector.svg?branch=master)](https://travis-ci.org/sub-digital/HURBAN-Selector)

## Prerequisites

* [Rust](https://rustup.rs/)
* [Dependencies for `shaderc-sys`](https://github.com/google/shaderc-rs#building-from-source)

## Running

Use `cargo run` with one of the following feature flags, depending on the platform:

* `vulkan` on Windows
* `metal` on macOS

Eg. `cargo run --features vulkan`