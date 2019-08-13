# HURBAN Selector

[![Build Status](https://travis-ci.org/sub-digital/HURBAN-Selector.svg?branch=master)](https://travis-ci.org/sub-digital/HURBAN-Selector)
[![Build Status](https://dev.azure.com/subdigital/HURBAN-selector/_apis/build/status/sub-digital.HURBAN-Selector?branchName=master)](https://dev.azure.com/subdigital/HURBAN-selector/_build/latest?definitionId=1&branchName=master)

## Prerequisites

- [Rust](https://rustup.rs/)
- [Dependencies for `shaderc-sys`](https://github.com/google/shaderc-rs#building-from-source)

## Running

Use `cargo run` with one of the following feature flags, depending on the platform:

- `vulkan` or `d3d12` on Windows
- `metal` on macOS

E.g. `cargo run --features vulkan`

Other rendering backends (d3d11, OpenGL) may become supported once
they are stable enough in `wgpu` and `gfx-hal`.
