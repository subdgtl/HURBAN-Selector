# HURBAN Selector

[![Build Status](https://dev.azure.com/subdigital/HURBAN-selector/_apis/build/status/sub-digital.HURBAN-Selector?branchName=master)](https://dev.azure.com/subdigital/HURBAN-selector/_build/latest?definitionId=1&branchName=master)

H.U.R.B.A.N. Selector is a software experiment sponsored by the [Slovak
Center of Design](https://www.scd.sk/). It is meant to test the
hypothesis that creating new designs and shapes is subconsciously
inspired by our previous experience. There is a trial and error phase
in the design process where many variations on the same shape are
prototyped and chosen from.

The software is currently in very early stages, but as it nears
completion, it will strive to be a tool for simple parametric
modeling, containing implementations of various hybridization
strategies for mesh models, allowing designers to smoothly interpolate
between multiple mesh geometries and select the result with the most
desired features.

_Screenshots coming soon™!_

## Getting the software

Currently the only option is to build from source (see development
guide down below).

## Developing

Make sure you have the following installed:

- [Rust](https://rustup.rs/)
- Clippy (`rustup component add clippy`)
- Rustfmt (`rustup component add rustfmt`)
- [Dependencies for `shaderc-sys`](https://github.com/google/shaderc-rs#building-from-source)

We use standard `cargo` workflows:

- `cargo clippy` to have a nice chat with 📎, the linter,
- `cargo fmt` to format the project,
- `cargo test` to run tests,
- `cargo doc --open` to build and open local documentation for the
  project and all dependencies (optionally pass
  `--document-private-items`),
- `cargo build` to build,
- `cargo run` to run.

### Testing

Apart from unit and integration tests, we do have a fair amount of
snapshot tests. These are used mostly to check for regressions in
operation implementations. The workflow is to always manually review
the new snapshot, if the operation's output has changed.

Snapshots are handled by the
[insta](https://docs.rs/insta/0.12.0/insta/) crate. The `cargo insta`
plugin, while not strictly necessary, is also useful in the
workflow. Get it with `cargo install cargo-insta`.

To make a new snapshot test, add a standard test and use
`insta::assert_json_snapshot("name_of_snapshot", &data)`. The test
will fail at first, as there is no snapshot to compare against. Use
the `cargo insta review` to review snapshot diffs or new snapshots.

### Environment Variables

**HS_GPU_BACKEND (optional)**: Force a GPU backend.

A gpu backend is automatically selected, but optionally a non-default
gpu backend can be specified with `HS_GPU_BACKEND`. Can take the
following values:

- `vulkan` on Windows, Linux, or macOS with VulkanSDK,
- `d3d12` on Windows,
- `metal` on macOS.

If working on the renderer, enabling Vulkan validation layers is
useful for additional validation:

``` shell
VK_LAYER_PATH=$VULKAN_SDK/bin \
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_VALIDATION \
cargo run
```

To run with Vulkan backend on macOS, you need to setup the VulkanSDK
(see macOS guide in [ash](https://crates.io/crates/ash)), and possibly
[disable
SIP](http://osxdaily.com/2015/10/05/disable-rootless-system-integrity-protection-mac-os-x/)

**HS_MSAA (optional)**: Force number of samples for multisampling, either 1,
4, 8, or 16.

**HS_VSYNC (optional)**: Explicitly enable (1) or disable (0) VSync.

**HS_LIBS_LOG_LEVEL (optional)**: Changes level of logging for external crates.
It is `warn` by default. Options are `error`, `warn`, `info`, `debug`, `trace`
and `off`.

### Licence

The editor source code is provided under the GNU GENERAL PUBLIC
LICENSE, Version 3. If the research or implementation yields
interesting results, those will be extracted from the editor and
published and licensed separately, most likely under a more permissive
license such as MIT.
