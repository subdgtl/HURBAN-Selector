# H.U.R.B.A.N. selector

[![Build Status](https://dev.azure.com/subdgtl/HURBAN-Selector/_apis/build/status/Master%20and%20PR?branchName=master)](https://dev.azure.com/subdgtl/HURBAN-Selector/_build/latest?definitionId=1&branchName=master)

H.U.R.B.A.N. selector is a software experiment sponsored by the [Slovak
Design Center](https://www.scd.sk/). It is meant to test the
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

![Screenshot](./hurban_selector-2020-02-03-15-20-24.png)

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

### Configuration

The application parses configuration from both command-line arguments and
environment variables. Run with `-h` to see the list of available options.

### Creating a Windows installer

Refer to [installer's readme](installer/README.md) to create Windows installer.

### Testing

Apart from unit and integration tests, we do have a fair amount of
snapshot tests. These are used mostly to check for regressions in
operation implementations. The workflow is to always manually review
the new snapshot, if the operation's output has changed.

Snapshots are handled by the
[insta](https://docs.rs/insta/) crate. The `cargo insta`
plugin, while not strictly necessary, is also useful in the
workflow. Get it with `cargo install cargo-insta`.

To make a new snapshot test, add a standard test and use
`insta::assert_json_snapshot("name_of_snapshot", &data)`. The test
will fail at first, as there is no snapshot to compare against. Use
the `cargo insta review` to review snapshot diffs or new snapshots.

### Renderer development

If working on the renderer, Vulkan validation layers can provide additional
validation. When running on the Vulkan backend and with `debug_assertions`
enabled, `gfx-hal` automatically enables Vulkan validation layers.

For the implementation to be able to load the validation layers, the [LunarG
Vulkan SDK](https://vulkan.lunarg.com/) must be installed. Note that the Vulkan
SDK can even be installed on macOS, and enables running on the Vulkan backend
there, but the setup is a bit more involved - see macOS guide in
[ash](https://crates.io/crates/ash)), and possibly [disable
SIP](http://osxdaily.com/2015/10/05/disable-rootless-system-integrity-protection-mac-os-x/)

### Licence

The editor source code is provided under the GNU GENERAL PUBLIC
LICENSE, Version 3. If the research or implementation yields
interesting results, those will be extracted from the editor and
published and licensed separately, most likely under a more permissive
license such as MIT.
