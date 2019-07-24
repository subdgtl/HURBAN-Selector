use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::ops::Deref;
use std::path::{Path, PathBuf};

use shaderc;

macro_rules! warn {
    ($msg:expr) => ({
        std::println!(concat!("cargo:warning=Build script warning: ", $msg))
    });
    ($fmt:expr, $($arg:tt)*) => ({
        std::println!(concat!("cargo:warning=Build script warning: ", $fmt), $($arg)*)
    });
}

fn main() {
    // Tell cargo to only rerun this script if it detects changes in `src/shaders`
    println!("cargo:rerun-if-changed=src/shaders");

    let current_dir = env::current_dir().expect("Build script needs current directory");
    let out_dir = std::env::var("OUT_DIR").expect("Build script expects an OUT_DIR");

    let src_dir: PathBuf = [&current_dir, Path::new("src/shaders")].iter().collect();
    let dst_dir: PathBuf = [&out_dir, "shaders"].iter().collect();

    let mut compile_options = shaderc::CompileOptions::new()
        .expect("Build script failed to create shaderc compile options");

    compile_options.set_target_env(shaderc::TargetEnv::Vulkan, 0);
    compile_options.set_source_language(shaderc::SourceLanguage::GLSL);

    #[cfg(debug_assertions)]
    compile_options.set_optimization_level(shaderc::OptimizationLevel::Zero);
    #[cfg(not(debug_assertions))]
    compile_options.set_optimization_level(shaderc::OptimizationLevel::Performance);

    let mut compiler =
        shaderc::Compiler::new().expect("Build script failed to create shaderc compiler");

    compile_shader_directory(&mut compiler, &compile_options, &src_dir, &dst_dir);
}

fn compile_shader_directory(
    compiler: &mut shaderc::Compiler,
    compile_options: &shaderc::CompileOptions,
    src_dir: &Path,
    dst_dir: &Path,
) {
    assert!(src_dir.is_dir(), "Source path must be a directory");
    if !dst_dir.is_dir() {
        fs::create_dir_all(dst_dir).expect("Build script failed creating shader output directory");
    }

    let src_dir_name = &*src_dir.to_string_lossy();

    for entry in fs::read_dir(src_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        let file_name_raw = path.file_name().expect("Shader file must have a file name");

        if path.is_dir() {
            warn!("Skipping directory {}", file_name_raw.to_string_lossy());
            continue;
        } else {
            let shader_file_name = match file_name_raw.to_str() {
                Some(file_name) => file_name,
                None => {
                    warn!(
                        "Skipping file ({}) with invalid UTF-8 name in {}",
                        file_name_raw.to_string_lossy(),
                        src_dir_name
                    );
                    continue;
                }
            };
            let shader_source =
                fs::read_to_string(&path).expect("Build script failed to read shader source");
            let shader_kind = match path.extension() {
                Some(ext) => match Deref::deref(&ext.to_string_lossy()) {
                    "vert" => shaderc::ShaderKind::Vertex,
                    "frag" => shaderc::ShaderKind::Fragment,
                    _ => {
                        warn!(
                            "Skipping file ({}) with unknown extension in {}",
                            file_name_raw.to_string_lossy(),
                            src_dir_name
                        );
                        continue;
                    }
                },
                None => {
                    warn!("Skipping file with no extension in {}", src_dir_name);
                    continue;
                }
            };

            let spirv = compile_shader_source(
                compiler,
                compile_options,
                shader_file_name,
                &shader_source,
                shader_kind,
            );

            let output_shader_file = shader_file_name.to_string() + ".spv";
            let output_file_path = dst_dir.join(output_shader_file);

            File::create(output_file_path)
                .expect("Build script failed to create output file descriptor")
                .write_all(&spirv)
                .expect("Build script failed to write shader file");
        }
    }
}

fn compile_shader_source(
    compiler: &mut shaderc::Compiler,
    compile_options: &shaderc::CompileOptions,
    shader_file_name: &str,
    shader_source: &str,
    shader_kind: shaderc::ShaderKind,
) -> Vec<u8> {
    match compiler.compile_into_spirv(
        shader_source,
        shader_kind,
        shader_file_name,
        "main",
        Some(compile_options),
    ) {
        Ok(compilation_artifact) => Vec::from(compilation_artifact.as_binary_u8()),
        Err(err) => panic!("Build script failed: shader compile error:\n\n {}", err),
    }
}
