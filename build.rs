use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let mlir_prefix =
        env::var("MLIR_SYS_230_PREFIX").expect("MLIR_SYS_230_PREFIX must be set");

    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cpp_dir = crate_dir.join("cpp");
    let build_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("cpp");
    // The dialect compiles against the driver's PermissiveInlinerInterface.hpp
    // rather than linking it, so this crate depends on the driver crate for its
    // source directory and recompiles when the driver's headers change.
    let lowering_driver_source_dir = env::var("DEP_LOWERING_DRIVER_SOURCE_DIR")
        .expect("DEP_LOWERING_DRIVER_SOURCE_DIR must be set by mlir-lowering-driver");

    // Keep generated TableGen files and native objects under Cargo's output
    // directory so profile builds and MLIR-prefix changes cannot share stale
    // source-tree artifacts.
    let status = Command::new("make")
        .arg("-j")
        .arg(format!("BUILD_DIR={}", build_dir.display()))
        .arg(format!("LOWERING_DRIVER_SOURCE_DIR={lowering_driver_source_dir}"))
        .current_dir(&cpp_dir)
        .status()
        .expect("Failed to run make in cpp/");

    if !status.success() {
        panic!("C++ build failed");
    }

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=printf_dialect");

    // The directory the plugin `.so` stands in beside the archive, exposed as
    // DEP_PRINTF_DIALECT_LIB_DIR to a dependent that links a staged program.
    println!("cargo:lib_dir={}", build_dir.display());

    // The dialect plugins this crate's lit rows load, each after the plugins it
    // references, published for the lit rows of the crates that depend on this one.
    let plugins = build_support::publish_dialect_plugins(build_dir.join("libprintf_dialect.so"));
    build_support::write_lit_site_config(
        &mlir_prefix,
        "LIT_SUITE",
        "cpp/tests",
        &[
            ("dialect_plugins", build_support::python_paths(&plugins)),
        ],
    );

    println!("cargo:rustc-link-arg=-Wl,-rpath,{mlir_prefix}/lib");

    println!("cargo:rerun-if-env-changed=MLIR_SYS_230_PREFIX");
    println!("cargo:rerun-if-changed=cpp");
    println!("cargo:rerun-if-changed={lowering_driver_source_dir}");
}

