use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Build the Go shared library
    println!("cargo:rerun-if-changed=main.go");
    println!("cargo:rerun-if-changed=go.mod");

    let status = Command::new("go")
        .args(["mod", "tidy"])
        .current_dir(&manifest_dir)
        .status()
        .expect("Failed to run go mod tidy");

    if !status.success() {
        panic!("go mod tidy failed");
    }

    let lib_path = out_dir.join("libanthropic.so");

    let status = Command::new("go")
        .args([
            "build",
            "-buildmode=c-shared",
            "-o",
            lib_path.to_str().unwrap(),
            "main.go",
        ])
        .current_dir(&manifest_dir)
        .status()
        .expect("Failed to build Go library");

    if !status.success() {
        panic!("Failed to build Go shared library");
    }

    // Link the library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=anthropic");

    // Set rpath to $ORIGIN so the library is found next to the binary
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");

    // Also set rpath to the build output dir for development
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out_dir.display());

    // Tell cargo to copy the library to the target directory
    // We output a path that the main crate's build.rs can use
    println!("cargo:rustc-env=ANTHROPIC_LIB_PATH={}", lib_path.display());
}
