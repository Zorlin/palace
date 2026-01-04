use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // OUT_DIR is like: target/debug/build/palace-xxx/out
    // We need to find the anthropic-ffi build dir and copy the .so
    let target_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("Could not find target dir");

    // Find libanthropic.so in the build directory
    let build_dir = target_dir.join("build");
    if let Ok(entries) = fs::read_dir(&build_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            if name.to_string_lossy().starts_with("anthropic-ffi-") {
                let lib_path = entry.path().join("out").join("libanthropic.so");
                if lib_path.exists() {
                    let dest = target_dir.join("libanthropic.so");
                    if let Err(e) = fs::copy(&lib_path, &dest) {
                        println!("cargo:warning=Failed to copy libanthropic.so: {}", e);
                    }
                    break;
                }
            }
        }
    }

    // Set rpath so the binary can find libanthropic.so next to itself
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
}
