// Removed: We now build libanthropic.so at runtime on-demand
// This makes Palace distributable without requiring Go at build time

fn main() {
    // No-op build script - FFI library is built lazily at runtime
    println!("cargo:rerun-if-changed=main.go");
    println!("cargo:rerun-if-changed=go.mod");
}
