fn main() {
    // Removed: libanthropic.so is now built on-demand at runtime in ~/.config/palace/lib/
    // This makes Palace distributable without requiring Go at build time

    // Set rpath so the binary can find shared libraries
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
}
