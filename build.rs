use std::{fs, path::PathBuf, process::Command};

fn main() {
    let cuda_dir = PathBuf::from("cuda");
    Command::new("./build_everything")
        .current_dir(&cuda_dir)
        .status()
        .unwrap();
    let cuda_so_base_name = "cuda_lin_alg";
    let cuda_so_name = format!("lib{cuda_so_base_name}.so");
    let lib_path = fs::read_dir(&cuda_dir)
        .unwrap()
        .filter_map(|entry| {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.file_name() == Some(cuda_so_name.as_ref()) {
                Some(path)
            } else {
                None
            }
        })
        .next()
        .expect("\ncuda shared lib not found\n");
    let so_dir = lib_path.parent().unwrap();
    // Link the cuda shared lib
    println!("cargo:rustc-link-search=native={}", so_dir.display());
    println!("cargo:rustc-link-lib=dylib=cuda_lin_alg");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", so_dir.display());
    // Rerun if the cuda directory changes
    println!("cargo:rerun-if-changed={}", cuda_dir.display());
}
