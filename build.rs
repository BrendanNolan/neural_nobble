use std::{fs, path::PathBuf, process::Command};

fn main() {
    let cuda_dir = fs::canonicalize("./cuda").unwrap();
    let cuda_so = create_cuda_so(&cuda_dir);
    let so_dir = cuda_so.parent().unwrap();
    println!("cargo:rustc-link-search=native={}", so_dir.display());
    println!("cargo:rustc-link-lib=dylib=cuda_lin_alg");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", so_dir.display());
    println!("cargo:rerun-if-changed={}", cuda_dir.display());
}

fn create_cuda_so(cuda_dir: &PathBuf) -> PathBuf {
    Command::new("./build_everything")
        .current_dir(cuda_dir)
        .status()
        .unwrap();
    let cuda_so_base_name = "cuda_lin_alg";
    let cuda_so_name = format!("lib{cuda_so_base_name}.so");
    let std_out_from_find = Command::new("find")
        .arg(cuda_dir)
        .arg("-name")
        .arg(&cuda_so_name)
        .output()
        .unwrap();
    let path = String::from_utf8(std_out_from_find.stdout)
        .unwrap()
        .lines()
        .next()
        .map(|s| s.to_owned())
        .unwrap_or_else(|| {
            panic!(
                "\n{} not found in {}\n",
                cuda_so_name,
                cuda_dir.to_string_lossy()
            )
        });
    PathBuf::from(path)
}
