//! Compiles `kernels/gravity.hip` into a HIP code object for the `rocm` feature.
//!
//! The toolchain lookup follows the one of candle-rocm-vulkan's `candle-core`
//! build script (Apache-2.0): `HIPCC`, then `ROCM_PATH` / `HIP_PATH`, then
//! `PATH`; the GPU targets come from `HIP_ARCH` / `CANDLE_ROCM_ARCHS`, else from
//! `amdgpu-arch`, else a list of common RDNA / CDNA targets.

use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_ARCHS: &[&str] = &["gfx1030", "gfx1031", "gfx1032", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201"];

fn env_path(var: &str) -> Option<PathBuf> {
    let v = std::env::var(var).ok()?;
    let v = v.trim().trim_end_matches(['\\', '/']);
    (!v.is_empty()).then(|| PathBuf::from(v))
}

fn exe(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

fn rocm_root() -> Option<PathBuf> {
    for var in ["ROCM_PATH", "HIP_PATH"] {
        if let Some(p) = env_path(var) {
            if p.is_dir() {
                return Some(p);
            }
        }
    }
    let fixed = if cfg!(windows) { "C:/Program Files/AMD/ROCm" } else { "/opt/rocm" };
    let p = PathBuf::from(fixed);
    if !p.is_dir() {
        return None;
    }
    if cfg!(windows) {
        // newest versioned folder
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(&p).ok()?.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
        dirs.sort();
        return dirs.pop();
    }
    Some(p)
}

fn hipcc(root: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = env_path("HIPCC") {
        return Some(p);
    }
    if let Some(root) = root {
        for name in ["hipcc", "hipcc.bin"] {
            let p = root.join("bin").join(exe(name));
            if p.is_file() {
                return Some(p);
            }
        }
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(exe("hipcc"))).find(|p| p.is_file())
}

fn archs(root: Option<&Path>) -> Vec<String> {
    for var in ["HIP_ARCH", "CANDLE_ROCM_ARCHS"] {
        if let Ok(v) = std::env::var(var) {
            let list: Vec<String> = v.split([',', ';', ' ']).map(|s| s.trim().to_string()).filter(|s| s.starts_with("gfx")).collect();
            if !list.is_empty() {
                return list;
            }
        }
    }
    if let Some(root) = root {
        for dir in [root.join("bin"), root.join("llvm").join("bin"), root.join("lib").join("llvm").join("bin")] {
            let tool = dir.join(exe("amdgpu-arch"));
            if !tool.is_file() {
                continue;
            }
            if let Ok(out) = Command::new(&tool).env("PATH", root.join("bin")).output() {
                let mut found: Vec<String> = Vec::new();
                for tok in String::from_utf8_lossy(&out.stdout).split_whitespace() {
                    let base = tok.split(':').next().unwrap_or(tok).to_string();
                    if base.starts_with("gfx") && base != "gfx000" && !found.contains(&base) {
                        found.push(base);
                    }
                }
                if !found.is_empty() {
                    return found;
                }
            }
        }
    }
    DEFAULT_ARCHS.iter().map(|s| s.to_string()).collect()
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/gravity.hip");
    for var in ["HIPCC", "ROCM_PATH", "HIP_PATH", "HIP_ARCH", "CANDLE_ROCM_ARCHS"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let out = out_dir.join("gravity.co");
    if std::env::var_os("CARGO_FEATURE_ROCM").is_none() {
        std::fs::write(&out, b"").unwrap();
        println!("cargo:rustc-env=GRAVITY_HIP_ARCHS=none");
        return;
    }
    let root = rocm_root();
    let Some(cc) = hipcc(root.as_deref()) else {
        panic!("hipcc not found: set HIPCC, ROCM_PATH or HIP_PATH (the rocm feature compiles kernels/gravity.hip)");
    };
    let archs = archs(root.as_deref());
    let mut cmd = Command::new(&cc);
    cmd.arg("--genco");
    for a in &archs {
        cmd.arg(format!("--offload-arch={a}"));
    }
    // the decoders must round like numpy: no contraction into fused multiply-adds
    cmd.args(["-O3", "-std=c++17", "-ffp-contract=off", "-o"]).arg(&out).arg("kernels/gravity.hip");
    let res = cmd.output().unwrap_or_else(|e| panic!("cannot run {cc:?}: {e}"));
    if !res.status.success() {
        panic!(
            "{cc:?} failed:\n{}\n{}",
            String::from_utf8_lossy(&res.stdout),
            String::from_utf8_lossy(&res.stderr)
        );
    }
    println!("cargo:rustc-env=GRAVITY_HIP_ARCHS={}", archs.join(","));
}
