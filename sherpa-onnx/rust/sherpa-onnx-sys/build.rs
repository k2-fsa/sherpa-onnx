use std::env;
use std::error::Error;
use std::ffi::OsStr;
use std::fs;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::{collections::HashSet, ffi::OsString};

use bzip2::read::BzDecoder;
use tar::Archive;

const RELEASE_BASE_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download";
const XCFRAMEWORK_RELEASE_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/xcframework";
const SHERPA_ONNX_STATIC_LIBS: &[&str] = &[
    "sherpa-onnx-c-api",
    "sherpa-onnx-core",
    "kaldi-decoder-core",
    "sherpa-onnx-kaldifst-core",
    "sherpa-onnx-fstfar",
    "sherpa-onnx-fst",
    "kaldi-native-fbank-core",
    "kissfft-float",
    "piper_phonemize",
    "espeak-ng",
    "ucd",
    "onnxruntime",
    "ssentencepiece_core",
];

type DynError = Box<dyn Error>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LinkMode {
    Static,
    Shared,
}

fn main() {
    if let Err(err) = try_main() {
        panic!("{err}");
    }
}

fn try_main() -> Result<(), DynError> {
    println!("cargo:rerun-if-env-changed=SHERPA_ONNX_LIB_DIR");
    println!("cargo:rerun-if-env-changed=SHERPA_ONNX_ARCHIVE_DIR");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    if env::var_os("DOCS_RS").is_some() {
        // docs.rs sets DOCS_RS=1; skip downloading/linking native libraries
        // so that `cargo doc` can succeed without the real C artifacts.
        return Ok(());
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS")?;
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")?;
    let link_mode = resolve_link_mode(&target_os)?;
    let lib_dir = resolve_lib_dir(link_mode, &target_os, &target_arch)?;

    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    if link_mode == LinkMode::Shared
        && matches!(target_os.as_str(), "linux" | "macos" | "android" | "ios")
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        emit_relative_rpath(&target_os);
        copy_unix_runtime_libs(&lib_dir, &target_os)?;
    }

    // For Android builds (e.g. via Tauri), copy .so files to the Tauri
    // Android project's jniLibs directory so Gradle bundles them into the APK.
    if target_os == "android" {
        copy_to_tauri_android_jnilibs(&lib_dir, &target_arch)?;
    }

    // For iOS builds (e.g. via Tauri), copy the xcframework to the Tauri
    // project directory so Xcode can find it via bundle.ios.frameworks.
    if target_os == "ios" {
        copy_xcframework_to_tauri_project(&lib_dir)?;
    }

    if link_mode == LinkMode::Shared && target_os == "windows" {
        copy_windows_runtime_dlls(&lib_dir)?;
    }

    match link_mode {
        LinkMode::Static => emit_static_link_directives(&target_os),
        LinkMode::Shared => emit_shared_link_directives(&target_os),
    }

    Ok(())
}

fn resolve_link_mode(target_os: &str) -> Result<LinkMode, DynError> {
    let static_enabled = env::var_os("CARGO_FEATURE_STATIC").is_some();
    let shared_enabled = env::var_os("CARGO_FEATURE_SHARED").is_some();

    if static_enabled && shared_enabled {
        return Err("Features `static` and `shared` cannot be enabled at the same time".into());
    }

    if shared_enabled {
        Ok(LinkMode::Shared)
    } else if target_os == "android" || target_os == "ios" {
        // Android and iOS only support shared linking.
        Ok(LinkMode::Shared)
    } else {
        Ok(LinkMode::Static)
    }
}

fn resolve_lib_dir(
    link_mode: LinkMode,
    target_os: &str,
    target_arch: &str,
) -> Result<PathBuf, DynError> {
    if let Some(path) = env::var_os("SHERPA_ONNX_LIB_DIR") {
        let path = PathBuf::from(path);
        if !path.is_dir() {
            return Err(format!(
                "SHERPA_ONNX_LIB_DIR does not exist or is not a directory: {}",
                path.display()
            )
            .into());
        }
        return Ok(path);
    }

    download_prebuilt_libs(link_mode, target_os, target_arch)
}

/// Return the download URL for a given archive.
fn download_url(archive_name: &str) -> String {
    // iOS xcframework archives live under the "xcframework" release tag;
    // everything else is under the versioned release tag.
    if archive_name.ends_with(".xcframework.zip") {
        format!("{XCFRAMEWORK_RELEASE_URL}/{archive_name}")
    } else {
        let version = env!("CARGO_PKG_VERSION");
        format!("{RELEASE_BASE_URL}/v{version}/{archive_name}")
    }
}

fn download_prebuilt_libs(
    link_mode: LinkMode,
    target_os: &str,
    target_arch: &str,
) -> Result<PathBuf, DynError> {
    let archive_name = archive_name(link_mode, target_os, target_arch)?;
    let archive_stem = archive_name
        .strip_suffix(".tar.bz2")
        .or_else(|| archive_name.strip_suffix(".xcframework.zip"))
        .unwrap_or(&archive_name);

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let cache_root = target_dir_from_out_dir(&out_dir)?.join("sherpa-onnx-prebuilt");
    let extracted_dir = cache_root.join(archive_stem);
    let lib_dir = extracted_dir.join("lib");

    if lib_dir.is_dir() {
        return Ok(lib_dir);
    }

    // Android archives use jniLibs/{abi}/ instead of lib/. Check both.
    let android_lib_dir = extracted_dir.join("jniLibs").join(android_abi(target_arch));
    if android_lib_dir.is_dir() {
        return Ok(android_lib_dir);
    }

    // iOS archives use xcframework bundles. Check for a symlink lib/ dir
    // created by a previous run.
    if target_os == "ios" {
        let ios_lib = extracted_dir.join("lib");
        if ios_lib.is_dir() {
            return Ok(ios_lib);
        }
    }

    fs::create_dir_all(&cache_root)?;

    let archive_path = cache_root.join(&archive_name);
    if !archive_path.is_file() {
        if let Some(local_archive_dir) = env::var_os("SHERPA_ONNX_ARCHIVE_DIR") {
            let local_archive_path = PathBuf::from(local_archive_dir).join(&archive_name);
            if !local_archive_path.is_file() {
                return Err(format!(
                    "SHERPA_ONNX_ARCHIVE_DIR does not contain expected archive: {}",
                    local_archive_path.display()
                )
                .into());
            }

            copy_file_atomically(&local_archive_path, &archive_path)?;
        } else {
            let url = download_url(&archive_name);
            eprintln!("Downloading sherpa-onnx libs from {url}");

            let response = ureq::builder()
                .try_proxy_from_env(true)
                .build()
                .get(&url)
                .call()
                .map_err(|e| format!("Failed to download sherpa-onnx archive from {url}: {e}"))?;
            let mut reader = response.into_reader();
            write_reader_atomically(&mut reader, &archive_path)?;
        }
    }

    if extracted_dir.exists() {
        fs::remove_dir_all(&extracted_dir)?;
    }

    let unpack_result: Result<(), DynError> = (|| {
        if archive_name.ends_with(".xcframework.zip") {
            // iOS archives are plain zip files containing the xcframework.
            // Extract to extracted_dir so the xcframework ends up at
            // extracted_dir/<XcframeworkName>.xcframework/.
            let zip_file = File::open(&archive_path)?;
            let mut archive = zip::ZipArchive::new(zip_file).map_err(|e| {
                format!("Failed to open zip archive {}: {e}", archive_path.display())
            })?;
            archive
                .extract(&extracted_dir)
                .map_err(|e| format!("Failed to extract zip archive: {e}"))?;
        } else {
            let tar_file = File::open(&archive_path)?;
            let decoder = BzDecoder::new(tar_file);
            let mut archive = Archive::new(decoder);
            archive.unpack(&cache_root)?;
        }
        Ok(())
    })();
    if let Err(err) = unpack_result {
        let _ = fs::remove_file(&archive_path);
        let _ = fs::remove_dir_all(&extracted_dir);
        return Err(format!(
            "Failed to unpack cached archive {}: {err}",
            archive_path.display()
        )
        .into());
    }

    if !lib_dir.is_dir() {
        // Android archives use jniLibs/{abi}/ instead of lib/.
        let android_lib_dir = extracted_dir
            .join("jniLibs")
            .join(android_abi(target_arch));
        if android_lib_dir.is_dir() {
            eprintln!("Downloaded sherpa-onnx Android libs to {}", android_lib_dir.display());
            return Ok(android_lib_dir);
        }

        // iOS archives contain xcframework bundles. Create a lib/ directory
        // with symlinks so Rust's linker can find the library under the
        // expected name (libsherpa-onnx-c-api.a / .dylib).
        if target_os == "ios" {
            if let Some(ios_lib) = setup_ios_lib_dir(&extracted_dir, link_mode)? {
                eprintln!("Downloaded sherpa-onnx iOS libs to {}", ios_lib.display());
                return Ok(ios_lib);
            }
        }

        return Err(format!(
            "Downloaded archive did not contain a lib directory: {}",
            lib_dir.display()
        )
        .into());
    }

    eprintln!("Downloaded sherpa-onnx libs to {}", extracted_dir.display());

    Ok(lib_dir)
}

/// Map a Rust target architecture to the Android ABI directory name used
/// in the prebuilt jniLibs/ layout.
fn android_abi(target_arch: &str) -> &str {
    match target_arch {
        "aarch64" => "arm64-v8a",
        "arm" => "armeabi-v7a",
        "x86" => "x86",
        "x86_64" => "x86_64",
        _ => "arm64-v8a",
    }
}

/// Find the SherpaOnnxC binary inside an extracted iOS xcframework archive
/// and create a `lib/` directory with a symlink named `libsherpa-onnx-c-api.{a,dylib}`
/// so that Rust's linker can find it under the expected name.
fn setup_ios_lib_dir(
    extracted_dir: &Path,
    link_mode: LinkMode,
) -> Result<Option<PathBuf>, DynError> {
    // The iOS archive structure is:
    //   build-ios/sherpa-onnx.xcframework/ios-arm64/SherpaOnnxC.framework/SherpaOnnxC
    // or for the shared-onnxruntime-static variant:
    //   sherpa-onnx.xcframework/ios-arm64/SherpaOnnxC.framework/SherpaOnnxC
    //   SherpaOnnxC.xcframework/ios-arm64/SherpaOnnxC.framework/SherpaOnnxC
    let candidates = [
        extracted_dir
            .join("build-ios")
            .join("sherpa-onnx.xcframework"),
        extracted_dir.join("sherpa-onnx.xcframework"),
        extracted_dir.join("SherpaOnnxC.xcframework"),
    ];

    let xcframework = match candidates.iter().find(|p| p.is_dir()) {
        Some(p) => p,
        None => return Ok(None),
    };

    let binary = xcframework
        .join("ios-arm64")
        .join("SherpaOnnxC.framework")
        .join("SherpaOnnxC");

    if !binary.is_file() {
        return Ok(None);
    }

    let lib_dir = extracted_dir.join("lib");
    fs::create_dir_all(&lib_dir)?;

    // For shared mode, we only need the sherpa-onnx-c-api dylib.
    // For static mode, the iOS xcframework contains a single merged archive
    // with all symbols. Create symlinks for every library name that
    // emit_static_link_directives() will ask for, so the linker finds them all.
    let link_names: Vec<&str> = match link_mode {
        LinkMode::Shared => vec!["libsherpa-onnx-c-api.dylib"],
        LinkMode::Static => SHERPA_ONNX_STATIC_LIBS
            .iter()
            .map(|lib| {
                // "sherpa-onnx-c-api" -> "libsherpa-onnx-c-api.a"
                // We leak a String to get a &'static str. This is fine for a
                // build script that runs once and exits.
                let s = format!("lib{lib}.a");
                Box::leak(s.into_boxed_str()) as &str
            })
            .collect(),
    };

    let abs_binary = fs::canonicalize(&binary)?;
    for link_name in &link_names {
        let link_path = lib_dir.join(link_name);
        if link_path.exists() {
            continue;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            symlink(&abs_binary, &link_path)?;
        }
        #[cfg(not(unix))]
        {
            fs::copy(&abs_binary, &link_path)?;
        }
    }

    Ok(Some(lib_dir))
}

fn archive_name(
    link_mode: LinkMode,
    target_os: &str,
    target_arch: &str,
) -> Result<String, DynError> {
    let version = env!("CARGO_PKG_VERSION");
    let name = match (link_mode, target_os, target_arch) {
        (LinkMode::Static, "linux", "x86_64") => {
            format!("sherpa-onnx-v{version}-linux-x64-static-lib.tar.bz2")
        }
        (LinkMode::Static, "linux", "aarch64") => {
            format!("sherpa-onnx-v{version}-linux-aarch64-static-lib.tar.bz2")
        }
        (LinkMode::Static, "macos", "x86_64") => {
            format!("sherpa-onnx-v{version}-osx-x64-static-lib.tar.bz2")
        }
        (LinkMode::Static, "macos", "aarch64") => {
            format!("sherpa-onnx-v{version}-osx-arm64-static-lib.tar.bz2")
        }
        (LinkMode::Static, "windows", "x86_64") => {
            format!("sherpa-onnx-v{version}-win-x64-static-MT-Release-lib.tar.bz2")
        }
        (LinkMode::Static, "windows", "aarch64") => {
            format!("sherpa-onnx-v{version}-win-arm64-static-MT-Release-lib.tar.bz2")
        }
        (LinkMode::Shared, "linux", "x86_64") => {
            format!("sherpa-onnx-v{version}-linux-x64-shared-lib.tar.bz2")
        }
        (LinkMode::Shared, "linux", "aarch64") => {
            format!("sherpa-onnx-v{version}-linux-aarch64-shared-cpu-lib.tar.bz2")
        }
        (LinkMode::Shared, "macos", "x86_64") => {
            format!("sherpa-onnx-v{version}-osx-x64-shared-lib.tar.bz2")
        }
        (LinkMode::Shared, "macos", "aarch64") => {
            format!("sherpa-onnx-v{version}-osx-arm64-shared-lib.tar.bz2")
        }
        (LinkMode::Shared, "windows", "x86_64") => {
            format!("sherpa-onnx-v{version}-win-x64-shared-MT-Release-lib.tar.bz2")
        }
        (LinkMode::Shared, "windows", "aarch64") => {
            format!("sherpa-onnx-v{version}-win-arm64-shared-MT-Release-lib.tar.bz2")
        }
        // Android: one archive with all ABIs under jniLibs/{abi}/.
        (LinkMode::Shared, "android", "aarch64" | "arm" | "x86" | "x86_64") => {
            format!("sherpa-onnx-v{version}-android.tar.bz2")
        }
        // iOS: shared xcframework from the xcframework release tag.
        // iOS only supports shared linking (onnxruntime statically linked
        // into the sherpa-onnx dylib).
        (LinkMode::Shared, "ios", "aarch64") => {
            format!("sherpa-onnx-v{version}-ios-shared-onnxruntime-static.xcframework.zip")
        }
        _ => return Err(format!(
            "Unsupported target for sherpa-onnx prebuilt libs: os={target_os}, arch={target_arch}"
        )
        .into()),
    };

    Ok(name)
}

fn emit_shared_link_directives(target_os: &str) {
    println!("cargo:rustc-link-lib=dylib=sherpa-onnx-c-api");
    // The iOS shared archive bundles onnxruntime statically into the
    // sherpa-onnx dylib, so no separate onnxruntime dylib is needed.
    if target_os != "ios" {
        println!("cargo:rustc-link-lib=dylib=onnxruntime");
    }
}

fn emit_static_link_directives(target_os: &str) {
    for lib in SHERPA_ONNX_STATIC_LIBS {
        println!("cargo:rustc-link-lib=static={lib}");
    }

    match target_os {
        "linux" => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=dylib=m");
            println!("cargo:rustc-link-lib=dylib=pthread");
            println!("cargo:rustc-link-lib=dylib=dl");
        }
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=dylib=c++");
            println!("cargo:rustc-link-lib=framework=Foundation");
        }
        _ => {}
    }
}

fn target_dir_from_out_dir(out_dir: &Path) -> Result<PathBuf, DynError> {
    if let Ok(explicit_target_dir) = env::var("CARGO_TARGET_DIR") {
        return Ok(PathBuf::from(explicit_target_dir));
    }

    if let Some(target_dir) = out_dir
        .ancestors()
        .find(|path| path.file_name() == Some(OsStr::new("target")))
    {
        return Ok(target_dir.to_path_buf());
    }

    Ok(out_dir.to_path_buf())
}

fn emit_relative_rpath(target_os: &str) {
    match target_os {
        "linux" | "android" => println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN"),
        "macos" | "ios" => println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path"),
        _ => {}
    }
}

fn profile_output_dirs() -> Result<[PathBuf; 2], DynError> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let profile = env::var("PROFILE")?;
    let profile_dir = out_dir
        .ancestors()
        .find(|path| path.file_name() == Some(OsStr::new(&profile)))
        .ok_or_else(|| {
            format!(
                "Could not locate Cargo profile directory from {}",
                out_dir.display()
            )
        })?
        .to_path_buf();

    Ok([profile_dir.clone(), profile_dir.join("examples")])
}

/// Copy Android .so files to the Tauri Android project's jniLibs directory
/// so that Gradle bundles them into the APK.
///
/// We need to find the Tauri project's `gen/android/` directory.  Since
/// `CARGO_MANIFEST_DIR` points to the *sherpa-onnx-sys* crate (not the Tauri
/// project), we walk up from `OUT_DIR` to locate the Tauri project root.
fn copy_to_tauri_android_jnilibs(lib_dir: &Path, target_arch: &str) -> Result<(), DynError> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let abi = android_abi(target_arch);

    // OUT_DIR is typically:
    //   <project>/src-tauri/target/<triple>/<profile>/build/sherpa-onnx-sys-<hash>/out
    // The Tauri project root (src-tauri/) is the parent of "target/".
    let target_dir = target_dir_from_out_dir(&out_dir)?;
    let project_dir = match target_dir.parent() {
        Some(p) => p,
        None => return Ok(()),
    };

    // Also try CARGO_MANIFEST_DIR as a fallback (works for non-Tauri setups).
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default());

    let jni_base_suffix = PathBuf::from("gen")
        .join("android")
        .join("app")
        .join("src")
        .join("main")
        .join("jniLibs");

    let candidates = [
        project_dir.join(&jni_base_suffix),
        manifest_dir.join(&jni_base_suffix),
    ];

    let tauri_jni_base = candidates.iter().find(|p| {
        // Check that the parent (main/) exists, meaning
        // `tauri android init` has been run.
        p.parent().map_or(false, |p| p.exists())
    });

    let tauri_jni_base = match tauri_jni_base {
        Some(p) => p,
        None => {
            // Not a Tauri Android build (or gen/android hasn't been created yet).
            return Ok(());
        }
    };

    let dest_dir = tauri_jni_base.join(abi);
    fs::create_dir_all(&dest_dir)?;

    let mut copied = 0;
    for entry in fs::read_dir(lib_dir)? {
        let entry = entry?;
        let path = entry.path();
        let is_so = path.file_name()
            .and_then(OsStr::to_str)
            .map(|name| name.contains(".so"))
            .unwrap_or(false);
        if !is_so {
            continue;
        }
        if let Some(file_name) = path.file_name() {
            let dest = dest_dir.join(file_name);
            fs::copy(&path, &dest)?;
            eprintln!("Copied {} -> {}", path.display(), dest.display());
            copied += 1;
        }
    }

    if copied > 0 {
        println!(
            "cargo:warning=Copied {copied} Android .so file(s) to Tauri jniLibs: {}",
            dest_dir.display()
        );
    }

    Ok(())
}

/// Copy the iOS xcframework to the Tauri project directory so that
/// Xcode can find it when linking.  `bundle.ios.frameworks` in
/// `tauri.conf.json` references the xcframework by name relative to the
/// project root.
fn copy_xcframework_to_tauri_project(lib_dir: &Path) -> Result<(), DynError> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    // lib_dir is something like
    //   target/.../sherpa-onnx-prebuilt/sherpa-onnx-v1.13.6-ios-shared-onnxruntime-static/lib
    // The xcframework sits next to lib/:
    //   .../sherpa-onnx.xcframework/   or   .../SherpaOnnxC.xcframework/
    let extracted_dir = lib_dir.parent().unwrap_or(lib_dir);

    let candidates = [
        extracted_dir.join("build-ios").join("sherpa-onnx.xcframework"),
        extracted_dir.join("sherpa-onnx.xcframework"),
        extracted_dir.join("SherpaOnnxC.xcframework"),
    ];

    let xcframework = match candidates.iter().find(|p| p.is_dir()) {
        Some(p) => p,
        None => return Ok(()),
    };

    // Navigate from OUT_DIR to the Tauri project root (src-tauri/).
    // OUT_DIR is typically: <project>/src-tauri/target/<triple>/<profile>/build/sherpa-onnx-sys-<hash>/out
    let target_dir = target_dir_from_out_dir(&out_dir)?;
    let project_dir = match target_dir.parent() {
        Some(p) => p,
        None => return Ok(()),
    };

    // Destination: src-tauri/SherpaOnnxC.xcframework
    let dest = project_dir.join("SherpaOnnxC.xcframework");
    if dest.exists() {
        // Already copied (e.g. from a previous build or CI pre-download).
        return Ok(());
    }

    eprintln!("Copying xcframework {} -> {}", xcframework.display(), dest.display());
    copy_dir_recursively(xcframework, &dest)?;
    println!(
        "cargo:warning=Copied SherpaOnnxC.xcframework to {}",
        dest.display()
    );
    Ok(())
}

/// Recursively copy a directory.
fn copy_dir_recursively(src: &Path, dst: &Path) -> Result<(), DynError> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let dest_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_recursively(&entry.path(), &dest_path)?;
        } else if ty.is_symlink() {
            let target = fs::read_link(entry.path())?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::symlink;
                symlink(&target, &dest_path)?;
            }
            #[cfg(not(unix))]
            {
                // On non-Unix, just copy the target file.
                fs::copy(entry.path(), &dest_path)?;
            }
        } else {
            fs::copy(entry.path(), &dest_path)?;
        }
    }
    Ok(())
}

fn copy_unix_runtime_libs(lib_dir: &Path, target_os: &str) -> Result<(), DynError> {
    let runtime_libs: Vec<PathBuf> = fs::read_dir(lib_dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.file_name()
                .and_then(OsStr::to_str)
                 .map(|name| match target_os {
                     "linux" | "android" => name.contains(".so"),
                     "macos" | "ios" => name.ends_with(".dylib"),
                    _ => false,
                })
                .unwrap_or(false)
        })
        .collect();

    if runtime_libs.is_empty() {
        return Err(format!(
            "No shared runtime libraries found in {}",
            lib_dir.display()
        )
        .into());
    }

    let mut copy_plan = Vec::<(PathBuf, OsString)>::new();
    let mut planned_names = HashSet::<OsString>::new();

    for lib in runtime_libs {
        if !lib.exists() {
            continue;
        }

        let lib_name = lib
            .file_name()
            .ok_or_else(|| format!("Invalid runtime library path: {}", lib.display()))?
            .to_os_string();

        let source = fs::canonicalize(&lib).unwrap_or(lib.clone());
        if planned_names.insert(lib_name.clone()) {
            copy_plan.push((source.clone(), lib_name));
        }

        if let Some(source_name) = source.file_name() {
            let source_name = source_name.to_os_string();
            if planned_names.insert(source_name.clone()) {
                copy_plan.push((source.clone(), source_name));
            }
        }
    }

    if copy_plan.is_empty() {
        return Err(format!(
            "No usable shared runtime libraries found in {}",
            lib_dir.display()
        )
        .into());
    }

    for dest_dir in profile_output_dirs()? {
        fs::create_dir_all(&dest_dir)?;
        for (source, dest_name) in &copy_plan {
            let dest = dest_dir.join(dest_name);
            fs::copy(source, &dest)?;
        }
    }

    Ok(())
}

fn temp_path_for(path: &Path) -> PathBuf {
    let mut temp_name = path
        .file_name()
        .map(OsStr::to_os_string)
        .unwrap_or_else(|| OsString::from("tmp"));
    temp_name.push(".part");
    path.with_file_name(temp_name)
}

fn copy_file_atomically(src: &Path, dst: &Path) -> Result<(), DynError> {
    let temp_path = temp_path_for(dst);
    if temp_path.exists() {
        let _ = fs::remove_file(&temp_path);
    }
    fs::copy(src, &temp_path)?;
    fs::rename(&temp_path, dst)?;
    Ok(())
}

fn write_reader_atomically(reader: &mut dyn io::Read, dst: &Path) -> Result<(), DynError> {
    let temp_path = temp_path_for(dst);
    if temp_path.exists() {
        let _ = fs::remove_file(&temp_path);
    }

    {
        let mut file = File::create(&temp_path)?;
        io::copy(reader, &mut file)?;
        file.sync_all()?;
    }

    fs::rename(&temp_path, dst)?;
    Ok(())
}

fn copy_windows_runtime_dlls(lib_dir: &Path) -> Result<(), DynError> {
    let dlls: Vec<PathBuf> = fs::read_dir(lib_dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension() == Some(OsStr::new("dll")))
        .collect();

    if dlls.is_empty() {
        println!(
            "cargo:warning=No runtime DLLs found in {}",
            lib_dir.display()
        );
        return Ok(());
    }

    let [profile_dir, examples_dir] = profile_output_dirs()?;
    for dest_dir in [profile_dir.clone(), examples_dir] {
        fs::create_dir_all(&dest_dir)?;
        for dll in &dlls {
            let dest = dest_dir.join(
                dll.file_name()
                    .ok_or_else(|| format!("Invalid DLL path: {}", dll.display()))?,
            );
            fs::copy(dll, &dest)?;
        }
    }

    println!(
        "cargo:warning=Copied Windows runtime DLLs to {} and {}/examples",
        profile_dir.display(),
        profile_dir.display()
    );

    Ok(())
}
