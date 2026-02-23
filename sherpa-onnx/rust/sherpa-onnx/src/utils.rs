use sherpa_onnx_sys as sys;
use std::ffi::{CStr, CString};

/// Returns the version string of sherpa-onnx
pub fn version() -> &'static str {
    unsafe {
        CStr::from_ptr(sys::SherpaOnnxGetVersionStr())
            .to_str()
            .unwrap()
    }
}

/// Returns the Git SHA1 of the build
pub fn git_sha1() -> &'static str {
    unsafe {
        CStr::from_ptr(sys::SherpaOnnxGetGitSha1())
            .to_str()
            .unwrap()
    }
}

/// Returns the Git date of the build
pub fn git_date() -> &'static str {
    unsafe {
        CStr::from_ptr(sys::SherpaOnnxGetGitDate())
            .to_str()
            .unwrap()
    }
}

/// Returns true if the given file exists
pub fn file_exists(filename: &str) -> bool {
    let cstr = CString::new(filename).unwrap();
    unsafe { sys::SherpaOnnxFileExists(cstr.as_ptr()) != 0 }
}
