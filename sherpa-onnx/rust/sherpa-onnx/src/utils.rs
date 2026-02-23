use sherpa_onnx_sys as sys;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// Safely convert a C string pointer to a `'static` Rust string slice.
///
/// If the pointer is null, an empty string is returned.
/// If the C string is not valid UTF-8, a lossy UTF-8 conversion is used
/// and the resulting string is leaked to obtain a `'static` lifetime.
fn c_str_to_static_str(ptr: *const c_char) -> &'static str {
    if ptr.is_null() {
        return "";
    }
    unsafe {
        match CStr::from_ptr(ptr).to_str() {
            Ok(s) => s,
            Err(_) => {
                // Fallback to a lossy conversion and leak the String to get a `'static` str.
                let owned = CStr::from_ptr(ptr)
                    .to_string_lossy()
                    .into_owned();
                Box::leak(owned.into_boxed_str())
            }
        }
    }
}

/// Returns the version string of sherpa-onnx
pub fn version() -> &'static str {
    let ptr = unsafe { sys::SherpaOnnxGetVersionStr() };
    c_str_to_static_str(ptr)
}

/// Returns the Git SHA1 of the build
pub fn git_sha1() -> &'static str {
    let ptr = unsafe { sys::SherpaOnnxGetGitSha1() };
    c_str_to_static_str(ptr)
}

/// Returns the Git date of the build
pub fn git_date() -> &'static str {
    let ptr = unsafe { sys::SherpaOnnxGetGitDate() };
    c_str_to_static_str(ptr)
}

/// Returns true if the given file exists
pub fn file_exists(filename: &str) -> bool {
    let cstr = match CString::new(filename) {
        Ok(cstr) => cstr,
        Err(_) => {
            // Invalid input (e.g., contains interior NUL); treat as non-existent.
            return false;
        }
    };

    unsafe { sys::SherpaOnnxFileExists(cstr.as_ptr()) != 0 }
}
