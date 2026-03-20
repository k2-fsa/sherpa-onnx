//! Speaker embedding extraction and speaker search utilities.
//!
//! See:
//!
//! - `rust-api-examples/examples/speaker_embedding_extractor.rs`
//! - `rust-api-examples/examples/speaker_embedding_manager.rs`
//! - `rust-api-examples/examples/speaker_embedding_cosine_similarity.rs`

use crate::{online_asr::OnlineStream, utils::to_c_ptr};
use sherpa_onnx_sys as sys;
use std::ffi::{CStr, CString};
use std::ptr;
use std::slice;

#[derive(Clone, Debug)]
/// Configuration for [`SpeakerEmbeddingExtractor`].
pub struct SpeakerEmbeddingExtractorConfig {
    pub model: Option<String>,
    pub num_threads: i32,
    pub debug: bool,
    pub provider: Option<String>,
}

impl Default for SpeakerEmbeddingExtractorConfig {
    fn default() -> Self {
        Self {
            model: None,
            num_threads: 1,
            debug: false,
            provider: Some("cpu".to_string()),
        }
    }
}

impl SpeakerEmbeddingExtractorConfig {
    pub(crate) fn to_sys(
        &self,
        cstrings: &mut Vec<CString>,
    ) -> sys::SpeakerEmbeddingExtractorConfig {
        sys::SpeakerEmbeddingExtractorConfig {
            model: to_c_ptr(&self.model, cstrings),
            num_threads: self.num_threads,
            debug: self.debug as i32,
            provider: to_c_ptr(&self.provider, cstrings),
        }
    }
}

#[derive(Clone, Debug)]
/// One speaker search result returned by [`SpeakerEmbeddingManager::get_best_matches`].
pub struct SpeakerEmbeddingMatch {
    pub score: f32,
    pub name: String,
}

/// Embedding extractor that consumes audio through an [`OnlineStream`].
pub struct SpeakerEmbeddingExtractor {
    ptr: *const sys::SpeakerEmbeddingExtractor,
    dim: i32,
}

unsafe impl Send for SpeakerEmbeddingExtractor {}

impl SpeakerEmbeddingExtractor {
    /// Create an extractor from `config`.
    pub fn create(config: &SpeakerEmbeddingExtractorConfig) -> Option<Self> {
        let mut cstrings = Vec::new();
        let sys_config = config.to_sys(&mut cstrings);
        let ptr = unsafe { sys::SherpaOnnxCreateSpeakerEmbeddingExtractor(&sys_config) };
        if ptr.is_null() {
            None
        } else {
            let dim = unsafe { sys::SherpaOnnxSpeakerEmbeddingExtractorDim(ptr) };
            Some(Self { ptr, dim })
        }
    }

    /// Return the embedding dimension.
    pub fn dim(&self) -> i32 {
        self.dim
    }

    /// Create an audio stream that can be filled with waveform chunks.
    pub fn create_stream(&self) -> Option<OnlineStream> {
        let ptr = unsafe { sys::SherpaOnnxSpeakerEmbeddingExtractorCreateStream(self.ptr) };
        if ptr.is_null() {
            None
        } else {
            Some(OnlineStream { ptr })
        }
    }

    /// Return `true` if enough audio has been accumulated to compute an embedding.
    pub fn is_ready(&self, stream: &OnlineStream) -> bool {
        unsafe { sys::SherpaOnnxSpeakerEmbeddingExtractorIsReady(self.ptr, stream.ptr) == 1 }
    }

    /// Compute the embedding for `stream`.
    pub fn compute(&self, stream: &OnlineStream) -> Option<Vec<f32>> {
        let p = unsafe {
            sys::SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(self.ptr, stream.ptr)
        };
        if p.is_null() {
            None
        } else {
            let ans = unsafe { slice::from_raw_parts(p, self.dim as usize) }.to_vec();
            unsafe { sys::SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(p) };
            Some(ans)
        }
    }
}

impl Drop for SpeakerEmbeddingExtractor {
    fn drop(&mut self) {
        unsafe {
            if !self
                .ptr
                .is_null()
            {
                sys::SherpaOnnxDestroySpeakerEmbeddingExtractor(self.ptr);
            }
        }
    }
}

/// In-memory index of named speaker embeddings.
pub struct SpeakerEmbeddingManager {
    ptr: *const sys::SpeakerEmbeddingManager,
    dim: i32,
}

unsafe impl Send for SpeakerEmbeddingManager {}

impl SpeakerEmbeddingManager {
    /// Create a manager for embeddings with the given dimension.
    pub fn create(dim: i32) -> Option<Self> {
        let ptr = unsafe { sys::SherpaOnnxCreateSpeakerEmbeddingManager(dim) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr, dim })
        }
    }

    /// Return the embedding dimension expected by the manager.
    pub fn dim(&self) -> i32 {
        self.dim
    }

    /// Add one embedding for `name`.
    pub fn add(&self, name: &str, embedding: &[f32]) -> bool {
        if embedding.len() != self.dim as usize {
            return false;
        }

        let c_name = match CString::new(name) {
            Ok(v) => v,
            Err(_) => return false,
        };

        unsafe {
            sys::SherpaOnnxSpeakerEmbeddingManagerAdd(self.ptr, c_name.as_ptr(), embedding.as_ptr())
                == 1
        }
    }

    /// Add multiple embeddings for `name`.
    pub fn add_list(&self, name: &str, embeddings: &[Vec<f32>]) -> bool {
        if embeddings.is_empty()
            || embeddings
                .iter()
                .any(|v| v.len() != self.dim as usize)
        {
            return false;
        }

        let c_name = match CString::new(name) {
            Ok(v) => v,
            Err(_) => return false,
        };

        let mut ptrs: Vec<*const f32> = embeddings
            .iter()
            .map(|v| v.as_ptr())
            .collect();
        ptrs.push(ptr::null());

        unsafe {
            sys::SherpaOnnxSpeakerEmbeddingManagerAddList(self.ptr, c_name.as_ptr(), ptrs.as_ptr())
                == 1
        }
    }

    /// Add multiple embeddings laid out as a flattened slice.
    pub fn add_list_flattened(&self, name: &str, embeddings: &[f32]) -> bool {
        if embeddings.is_empty() || embeddings.len() % self.dim as usize != 0 {
            return false;
        }

        let c_name = match CString::new(name) {
            Ok(v) => v,
            Err(_) => return false,
        };

        let n = (embeddings.len() / self.dim as usize) as i32;
        unsafe {
            sys::SherpaOnnxSpeakerEmbeddingManagerAddListFlattened(
                self.ptr,
                c_name.as_ptr(),
                embeddings.as_ptr(),
                n,
            ) == 1
        }
    }

    /// Remove all embeddings stored under `name`.
    pub fn remove(&self, name: &str) -> bool {
        let c_name = match CString::new(name) {
            Ok(v) => v,
            Err(_) => return false,
        };

        unsafe { sys::SherpaOnnxSpeakerEmbeddingManagerRemove(self.ptr, c_name.as_ptr()) == 1 }
    }

    /// Search for the best matching speaker name above `threshold`.
    pub fn search(&self, embedding: &[f32], threshold: f32) -> Option<String> {
        if embedding.len() != self.dim as usize {
            return None;
        }

        unsafe {
            let p = sys::SherpaOnnxSpeakerEmbeddingManagerSearch(
                self.ptr,
                embedding.as_ptr(),
                threshold,
            );
            if p.is_null() {
                None
            } else {
                let ans = CStr::from_ptr(p)
                    .to_string_lossy()
                    .into_owned();
                sys::SherpaOnnxSpeakerEmbeddingManagerFreeSearch(p);
                Some(ans)
            }
        }
    }

    /// Return up to `n` best matches above `threshold`.
    pub fn get_best_matches(
        &self,
        embedding: &[f32],
        threshold: f32,
        n: i32,
    ) -> Vec<SpeakerEmbeddingMatch> {
        if embedding.len() != self.dim as usize {
            return Vec::new();
        }

        unsafe {
            let r = sys::SherpaOnnxSpeakerEmbeddingManagerGetBestMatches(
                self.ptr,
                embedding.as_ptr(),
                threshold,
                n,
            );
            if r.is_null() {
                return Vec::new();
            }

            let result = &*r;
            let matches = slice::from_raw_parts(result.matches, result.count as usize)
                .iter()
                .map(|m| SpeakerEmbeddingMatch {
                    score: m.score,
                    name: if m
                        .name
                        .is_null()
                    {
                        String::new()
                    } else {
                        CStr::from_ptr(m.name)
                            .to_string_lossy()
                            .into_owned()
                    },
                })
                .collect::<Vec<_>>();
            sys::SherpaOnnxSpeakerEmbeddingManagerFreeBestMatches(r);
            matches
        }
    }

    pub fn verify(&self, name: &str, embedding: &[f32], threshold: f32) -> bool {
        if embedding.len() != self.dim as usize {
            return false;
        }

        let c_name = match CString::new(name) {
            Ok(v) => v,
            Err(_) => return false,
        };

        unsafe {
            sys::SherpaOnnxSpeakerEmbeddingManagerVerify(
                self.ptr,
                c_name.as_ptr(),
                embedding.as_ptr(),
                threshold,
            ) == 1
        }
    }

    pub fn contains(&self, name: &str) -> bool {
        let c_name = match CString::new(name) {
            Ok(v) => v,
            Err(_) => return false,
        };

        unsafe { sys::SherpaOnnxSpeakerEmbeddingManagerContains(self.ptr, c_name.as_ptr()) == 1 }
    }

    pub fn num_speakers(&self) -> i32 {
        unsafe { sys::SherpaOnnxSpeakerEmbeddingManagerNumSpeakers(self.ptr) }
    }

    pub fn get_all_speakers(&self) -> Vec<String> {
        unsafe {
            let names = sys::SherpaOnnxSpeakerEmbeddingManagerGetAllSpeakers(self.ptr);
            if names.is_null() {
                return Vec::new();
            }

            let mut ans = Vec::new();
            let mut p = names;
            while !(*p).is_null() {
                ans.push(
                    CStr::from_ptr(*p)
                        .to_string_lossy()
                        .into_owned(),
                );
                p = p.add(1);
            }
            sys::SherpaOnnxSpeakerEmbeddingManagerFreeAllSpeakers(names);
            ans
        }
    }
}

impl Drop for SpeakerEmbeddingManager {
    fn drop(&mut self) {
        unsafe {
            if !self
                .ptr
                .is_null()
            {
                sys::SherpaOnnxDestroySpeakerEmbeddingManager(self.ptr);
            }
        }
    }
}
