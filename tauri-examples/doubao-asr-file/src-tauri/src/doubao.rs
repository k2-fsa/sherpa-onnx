// A thin client for the Doubao (豆包) bigmodel recording-file recognition API.
//
// The API is asynchronous: submit a task, then poll it until it is done. The
// audio is uploaded as base64 encoded wav data, so no public URL is needed.
//
// Ported from python-api-examples/generate-subtitles-doubao.py.

use std::time::{Duration, Instant};

pub const SUBMIT_URL: &str = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit";
pub const QUERY_URL: &str = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query";

/// X-Api-Status-Code returned in the response headers
pub const STATUS_OK: &str = "20000000";
/// processing / queued
pub const STATUS_RUNNING: [&str; 2] = ["20000001", "20000002"];
pub const STATUS_SILENCE: &str = "20000003";

pub const DEFAULT_RESOURCE_ID: &str = "volc.seedasr.auc";

#[derive(Debug, Clone)]
pub struct DoubaoClient {
    pub api_key: String,
    pub resource_id: String,
    pub enable_itn: bool,
    pub enable_punc: bool,
    pub query_interval: f32,
    pub query_timeout: f32,
}

impl DoubaoClient {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            resource_id: DEFAULT_RESOURCE_ID.to_string(),
            enable_itn: true,
            enable_punc: true,
            query_interval: 0.5,
            query_timeout: 120.0,
        }
    }
}

/// Headers shared by the submit and query calls. `log_id` is only sent on the
/// query call and must be the X-Tt-Logid returned by the submit call.
pub fn build_headers(
    api_key: &str,
    resource_id: &str,
    request_id: &str,
    log_id: &str,
) -> Vec<(String, String)> {
    let mut headers = vec![
        ("Content-Type".to_string(), "application/json".to_string()),
        ("X-Api-Key".to_string(), api_key.to_string()),
        ("X-Api-Resource-Id".to_string(), resource_id.to_string()),
        ("X-Api-Request-Id".to_string(), request_id.to_string()),
        ("X-Api-Sequence".to_string(), "-1".to_string()),
    ];
    if !log_id.is_empty() {
        headers.push(("X-Tt-Logid".to_string(), log_id.to_string()));
    }
    headers
}

/// JSON body of the submit call. `wav_data` is base64 of a 16 kHz mono wav.
pub fn build_submit_body(wav_data: &str, enable_itn: bool, enable_punc: bool) -> String {
    format!(
        r#"{{"user":{{"uid":"sherpa-onnx"}},"audio":{{"data":"{wav_data}","format":"wav","codec":"raw","rate":16000,"bits":16,"channel":1}},"request":{{"model_name":"bigmodel","enable_itn":{enable_itn},"enable_punc":{enable_punc},"enable_ddc":false,"enable_speaker_info":false,"enable_channel_split":false,"show_utterances":true,"vad_segment":false,"sensitive_words_filter":""}}}}"#
    )
}

/// Extract the transcript from a successful query response body.
pub fn parse_result_json(body: &str) -> String {
    let value: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => return String::new(),
    };
    let result = value.get("result").cloned().unwrap_or_default();
    let text = result
        .get("text")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if !text.is_empty() {
        return text;
    }
    // Fallback: join the texts of the utterances
    result
        .get("utterances")
        .and_then(|u| u.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|u| u.get("text").and_then(|t| t.as_str()))
                .collect::<String>()
        })
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// Send one HTTP POST and return (status_code, log_id, body).
fn post(url: &str, headers: &[(String, String)], body: &str) -> Result<(String, String, String), String> {
    let mut req = ureq::post(url).timeout(Duration::from_secs(60));
    for (k, v) in headers {
        req = req.set(k, v);
    }
    let resp = req
        .send_string(body)
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    let status = resp
        .header("X-Api-Status-Code")
        .unwrap_or("")
        .to_string();
    let log_id = resp.header("X-Tt-Logid").unwrap_or("").to_string();
    let body = resp.into_string().map_err(|e| format!("read body: {e}"))?;
    Ok((status, log_id, body))
}

impl DoubaoClient {
    /// Submit a task and return the X-Tt-Logid of the response.
    fn submit(&self, request_id: &str, wav_bytes: &[u8]) -> Result<String, String> {
        let wav_data = base64_encode(wav_bytes);
        let body = build_submit_body(&wav_data, self.enable_itn, self.enable_punc);
        let headers = build_headers(&self.api_key, &self.resource_id, request_id, "");

        let mut last_err = String::new();
        for i in 0..3 {
            match post(SUBMIT_URL, &headers, &body) {
                Ok((status, log_id, body)) => {
                    if status == STATUS_OK {
                        return Ok(log_id);
                    }
                    let message = header_message(&body, &status);
                    last_err = format!("submit failed: X-Api-Status-Code={status}, {message}");
                }
                Err(e) => {
                    last_err = format!("submit failed: {e}");
                }
            }
            std::thread::sleep(Duration::from_millis(1000 * (i + 1) as u64));
        }
        Err(last_err)
    }

    /// Poll the task until it is done. Returns (transcript, diagnostic) — the
    /// diagnostic is what the API said (status + body) so an empty transcript
    /// is never a mystery.
    fn query(&self, request_id: &str, log_id: &str) -> Result<(String, String), String> {
        let headers = build_headers(&self.api_key, &self.resource_id, request_id, log_id);
        let deadline = Instant::now() + Duration::from_secs_f32(self.query_timeout);
        while Instant::now() < deadline {
            let (status, _, body) = post(QUERY_URL, &headers, "{}")?;
            let note = format!("status={status}, body={}", truncate(&body, 160));
            if status == STATUS_OK {
                return Ok((parse_result_json(&body), note));
            }
            if status == STATUS_SILENCE {
                return Ok((String::new(), note));
            }
            if !STATUS_RUNNING.contains(&status.as_str()) {
                return Err(format!("query failed: {note}"));
            }
            std::thread::sleep(Duration::from_secs_f32(self.query_interval));
        }
        Err(format!(
            "query timed out after {} seconds: {request_id}",
            self.query_timeout
        ))
    }

    /// Recognize a single audio clip. Returns (transcript, diagnostic).
    pub fn recognize(&self, wav_bytes: &[u8]) -> Result<(String, String), String> {
        let request_id = uuid::Uuid::new_v4().to_string();
        let log_id = self.submit(&request_id, wav_bytes)?;
        self.query(&request_id, &log_id)
    }
}

fn header_message(body: &str, status: &str) -> String {
    format!("status={status}, body={}", truncate(body, 200))
}

fn truncate(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Minimal base64 encoder (standard alphabet with padding) so the module has
/// no extra dependency.
pub fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn headers_include_log_id_only_when_given() {
        let h = build_headers("k", "r", "req", "");
        assert!(h.iter().any(|(k, v)| k == "X-Api-Key" && v == "k"));
        assert!(h.iter().any(|(k, v)| k == "X-Api-Sequence" && v == "-1"));
        assert!(!h.iter().any(|(k, _)| k == "X-Tt-Logid"));

        let h = build_headers("k", "r", "req", "log");
        assert!(h.iter().any(|(k, v)| k == "X-Tt-Logid" && v == "log"));
    }

    #[test]
    fn submit_body_has_audio_and_request_fields() {
        let body = build_submit_body("AAAA", true, false);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["audio"]["data"], "AAAA");
        assert_eq!(v["audio"]["format"], "wav");
        assert_eq!(v["audio"]["rate"], 16000);
        assert_eq!(v["request"]["model_name"], "bigmodel");
        assert_eq!(v["request"]["enable_itn"], true);
        assert_eq!(v["request"]["enable_punc"], false);
    }

    #[test]
    fn parse_result_prefers_text() {
        assert_eq!(
            parse_result_json(r#"{"result":{"text":" 你好 ","utterances":[{"text":"IGNORED"}]}}"#),
            "你好"
        );
    }

    #[test]
    fn parse_result_falls_back_to_utterances() {
        assert_eq!(
            parse_result_json(r#"{"result":{"text":"","utterances":[{"text":"你"},{"text":"好"}]}}"#),
            "你好"
        );
    }

    #[test]
    fn parse_result_garbage_is_empty() {
        assert_eq!(parse_result_json("not json"), "");
    }
}
