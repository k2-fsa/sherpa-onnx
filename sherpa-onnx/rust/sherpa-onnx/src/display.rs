//! Small terminal display helper for streaming ASR demos.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Stores finalized sentences with timestamps and the current partial hypothesis for terminal UIs.
#[derive(Debug)]
pub struct DisplayManager {
    sentences: Vec<(String, String)>,
    current_text: String,
    last_render: Instant,
}

fn current_datetime() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Seconds since start of UTC day
    let secs_today = now % 86400;
    let hour = secs_today / 3600;
    let minute = (secs_today % 3600) / 60;
    let second = secs_today % 60;

    // Days since 1970-01-01
    let days = (now / 86400) as i64;

    // Convert days-since-epoch to Y-M-D (proleptic Gregorian)
    let (year, month, day) = days_to_ymd(days);

    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        year, month, day, hour, minute, second
    )
}

/// Convert a day-count (days since 1970-01-01, epoch = 0) to (year, month, day).
fn days_to_ymd(days: i64) -> (i32, u32, u32) {
    // Algorithm from Howard Hinnant's date library.
    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u32; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m, d)
}

impl DisplayManager {
    /// Create an empty display manager.
    pub fn new() -> Self {
        Self {
            sentences: Vec::new(),
            current_text: String::new(),
            last_render: Instant::now(),
        }
    }

    /// Replace the current partial text shown in the display.
    pub fn update_text(&mut self, text: &str) {
        self.current_text = text.to_string();
    }

    /// Move the current partial text into the finalized sentence list with a timestamp.
    pub fn finalize_sentence(&mut self) {
        let trimmed = self.current_text.trim();
        if !trimmed.is_empty()
            && (self.current_text.as_bytes()[0] != b' ' || self.current_text.len() > 1)
        {
            self.sentences
                .push((current_datetime(), trimmed.to_string()));
        }
        self.current_text.clear();
    }

    /// Render the current state to stdout.
    ///
    /// Rendering is throttled slightly to reduce terminal flicker.
    pub fn render(&mut self) {
        // Throttle rendering to reduce flicker (200ms)
        if self.last_render.elapsed() < Duration::from_millis(200) {
            return;
        }

        if self.sentences.is_empty() && self.current_text.is_empty() {
            return;
        }

        self.last_render = Instant::now();

        // Only clear screen if there is content to show
        if !self.sentences.is_empty() || !self.current_text.is_empty() {
            print!("\x1B[2J\x1B[1;1H");
        }

        println!("=== Speech Recognition with Next-gen Kaldi ===");
        println!("------------------------------");

        if !self.sentences.is_empty() {
            for (i, (ts, text)) in self.sentences.iter().enumerate() {
                println!("[{}] {}. {}", ts, i + 1, text);
            }
            println!("------------------------------");
        }

        if !self.current_text.is_empty() {
            println!("Recognizing: {}", self.current_text);
        }
    }

    /// Return `true` if at least one sentence has been finalized.
    pub fn has_sentences(&self) -> bool {
        !self.sentences.is_empty()
    }

    /// Borrow the current partial text.
    pub fn current_text(&self) -> &str {
        &self.current_text
    }
}

impl Default for DisplayManager {
    fn default() -> Self {
        Self::new()
    }
}
