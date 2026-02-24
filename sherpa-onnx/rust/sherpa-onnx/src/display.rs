// Display manager for streaming ASR, inspired by sherpa-display.h
// Handles finalized sentences and current partial text.

use std::time::{Duration, Instant};

/// DisplayManager stores finalized sentences and current partial text
#[derive(Debug)]
pub struct DisplayManager {
    sentences: Vec<String>,
    current_text: String,
    last_render: Instant,
}

impl DisplayManager {
    /// Create a new DisplayManager
    pub fn new() -> Self {
        Self {
            sentences: Vec::new(),
            current_text: String::new(),
            last_render: Instant::now(),
        }
    }

    /// Update the current partial text
    pub fn update_text(&mut self, text: &str) {
        self.current_text = text.to_string();
    }

    /// Finalize the current sentence and move it to `sentences`
    pub fn finalize_sentence(&mut self) {
        let trimmed = self
            .current_text
            .trim();
        if !trimmed.is_empty() {
            self.sentences
                .push(trimmed.to_string());
        }
        self.current_text
            .clear();
    }

    /// Render the display to stdout
    /// Clears the screen and prints finalized + current text
    pub fn render(&mut self) {
        // Throttle rendering to reduce flicker (200ms)
        if self
            .last_render
            .elapsed()
            < Duration::from_millis(200)
        {
            return;
        }
        self.last_render = Instant::now();

        // Clear screen (ANSI escape)
        print!("\x1B[2J\x1B[1;1H");
        println!("=== Speech Recognition with Next-gen Kaldi ===");
        println!("-----------------------------------------------");

        for (i, s) in self
            .sentences
            .iter()
            .enumerate()
        {
            println!("{}: {}", i + 1, s);
        }

        if !self
            .current_text
            .is_empty()
        {
            println!("-----------------------------------------------");
            println!("Recognizing: {}", self.current_text);
        }
    }

    /// Returns true if there are finalized sentences
    pub fn has_sentences(&self) -> bool {
        !self
            .sentences
            .is_empty()
    }

    /// Returns current partial text
    pub fn current_text(&self) -> &str {
        &self.current_text
    }
}
