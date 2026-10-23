// Subtitle/text export in the three formats used by the reference app:
//
//  1. SRT             00:00:17,030 --> 00:00:19,942  (hh:mm:ss,mmm, hours zero-padded)
//  2. Timestamped TXT 0:00:17,030 --> 0:00:19,942  大家好…  (h:mm:ss,mmm + two spaces)
//  3. Plain TXT       all transcript text concatenated

#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

/// Clamp to >= 0 before formatting.
fn clamp(t: f32) -> f32 {
    if t.is_finite() && t > 0.0 {
        t
    } else {
        0.0
    }
}

/// Split seconds into (hours, minutes, seconds, milliseconds).
fn split_time(total_seconds: f32) -> (u32, u32, u32, u32) {
    let t = clamp(total_seconds);
    let h = (t / 3600.0).floor() as u32;
    let m = ((t % 3600.0) / 60.0).floor() as u32;
    let s = (t % 60.0).floor() as u32;
    // Same rounding as the reference: Math.round((t - floor(t)) * 1000),
    // then clamp to 999.
    let ms = ((t - t.floor()) * 1000.0).round() as u32;
    let ms = ms.min(999);
    (h, m, s, ms)
}

/// seconds -> h:mm:ss,mmm (timestamped TXT; hour not zero-padded)
pub fn ts_loose(total_seconds: f32) -> String {
    let (h, m, s, ms) = split_time(total_seconds);
    format!("{h}:{m:02}:{s:02},{ms:03}")
}

/// seconds -> hh:mm:ss,mmm (SRT; hour zero-padded to two digits)
pub fn ts_srt(total_seconds: f32) -> String {
    let (h, m, s, ms) = split_time(total_seconds);
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// Format 1: standard SRT subtitles.
pub fn to_srt(segments: &[Segment]) -> String {
    segments
        .iter()
        .enumerate()
        .map(|(i, seg)| {
            format!(
                "{}\n{} --> {}\n{}",
                i + 1,
                ts_srt(seg.start),
                ts_srt(seg.end),
                seg.text.trim()
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Format 2: timestamped text: `h:mm:ss,mmm --> h:mm:ss,mmm  text`.
pub fn to_timestamped_txt(segments: &[Segment]) -> String {
    segments
        .iter()
        .map(|seg| {
            format!(
                "{} --> {}  {}",
                ts_loose(seg.start),
                ts_loose(seg.end),
                seg.text.trim()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Format 3: plain text; timestamps dropped, texts concatenated.
pub fn to_plain_text(segments: &[Segment]) -> String {
    segments
        .iter()
        .map(|seg| seg.text.trim())
        .collect::<Vec<_>>()
        .join("")
}

/// Parse `hh:mm:ss,mmm` or `h:mm:ss,mmm` into seconds. Returns None if the
/// string does not look like a timestamp.
pub fn parse_time(s: &str) -> Option<f32> {
    let s = s.trim();
    let (hms, ms) = s.split_once(',')?;
    let ms: f32 = ms.trim().parse().ok()?;
    let mut parts = hms.split(':');
    let h: f32 = parts.next()?.trim().parse().ok()?;
    let m: f32 = parts.next()?.trim().parse().ok()?;
    let sec: f32 = parts.next()?.trim().parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some(h * 3600.0 + m * 60.0 + sec + ms / 1000.0)
}

/// Parse standard SRT content into segments (the inverse of `to_srt`).
pub fn parse_srt(content: &str) -> Result<Vec<Segment>, String> {
    let mut out = Vec::new();
    for block in content.split("\n\n") {
        let lines: Vec<&str> = block.lines().collect();
        if lines.is_empty() {
            continue;
        }
        // Block layout: index, "start --> end", then one or more text lines.
        let (times_idx, text_start) = if lines[0].trim().parse::<u32>().is_ok() {
            (1, 2)
        } else {
            (0, 1)
        };
        let times = lines
            .get(times_idx)
            .ok_or_else(|| "missing time line in srt block".to_string())?;
        let (start, end) = times
            .split_once("-->")
            .ok_or_else(|| format!("bad srt time line: {times}"))?;
        let start = parse_time(start).ok_or_else(|| format!("bad start time: {start}"))?;
        let end = parse_time(end).ok_or_else(|| format!("bad end time: {end}"))?;
        let text = lines[text_start.min(lines.len())..].join("\n").trim().to_string();
        out.push(Segment { start, end, text });
    }
    Ok(out)
}

/// Parse timestamped TXT content into segments (the inverse of
/// `to_timestamped_txt`): each line is
/// `h:mm:ss,mmm --> h:mm:ss,mmm  text`.
pub fn parse_timestamped_txt(content: &str) -> Result<Vec<Segment>, String> {
    let mut out = Vec::new();
    for line in content.lines() {
        let line = line.trim_end();
        if line.trim().is_empty() {
            continue;
        }
        let (left, rest) = line
            .split_once("-->")
            .ok_or_else(|| format!("bad timestamped line: {line}"))?;
        let start = parse_time(left).ok_or_else(|| format!("bad start time: {left}"))?;
        // rest = "h:mm:ss,mmm  text"; the text is separated by two spaces.
        let (end_str, text) = match rest.split_once("  ") {
            Some((t, text)) => (t, text),
            None => (rest.trim(), ""),
        };
        let end = parse_time(end_str).ok_or_else(|| format!("bad end time: {end_str}"))?;
        out.push(Segment {
            start,
            end,
            text: text.trim().to_string(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two sample segments used by the tests.
    fn segs() -> Vec<Segment> {
        vec![
            Segment {
                start: 17.03,
                end: 19.942,
                text: "大家好".to_string(),
            },
            Segment {
                start: 61.5,
                end: 62.0,
                text: "  hello  ".to_string(),
            },
        ]
    }

    #[test]
    /// Tests SRT timestamps zero-pad hours.
    fn srt_hours_are_zero_padded() {
        assert_eq!(ts_srt(17.03), "00:00:17,030");
        assert_eq!(ts_srt(3723.5), "01:02:03,500");
    }

    #[test]
    /// Tests timestamped-TXT timestamps leave hours unpadded.
    fn loose_hours_are_not_zero_padded() {
        assert_eq!(ts_loose(17.03), "0:00:17,030");
        assert_eq!(ts_loose(3723.5), "1:02:03,500");
    }

    #[test]
    /// Tests invalid times clamp to zero.
    fn negative_and_nan_times_clamp_to_zero() {
        assert_eq!(ts_srt(-5.0), "00:00:00,000");
        assert_eq!(ts_srt(f32::NAN), "00:00:00,000");
    }

    #[test]
    /// Tests the SRT layout matches the reference app.
    fn srt_matches_reference_layout() {
        let expected = "1\n00:00:17,030 --> 00:00:19,942\n大家好\n\n\
                        2\n00:01:01,500 --> 00:01:02,000\nhello";
        assert_eq!(to_srt(&segs()), expected);
    }

    #[test]
    /// Tests the timestamped-TXT layout (two spaces before text).
    fn timestamped_txt_uses_loose_times_and_two_spaces() {
        assert_eq!(
            to_timestamped_txt(&segs()),
            "0:00:17,030 --> 0:00:19,942  大家好\n0:01:01,500 --> 0:01:02,000  hello"
        );
    }

    #[test]
    /// Tests plain text concatenates trimmed segment texts.
    fn plain_txt_concatenates_trimmed_texts() {
        assert_eq!(to_plain_text(&segs()), "大家好hello");
    }

    #[test]
    /// Tests timestamp parsing for both hour widths.
    fn parse_time_accepts_padded_and_unpadded_hours() {
        assert_eq!(parse_time("00:00:17,030"), Some(17.03));
        assert_eq!(parse_time("0:00:17,030"), Some(17.03));
        assert_eq!(parse_time("1:02:03,500"), Some(3723.5));
        assert_eq!(parse_time("nonsense"), None);
    }

    #[test]
    /// Tests SRT format -> parse round-trip.
    fn srt_round_trips() {
        let srt = to_srt(&segs());
        let parsed = parse_srt(&srt).unwrap();
        assert_eq!(parsed.len(), 2);
        assert!((parsed[0].start - 17.03).abs() < 0.001);
        assert!((parsed[0].end - 19.942).abs() < 0.001);
        assert_eq!(parsed[0].text, "大家好");
        assert_eq!(parsed[1].text, "hello");
    }

    #[test]
    /// Tests timestamped-TXT format -> parse round-trip.
    fn timestamped_txt_round_trips() {
        let txt = to_timestamped_txt(&segs());
        let parsed = parse_timestamped_txt(&txt).unwrap();
        assert_eq!(parsed.len(), 2);
        assert!((parsed[1].start - 61.5).abs() < 0.001);
        assert_eq!(parsed[1].text, "hello");
    }
}
