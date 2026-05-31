// ── mlp::clips — Build strongly-typed LabeledClips from saved session files ──
//
// Reads the same session files that the Python pipeline reads (session.json +
// frames.ndjson) and produces a Vec<LabeledClip> where every field has a
// compile-time-checked type.

use std::collections::HashMap;
use std::path::Path;

use crate::mlp::types::{ActionLabel, FrameMeta, LabeledClip};
use crate::session_loader::LoadedSession;

// ── Load frames from one session ─────────────────────────────────────────────

/// Load per-frame metadata from a single session recording directory.
/// Returns only frames that have an image_path and a valid action label.
pub fn load_frames_from_session(session_dir: &Path) -> Result<Vec<FrameMeta>, String> {
    let session = LoadedSession::load(session_dir)?;
    let session_id = session.meta.session_id.clone();

    let mut frames = Vec::with_capacity(session.frames.len());

    for fr in &session.frames {
        let image_path = match &fr.image_path {
            Some(p) => session_dir.join(p),
            None => continue, // skip frames without saved images
        };

        if !image_path.exists() {
            continue;
        }

        let action: ActionLabel = match fr.cursor.action.parse() {
            Ok(a) => a,
            Err(e) => {
                eprintln!(
                    "  [warn] frame {}: unknown action '{}': {e} — skipping",
                    fr.frame_index, fr.cursor.action
                );
                continue;
            }
        };

        frames.push(FrameMeta {
            image_path,
            cursor_pos_norm: fr.cursor.position_norm,
            action,
            session_id: session_id.clone(),
        });
    }

    Ok(frames)
}

// ── Load multiple sessions ───────────────────────────────────────────────────

/// Load frames from one or more session directories.
///
/// Classes with fewer than `min_class_count` frames are dropped.
///
/// JitterClick is already merged into Click by `ActionLabel::from_str` at
/// parse time, so no separate merge step is needed.
///
/// Returns `(all_frames, action_labels)` where `action_labels` is the sorted
/// list of action labels that survived filtering.
pub fn load_all_sessions(
    dirs: &[impl AsRef<Path>],
    min_class_count: usize,
) -> Result<(Vec<FrameMeta>, Vec<ActionLabel>), String> {
    let mut all_frames: Vec<FrameMeta> = Vec::new();

    for d in dirs {
        let d = d.as_ref();
        if !d.is_dir() {
            eprintln!("  [warn] not a directory — skipping: {}", d.display());
            continue;
        }
        let frames = load_frames_from_session(d)?;
        eprintln!(
            "  Loaded {:>5} frames from {}",
            frames.len(),
            d.file_name().unwrap_or_default().to_string_lossy()
        );
        all_frames.extend(frames);
    }

    if all_frames.is_empty() {
        return Err("No frames with images found in any session directory".into());
    }

    // Drop rare classes
    let mut counts: HashMap<ActionLabel, usize> = HashMap::new();
    for f in &all_frames {
        *counts.entry(f.action).or_insert(0) += 1;
    }

    let mut action_labels: Vec<ActionLabel> = counts
        .into_iter()
        .filter(|(_, n)| *n >= min_class_count)
        .map(|(a, _)| a)
        .collect();
    action_labels.sort_by_key(|a| a.as_index());

    let keep: Vec<ActionLabel> = action_labels.clone();
    all_frames.retain(|f| keep.contains(&f.action));

    if all_frames.is_empty() {
        return Err("All classes dropped by min_class_count filter".into());
    }

    // Print action counts
    let mut final_counts: HashMap<String, usize> = HashMap::new();
    for f in &all_frames {
        *final_counts.entry(f.action.to_string()).or_insert(0) += 1;
    }
    eprintln!("  Action counts:  {final_counts:?}");

    Ok((all_frames, action_labels))
}

// ── Clip builder ─────────────────────────────────────────────────────────────

/// Build T-frame temporal clips from consecutive saved frames within each
/// session.
///
/// Clips with large temporal gaps (from pauses or session restarts) are
/// skipped.  The label is taken from the **last** frame in the clip.
pub fn build_clips(
    frames: &[FrameMeta],
    clip_len: usize,
    stride: usize,
) -> Vec<LabeledClip> {
    // Group by session_id
    let mut by_session: HashMap<&str, Vec<&FrameMeta>> = HashMap::new();
    for f in frames {
        by_session
            .entry(f.session_id.as_str())
            .or_default()
            .push(f);
    }

    let mut clips: Vec<LabeledClip> = Vec::new();

    for (_sid, session_frames) in &by_session {
        if session_frames.len() < clip_len {
            continue;
        }

        // Session frames are already in temporal order from LoadedSession,
        // but we sort by path to be safe (paths embed timestamps).
        let mut sorted: Vec<&&FrameMeta> = session_frames.iter().collect();
        sorted.sort_by_key(|f| f.image_path.as_os_str().to_os_string());

        // Estimate typical frame step for temporal-gap detection.
        // We don't have explicit frame_index in FrameMeta, so we use the
        // frame numbering embedded in the image path filenames as a proxy.
        let indices: Vec<usize> = sorted
            .iter()
            .filter_map(|f| extract_frame_number(&f.image_path))
            .collect();

        let typical_step = if indices.len() >= 2 {
            let steps: Vec<usize> = indices
                .windows(2)
                .map(|w| w[1].saturating_sub(w[0]))
                .collect();
            if steps.is_empty() {
                1
            } else {
                // median
                let mut s = steps.clone();
                s.sort();
                s[s.len() / 2]
            }
        } else {
            1
        };
        let max_gap = (typical_step * 4).max(1);

        let n = sorted.len();

        for start in (0..n.saturating_sub(clip_len - 1)).step_by(stride.max(1)) {
            let end = start + clip_len;
            if end > n {
                break;
            }
            let window = &sorted[start..end];

            // Temporal gap check
            let idxs: Vec<usize> = window
                .iter()
                .filter_map(|f| extract_frame_number(&f.image_path))
                .collect();
            if idxs.len() == clip_len {
                let gaps: Vec<usize> = idxs
                    .windows(2)
                    .map(|w| w[1].saturating_sub(w[0]))
                    .collect();
                if gaps.iter().any(|&g| g > max_gap) {
                    continue; // temporal gap too large — skip
                }
            } else {
                // Could not extract frame numbers from all paths — we can't
                // verify temporal contiguity.  Accept the clip but warn.
                eprintln!(
                    "  [warn] clip starting at frame index {} in session '{}': \
                     could not extract frame numbers from {}/{} paths — \
                     temporal gap check skipped",
                    start,
                    _sid,
                    clip_len - idxs.len(),
                    clip_len,
                );
            }

            let last = window[clip_len - 1];

            clips.push(LabeledClip {
                frames: window.iter().map(|f| f.image_path.clone()).collect(),
                cursor_pos: last.cursor_pos_norm,
                action: last.action,
                session_id: last.session_id.clone(),
            });
        }
    }

    clips
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Try to extract a frame number from an image path like
/// `frames/frame_000042.jpg` or `frames/000042.png`.
fn extract_frame_number(path: &Path) -> Option<usize> {
    let stem = path.file_stem()?.to_str()?;
    // Strip "frame_" prefix if present
    let num_str = stem.strip_prefix("frame_").unwrap_or(stem);
    num_str.parse::<usize>().ok()
}
