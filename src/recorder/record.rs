use std::collections::VecDeque;
use std::io::{BufWriter, Write as IoWrite};
use std::fs::File;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::coords;
use crate::cursor_action::{action_kind_str, CursorAction};
use crate::windows::WindowLayer;

// ── Serializable record types ────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct CursorRecord {
    pub class:         String,
    pub action:        String,
    pub position_px:   [i32; 2],
    pub position_norm: [f32; 2],
    pub bbox_px:       [i32; 4],
    pub bbox_norm:     [f32; 4],
    pub window_id:     Option<u32>,
    pub z_index:       Option<usize>,
    pub confidence:    f32,
}

#[derive(Serialize, Deserialize)]
pub struct MaskPaths {
    pub windows_label:  String,
    pub cursor_mask:    String,
    pub combined_label: String,
    /// Dense per-frame u16 instance map (0=bg, 1..N front-to-back).
    #[serde(default)]
    pub instance_map:   String,
    /// u8 semantic class mask (class_label::* values).
    #[serde(default)]
    pub class_mask:     String,
    /// JSON array of VisibleWindowMask — authoritative per-window metadata.
    #[serde(default)]
    pub windows_json:   String,
}

#[derive(Serialize, Deserialize)]
pub struct TypingAreaRecord {
    pub bbox:       [i32; 4],
    pub confidence: f32,
    pub source:     String,
}

#[derive(Serialize, Deserialize)]
pub struct FrameRecord {
    pub frame_index:     u64,
    pub timestamp_ns:    u64,
    pub image_path:      Option<String>,
    pub capture_size_px: [u32; 2],
    pub windows:         Vec<OwnedWindowRecord>,
    pub cursor:          CursorRecord,
    pub mask_paths:      MaskPaths,
    pub label_map:       Vec<LabelEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typing_area:     Option<TypingAreaRecord>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct OwnedWindowRecord {
    pub class:       String,
    pub action:      String,
    pub window_id:   u32,
    pub z_index:     usize,
    pub cg_layer:    i32,
    pub owner_pid:   i32,
    pub owner_name:  String,
    pub window_name: Option<String>,
    pub bbox_px:     [i32; 4],
    pub bbox_norm:   [f32; 4],
    pub visible:     bool,
    pub confidence:  f32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LabelEntry {
    pub label_id:   u16,
    pub window_id:  u32,
    pub owner_name: String,
}

#[derive(Serialize, Deserialize)]
pub struct EventRecord {
    pub event_id:    String,
    pub kind:        String,
    pub timestamp_ns: u64,
    pub frame_index: u64,
    pub position_px:   [i32; 2],
    pub position_norm: [f32; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox_px:     Option<[i32; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox_norm:   Option<[f32; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_position_px: Option<[i32; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_position_px:   Option<[i32; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path_px:     Option<Vec<[i32; 2]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distance_px: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub button:      Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub click_count: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window_id:   Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z_index:     Option<usize>,
    pub confidence:  f32,
    pub source:      String,
}

// ── NDJSON writer ─────────────────────────────────────────────────────────────

pub struct NdjsonWriter {
    file: BufWriter<File>,
}

impl NdjsonWriter {
    pub fn create(path: &Path) -> std::io::Result<Self> {
        let f = File::create(path)?;
        Ok(Self { file: BufWriter::new(f) })
    }

    pub fn write<T: serde::Serialize>(&mut self, record: &T) {
        if let Ok(s) = serde_json::to_string(record) {
            let _ = writeln!(self.file, "{}", s);
        }
    }

    pub fn flush(&mut self) { let _ = self.file.flush(); }
}

// ── Record builders ───────────────────────────────────────────────────────────

pub fn build_window_records(
    windows:    &[WindowLayer],
    focused_id: Option<u32>,
    cap_w:      u32,
    cap_h:      u32,
) -> Vec<OwnedWindowRecord> {
    windows.iter()
        .filter(|w| w.include_in_segmentation)
        .map(|w| {
            let bbox = coords::recti_to_bbox(w.bounds_pixels);
            let action = if w.category.is_popup_like() {
                "popup"
            } else if Some(w.window_id) == focused_id {
                "focused"
            } else {
                "unfocused"
            };
            OwnedWindowRecord {
                class:      "window".to_string(),
                action:     action.to_string(),
                window_id:  w.window_id,
                z_index:    w.z_index,
                cg_layer:   w.cg_layer,
                owner_pid:  w.owner_pid,
                owner_name: w.owner_name.clone(),
                window_name: w.window_name.clone(),
                bbox_px:    bbox,
                bbox_norm:  coords::norm_bbox(bbox, cap_w, cap_h),
                visible:    w.is_onscreen,
                confidence: 1.0,
            }
        })
        .collect()
}

pub fn build_cursor_record(
    pos_px:    (i32, i32),
    bbox_px:   [i32; 4],
    action:    &str,
    window_id: Option<u32>,
    z_index:   Option<usize>,
    cap_w:     u32,
    cap_h:     u32,
) -> CursorRecord {
    CursorRecord {
        class:         "cursor".to_string(),
        action:        action.to_string(),
        position_px:   [pos_px.0, pos_px.1],
        position_norm: coords::norm_pt(pos_px.0, pos_px.1, cap_w, cap_h),
        bbox_px,
        bbox_norm:     coords::norm_bbox(bbox_px, cap_w, cap_h),
        window_id,
        z_index,
        confidence:    1.0,
    }
}

pub fn build_event_record(
    action:      &CursorAction,
    event_id:    u64,
    frame_index: u64,
    cap_w:       u32,
    cap_h:       u32,
) -> EventRecord {
    let pos   = action.position;
    let bbox  = action.bbox;
    let sp    = action.start_position;
    let ep    = action.end_position;
    let path  = if action.path.is_empty() { None }
                else { Some(action.path.iter().map(|&(x, y)| [x, y]).collect()) };

    EventRecord {
        event_id:    format!("evt_{:06}", event_id),
        kind:        action_kind_str(action.kind).to_string(),
        timestamp_ns: action.timestamp_ns,
        frame_index,
        position_px:   [pos.0, pos.1],
        position_norm: coords::norm_pt(pos.0, pos.1, cap_w, cap_h),
        bbox_px:   bbox,
        bbox_norm: bbox.map(|b| coords::norm_bbox(b, cap_w, cap_h)),
        start_position_px: sp.map(|(x, y)| [x, y]),
        end_position_px:   ep.map(|(x, y)| [x, y]),
        path_px:   path,
        duration_ms: action.duration_ms,
        distance_px: action.distance_px,
        button:      action.button.clone(),
        click_count: action.click_count,
        window_id:   action.window_id,
        z_index:     action.z_index,
        confidence:  action.confidence,
        source:      action.source.clone(),
    }
}

// ── New entity event records ──────────────────────────────────────────────────

/// One keyboard KeyDown event written to events.ndjson.
#[derive(Serialize, Deserialize, Clone)]
pub struct KeyEventRecord {
    pub event_id:    String,
    pub class:       String,   // "keyboard"
    pub kind:        String,   // "KeyDown"
    pub timestamp_ns: u64,
    pub frame_index: u64,
    pub key_code:    u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window_id:   Option<u32>,   // focused window at key time
}

/// Window focus change event written to events.ndjson.
#[derive(Serialize, Deserialize, Clone)]
pub struct FocusEventRecord {
    pub event_id:    String,
    pub class:       String,   // "window"
    pub kind:        String,   // "FocusChange"
    pub timestamp_ns: u64,
    pub frame_index: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_window: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_window:   Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_owner:    Option<String>,
}

// ── Frame timestamp tracker for nearest-frame lookup ─────────────────────────

pub struct FrameTimestamps {
    entries: VecDeque<(u64, u64)>,   // (frame_index, timestamp_ns)
}

impl FrameTimestamps {
    pub fn new() -> Self { Self { entries: VecDeque::with_capacity(60) } }

    pub fn push(&mut self, frame_index: u64, ts_ns: u64) {
        if self.entries.len() >= 60 { self.entries.pop_front(); }
        self.entries.push_back((frame_index, ts_ns));
    }

    /// Return frame_index whose timestamp is nearest to `event_ts`.
    pub fn nearest(&self, event_ts: u64) -> u64 {
        self.entries.iter()
            .min_by_key(|(_, ts)| event_ts.abs_diff(*ts))
            .map(|(fi, _)| *fi)
            .unwrap_or(0)
    }
}
