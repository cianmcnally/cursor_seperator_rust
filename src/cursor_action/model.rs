use serde::{Deserialize, Serialize};

/// The seven mutually-exclusive action classes the prediction model consumes.
/// A held or jittery press is still a `Click`; a select-drag is still a `Drag`.
/// Sub-signal (hold duration, jitter, click count, drag path) is preserved in the
/// `CursorAction` fields, not as separate labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum CursorActionKind {
    #[default]
    Idle,
    Move,
    Click,
    DoubleClick,
    Drag,
    Scroll,
    Typing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorAction {
    pub kind:           CursorActionKind,
    pub timestamp_ns:   u64,
    pub position:       (i32, i32),
    pub start_position: Option<(i32, i32)>,
    pub end_position:   Option<(i32, i32)>,
    pub path:           Vec<(i32, i32)>,
    /// [x, y, w, h] in capture-pixel coordinates.
    pub bbox:           Option<[i32; 4]>,
    pub duration_ms:    Option<f64>,
    pub distance_px:    Option<f64>,
    pub button:         Option<String>,
    pub click_count:    Option<u8>,
    pub window_id:      Option<u32>,
    pub z_index:        Option<usize>,
    pub confidence:     f32,
    pub source:         String,
}
