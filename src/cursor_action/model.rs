use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CursorActionKind {
    Move,
    SingleClick,
    DoubleClick,
    ClickAndHold,
    DragStart,
    DragMove,
    DragEnd,
    DragSelect,
    Scroll,
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
