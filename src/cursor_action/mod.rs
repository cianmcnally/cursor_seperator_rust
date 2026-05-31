mod detector;
mod model;
mod tap;

pub use detector::{ActionSnapshot, CursorActionArgs, start_cursor_action_detector};
pub use model::{CursorAction, CursorActionKind};

/// Snake-case label that matches the model's prediction space exactly
/// (see `predictions.rs`).
pub fn action_kind_str(k: CursorActionKind) -> &'static str {
    match k {
        CursorActionKind::Idle        => "idle",
        CursorActionKind::Move        => "move",
        CursorActionKind::Click       => "click",
        CursorActionKind::DoubleClick => "double_click",
        CursorActionKind::Drag        => "drag",
        CursorActionKind::Scroll      => "scroll",
        CursorActionKind::Typing      => "typing",
    }
}
