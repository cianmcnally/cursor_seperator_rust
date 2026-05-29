use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct RectF {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct RectI {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl RectI {
    pub fn area(&self) -> i64 {
        (self.w.max(0) as i64) * (self.h.max(0) as i64)
    }
    pub fn is_empty(&self) -> bool {
        self.w <= 0 || self.h <= 0
    }
    /// Clip to [0, canvas_w) × [0, canvas_h).  Returns None if fully outside.
    pub fn clip(&self, canvas_w: i32, canvas_h: i32) -> Option<RectI> {
        let x0 = self.x.max(0);
        let y0 = self.y.max(0);
        let x1 = (self.x + self.w).min(canvas_w);
        let y1 = (self.y + self.h).min(canvas_h);
        if x1 <= x0 || y1 <= y0 {
            None
        } else {
            Some(RectI { x: x0, y: y0, w: x1 - x0, h: y1 - y0 })
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WindowCategory {
    NormalAppWindow,
    SystemUi,
    MenuBar,
    Dock,
    Desktop,
    TooltipPopover,
    Overlay,
    TinyJunk,
    Unknown,
}

impl WindowCategory {
    pub fn from_layer_and_owner(cg_layer: i32, owner: &str) -> Self {
        match cg_layer {
            i32::MIN..=-1 => WindowCategory::Desktop,
            0 => WindowCategory::NormalAppWindow,
            1..=19 => WindowCategory::TooltipPopover,
            20 if owner == "Dock" => WindowCategory::Dock,
            20..=30 => {
                if owner.contains("Window Server") || owner.contains("Dock") {
                    WindowCategory::SystemUi
                } else {
                    WindowCategory::Overlay
                }
            }
            31..=499 => WindowCategory::Overlay,
            _ => WindowCategory::SystemUi,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowLayer {
    pub window_id: u32,
    /// Position in the array returned by CGWindowListCopyWindowInfo; 0 = frontmost.
    pub z_index: usize,
    /// kCGWindowLayer — NOT the same as z_index.
    pub cg_layer: i32,
    pub owner_pid: i32,
    pub owner_name: String,
    pub window_name: Option<String>,
    pub bounds_points: RectF,
    pub bounds_pixels: RectI,
    pub alpha: f64,
    pub is_onscreen: bool,
    pub sharing_state: Option<i32>,
    pub store_type: Option<i32>,
    pub memory_usage: Option<i64>,
    pub category: WindowCategory,
    pub include_in_segmentation: bool,
}

/// Timing snapshot from one sample + composite cycle.
#[derive(Debug, Clone, Default)]
pub struct WindowTimings {
    pub window_sample_ms: f64,
    pub mask_build_ms: f64,
    pub composite_ms: f64,
    pub render_ms: f64,
    pub raw_window_count: usize,
    pub segmentation_window_count: usize,
}
