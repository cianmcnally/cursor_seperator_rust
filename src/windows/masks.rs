use super::model::{RectI, WindowLayer};

/// Single-channel (u8) mask for one window.  0 = background, 255 = window region.
pub struct LayerMask {
    pub layer_id: u32,
    pub z_index: usize,
    pub rect: RectI,
    pub width: u32,
    pub height: u32,
    /// Row-major pixels, length = width * height.
    pub data: Vec<u8>,
}

impl LayerMask {
    /// Build a filled-rectangle mask from a window's pixel bounds, clipped to
    /// the capture canvas.
    pub fn from_window(win: &WindowLayer, canvas_w: u32, canvas_h: u32) -> Option<Self> {
        let clipped = win.bounds_pixels.clip(canvas_w as i32, canvas_h as i32)?;
        let w = clipped.w as u32;
        let h = clipped.h as u32;
        let data = vec![255u8; (w * h) as usize];
        Some(LayerMask {
            layer_id: win.window_id,
            z_index: win.z_index,
            rect: clipped,
            width: w,
            height: h,
            data,
        })
    }
}

/// Build per-window masks for all windows that should be in segmentation.
pub fn build_masks(windows: &[WindowLayer], canvas_w: u32, canvas_h: u32) -> Vec<LayerMask> {
    windows
        .iter()
        .filter(|w| w.include_in_segmentation)
        .filter_map(|w| LayerMask::from_window(w, canvas_w, canvas_h))
        .collect()
}
