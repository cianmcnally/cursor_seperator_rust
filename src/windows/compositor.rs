use super::model::WindowLayer;

/// Composite all segmentation windows back-to-front into a u32 label mask.
///
/// Windows are stored front-to-back (z_index 0 = frontmost).  To produce a
/// topmost-visible-window segmentation we iterate in reverse so front windows
/// paint last and overwrite back windows.
///
/// Output: flat row-major array, length = canvas_w * canvas_h.
/// Each cell holds the window_id of the topmost window covering that pixel,
/// or 0 for background.
pub fn composite_label_mask(
    windows: &[WindowLayer],
    canvas_w: usize,
    canvas_h: usize,
    cursor_rect: Option<(i32, i32, i32, i32)>, // (x, y, w, h) in pixels
    cursor_label: u32,
) -> Vec<u32> {
    let mut mask = vec![0u32; canvas_w * canvas_h];

    // Back-to-front: reversed iterator (highest z_index first).
    for win in windows.iter().rev() {
        if !win.include_in_segmentation { continue; }
        fill_rect_label(&mut mask, canvas_w, canvas_h, win.bounds_pixels, win.window_id);
    }

    // Cursor always on top.
    if let Some((cx, cy, cw, ch)) = cursor_rect {
        let rect = super::model::RectI { x: cx, y: cy, w: cw, h: ch };
        fill_rect_label(&mut mask, canvas_w, canvas_h, rect, cursor_label);
    }

    mask
}

fn fill_rect_label(
    mask: &mut [u32],
    canvas_w: usize,
    canvas_h: usize,
    rect: super::model::RectI,
    label: u32,
) {
    let x0 = rect.x.max(0) as usize;
    let y0 = rect.y.max(0) as usize;
    let x1 = (rect.x + rect.w).min(canvas_w as i32).max(0) as usize;
    let y1 = (rect.y + rect.h).min(canvas_h as i32).max(0) as usize;

    for row in y0..y1 {
        let base = row * canvas_w + x0;
        mask[base..base + (x1 - x0)].fill(label);
    }
}

/// Convert a u32 label mask to a visible BGRA u32 framebuffer for debug preview.
/// Each unique non-zero label gets a deterministic colour.  Zero (background) → black.
/// Cursor label → bright cyan.
pub fn label_mask_to_fb(mask: &[u32], cursor_label: u32) -> Vec<u32> {
    mask.iter()
        .map(|&id| {
            if id == 0 {
                0xFF000000
            } else if id == cursor_label {
                0xFF00FFFF // cyan for cursor
            } else {
                label_color(id)
            }
        })
        .collect()
}

/// Deterministic pseudo-random colour for a window id.
fn label_color(id: u32) -> u32 {
    // Murmur-style hash to spread IDs across the colour space.
    let mut h = id.wrapping_mul(0x9e37_79b9);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    // Force decent brightness: set high bits of each channel.
    let r = 0x80 | ((h       ) & 0x7F);
    let g = 0x80 | ((h >>  8) & 0x7F);
    let b = 0x80 | ((h >> 16) & 0x7F);
    0xFF00_0000 | (r << 16) | (g << 8) | b
}
