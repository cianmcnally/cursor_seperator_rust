use std::cmp::Reverse;
use super::model::WindowLayer;

/// Composite all segmentation windows into a u32 label mask.
///
/// Sorts by z_index descending (backmost first) internally — does not trust
/// caller ordering.  z_index 0 = frontmost, so highest z_index paints first
/// and lower z_index (front) overwrites.
///
/// Output: flat row-major, length = canvas_w * canvas_h.
/// Each cell = window_id of topmost window, 0 = background.
pub fn composite_label_mask(
    windows: &[WindowLayer],
    canvas_w: usize,
    canvas_h: usize,
    cursor_rect: Option<(i32, i32, i32, i32)>,
    cursor_label: u32,
) -> Vec<u32> {
    let mut mask = vec![0u32; canvas_w * canvas_h];

    let mut wins: Vec<&WindowLayer> = windows
        .iter()
        .filter(|w| w.include_in_segmentation)
        .collect();

    // Paint backmost first; frontmost (z_index 0) paints last and wins.
    wins.sort_by_key(|w| Reverse(w.z_index));

    for win in wins {
        fill_rect_label(&mut mask, canvas_w, canvas_h, win.bounds_pixels, win.window_id);
    }

    if let Some((cx, cy, cw, ch)) = cursor_rect {
        fill_rect_label(
            &mut mask, canvas_w, canvas_h,
            super::model::RectI { x: cx, y: cy, w: cw, h: ch },
            cursor_label,
        );
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
    if rect.w <= 0 || rect.h <= 0 || canvas_w == 0 || canvas_h == 0 {
        return;
    }

    let x0 = rect.x.max(0).min(canvas_w as i32) as usize;
    let y0 = rect.y.max(0).min(canvas_h as i32) as usize;
    let x1 = (rect.x + rect.w).max(0).min(canvas_w as i32) as usize;
    let y1 = (rect.y + rect.h).max(0).min(canvas_h as i32) as usize;

    if x0 >= x1 || y0 >= y1 { return; }

    for row in y0..y1 {
        let base = row * canvas_w + x0;
        mask[base..base + (x1 - x0)].fill(label);
    }
}

/// Convert one label id to a BGRA u32 pixel — no allocation.
pub fn label_to_fb_pixel(id: u32, cursor_label: u32) -> u32 {
    if id == 0 {
        0xFF000000
    } else if id == cursor_label {
        0xFF00FFFF
    } else {
        label_color(id)
    }
}

/// Convert a full label mask to a BGRA framebuffer Vec (debug/export use).
pub fn label_mask_to_fb(mask: &[u32], cursor_label: u32) -> Vec<u32> {
    mask.iter().map(|&id| label_to_fb_pixel(id, cursor_label)).collect()
}

fn label_color(id: u32) -> u32 {
    let mut h = id.wrapping_mul(0x9e37_79b9);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    let r = 0x80 | (h         & 0x7F);
    let g = 0x80 | ((h >>  8) & 0x7F);
    let b = 0x80 | ((h >> 16) & 0x7F);
    0xFF00_0000 | (r << 16) | (g << 8) | b
}
