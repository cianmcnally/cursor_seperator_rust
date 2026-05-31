use crate::windows::model::RectI;

/// Normalize a `[x, y, w, h]` pixel bbox to `[0,1]` relative to the capture canvas.
#[inline]
pub fn norm_bbox(bbox: [i32; 4], cap_w: u32, cap_h: u32) -> [f32; 4] {
    let cw = cap_w as f32;
    let ch = cap_h as f32;
    [
        bbox[0] as f32 / cw,
        bbox[1] as f32 / ch,
        bbox[2] as f32 / cw,
        bbox[3] as f32 / ch,
    ]
}

/// Normalize a pixel point to `[0,1]` relative to the capture canvas.
#[inline]
pub fn norm_pt(x: i32, y: i32, cap_w: u32, cap_h: u32) -> [f32; 2] {
    [x as f32 / cap_w as f32, y as f32 / cap_h as f32]
}

/// Convert `RectI` to `[x, y, w, h]` array.
#[inline]
pub fn recti_to_bbox(r: RectI) -> [i32; 4] {
    [r.x, r.y, r.w, r.h]
}


