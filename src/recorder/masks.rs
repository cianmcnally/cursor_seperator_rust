use std::collections::HashMap;
use std::io::BufWriter;
use std::path::Path;

use crate::windows::WindowLayer;
use crate::windows::model::RectI;
use super::record::LabelEntry;

const CURSOR_LABEL: u16 = 0xFFFF;

// ── Semantic class labels for visible_class_mask ──────────────────────────────
//
// NOTE: these encode window *state* (focus / popup-of), not pure object type.
// The same window changes label when focus changes. If you later need a stable
// object-type channel (window / popup / menu / cursor), add a second mask —
// don't overload this one.

pub mod class_label {
    pub const _BACKGROUND:         u8 = 0;
    pub const FOCUSED_WINDOW:     u8 = 1;
    pub const UNFOCUSED_WINDOW:   u8 = 2;
    pub const POPUP_OF_FOCUSED:   u8 = 3;
    pub const POPUP_OF_UNFOCUSED: u8 = 4;
    pub const OCCLUDER:           u8 = 5;
}

// ── Per-window visible mask metadata ─────────────────────────────────────────

/// Rich per-window segmentation record for one frame, written to `*_windows.json`.
///
/// instance_id is dense per-frame (1..N, frontmost=1); NOT session-stable.
/// windows.json maps instance_id back to window_id and all other metadata.
#[derive(serde::Serialize)]
pub struct VisibleWindowMask {
    pub frame_id:                 u64,
    pub instance_id:              u16,
    pub window_id:                u32,
    pub app_name:                 String,
    pub title:                    Option<String>,
    pub z_index:                  usize,
    pub focused:                  bool,
    pub class_label:              u8,

    pub full_rect_px:             [i32; 4],
    pub full_area_px:             i64,

    pub visible_bbox_px:          [i32; 4],
    pub visible_area_px:          i64,
    /// visible_area / full window rect area (counts off-canvas area as "lost").
    pub visible_ratio:            f32,
    /// visible_area / on-canvas-clipped rect area. 1.0 == fully unobscured where
    /// it overlaps the capture region, even if part of the window is off-screen.
    pub visible_ratio_onscreen:   f32,

    /// Always "rle". Custom row-major, alternating bg/fg counts from bg.
    /// NOT COCO-compatible (COCO RLE is column-major / Fortran order). Decode
    /// with a matching row-major reader: see encode_rle_crop.
    pub visible_mask_encoding:    &'static str,
    /// RLE counts within the visible_bbox crop. Reconstruct by placing at
    /// visible_mask_bbox_origin in screen coords.
    pub visible_mask_rle:         Vec<u32>,
    pub visible_mask_bbox_origin: [i32; 2],
    pub visible_mask_bbox_size:   [u32; 2],

    pub occluded_by_window_ids:   Vec<u32>,
    pub is_visible:               bool,

    /// For a popup/tooltip: window_id of the frontmost layer-0 normal window
    /// owned by the same process (its visual parent). For a normal window:
    /// its own window_id. None if no parent could be resolved. Lets a consumer
    /// group "Chrome window + its dropdown" without collapsing their instances.
    pub root_window_id:           Option<u32>,
}

// ── Instance mask builder ─────────────────────────────────────────────────────

/// Compute the visible instance map, semantic class mask, and per-window metadata.
///
/// Returns (instance_map u16, class_mask u8, window_metadata):
///   instance_map — 0=bg, 1..N dense per-frame IDs, frontmost window gets id=1
///   class_mask   — 0=bg, see class_label::* constants
///   window_metadata — Vec<VisibleWindowMask>, one entry per segmented window
///
/// z_index=0 is frontmost. Sorted ascending so frontmost is processed first
/// and the occupied mask accumulates front-to-back (correct occlusion).
pub fn build_instance_mask(
    windows:  &[WindowLayer],
    frame_id: u64,
    cap_w:    usize,
    cap_h:    usize,
) -> (Vec<u16>, Vec<u8>, Vec<VisibleWindowMask>) {
    use crate::windows::model::WindowMaskRole;

    let mut instance_map = vec![0u16; cap_w * cap_h];
    let mut class_mask   = vec![0u8;  cap_w * cap_h];
    let mut occupied     = vec![false; cap_w * cap_h];

    let mut wins: Vec<&WindowLayer> = windows
        .iter()
        .filter(|w| w.include_in_segmentation)
        .collect();
    wins.sort_by_key(|w| w.z_index);  // z=0 first = frontmost first

    let mut result = Vec::with_capacity(wins.len());

    for (idx, win) in wins.iter().enumerate() {
        let instance_id = (idx + 1) as u16;
        let cl = match win.mask_role {
            WindowMaskRole::FocusedRoot      => class_label::FOCUSED_WINDOW,
            WindowMaskRole::UnfocusedRoot    => class_label::UNFOCUSED_WINDOW,
            WindowMaskRole::PopupOfFocused   => class_label::POPUP_OF_FOCUSED,
            WindowMaskRole::PopupOfUnfocused => class_label::POPUP_OF_UNFOCUSED,
            _                                => class_label::OCCLUDER,
        };

        let fr = win.bounds_pixels;
        let x0 = fr.x.max(0).min(cap_w as i32) as usize;
        let y0 = fr.y.max(0).min(cap_h as i32) as usize;
        let x1 = (fr.x + fr.w).max(0).min(cap_w as i32) as usize;
        let y1 = (fr.y + fr.h).max(0).min(cap_h as i32) as usize;

        let mut vx0 = usize::MAX;
        let mut vy0 = usize::MAX;
        let mut vx1 = 0usize;
        let mut vy1 = 0usize;
        let mut visible_area = 0i64;

        for py in y0..y1 {
            for px in x0..x1 {
                let i = py * cap_w + px;
                if !occupied[i] {
                    instance_map[i] = instance_id;
                    class_mask[i]   = cl;
                    occupied[i]     = true;
                    visible_area   += 1;
                    if px < vx0 { vx0 = px; }
                    if py < vy0 { vy0 = py; }
                    if px + 1 > vx1 { vx1 = px + 1; }
                    if py + 1 > vy1 { vy1 = py + 1; }
                }
            }
        }

        let is_visible = visible_area > 0;
        let visible_bbox_px = if is_visible {
            [vx0 as i32, vy0 as i32, (vx1 - vx0) as i32, (vy1 - vy0) as i32]
        } else {
            [0, 0, 0, 0]
        };

        // Windows with lower z_index (more front) whose rect overlaps this window.
        let occluded_by: Vec<u32> = wins[..idx]
            .iter()
            .filter(|w| rects_overlap(w.bounds_pixels, win.bounds_pixels))
            .map(|w| w.window_id)
            .collect();

        let (rle, bbox_origin, bbox_size) = if is_visible {
            (
                encode_rle_crop(&instance_map, cap_w, vx0, vy0, vx1, vy1, instance_id),
                [vx0 as i32, vy0 as i32],
                [(vx1 - vx0) as u32, (vy1 - vy0) as u32],
            )
        } else {
            (vec![], [0i32, 0i32], [0u32, 0u32])
        };

        // On-canvas clipped rect area (denominator for visible_ratio_onscreen).
        let onscreen_area =
            (x1.saturating_sub(x0) as i64) * (y1.saturating_sub(y0) as i64);

        let root_window_id = resolve_root_window_id(win, windows);

        let full_area = fr.area();
        result.push(VisibleWindowMask {
            frame_id,
            instance_id,
            window_id:    win.window_id,
            app_name:     win.owner_name.clone(),
            title:        win.window_name.clone(),
            z_index:      win.z_index,
            focused:      matches!(win.mask_role, WindowMaskRole::FocusedRoot),
            class_label:  cl,
            full_rect_px: [fr.x, fr.y, fr.w, fr.h],
            full_area_px: full_area,
            visible_bbox_px,
            visible_area_px: visible_area,
            visible_ratio: if full_area > 0 {
                visible_area as f32 / full_area as f32
            } else {
                0.0
            },
            visible_ratio_onscreen: if onscreen_area > 0 {
                visible_area as f32 / onscreen_area as f32
            } else {
                0.0
            },
            visible_mask_encoding:    "rle",
            visible_mask_rle:         rle,
            visible_mask_bbox_origin: bbox_origin,
            visible_mask_bbox_size:   bbox_size,
            occluded_by_window_ids:   occluded_by,
            is_visible,
            root_window_id,
        });
    }

    (instance_map, class_mask, result)
}

/// Resolve a window's visual parent (root). For popups/tooltips this is the
/// frontmost layer-0 NormalAppWindow owned by the same process; for normal
/// windows it is the window itself. Mirrors the label-sharing logic in
/// build_windows_label_mask so instance metadata and stable labels agree.
fn resolve_root_window_id(win: &WindowLayer, windows: &[WindowLayer]) -> Option<u32> {
    use crate::windows::model::WindowCategory;
    if !win.category.is_popup_like() {
        return Some(win.window_id);
    }
    windows.iter()
        .filter(|w| {
            w.include_in_segmentation
                && w.owner_pid == win.owner_pid
                && w.cg_layer == 0
                && matches!(w.category, WindowCategory::NormalAppWindow)
        })
        .min_by_key(|w| w.z_index)
        .map(|parent| parent.window_id)
}

fn rects_overlap(a: RectI, b: RectI) -> bool {
    a.w > 0 && a.h > 0 && b.w > 0 && b.h > 0
        && a.x < b.x + b.w
        && a.x + a.w > b.x
        && a.y < b.y + b.h
        && a.y + a.h > b.y
}

/// Custom ROW-MAJOR RLE of the binary mask `instance_map[..] == instance_id`
/// within the crop [x0,x1) × [y0,y1).
/// Alternating bg/fg run lengths starting from bg. First element is the
/// background count, which may be 0 if the first pixel is foreground.
/// NOT COCO-compatible: COCO RLE is column-major (Fortran order). Decode only
/// with a matching row-major reader.
fn encode_rle_crop(
    instance_map: &[u16],
    cap_w:        usize,
    x0: usize, y0: usize, x1: usize, y1: usize,
    instance_id:  u16,
) -> Vec<u32> {
    let mut runs  = Vec::new();
    let mut is_fg = false;
    let mut run   = 0u32;
    for y in y0..y1 {
        for x in x0..x1 {
            let fg = instance_map[y * cap_w + x] == instance_id;
            if fg == is_fg {
                run += 1;
            } else {
                runs.push(run);
                is_fg = fg;
                run   = 1;
            }
        }
    }
    if run > 0 { runs.push(run); }
    runs
}

// ── Label assignment ──────────────────────────────────────────────────────────

/// Assigns stable sequential labels (1..N) to window_ids across the session.
pub struct LabelAssigner {
    map:  HashMap<u32, u16>,
    next: u16,
}

impl LabelAssigner {
    pub fn new() -> Self {
        Self { map: HashMap::new(), next: 1 }
    }

    pub fn label_for(&mut self, window_id: u32) -> u16 {
        if let Some(&l) = self.map.get(&window_id) { return l; }
        // Valid labels are 1..CURSOR_LABEL. Saturate at the top rather than
        // wrapping_add into 0 (background) or CURSOR_LABEL (cursor sentinel),
        // which would silently corrupt the mask. 65k windows/session is
        // unreachable in practice; if it ever happens, reuse the last label
        // and warn instead of producing background-labeled windows.
        if self.next >= CURSOR_LABEL {
            eprintln!("[recorder] LabelAssigner exhausted u16 label space; \
                       reusing label {} (window {window_id})", CURSOR_LABEL - 1);
            return CURSOR_LABEL - 1;
        }
        let l = self.next;
        self.next += 1;
        self.map.insert(window_id, l);
        l
    }

}

// ── Mask builders ─────────────────────────────────────────────────────────────

/// Assign stable session labels to each segmentation window, in back-to-front
/// paint order. Calls the (stateful) `assigner` exactly once per window — this is
/// the only label-mask step that must run during capture, in frame order. The
/// returned label list pairs with the windows (same filter + sort) and is handed
/// to `paint_windows_label_mask` so the heavy pixel fill can run later.
///
/// Returns `(labels_in_paint_order, label_entries)`.
pub fn assign_window_labels(
    windows:  &[WindowLayer],
    assigner: &mut LabelAssigner,
) -> (Vec<u16>, Vec<LabelEntry>) {
    use crate::windows::model::WindowCategory;

    let mut wins: Vec<&WindowLayer> = windows.iter()
        .filter(|w| w.include_in_segmentation)
        .collect();
    wins.sort_by_key(|w| std::cmp::Reverse(w.z_index));

    let mut labels  = Vec::with_capacity(wins.len());
    let mut entries = Vec::with_capacity(wins.len());
    for win in &wins {
        let label = if win.category.is_popup_like() {
            // Popups share the label of the frontmost layer-0 window owned by
            // the same process. Falls back to its own label if no parent found.
            windows.iter()
                .filter(|w| {
                    w.include_in_segmentation
                        && w.owner_pid == win.owner_pid
                        && w.cg_layer == 0
                        && matches!(w.category, WindowCategory::NormalAppWindow)
                })
                .min_by_key(|w| w.z_index)
                .map(|parent| assigner.label_for(parent.window_id))
                .unwrap_or_else(|| assigner.label_for(win.window_id))
        } else {
            assigner.label_for(win.window_id)
        };
        labels.push(label);
        entries.push(LabelEntry {
            label_id:  label,
            window_id: win.window_id,
            owner_name: win.owner_name.clone(),
        });
    }
    (labels, entries)
}

/// Paint the u16 label mask at `cap_w × cap_h` from `assign_window_labels` output
/// (same window filter + back-to-front sort, so `labels` lines up by index).
/// Geometry only — no assigner — so it is safe in the post-recording pass at any
/// resolution. Frontmost window's label survives (painted last).
pub fn paint_windows_label_mask(
    windows: &[WindowLayer],
    labels:  &[u16],
    cap_w:   usize,
    cap_h:   usize,
) -> Vec<u16> {
    let mut mask = vec![0u16; cap_w * cap_h];
    let mut wins: Vec<&WindowLayer> = windows.iter()
        .filter(|w| w.include_in_segmentation)
        .collect();
    wins.sort_by_key(|w| std::cmp::Reverse(w.z_index));
    for (win, &label) in wins.iter().zip(labels) {
        fill_rect_u16(&mut mask, cap_w, cap_h, win.bounds_pixels, label);
    }
    mask
}

/// Build an 8-bit cursor mask (255 where cursor sprite is opaque; 0 elsewhere).
///
/// If `sprite_pixels` is provided (BGRA), the arrow outline is alpha-thresholded
/// and scaled into the bounding box. Falls back to a solid rectangle if no sprite.
pub fn build_cursor_mask(
    cursor_bbox:   Option<[i32; 4]>,
    sprite_pixels: Option<&[u8]>,
    sprite_w:      usize,
    sprite_h:      usize,
    cap_w:         usize,
    cap_h:         usize,
) -> Vec<u8> {
    let mut mask = vec![0u8; cap_w * cap_h];
    let (bbox, sprite) = match (cursor_bbox, sprite_pixels) {
        (Some(b), Some(p)) if b[2] > 0 && b[3] > 0 && sprite_w > 0 && sprite_h > 0 => (b, p),
        (Some(b), _) => {
            // Fallback: solid rect
            let x0 = b[0].max(0).min(cap_w as i32) as usize;
            let y0 = b[1].max(0).min(cap_h as i32) as usize;
            let x1 = (b[0] + b[2]).max(0).min(cap_w as i32) as usize;
            let y1 = (b[1] + b[3]).max(0).min(cap_h as i32) as usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    mask[y * cap_w + x] = 255;
                }
            }
            return mask;
        }
        _ => return mask,
    };

    let [cx, cy, cw, ch] = bbox;
    for dy in 0..ch as usize {
        let sy = dy * sprite_h / ch as usize;
        let my = cy + dy as i32;
        if my < 0 || my >= cap_h as i32 { continue; }
        for dx in 0..cw as usize {
            let sx = dx * sprite_w / cw as usize;
            let alpha = sprite[(sy * sprite_w + sx) * 4 + 3];
            if alpha > 64 {
                let mx = cx + dx as i32;
                if mx >= 0 && mx < cap_w as i32 {
                    mask[my as usize * cap_w + mx as usize] = 255;
                }
            }
        }
    }
    mask
}

/// Build combined u16 label mask: windows first, then cursor sprite drawn on top.
///
/// If `sprite_pixels` is provided, the cursor arrow shape is alpha-thresholded
/// into the bbox. Falls back to a solid CURSOR_LABEL rectangle if no sprite.
pub fn build_combined_label_mask(
    windows_mask:  &[u16],
    cursor_bbox:   Option<[i32; 4]>,
    sprite_pixels: Option<&[u8]>,
    sprite_w:      usize,
    sprite_h:      usize,
    cap_w:         usize,
    cap_h:         usize,
) -> Vec<u16> {
    let mut mask = windows_mask.to_vec();
    let (bbox, sprite) = match (cursor_bbox, sprite_pixels) {
        (Some(b), Some(p)) if b[2] > 0 && b[3] > 0 && sprite_w > 0 && sprite_h > 0 => (b, p),
        (Some(b), _) => {
            // Fallback: solid rect of CURSOR_LABEL
            let x0 = b[0].max(0).min(cap_w as i32) as usize;
            let y0 = b[1].max(0).min(cap_h as i32) as usize;
            let x1 = (b[0] + b[2]).max(0).min(cap_w as i32) as usize;
            let y1 = (b[1] + b[3]).max(0).min(cap_h as i32) as usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    mask[y * cap_w + x] = CURSOR_LABEL;
                }
            }
            return mask;
        }
        _ => return mask,
    };

    let [cx, cy, cw, ch] = bbox;
    for dy in 0..ch as usize {
        let sy = dy * sprite_h / ch as usize;
        let my = cy + dy as i32;
        if my < 0 || my >= cap_h as i32 { continue; }
        for dx in 0..cw as usize {
            let sx = dx * sprite_w / cw as usize;
            let alpha = sprite[(sy * sprite_w + sx) * 4 + 3];
            if alpha > 64 {
                let mx = cx + dx as i32;
                if mx >= 0 && mx < cap_w as i32 {
                    mask[my as usize * cap_w + mx as usize] = CURSOR_LABEL;
                }
            }
        }
    }
    mask
}

fn fill_rect_u16(
    mask:   &mut [u16],
    cap_w:  usize,
    cap_h:  usize,
    rect:   crate::windows::model::RectI,
    label:  u16,
) {
    if rect.w <= 0 || rect.h <= 0 { return; }
    let x0 = rect.x.max(0).min(cap_w as i32) as usize;
    let y0 = rect.y.max(0).min(cap_h as i32) as usize;
    let x1 = (rect.x + rect.w).max(0).min(cap_w as i32) as usize;
    let y1 = (rect.y + rect.h).max(0).min(cap_h as i32) as usize;
    for y in y0..y1 {
        let base = y * cap_w + x0;
        mask[base..base + (x1 - x0)].fill(label);
    }
}

// ── Label-mask downsampling ───────────────────────────────────────────────────

// ── PNG writers ───────────────────────────────────────────────────────────────

/// Write a JPEG from a BGRA capture buffer (handles stride padding). Quality 85.
pub fn write_frame_jpeg(
    path:          &Path,
    pixels:        &[u8],
    width:         usize,
    height:        usize,
    bytes_per_row: usize,
) -> std::io::Result<()> {
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        let rs = y * bytes_per_row;
        for x in 0..width {
            let i = rs + x * 4;
            rgb.push(pixels[i + 2]); // R
            rgb.push(pixels[i + 1]); // G
            rgb.push(pixels[i]);     // B
        }
    }
    let file = std::fs::File::create(path)?;
    let w = BufWriter::new(file);
    image::codecs::jpeg::JpegEncoder::new_with_quality(w, 85)
        .encode(&rgb, width as u32, height as u32, image::ExtendedColorType::Rgb8)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

/// Write a 16-bit grayscale PNG from a u16 label mask (big-endian).
pub fn write_label_png(path: &Path, mask: &[u16], width: usize, height: usize) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let w = BufWriter::new(file);
    let mut enc = png::Encoder::new(w, width as u32, height as u32);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Sixteen);
    let mut pw = enc.write_header()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    let bytes: Vec<u8> = mask.iter()
        .flat_map(|&v| [(v >> 8) as u8, (v & 0xFF) as u8])
        .collect();
    pw.write_image_data(&bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

/// Write an 8-bit grayscale PNG for the semantic class mask (class_label::* values).
pub fn write_class_mask_png(
    path:   &Path,
    mask:   &[u8],
    width:  usize,
    height: usize,
) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let w = BufWriter::new(file);
    let mut enc = png::Encoder::new(w, width as u32, height as u32);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    let mut pw = enc.write_header()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    pw.write_image_data(mask)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

/// Write per-frame windows.json — pretty-printed array of VisibleWindowMask.
pub fn write_windows_json(path: &Path, masks: &[VisibleWindowMask]) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(masks)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(path, json.as_bytes())
}

/// Write an 8-bit grayscale PNG from a u8 cursor mask.
pub fn write_cursor_png(path: &Path, mask: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let w = BufWriter::new(file);
    let mut enc = png::Encoder::new(w, width as u32, height as u32);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    let mut pw = enc.write_header()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    pw.write_image_data(mask)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode the custom row-major RLE back into a flat fg/bg bitmap of size w*h.
    fn decode_rle_row_major(runs: &[u32], w: usize, h: usize) -> Vec<bool> {
        let mut out = Vec::with_capacity(w * h);
        let mut is_fg = false;
        for &run in runs {
            for _ in 0..run { out.push(is_fg); }
            is_fg = !is_fg;
        }
        assert_eq!(out.len(), w * h, "RLE total run length must equal crop area");
        out
    }

    #[test]
    fn rle_roundtrips_row_major() {
        // 4x3 canvas; instance_id=1 occupies an L-shape. cap_w == crop width here.
        let w = 4;
        let h = 3;
        let id = 1u16;
        // row0: . X X .   row1: . X . .   row2: X X X .
        let map: Vec<u16> = vec![
            0, 1, 1, 0,
            0, 1, 0, 0,
            1, 1, 1, 0,
        ];
        let runs = encode_rle_crop(&map, w, 0, 0, w, h, id);
        // First run is the leading background count (1 here, not 0).
        assert_eq!(runs[0], 1, "must start with a bg run (COCO-style start-from-bg)");
        let decoded = decode_rle_row_major(&runs, w, h);
        let expected: Vec<bool> = map.iter().map(|&v| v == id).collect();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn rle_leading_foreground_emits_zero_bg_run() {
        // First pixel is foreground → first element must be 0.
        let w = 2;
        let h = 1;
        let map: Vec<u16> = vec![1, 0];
        let runs = encode_rle_crop(&map, w, 0, 0, w, h, 1);
        assert_eq!(runs[0], 0, "leading fg pixel must produce a 0-length bg run");
        assert_eq!(decode_rle_row_major(&runs, w, h), vec![true, false]);
    }

    #[test]
    fn label_assigner_skips_zero_and_cursor_sentinel() {
        let mut a = LabelAssigner::new();
        let l1 = a.label_for(100);
        let l2 = a.label_for(200);
        assert_eq!(l1, 1);
        assert_eq!(l2, 2);
        assert_eq!(a.label_for(100), 1, "stable per window_id");
        assert_ne!(l1, 0);
        assert_ne!(l1, CURSOR_LABEL);
    }
}
