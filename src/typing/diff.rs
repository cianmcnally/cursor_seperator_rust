use crate::windows::model::RectI;

// ── Grayscale extraction ────────────────────────────────────────────────────

/// Extract a ROI from a BGRA frame into a flat grayscale buffer.
/// BGRA layout: [B=0, G=1, R=2, A=3].
pub fn extract_gray_roi(
    pixels: &[u8],
    bpr:    usize,
    roi:    RectI,
    out:    &mut Vec<u8>,
) {
    let w = roi.w as usize;
    let h = roi.h as usize;
    out.resize(w * h, 0);
    for row in 0..h {
        let src_y   = (roi.y as usize) + row;
        let src_off = src_y * bpr + (roi.x as usize) * 4;
        let dst_off = row * w;
        for col in 0..w {
            let si = src_off + col * 4;
            if si + 2 < pixels.len() {
                let b = pixels[si]     as u32;
                let g = pixels[si + 1] as u32;
                let r = pixels[si + 2] as u32;
                out[dst_off + col] = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
            }
        }
    }
}

// ── Absolute difference ─────────────────────────────────────────────────────

pub fn absdiff(a: &[u8], b: &[u8], out: &mut Vec<u8>) {
    let len = a.len().min(b.len());
    out.resize(len, 0);
    for i in 0..len {
        out[i] = a[i].abs_diff(b[i]);
    }
}

// ── Cursor masking ──────────────────────────────────────────────────────────

/// Zero out pixels in ROI-space buffer that overlap with `cursor` (frame-pixel coords).
pub fn mask_cursor(buf: &mut [u8], roi_w: usize, roi: RectI, cursor: RectI) {
    let cx0 = ((cursor.x - roi.x).max(0) as usize).min(roi.w as usize);
    let cy0 = ((cursor.y - roi.y).max(0) as usize).min(roi.h as usize);
    let cx1 = ((cursor.x + cursor.w - roi.x).max(0) as usize).min(roi.w as usize);
    let cy1 = ((cursor.y + cursor.h - roi.y).max(0) as usize).min(roi.h as usize);
    for row in cy0..cy1 {
        for col in cx0..cx1 {
            buf[row * roi_w + col] = 0;
        }
    }
}

// ── Adaptive threshold ──────────────────────────────────────────────────────

/// (threshold, mean, std) from diff buffer. threshold = max(12, mean + 3*std).
pub fn adaptive_threshold(diff: &[u8]) -> (u8, f32, f32) {
    if diff.is_empty() { return (12, 0.0, 0.0); }
    let n   = diff.len() as f64;
    let sum = diff.iter().map(|&v| v as u64).sum::<u64>();
    let mean = sum as f64 / n;
    let var  = diff.iter()
        .map(|&v| { let d = v as f64 - mean; d * d })
        .sum::<f64>() / n;
    let std  = var.sqrt();
    let thresh = ((mean + 3.0 * std).round() as u8).max(12);
    (thresh, mean as f32, std as f32)
}

/// pixels >= threshold → 255, else → 0.
pub fn binarize(diff: &[u8], threshold: u8, out: &mut Vec<u8>) {
    out.resize(diff.len(), 0);
    for i in 0..diff.len() {
        out[i] = if diff[i] >= threshold { 255 } else { 0 };
    }
}

// ── Morphology ──────────────────────────────────────────────────────────────

/// Horizontal binary closing: dilate then erode (bridges gaps up to `radius` px).
pub fn horiz_close(binary: &mut Vec<u8>, tmp: &mut Vec<u8>, w: usize, h: usize, radius: usize) {
    tmp.resize(binary.len(), 0);
    // Dilate horizontally
    for row in 0..h {
        let base = row * w;
        for col in 0..w {
            let lo = col.saturating_sub(radius);
            let hi = (col + radius + 1).min(w);
            tmp[base + col] = if binary[base + lo..base + hi].iter().any(|&v| v > 0) { 255 } else { 0 };
        }
    }
    // Erode back
    for row in 0..h {
        let base = row * w;
        for col in 0..w {
            let lo = col.saturating_sub(radius);
            let hi = (col + radius + 1).min(w);
            binary[base + col] = if tmp[base + lo..base + hi].iter().all(|&v| v > 0) { 255 } else { 0 };
        }
    }
}

// ── Connected components ────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Blob {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
    pub pixel_count: u32,
}

impl Blob {
    pub fn width(&self)  -> i32 { (self.x1 - self.x0 + 1).max(1) }
    pub fn height(&self) -> i32 { (self.y1 - self.y0 + 1).max(1) }

    pub fn is_caret_like(&self) -> bool {
        self.width() <= 4 && self.height() >= 10
    }

    pub fn is_text_like(&self, roi_w: i32, roi_h: i32) -> bool {
        let area_frac = self.pixel_count as f32 / (roi_w * roi_h).max(1) as f32;
        if area_frac > 0.20 { return false; }
        let w = self.width();
        let h = self.height();
        if self.is_caret_like() { return true; }
        // Reject full-width bands (scrollbar redraw, window chrome)
        if w > roi_w * 3 / 4 { return false; }
        // Reject tiny specks (< 2x2 px)
        if w < 2 && h < 2 { return false; }
        // Text glyphs: height in reasonable range, not absurdly tall
        h >= 3 && h <= 120
    }
}

fn find_root(parent: &mut Vec<u32>, mut x: u32) -> u32 {
    // Iterative path compression (avoids stack overflow on long runs)
    let mut root = x;
    while parent[root as usize] != root {
        root = parent[root as usize];
    }
    while parent[x as usize] != root {
        let next = parent[x as usize];
        parent[x as usize] = root;
        x = next;
    }
    root
}

pub fn find_blobs(binary: &[u8], width: usize, height: usize) -> Vec<Blob> {
    if width == 0 || height == 0 { return Vec::new(); }
    let len = width * height;
    let mut labels = vec![0u32; len];
    // parent[0] = background sentinel; labels start at 1
    let mut parent: Vec<u32> = vec![0u32];
    let mut next_label = 1u32;

    // First pass: label + union
    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            if binary[idx] == 0 { continue; }

            let above = if row > 0 && binary[(row - 1) * width + col] > 0 {
                labels[(row - 1) * width + col]
            } else { 0 };
            let left = if col > 0 && binary[row * width + col - 1] > 0 {
                labels[row * width + col - 1]
            } else { 0 };

            labels[idx] = match (above, left) {
                (0, 0) => {
                    let l = next_label;
                    next_label += 1;
                    parent.push(l);
                    l
                }
                (a, 0) => a,
                (0, b) => b,
                (a, b) if a == b => a,
                (a, b) => {
                    let ra = find_root(&mut parent, a);
                    let rb = find_root(&mut parent, b);
                    if ra != rb { parent[ra as usize] = rb; }
                    rb
                }
            };
        }
    }

    // Second pass: collect bounding boxes
    let mut blobs: std::collections::HashMap<u32, Blob> = std::collections::HashMap::new();
    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            if binary[idx] == 0 { continue; }
            let root = find_root(&mut parent, labels[idx]);
            let c = col as i32;
            let r = row as i32;
            let e = blobs.entry(root).or_insert(Blob { x0: c, y0: r, x1: c, y1: r, pixel_count: 0 });
            if c < e.x0 { e.x0 = c; }
            if c > e.x1 { e.x1 = c; }
            if r < e.y0 { e.y0 = r; }
            if r > e.y1 { e.y1 = r; }
            e.pixel_count += 1;
        }
    }

    blobs.into_values().collect()
}
