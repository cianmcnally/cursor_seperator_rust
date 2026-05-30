use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::windows::model::RectI;
use crate::windows::WindowLayer;

use super::diff::{
    Blob, absdiff, adaptive_threshold, binarize, extract_gray_roi,
    find_blobs, horiz_close, mask_cursor,
};
use super::ring::{FrameRingBuffer, RingEntry};
use super::tap::{TapEvent, start_key_tap};

// ── Public types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct InteractionRegion {
    pub bbox:         RectI,
    pub confidence:   f32,
    pub source:       String,
    pub timestamp_ns: u64,
    pub window_id:    Option<u32>,
    pub z_index:      Option<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct TypingDetectorResult {
    pub region:    Option<InteractionRegion>,
    /// ROI searched (in capture-pixel coords).
    pub roi:       Option<RectI>,
    /// Grayscale absdiff within ROI, plus its position.
    pub diff_gray: Option<(Vec<u8>, RectI)>,
}

pub struct SharedTypingState(Mutex<TypingDetectorResult>);

impl SharedTypingState {
    fn new() -> Arc<Self> {
        Arc::new(Self(Mutex::new(TypingDetectorResult::default())))
    }

    pub fn read(&self) -> TypingDetectorResult {
        self.0.lock().unwrap().clone()
    }

    fn write(&self, r: TypingDetectorResult) {
        *self.0.lock().unwrap() = r;
    }
}

pub struct TypingArgs {
    /// Capture the raw diff buffer for the debug overlay.
    pub show_diff:    bool,
    /// Backing scale factor (backingScaleFactor), converts CG-point mouse
    /// coords to capture-pixel coords.
    pub scale:        f64,
    /// Capture origin in pixels (0 for full-display).
    pub cap_origin_x: f64,
    pub cap_origin_y: f64,
}

pub fn start_typing_detector(
    ring: Arc<FrameRingBuffer>,
    args: TypingArgs,
) -> Arc<SharedTypingState> {
    let state  = SharedTypingState::new();
    let state2 = state.clone();
    let (tx, rx) = std::sync::mpsc::channel::<TapEvent>();
    start_key_tap(tx);
    std::thread::spawn(move || run_detector(ring, rx, state2, args));
    state
}

// ── Detector loop ─────────────────────────────────────────────────────────────

struct DetState {
    last_typing_area: Option<RectI>,
    last_click_px:    Option<(f64, f64)>,
    // Reusable image buffers — resized as needed, no per-keypress alloc
    gray_bef: Vec<u8>,
    gray_aft: Vec<u8>,
    diff:     Vec<u8>,
    binary:   Vec<u8>,
    tmp:      Vec<u8>,
}

impl DetState {
    fn new() -> Self {
        Self {
            last_typing_area: None,
            last_click_px: None,
            gray_bef: Vec::new(),
            gray_aft: Vec::new(),
            diff:     Vec::new(),
            binary:   Vec::new(),
            tmp:      Vec::new(),
        }
    }
}

fn run_detector(
    ring:  Arc<FrameRingBuffer>,
    rx:    std::sync::mpsc::Receiver<TapEvent>,
    state: Arc<SharedTypingState>,
    args:  TypingArgs,
) {
    let mut ds          = DetState::new();
    let mut pending_key: Option<Instant> = None;

    loop {
        // Drain all pending events (keep latest key, track last click)
        loop {
            match rx.try_recv() {
                Ok(TapEvent::KeyDown { at }) => pending_key = Some(at),
                Ok(TapEvent::MouseDown { x_pts, y_pts, .. }) => {
                    let x_px = x_pts * args.scale - args.cap_origin_x;
                    let y_px = y_pts * args.scale - args.cap_origin_y;
                    // Click far from cached typing area → clear box from overlay
                    // and invalidate cache so next keyDown uses click-based ROI.
                    if let Some(ta) = ds.last_typing_area {
                        let cx = (ta.x + ta.w / 2) as f64;
                        let cy = (ta.y + ta.h / 2) as f64;
                        let dx = x_px - cx;
                        let dy = y_px - cy;
                        if dx * dx + dy * dy > 250.0 * 250.0 {
                            ds.last_typing_area = None;
                            state.write(TypingDetectorResult::default());
                        }
                    }
                    ds.last_click_px = Some((x_px, y_px));
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(_) => return,
            }
        }

        if let Some(key_at) = pending_key {
            let deadline = key_at + Duration::from_millis(200);

            if Instant::now() > deadline {
                pending_key = None;
                continue;
            }

            let before = ring.latest_before(key_at);
            let after  = ring.earliest_after(key_at);

            if let (Some(bef), Some(aft)) = (before, after) {
                pending_key = None;
                let result = detect(&mut ds, &bef, &aft, &args);

                if let Some(ref reg) = result.region {
                    // JSON to stdout
                    println!("{{\"timestamp_ns\":{},\"interaction_regions\":[{{\"kind\":\"TypingArea\",\"bbox\":[{},{},{},{}],\"confidence\":{:.3},\"source\":\"{}\",\"window_id\":{},\"z_index\":{}}}]}}",
                        reg.timestamp_ns,
                        reg.bbox.x, reg.bbox.y, reg.bbox.w, reg.bbox.h,
                        reg.confidence,
                        reg.source,
                        reg.window_id.map_or("null".to_string(), |v| v.to_string()),
                        reg.z_index.map_or("null".to_string(), |v| v.to_string()),
                    );
                    ds.last_typing_area = Some(reg.bbox);
                }

                state.write(result);
            }
        }

        std::thread::sleep(Duration::from_millis(1));
    }
}

// ── Detection pipeline ───────────────────────────────────────────────────────

fn detect(
    ds:  &mut DetState,
    bef: &RingEntry,
    aft: &RingEntry,
    args: &TypingArgs,
) -> TypingDetectorResult {
    let fw = bef.width  as i32;
    let fh = bef.height as i32;

    let roi_unclamped = choose_roi(ds, bef, fw, fh);
    let roi = match roi_unclamped.clip(fw, fh) {
        Some(r) => r,
        None => return TypingDetectorResult { region: None, roi: Some(roi_unclamped), diff_gray: None },
    };
    let roi_w = roi.w as usize;
    let roi_h = roi.h as usize;

    // Extract grayscale ROIs
    extract_gray_roi(&bef.pixels, bef.bytes_per_row, roi, &mut ds.gray_bef);
    extract_gray_roi(&aft.pixels, aft.bytes_per_row, roi, &mut ds.gray_aft);

    // Absdiff
    absdiff(&ds.gray_bef, &ds.gray_aft, &mut ds.diff);

    // Mask cursor positions (both before and after frame)
    if let Some(cr) = bef.cursor_rect { mask_cursor(&mut ds.diff, roi_w, roi, expanded(cr, 8)); }
    if let Some(cr) = aft.cursor_rect { mask_cursor(&mut ds.diff, roi_w, roi, expanded(cr, 8)); }

    let diff_gray = if args.show_diff {
        Some((ds.diff.clone(), roi))
    } else {
        None
    };

    // Adaptive threshold + binarize
    let (thresh, _mean, _std) = adaptive_threshold(&ds.diff);
    binarize(&ds.diff, thresh, &mut ds.binary);

    // Horizontal closing: bridges char gaps up to ~6px
    horiz_close(&mut ds.binary, &mut ds.tmp, roi_w, roi_h, 3);

    // Connected components
    let blobs = find_blobs(&ds.binary, roi_w, roi_h);

    let roi_area_px = roi.w * roi.h;
    let mut has_caret  = false;
    let mut huge_diff  = false;

    let kept: Vec<&Blob> = blobs.iter().filter(|b| {
        let frac = b.pixel_count as f32 / roi_area_px as f32;
        if frac > 0.20 { huge_diff = true; return false; }
        if b.is_caret_like() { has_caret = true; return true; }
        b.is_text_like(roi.w, roi.h)
    }).collect();

    if kept.is_empty() {
        // Return cached area with reduced confidence
        if let Some(ta) = ds.last_typing_area {
            let (wid, zidx) = window_for_rect(&aft.windows, ta);
            let conf = (0.35f32 + 0.10 - 0.20).clamp(0.0, 1.0);
            return TypingDetectorResult {
                region: Some(InteractionRegion {
                    bbox: ta,
                    confidence: conf,
                    source: "active".into(),
                    timestamp_ns: now_ns(),
                    window_id: wid,
                    z_index: zidx,
                }),
                roi: Some(roi),
                diff_gray,
            };
        }
        return TypingDetectorResult { region: None, roi: Some(roi), diff_gray };
    }

    // Group blobs into text-line bands, pick largest
    let bbox_roi = group_band(&kept);

    // Translate ROI-space bbox to frame-pixel space + padding
    let pad_x = 20i32;
    let pad_y = 10i32;
    let bbox = RectI {
        x: roi.x + bbox_roi.x0 - pad_x,
        y: roi.y + bbox_roi.y0 - pad_y,
        w: (bbox_roi.x1 - bbox_roi.x0 + 1 + pad_x * 2).max(1),
        h: (bbox_roi.y1 - bbox_roi.y0 + 1 + pad_y * 2).max(1),
    };

    // Confidence
    let mut conf = 0.35f32;   // keyDown triggered
    conf += 0.25;             // localized diff found
    if has_caret { conf += 0.20; }
    if huge_diff { conf -= 0.30; }
    if !is_text_like_geometry(bbox) { conf -= 0.20; }

    let (wid, zidx) = window_for_rect(&aft.windows, bbox);
    if zidx.map_or(false, |z| z == 0) { conf += 0.10; }
    if near_cache_or_click(ds, bbox)   { conf += 0.10; }

    let conf = conf.clamp(0.0, 1.0);

    TypingDetectorResult {
        region: Some(InteractionRegion {
            bbox,
            confidence: conf,
            source: "typing".into(),
            timestamp_ns: now_ns(),
            window_id: wid,
            z_index: zidx,
        }),
        roi: Some(roi),
        diff_gray,
    }
}

// ── ROI selection ─────────────────────────────────────────────────────────────

fn choose_roi(ds: &DetState, bef: &RingEntry, fw: i32, fh: i32) -> RectI {
    // Priority 1: last known typing area + 80px
    if let Some(ta) = ds.last_typing_area {
        return expanded(ta, 80);
    }

    // Priority 3: 700×260 around last click
    if let Some((cx, cy)) = ds.last_click_px {
        return RectI { x: cx as i32 - 350, y: cy as i32 - 130, w: 700, h: 260 };
    }

    // Priority 4: frontmost segmentation window
    if let Some(w) = bef.windows.iter().find(|w| w.include_in_segmentation) {
        return w.bounds_pixels;
    }

    // Priority 5: window containing current cursor position
    if let Some(cr) = bef.cursor_rect {
        let mid = (cr.x + cr.w / 2, cr.y + cr.h / 2);
        for win in &bef.windows {
            let b = win.bounds_pixels;
            if mid.0 >= b.x && mid.0 < b.x + b.w && mid.1 >= b.y && mid.1 < b.y + b.h {
                return b;
            }
        }
    }

    RectI { x: 0, y: 0, w: fw, h: fh }
}

// ── Band grouping ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Band { x0: i32, y0: i32, x1: i32, y1: i32 }

fn group_band(blobs: &[&Blob]) -> Band {
    let merge_gap = 16i32;
    let mut bands: Vec<Band> = Vec::new();

    let mut sorted: Vec<&&Blob> = blobs.iter().collect();
    sorted.sort_by_key(|b| b.y0);

    for blob in &sorted {
        let mut merged = false;
        for band in &mut bands {
            if blob.y0 <= band.y1 + merge_gap && blob.y1 >= band.y0 - merge_gap {
                band.x0 = band.x0.min(blob.x0);
                band.y0 = band.y0.min(blob.y0);
                band.x1 = band.x1.max(blob.x1);
                band.y1 = band.y1.max(blob.y1);
                merged = true;
                break;
            }
        }
        if !merged {
            bands.push(Band { x0: blob.x0, y0: blob.y0, x1: blob.x1, y1: blob.y1 });
        }
    }

    // Largest by y-extent
    bands.into_iter()
        .max_by_key(|b| b.y1 - b.y0)
        .unwrap_or(Band { x0: 0, y0: 0, x1: 0, y1: 0 })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn expanded(r: RectI, pad: i32) -> RectI {
    RectI { x: r.x - pad, y: r.y - pad, w: r.w + pad * 2, h: r.h + pad * 2 }
}

fn window_for_rect(windows: &[WindowLayer], rect: RectI) -> (Option<u32>, Option<usize>) {
    let cx = rect.x + rect.w / 2;
    let cy = rect.y + rect.h / 2;
    for win in windows {
        let b = win.bounds_pixels;
        if cx >= b.x && cx < b.x + b.w && cy >= b.y && cy < b.y + b.h {
            return (Some(win.window_id), Some(win.z_index));
        }
    }
    (None, None)
}

fn is_text_like_geometry(r: RectI) -> bool {
    // Text areas are wider than they are tall, and not a single pixel
    r.w > 10 && r.h > 5 && (r.w as f32 / r.h as f32) > 0.5
}

fn near_cache_or_click(ds: &DetState, bbox: RectI) -> bool {
    let cx = bbox.x + bbox.w / 2;
    let cy = bbox.y + bbox.h / 2;

    if let Some(ta) = ds.last_typing_area {
        let tx = ta.x + ta.w / 2;
        let ty = ta.y + ta.h / 2;
        if (cx - tx).pow(2) + (cy - ty).pow(2) < 300 * 300 { return true; }
    }
    if let Some((lx, ly)) = ds.last_click_px {
        let dx = cx as f64 - lx;
        let dy = cy as f64 - ly;
        if dx * dx + dy * dy < 200.0 * 200.0 { return true; }
    }
    false
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}
