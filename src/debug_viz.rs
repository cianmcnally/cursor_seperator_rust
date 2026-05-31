use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// All imagery streamed to Rerun is downsampled so its longest side is ≤ this.
/// Native capture can be 2.5–4K; logging that every frame (frame + N masks) buries
/// the viewer in RAM. The dataset PNGs on disk stay full-resolution — this only
/// affects the visualizer.
const VIZ_MAX_DIM: u32 = 1280;

/// Memory budget for the spawned live viewer. It drops old data past this instead
/// of growing to the default 75%-of-RAM (which was hitting 18 GiB).
const VIZ_VIEWER_MEMORY_LIMIT: &str = "2GB";

use crate::recorder::masks::VisibleWindowMask;
use crate::recorder::record::{EventRecord, FrameRecord, FocusEventRecord, KeyEventRecord};

// ── DebugViz — Rerun recording-time visualizer ───────────────────────────────
///
/// Logs structured data to a Rerun .rrd file during live recording.
/// Namespace layout:
///
///   /recorded/frame           — RGB image of captured frame
///   /recorded/cursor          — cursor position point
///   /recorded/cursor_box      — cursor bounding box
///   /recorded/window_boxes    — window rectangles
///   /recorded/focused_window  — focused window highlight
///   /recorded/events          — event markers on timeline
///   /recorded/masks           — label masks (if saved)
///
///   /checks/frame_hash_match  — 1 if frame changed, 0 if frozen
///   /checks/event_count       — cumulative event count
///   /checks/window_count      — number of segmented windows
///   /checks/frame_interval_ms — time since previous frame
///   /checks/dropped_frames    — detected frame drops
///   /checks/writer_queue_depth— approximate writer backlog
///   /checks/warnings          — text warnings

/// Where the Rerun stream goes.
pub enum VizSink {
    /// Spawn a live Rerun viewer and stream to it (ephemeral — nothing on disk).
    Spawn,
    /// Write a `.rrd` file for later `rerun <file>`.
    Save(PathBuf),
}

pub struct DebugViz {
    rec: rerun::RecordingStream,
    last_frame: Mutex<Option<(u64, u64)>>,       // (frame_idx, capture_time_ns)
    last_frame_hash: Mutex<Option<u64>>,           // previous frame's FNV-1a hash
    last_window_labels: Mutex<HashMap<u16, String>>, // instance_id → last logged label
    event_count: Mutex<u64>,
    /// Per-kind cumulative event counts → /checks/counts/<kind> (replaces the HUD).
    event_kind_counts: Mutex<HashMap<String, u64>>,
    /// Downsample factor applied to the most recent frame — so between-frame event
    /// markers scale their coordinates to match the downsampled imagery.
    last_scale: AtomicUsize,
}

impl DebugViz {
    pub fn new(
        session_id: &str,
        sink: VizSink,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let builder = rerun::RecordingStreamBuilder::new(session_id.to_string());
        let rec = match sink {
            VizSink::Spawn => {
                let opts = rerun::SpawnOptions {
                    memory_limit: VIZ_VIEWER_MEMORY_LIMIT.to_string(),
                    ..Default::default()
                };
                builder.spawn_opts(&opts)?
            }
            VizSink::Save(path) => builder.save(path)?,
        };

        Ok(Self {
            rec,
            last_frame: Mutex::new(None),
            last_frame_hash: Mutex::new(None),
            last_window_labels: Mutex::new(HashMap::new()),
            event_count: Mutex::new(0),
            event_kind_counts: Mutex::new(HashMap::new()),
            last_scale: AtomicUsize::new(1),
        })
    }

    // ── Frame logging ─────────────────────────────────────────────────────

    pub fn log_frame(
        &self,
        frame_idx: u64,
        capture_time_ns: u64,
        frame_rec: &FrameRecord,
        pixels_bgra: &[u8],
        width: u32,
        height: u32,
        // Log the (heavy) RGB image this frame. When recording, callers pass the
        // sampled cadence so the .rrd doesn't re-store every captured frame.
        log_image: bool,
    ) {
        self.set_time(frame_idx, capture_time_ns);

        let s = viz_scale(width, height);
        self.last_scale.store(s, Ordering::Relaxed);

        // Integrity checks
        self.log_integrity(frame_idx, capture_time_ns, pixels_bgra, width, height, frame_rec);

        // Frame image (gated + downsampled for the viewer)
        if log_image {
            if pixels_bgra.len() == (width as usize * height as usize * 4) {
                let rgb = bgra_to_rgb(pixels_bgra);
                let (rgb, dw, dh) = downsample_rgb(&rgb, width as usize, height as usize, s);
                let _ = self.rec.log(
                    "/recorded/frame",
                    &rerun::Image::from_rgb24(rgb, [dw, dh]),
                );
            } else {
                self.log_warning(
                    frame_idx,
                    capture_time_ns,
                    &format!(
                        "bad pixel buffer len: got {}, expected {}",
                        pixels_bgra.len(),
                        width as usize * height as usize * 4
                    ),
                );
            }
        }

        self.log_windows(frame_rec, s);
        self.log_cursor(frame_rec, s);
        self.log_focused_window(frame_idx, capture_time_ns, frame_rec, s);
    }

    // ── Event markers ─────────────────────────────────────────────────────

    /// Log a cursor action event marker on the timeline.
    pub fn log_event(
        &self,
        frame_idx: u64,
        event_time_ns: u64,
        event: &EventRecord,
    ) {
        self.set_time(frame_idx, event_time_ns);

        let s = self.last_scale.load(Ordering::Relaxed).max(1) as f32;
        let x = event.position_px[0] as f32 / s;
        let y = event.position_px[1] as f32 / s;

        // event.kind is already the snake_case 7-class label (click/double_click/drag/scroll).
        let marker_kind = event.kind.as_str();
        let label = marker_kind.to_string();

        // Log as a labeled point on the 2D canvas.
        let points = rerun::Points2D::new([(x, y)])
            .with_labels([label.clone()])
            .with_radii([8.0]);

        let _ = self.rec.log("/recorded/events", &points);

        // Also log as a text entry for the timeline view.
        let text = format!("[{marker_kind}] @ ({:.0}, {:.0})", x, y);
        let _ = self.rec.log(
            "/recorded/events",
            &rerun::TextLog::new(text),
        );

        // Track total + per-kind event counts. The per-kind scalars under
        // /checks/counts/<kind> replace the old floating NSWindow HUD.
        {
            let mut count = self.event_count.lock().unwrap();
            *count += 1;
            let _ = self.rec.log("/checks/event_count", &rerun::Scalars::new([*count as f64]));
        }
        {
            let mut counts = self.event_kind_counts.lock().unwrap();
            let c = counts.entry(label.clone()).or_insert(0);
            *c += 1;
            let _ = self.rec.log(
                format!("/checks/counts/{label}"),
                &rerun::Scalars::new([*c as f64]),
            );
        }
    }

    /// Log a focus-change event marker.
    pub fn log_focus_event(
        &self,
        frame_idx: u64,
        event_time_ns: u64,
        event: &FocusEventRecord,
    ) {
        self.set_time(frame_idx, event_time_ns);

        let label = format!(
            "focus_changed: {:?} → {:?}",
            event.from_window, event.to_window
        );

        let _ = self.rec.log(
            "/recorded/events",
            &rerun::TextLog::new(label.as_str()),
        );
    }

    /// Log a key-down event marker.
    pub fn log_key_event(
        &self,
        frame_idx: u64,
        event_time_ns: u64,
        event: &KeyEventRecord,
    ) {
        self.set_time(frame_idx, event_time_ns);

        let label = format!("typing: key_code={}", event.key_code);

        let _ = self.rec.log(
            "/recorded/events",
            &rerun::TextLog::new(label.as_str()),
        );
    }

    /// Log a window created/destroyed marker.
    pub fn log_window_lifecycle(
        &self,
        frame_idx: u64,
        event_time_ns: u64,
        kind: &str,        // "window_created" or "window_destroyed"
        window_id: u32,
        owner_name: &str,
    ) {
        self.set_time(frame_idx, event_time_ns);

        let label = format!("{kind}: id={window_id} owner={owner_name}");

        let _ = self.rec.log(
            "/recorded/events",
            &rerun::TextLog::new(label.as_str()),
        );
    }

    // ── Warnings ──────────────────────────────────────────────────────────

    pub fn log_warning(&self, frame_idx: u64, capture_time_ns: u64, msg: &str) {
        self.set_time(frame_idx, capture_time_ns);

        let _ = self.rec.log(
            "/checks/warnings",
            &rerun::TextLog::new(msg),
        );
    }

    // ── Segmentation visualization ────────────────────────────────────────

    /// Log instance map, class mask, per-window geometry, and cursor mask.
    ///
    /// AnnotationContext logged static → explicit hashed colors, no pink/pastel bleed.
    /// Per-window: opacity 1.0, draw_order 1.0 (class-0 fully transparent).
    /// Cursor: bright cyan via AnnotationContext, draw_order 100.0 (always on top).
    pub fn log_segmentation(
        &self,
        frame_idx:        u64,
        capture_time_ns:  u64,
        instance_map:     &[u16],
        class_mask:       &[u8],
        window_masks:     &[VisibleWindowMask],
        cursor_mask:      &[u8],
        width:            u32,
        height:           u32,
    ) {
        self.set_time(frame_idx, capture_time_ns);

        // Everything below is downsampled by `s` (and box coords divided by `s`)
        // so native-res capture doesn't bury the viewer in RAM.
        let s  = viz_scale(width, height);
        let sf = s as f32;
        let w  = width as usize;
        let dw = (w / s).max(1);
        let dh = (height as usize / s).max(1);
        let seg_fmt = rerun::ImageFormat::segmentation([dw as u32, dh as u32], rerun::ChannelDatatype::U8);

        // ── Annotation contexts (static): explicit hashed colours per window id ──
        let win_anns: Vec<(u16, String, rerun::Rgba32)> = window_masks.iter()
            .map(|wm| {
                let [r, g, b] = instance_id_color(wm.instance_id as u32);
                (wm.instance_id, format!("{} id={}", wm.app_name, wm.window_id), rerun::Rgba32::from_rgb(r, g, b))
            })
            .collect();
        let _ = self.rec.log_static(
            "/windows",
            &rerun::AnnotationContext::new(
                win_anns.iter().map(|(id, label, color)| (*id, label.as_str(), *color))
            ),
        );
        let _ = self.rec.log_static(
            "/cursor",
            &rerun::AnnotationContext::new([(255u16, "cursor", rerun::Rgba32::from_rgb(0, 220, 255))]),
        );

        // ── Combined masks (occlusion-correct), nearest-downsampled ────────────
        let mut seg_combined = vec![0u8; dw * dh];
        let mut class_ds     = vec![0u8; dw * dh];
        for dy in 0..dh {
            let sy = dy * s;
            for dx in 0..dw {
                let si = sy * w + dx * s;
                seg_combined[dy * dw + dx] = instance_map[si].min(255) as u8;
                class_ds[dy * dw + dx]     = class_mask[si];
            }
        }
        let _ = self.rec.log(
            "/frame/visible_instances",
            &rerun::SegmentationImage::new(seg_combined, seg_fmt.clone()).with_opacity(0.7_f32),
        );
        let _ = self.rec.log(
            "/frame/visible_classes",
            &rerun::SegmentationImage::new(class_ds, seg_fmt.clone()).with_opacity(0.7_f32),
        );

        // ── Per-window registration layers (cheap boxes only) ──────────────────
        // Per-window masks are fully encoded in /frame/visible_instances above.
        // We deliberately do NOT log a full-frame SegmentationImage per window —
        // that was N full-frame images/frame, the dominant writer cost and what
        // blew up RAM/.rrd. The full_rect + visible_bbox boxes give the same
        // "did this window register" signal for a few bytes.
        for wm in window_masks {
            let [fx, fy, fw, fh] = wm.full_rect_px;
            let _ = self.rec.log(
                format!("/windows/{}/full_rect", wm.instance_id),
                &rerun::Boxes2D::from_mins_and_sizes(
                    [(fx as f32 / sf, fy as f32 / sf)],
                    [(fw as f32 / sf, fh as f32 / sf)],
                ),
            );

            if wm.is_visible {
                let [vx, vy, vw, vh] = wm.visible_bbox_px;
                let _ = self.rec.log(
                    format!("/windows/{}/visible_bbox", wm.instance_id),
                    &rerun::Boxes2D::from_mins_and_sizes(
                        [(vx as f32 / sf, vy as f32 / sf)],
                        [(vw as f32 / sf, vh as f32 / sf)],
                    ),
                );
            }

            let label = format!(
                "id={} z={} focused={} vis={:.2} {}",
                wm.window_id, wm.z_index, wm.focused, wm.visible_ratio, wm.app_name,
            );
            let mut label_cache = self.last_window_labels.lock().unwrap();
            if label_cache.get(&wm.instance_id).map(|ls| ls.as_str()) != Some(&label) {
                let _ = self.rec.log(
                    format!("/windows/{}/label", wm.instance_id),
                    &rerun::TextLog::new(label.clone()),
                );
                label_cache.insert(wm.instance_id, label);
            }
        }

        // ── Cursor mask: downsampled, draw_order 100 = on top ──────────────────
        if !cursor_mask.is_empty() {
            let mut cur = vec![0u8; dw * dh];
            for dy in 0..dh {
                let sy = dy * s;
                for dx in 0..dw { cur[dy * dw + dx] = cursor_mask[sy * w + dx * s]; }
            }
            let _ = self.rec.log(
                "/cursor/mask",
                &rerun::SegmentationImage::new(cur, seg_fmt)
                    .with_opacity(1.0_f32)
                    .with_draw_order(100.0_f32),
            );
        }
    }

    pub fn flush(&self) {
        let _ = self.rec.flush_blocking();
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    fn set_time(&self, frame_idx: u64, capture_time_ns: u64) {
        self.rec.set_time_sequence("frame_idx", frame_idx as i64);
        self.rec
            .set_timestamp_nanos_since_epoch("capture_time", capture_time_ns as i64);
    }

    fn log_integrity(
        &self,
        frame_idx: u64,
        capture_time_ns: u64,
        pixels: &[u8],
        width: u32,
        height: u32,
        frame_rec: &FrameRecord,
    ) {
        let mut last = self.last_frame.lock().unwrap();

        // ── Monotonicity checks ─────────────────────────────────────────
        if let Some((prev_idx, prev_ts)) = *last {
            if frame_idx <= prev_idx {
                self.log_warning(
                    frame_idx,
                    capture_time_ns,
                    &format!("non-monotonic frame_idx: prev={prev_idx}, now={frame_idx}"),
                );
            }

            if capture_time_ns <= prev_ts {
                self.log_warning(
                    frame_idx,
                    capture_time_ns,
                    &format!("non-monotonic capture_time_ns: prev={prev_ts}, now={capture_time_ns}"),
                );
            }

            let frame_interval_ms =
                (capture_time_ns.saturating_sub(prev_ts)) as f64 / 1_000_000.0;

            let _ = self.rec.log(
                "/checks/frame_interval_ms",
                &rerun::Scalars::new([frame_interval_ms]),
            );

            // Dropped frame detection: log scalar, warn only on severe gaps (>500ms)
            if frame_interval_ms > 60.0 {
                let dropped_est = (frame_interval_ms / 33.33).round() as u64 - 1;
                let _ = self.rec.log(
                    "/checks/dropped_frames",
                    &rerun::Scalars::new([dropped_est as f64]),
                );
                if frame_interval_ms > 500.0 {
                    self.log_warning(
                        frame_idx,
                        capture_time_ns,
                        &format!("large gap: {:.1}ms (~{dropped_est} frames)", frame_interval_ms),
                    );
                }
            }
        }

        *last = Some((frame_idx, capture_time_ns));

        // ── Hash match: 1 = frame changed (expected), 0 = frozen ────────
        // Strided sample, not the whole buffer: hashing all ~100 MB of BGRA per
        // frame cost ~50–100 ms and was the recorder's dominant per-frame cost,
        // dropping frames. A 4 KB stride still reliably flips when pixels change.
        let hash = fnv1a64_strided(pixels, 4096);
        let mut last_hash = self.last_frame_hash.lock().unwrap();
        let hash_match: f64 = match *last_hash {
            Some(prev) if prev == hash => 0.0,  // frozen — same hash
            Some(_) => 1.0,                      // changed — expected
            None => 1.0,                         // first frame
        };
        let _ = self.rec.log(
            "/checks/frame_hash_match",
            &rerun::Scalars::new([hash_match]),
        );
        *last_hash = Some(hash);

        // ── Window count ────────────────────────────────────────────────
        let window_count = frame_rec.windows.len() as f64;
        let _ = self.rec.log(
            "/checks/window_count",
            &rerun::Scalars::new([window_count]),
        );

        // ── Buffer size sanity ──────────────────────────────────────────
        if pixels.len() != width as usize * height as usize * 4 {
            self.log_warning(
                frame_idx,
                capture_time_ns,
                "pixel buffer size does not match width * height * 4",
            );
        }
    }

    fn log_windows(&self, frame_rec: &FrameRecord, s: usize) {
        let sf = s as f32;
        let mut mins = Vec::new();
        let mut sizes = Vec::new();
        let mut labels = Vec::new();

        for w in &frame_rec.windows {
            let bbox = w.bbox_px;
            mins.push([bbox[0] as f32 / sf, bbox[1] as f32 / sf]);
            sizes.push([bbox[2] as f32 / sf, bbox[3] as f32 / sf]);

            labels.push(format!(
                "id={} z={} {}",
                w.window_id,
                w.z_index,
                w.action,
            ));
        }

        if mins.is_empty() {
            return;
        }

        let boxes_2d = rerun::Boxes2D::from_mins_and_sizes(mins, sizes)
            .with_labels(labels);

        let _ = self.rec.log("/recorded/window_boxes", &boxes_2d);
    }

    fn log_cursor(&self, frame_rec: &FrameRecord, s: usize) {
        let sf = s as f32;
        let pos = frame_rec.cursor.position_px;
        let bbox = frame_rec.cursor.bbox_px;

        // Cursor position as a point
        let points = rerun::Points2D::new([(pos[0] as f32 / sf, pos[1] as f32 / sf)])
            .with_labels([frame_rec.cursor.action.clone()])
            .with_radii([6.0]);

        let _ = self.rec.log("/recorded/cursor", &points);

        // Cursor bounding box as a rectangle
        if bbox[2] > 0 && bbox[3] > 0 {
            let boxes = rerun::Boxes2D::from_mins_and_sizes(
                [(bbox[0] as f32 / sf, bbox[1] as f32 / sf)],
                [(bbox[2] as f32 / sf, bbox[3] as f32 / sf)],
            );
            let _ = self.rec.log("/recorded/cursor_box", &boxes);
        }
    }

    fn log_focused_window(
        &self,
        frame_idx: u64,
        capture_time_ns: u64,
        frame_rec: &FrameRecord,
        s: usize,
    ) {
        let _ = (frame_idx, capture_time_ns);
        let sf = s as f32;
        // Highlight the focused window with a distinct color
        if let Some(focused) = frame_rec.windows.iter().find(|w| w.action == "focused") {
            let bbox = focused.bbox_px;
            let boxes = rerun::Boxes2D::from_mins_and_sizes(
                [(bbox[0] as f32 / sf, bbox[1] as f32 / sf)],
                [(bbox[2] as f32 / sf, bbox[3] as f32 / sf)],
            )
            .with_labels([format!("focused: id={} {}", focused.window_id, focused.owner_name)]);

            let _ = self.rec.log("/recorded/focused_window", &boxes);
        }
    }
}

// ── Utility functions ─────────────────────────────────────────────────────────

/// Integer downsample factor so the longest side is ≤ VIZ_MAX_DIM.
#[inline]
fn viz_scale(w: u32, h: u32) -> usize {
    (w.max(h) as usize / VIZ_MAX_DIM as usize).max(1)
}

/// Nearest-neighbour downsample of an RGB24 buffer by integer factor `s`.
fn downsample_rgb(rgb: &[u8], w: usize, h: usize, s: usize) -> (Vec<u8>, u32, u32) {
    if s <= 1 { return (rgb.to_vec(), w as u32, h as u32); }
    let dw = (w / s).max(1);
    let dh = (h / s).max(1);
    let mut out = Vec::with_capacity(dw * dh * 3);
    for dy in 0..dh {
        let sy = dy * s;
        for dx in 0..dw {
            let si = (sy * w + dx * s) * 3;
            out.push(rgb[si]);
            out.push(rgb[si + 1]);
            out.push(rgb[si + 2]);
        }
    }
    (out, dw as u32, dh as u32)
}

pub fn bgra_to_rgb(bgra: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(bgra.len() / 4 * 3);

    for px in bgra.chunks_exact(4) {
        let b = px[0];
        let g = px[1];
        let r = px[2];

        rgb.push(r);
        rgb.push(g);
        rgb.push(b);
    }

    rgb
}

/// Deterministic pastel RGB from a dense instance id (same hash as compositor::label_color).
fn instance_id_color(id: u32) -> [u8; 3] {
    let mut h = id.wrapping_mul(0x9e37_79b9);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    [
        0x80 | (h         & 0x7F) as u8,
        0x80 | ((h >>  8) & 0x7F) as u8,
        0x80 | ((h >> 16) & 0x7F) as u8,
    ]
}

/// FNV-1a over a strided sample of `bytes` (every `step`-th byte) plus the length.
/// O(len/step) instead of O(len) — for freeze detection on large frame buffers,
/// where hashing every byte is far too slow to run per frame.
pub fn fnv1a64_strided(bytes: &[u8], step: usize) -> u64 {
    let step = step.max(1);
    let mut hash: u64 = 0xcbf29ce484222325;
    // Fold in the length so differently-sized buffers never collide.
    hash ^= bytes.len() as u64;
    hash = hash.wrapping_mul(0x100000001b3);
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(0x100000001b3);
        i += step;
    }
    hash
}
