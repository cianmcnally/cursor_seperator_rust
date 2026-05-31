use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use crate::cursor_action::{action_kind_str, ActionSnapshot, CursorAction, CursorActionKind};
use crate::debug_viz::{DebugViz, VizSink};
use crate::typing::TypingDetectorResult;
use crate::windows::WindowLayer;
use super::KeyTapEvent;

use super::masks::{
    assign_window_labels, build_combined_label_mask, build_cursor_mask,
    build_instance_mask, paint_windows_label_mask,
    write_class_mask_png, write_cursor_png, write_frame_jpeg, write_label_png,
    write_windows_json, LabelAssigner,
};
use super::record::{
    build_cursor_record, build_event_record, build_window_records,
    FocusEventRecord, FrameRecord, FrameTimestamps, KeyEventRecord, MaskPaths, NdjsonWriter,
    TypingAreaRecord,
};
use super::session::SessionMeta;
use super::RecordArgs;

/// When recording, log the heavy Rerun image + segmentation only every Nth saved
/// frame. The full-res RGB+mask log dominates the writer's per-frame cost (~650 KB
/// and tens of ms each); at the capture rate it backs up the frame queue and drops
/// most frames. PNG/JPEG (downsampled) + event/overlay markers still log every
/// frame — only the .rrd imagery is thinned. Live mode logs every frame as before.
const RECORD_VIZ_EVERY: u64 = 5;

// ── Task types ────────────────────────────────────────────────────────────────

pub struct FrameWriteTask {
    pub frame_index:   u64,
    pub timestamp_ns:  u64,
    /// BGRA pixels, full capture resolution.  Shared with the capture loop via Arc.
    pub pixels:        Arc<Vec<u8>>,
    pub width:         usize,
    pub height:        usize,
    pub bytes_per_row: usize,
    pub windows:       Arc<Vec<WindowLayer>>,
    pub cursor_pos_px: (i32, i32),
    /// Cursor bounding box [x, y, w, h] in capture pixels.
    pub cursor_bbox:   Option<[i32; 4]>,
    /// BGRA cursor sprite pixels. None if sprite failed to load.
    /// Used to shape the cursor mask (arrow outline instead of solid rect).
    pub cursor_sprite:     Option<Arc<Vec<u8>>>,
    pub cursor_sprite_w:   usize,
    pub cursor_sprite_h:   usize,
    pub action_snap:   ActionSnapshot,
    pub typing_result: TypingDetectorResult,
    pub cap_w:         u32,
    pub cap_h:         u32,
    /// Focused window id at capture time — from the captured window list
    /// (mask_role == FocusedRoot), never from a live AX query.
    pub focused_window_id_at_capture: Option<u32>,
}

/// Per-frame mask-building work deferred to the post-recording pass.
///
/// During capture the hot path does only the cheap, time-critical work (write the
/// frame JPEG, append events, frames/windows.ndjson, light Rerun) and stashes one
/// of these — all `Arc`/`Copy`/small `Vec`, NO pixel buffer — so the frame queue
/// never backs up and frames are never dropped. After capture stops we replay the
/// stash to build the masks (at save resolution) and write the mask PNGs/JSON the
/// frame records already point at. `labels` come from `assign_window_labels`
/// (computed in frame order during capture so the session-stable assigner stays
/// consistent); the pixel fill happens here.
struct MaskJob {
    frame_index:     u64,
    timestamp_ns:    u64,
    windows:         Arc<Vec<WindowLayer>>,
    labels:          Vec<u16>,
    cursor_bbox:     Option<[i32; 4]>,
    cursor_sprite:   Option<Arc<Vec<u8>>>,
    cursor_sprite_w: usize,
    cursor_sprite_h: usize,
    cap_w:           u32,
    cap_h:           u32,
}

// ── Writer thread ─────────────────────────────────────────────────────────────

pub fn start_writer(
    session_dir: Option<PathBuf>,
    session:     SessionMeta,
    frame_rx:    mpsc::Receiver<FrameWriteTask>,
    event_rx:    mpsc::Receiver<CursorAction>,
    key_rx:      mpsc::Receiver<KeyTapEvent>,
    shutdown_rx: mpsc::Receiver<()>,
    args:        RecordArgs,
    viz_sink:    VizSink,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("recorder-writer".into())
        .spawn(move || {
            run_writer(session_dir, session, frame_rx, event_rx, key_rx, shutdown_rx, args, viz_sink);
        })
        .expect("spawn recorder writer")
}

/// NDJSON sinks — present only when recording to disk.
struct DiskWriters {
    frames:  NdjsonWriter,
    events:  NdjsonWriter,
    windows: NdjsonWriter,
    frames_dir: PathBuf,
    masks_dir:  PathBuf,
}

fn run_writer(
    session_dir: Option<PathBuf>,
    session:     SessionMeta,
    frame_rx:    mpsc::Receiver<FrameWriteTask>,
    event_rx:    mpsc::Receiver<CursorAction>,
    key_rx:      mpsc::Receiver<KeyTapEvent>,
    shutdown_rx: mpsc::Receiver<()>,
    args:        RecordArgs,
    viz_sink:    VizSink,
) {
    let recording = session_dir.is_some();

    // ── Disk sinks — only when recording ──────────────────────────────────────
    let mut disk: Option<DiskWriters> = if let Some(ref dir) = session_dir {
        let frames_dir = dir.join("frames");
        let masks_dir  = dir.join("masks");
        if args.save_frames { let _ = std::fs::create_dir_all(&frames_dir); }
        if args.save_masks  { let _ = std::fs::create_dir_all(&masks_dir); }
        if let Err(e) = session.write_json(dir) {
            eprintln!("[recorder] session.json write failed: {e}");
        }
        match (
            NdjsonWriter::create(&dir.join("frames.ndjson")),
            NdjsonWriter::create(&dir.join("events.ndjson")),
            NdjsonWriter::create(&dir.join("windows.ndjson")),
        ) {
            (Ok(frames), Ok(events), Ok(windows)) =>
                Some(DiskWriters { frames, events, windows, frames_dir, masks_dir }),
            _ => { eprintln!("[recorder] failed to open ndjson writers"); return; }
        }
    } else {
        None
    };

    // ── Rerun — always on (the single viewer) ────────────────────────────────
    let debug_viz: Option<DebugViz> = match DebugViz::new(&session.session_id, viz_sink) {
        Ok(v) => {
            if recording {
                println!("[recorder] rerun → {}/debug.rrd",
                    session_dir.as_ref().unwrap().display());
            } else {
                println!("[viewer] streaming live to a spawned Rerun viewer");
            }
            Some(v)
        }
        Err(e) => { eprintln!("[viewer] failed to start Rerun: {e}"); None }
    };

    let mut label_assigner   = LabelAssigner::new();
    let mut frame_timestamps = FrameTimestamps::new();
    let mut event_counter    = 0u64;
    // Frame index up to which we force-save every frame (post-event radius).
    let mut force_save_until: u64 = 0;
    let mut last_focused_window: Option<u32> = None;
    let mut focus_initialized = false;
    let mut known_window_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();

    let cap_w = session.capture_size_px[0];
    let cap_h = session.capture_size_px[1];
    let mut stopping = false;
    // Deferred mask-building jobs (recording only). Filled on the hot path,
    // drained in one post pass after capture stops — see MaskJob.
    let mut pending: Vec<MaskJob> = Vec::new();

    loop {
        let mut did_work = false;

        // ── Cursor action + key events (drained EVERY iteration, before frames,
        //    so a slow PNG-writing frame pass can't starve them) ───────────────
        let (ev_work, ev_disc) = drain_events(
            &event_rx, &mut disk, debug_viz.as_ref(), &frame_timestamps,
            &mut event_counter, &mut force_save_until, &args, cap_w, cap_h);
        if ev_disc { flush(&mut disk, debug_viz.as_ref()); return; }
        did_work |= ev_work;
        did_work |= drain_keys(
            &key_rx, &mut disk, debug_viz.as_ref(), &frame_timestamps,
            &mut event_counter, &mut force_save_until, &args, last_focused_window);

        // ── Frames: process at most ONE per iteration. When PNG writes fall
        //    behind the capture rate the queue stays full, but events still get
        //    a drain pass between every frame instead of being starved. ────────
        let mut got_frame = false;
        {
            match frame_rx.try_recv() {
                Ok(task) => {
                    got_frame = true;
                    frame_timestamps.push(task.frame_index, task.timestamp_ns);

                    // Window lifecycle: created / destroyed (Rerun markers only).
                    let current_ids: std::collections::HashSet<u32> = task.windows
                        .iter()
                        .filter(|w| w.include_in_segmentation)
                        .map(|w| w.window_id)
                        .collect();
                    if let Some(viz) = debug_viz.as_ref() {
                        for id in current_ids.difference(&known_window_ids) {
                            if let Some(w) = task.windows.iter().find(|w| w.window_id == *id) {
                                viz.log_window_lifecycle(
                                    task.frame_index, task.timestamp_ns,
                                    "window_created", *id, &w.owner_name,
                                );
                            }
                        }
                        for id in known_window_ids.difference(&current_ids) {
                            viz.log_window_lifecycle(
                                task.frame_index, task.timestamp_ns,
                                "window_destroyed", *id, "(closed)",
                            );
                        }
                    }
                    known_window_ids = current_ids;

                    // Focus change — from captured data, not live AX.
                    if focus_initialized
                        && task.focused_window_id_at_capture != last_focused_window
                    {
                        event_counter += 1;
                        let to_owner = task.focused_window_id_at_capture.and_then(|wid| {
                            task.windows.iter()
                                .find(|w| w.window_id == wid)
                                .map(|w| w.owner_name.clone())
                        });
                        let rec = FocusEventRecord {
                            event_id:    format!("evt_{:06}", event_counter),
                            class:       "window".to_string(),
                            kind:        "FocusChange".to_string(),
                            timestamp_ns: task.timestamp_ns,
                            frame_index: task.frame_index,
                            from_window: last_focused_window,
                            to_window:   task.focused_window_id_at_capture,
                            to_owner,
                        };
                        if let Some(d) = disk.as_mut() { d.events.write(&rec); }
                        if let Some(viz) = debug_viz.as_ref() {
                            viz.log_focus_event(task.frame_index, task.timestamp_ns, &rec);
                        }
                        force_save_until =
                            force_save_until.max(task.frame_index + args.event_save_radius);
                    }
                    last_focused_window = task.focused_window_id_at_capture;
                    focus_initialized = true;

                    record_frame_hot(
                        &task, recording, &mut label_assigner, disk.as_mut(),
                        &args, debug_viz.as_ref(), &mut pending,
                    );
                    did_work = true;
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(_) => { flush(&mut disk, debug_viz.as_ref()); return; }
            }
        }

        // ── Shutdown: latch the signal, then keep draining the frame queue (one
        //    per iteration, events drained each pass) until it is empty, so no
        //    queued frames OR in-flight events are dropped on exit. ────────────
        if !stopping && shutdown_rx.try_recv().is_ok() {
            stopping = true;
        }
        if stopping && !got_frame {
            // Frame queue drained — final straggler-event pass.
            let _ = drain_events(
                &event_rx, &mut disk, debug_viz.as_ref(), &frame_timestamps,
                &mut event_counter, &mut force_save_until, &args, cap_w, cap_h);
            let _ = drain_keys(
                &key_rx, &mut disk, debug_viz.as_ref(), &frame_timestamps,
                &mut event_counter, &mut force_save_until, &args, last_focused_window);
            // Post-recording pass: build every stashed frame's masks now that the
            // capture loop is done, so nothing competed with it for CPU.
            run_mask_post_pass(&pending, &mut disk, &args, debug_viz.as_ref());
            flush(&mut disk, debug_viz.as_ref());
            return;
        }

        if !did_work && !got_frame {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

/// Drain all pending cursor-action events to disk + Rerun.
/// Returns `(did_work, disconnected)`.
fn drain_events(
    event_rx:         &mpsc::Receiver<CursorAction>,
    disk:             &mut Option<DiskWriters>,
    viz:              Option<&DebugViz>,
    frame_timestamps: &FrameTimestamps,
    event_counter:    &mut u64,
    force_save_until: &mut u64,
    args:             &RecordArgs,
    cap_w: u32, cap_h: u32,
) -> (bool, bool) {
    let mut did_work = false;
    loop {
        match event_rx.try_recv() {
            Ok(action) => {
                *event_counter += 1;
                let fi = frame_timestamps.nearest(action.timestamp_ns);
                let rec = build_event_record(&action, *event_counter, fi, cap_w, cap_h);
                if let Some(d) = disk.as_mut() { d.events.write(&rec); }
                if let Some(v) = viz { v.log_event(fi, rec.timestamp_ns, &rec); }
                *force_save_until = (*force_save_until).max(fi + args.event_save_radius);
                did_work = true;
            }
            Err(mpsc::TryRecvError::Empty) => return (did_work, false),
            Err(_) => return (did_work, true),
        }
    }
}

/// Drain all pending key-down events to disk + Rerun. Returns `did_work`.
fn drain_keys(
    key_rx:           &mpsc::Receiver<KeyTapEvent>,
    disk:             &mut Option<DiskWriters>,
    viz:              Option<&DebugViz>,
    frame_timestamps: &FrameTimestamps,
    event_counter:    &mut u64,
    force_save_until: &mut u64,
    args:             &RecordArgs,
    last_focused_window: Option<u32>,
) -> bool {
    let mut did_work = false;
    loop {
        match key_rx.try_recv() {
            Ok((timestamp_ns, key_code)) => {
                *event_counter += 1;
                let fi = frame_timestamps.nearest(timestamp_ns);
                let rec = KeyEventRecord {
                    event_id:    format!("evt_{:06}", *event_counter),
                    class:       "keyboard".to_string(),
                    kind:        "KeyDown".to_string(),
                    timestamp_ns,
                    frame_index: fi,
                    key_code,
                    window_id:   last_focused_window,
                };
                if let Some(d) = disk.as_mut() { d.events.write(&rec); }
                if let Some(v) = viz { v.log_key_event(fi, timestamp_ns, &rec); }
                *force_save_until = (*force_save_until).max(fi + args.event_save_radius);
                did_work = true;
            }
            Err(mpsc::TryRecvError::Empty) => return did_work,
            Err(_) => return did_work,
        }
    }
}

fn flush(disk: &mut Option<DiskWriters>, viz: Option<&DebugViz>) {
    if let Some(d) = disk.as_mut() {
        d.frames.flush(); d.events.flush(); d.windows.flush();
    }
    if let Some(v) = viz { v.flush(); }
}

/// Nearest-neighbour BGRA downsample. Returns `(pixels, width, height, bpr)`.
fn downsample_bgra(
    src:    &[u8],
    src_w:  usize,
    src_h:  usize,
    src_bpr: usize,
    dst_w:  usize,
    dst_h:  usize,
) -> (Vec<u8>, usize, usize, usize) {
    let mut out = Vec::with_capacity(dst_w * dst_h * 4);
    for dy in 0..dst_h {
        let sy = dy * src_h / dst_h;
        let row_off = sy * src_bpr;
        for dx in 0..dst_w {
            let sx = dx * src_w / dst_w;
            let si = row_off + sx * 4;
            out.push(src[si]);
            out.push(src[si + 1]);
            out.push(src[si + 2]);
            out.push(src[si + 3]);
        }
    }
    let bpr = dst_w * 4;
    (out, dst_w, dst_h, bpr)
}

/// Hot path: the cheap, time-critical per-frame work done DURING capture.
///
/// Writes the frame JPEG, frames.ndjson + windows.ndjson, and a light Rerun log
/// (RGB at cadence + box/cursor overlays — NO segmentation image). The expensive
/// mask building is NOT done here; instead a `MaskJob` is stashed for the post
/// pass. Live mode (no disk) still builds + logs masks inline so the viewer is
/// complete — there is no capture-rate pressure when nothing is written to disk.
fn record_frame_hot(
    task:      &FrameWriteTask,
    recording: bool,
    assigner:  &mut LabelAssigner,
    disk:      Option<&mut DiskWriters>,
    args:      &RecordArgs,
    viz:       Option<&DebugViz>,
    pending:   &mut Vec<MaskJob>,
) {
    let fi    = task.frame_index;
    let cap_w = task.cap_w as usize;
    let cap_h = task.cap_h as usize;
    let cw    = task.cap_w;
    let ch    = task.cap_h;
    let save_every = args.save_frames_every.max(1);
    let is_sampled = fi % save_every == 0;
    let frame_stem = format!("frame_{:06}", fi);
    let sprite_data = task.cursor_sprite.as_ref().map(|a| a.as_slice());

    // Session-stable labels — the assigner is stateful and MUST run here, in
    // frame order. The label list is handed to the post pass for the pixel fill.
    let (labels, label_entries) = assign_window_labels(&task.windows, assigner);

    // ── JPEG (the only per-pixel work on the hot path) + deterministic paths ──
    let mut image_path = None;
    let mut mask_paths = MaskPaths {
        windows_label: String::new(), cursor_mask: String::new(),
        combined_label: String::new(), instance_map: String::new(),
        class_mask: String::new(), windows_json: String::new(),
    };
    if let Some(d) = disk.as_ref().map(|d| &**d) {
        if args.save_frames && is_sampled && !task.pixels.is_empty() {
            let frame_path = d.frames_dir.join(format!("{}.jpg", frame_stem));
            let (px, pw, ph, pbpr) = match args.frame_save_size {
                Some((tw, th)) if tw < task.width || th < task.height =>
                    downsample_bgra(&task.pixels, task.width, task.height, task.bytes_per_row, tw, th),
                _ => ((*task.pixels).clone(), task.width, task.height, task.bytes_per_row),
            };
            if let Err(e) = write_frame_jpeg(&frame_path, &px, pw, ph, pbpr) {
                eprintln!("[recorder] frame {fi} jpeg write failed: {e}");
            }
            image_path = Some(rel_path(&frame_path));
        }
        if args.save_masks && is_sampled {
            // The mask files don't exist yet — the post pass writes them — but
            // their names are deterministic, so the frame record can point at
            // them now and frames.ndjson is complete the moment capture ends.
            mask_paths = mask_paths_for(&d.masks_dir, &frame_stem);
        }
    }

    // ── Per-frame ground-truth label (always one of the 7; typing folds in) ───
    let cur = task.action_snap.current;
    let typing_now = task.typing_result.region.as_ref()
        .map_or(false, |r| r.source == "typing" && r.confidence > 0.4);
    let cursor_action_str = if typing_now
        && matches!(cur, CursorActionKind::Idle | CursorActionKind::Move) {
        "typing".to_string()
    } else {
        action_kind_str(cur).to_string()
    };

    let focused = task.focused_window_id_at_capture;
    let (wid_under, zidx_under) = window_at(task.cursor_pos_px, &task.windows);
    let cursor_rec = build_cursor_record(
        task.cursor_pos_px,
        task.cursor_bbox.unwrap_or([task.cursor_pos_px.0 - 12, task.cursor_pos_px.1 - 12, 24, 24]),
        &cursor_action_str, wid_under, zidx_under, cw, ch,
    );
    let window_recs = build_window_records(&task.windows, focused, cw, ch);
    let typing_area = task.typing_result.region.as_ref().map(|r| TypingAreaRecord {
        bbox:       [r.bbox.x, r.bbox.y, r.bbox.w, r.bbox.h],
        confidence: r.confidence,
        source:     r.source.clone(),
    });

    let frame_rec = FrameRecord {
        frame_index:     fi,
        timestamp_ns:    task.timestamp_ns,
        image_path,
        capture_size_px: [cw, ch],
        windows:         window_recs.clone(),
        cursor:          cursor_rec,
        mask_paths,
        label_map:       label_entries,
        typing_area,
    };

    // ── Disk: frames.ndjson + windows.ndjson (recording only) ─────────────────
    if let Some(d) = disk {
        d.frames.write(&frame_rec);
        #[derive(serde::Serialize)]
        struct WindowsEntry<'a> {
            frame_index: u64,
            timestamp_ns: u64,
            windows: &'a Vec<super::record::OwnedWindowRecord>,
        }
        d.windows.write(&WindowsEntry {
            frame_index: fi, timestamp_ns: task.timestamp_ns, windows: &window_recs,
        });
    }

    // ── Stash the mask build for the post pass (recording only) ───────────────
    if recording && args.save_masks && is_sampled {
        pending.push(MaskJob {
            frame_index:     fi,
            timestamp_ns:    task.timestamp_ns,
            windows:         Arc::clone(&task.windows),
            labels,
            cursor_bbox:     task.cursor_bbox,
            cursor_sprite:   task.cursor_sprite.clone(),
            cursor_sprite_w: task.cursor_sprite_w,
            cursor_sprite_h: task.cursor_sprite_h,
            cap_w:           cw,
            cap_h:           ch,
        });
    }

    // ── Rerun ─────────────────────────────────────────────────────────────────
    if let Some(viz) = viz {
        if recording {
            // RGB at cadence + box/cursor overlays. The segmentation image is
            // intentionally omitted from a recording .rrd (it needs the masks we
            // deferred); review masks via the PNGs on disk.
            let log_img = fi % RECORD_VIZ_EVERY == 0;
            viz.log_frame(fi, task.timestamp_ns, &frame_rec, &task.pixels, cw, ch, log_img);
        } else {
            // Live: build masks inline (no disk pressure) and log the full view.
            let (instance_map, class_mask_data, window_masks) =
                build_instance_mask(&task.windows, fi, cap_w, cap_h);
            let cursor_mask = build_cursor_mask(
                task.cursor_bbox, sprite_data,
                task.cursor_sprite_w, task.cursor_sprite_h, cap_w, cap_h,
            );
            viz.log_frame(fi, task.timestamp_ns, &frame_rec, &task.pixels, cw, ch, true);
            viz.log_segmentation(
                fi, task.timestamp_ns,
                &instance_map, &class_mask_data, &window_masks, &cursor_mask, cw, ch,
            );
        }
    }
}

/// Post-recording pass: build every stashed frame's masks and write the PNG/JSON
/// files the frame records already reference. Runs after the capture loop has
/// stopped, so it has the CPU to itself and never causes a dropped frame.
fn run_mask_post_pass(
    pending: &[MaskJob],
    disk:    &mut Option<DiskWriters>,
    args:    &RecordArgs,
    viz:     Option<&DebugViz>,
) {
    if pending.is_empty() || !args.save_masks { return; }
    let masks_dir = match disk.as_ref() {
        Some(d) => d.masks_dir.clone(),
        None => return,
    };
    println!("[recorder] building masks for {} frames…", pending.len());
    let t0 = std::time::Instant::now();
    for job in pending {
        build_frame_masks(job, args, &masks_dir, viz);
    }
    println!("[recorder] masks done in {:.1}s", t0.elapsed().as_secs_f64());
}

/// Build one frame's masks at the save resolution and write all six files.
/// Window rects + cursor bbox are scaled into save space; labels come from the
/// hot-path `assign_window_labels` so session-stable IDs stay consistent.
///
/// If `viz` is present, also log the segmentation to the .rrd for every
/// `RECORD_VIZ_EVERY`-th frame — at CAPTURE resolution so it aligns with the
/// capture-coord RGB image + box overlays the hot path logged. (Rebuilding at
/// capture res here is cheap because the post pass has the CPU to itself.)
fn build_frame_masks(job: &MaskJob, args: &RecordArgs, masks_dir: &Path, viz: Option<&DebugViz>) {
    let fi    = job.frame_index;
    let cap_w = job.cap_w as usize;
    let cap_h = job.cap_h as usize;
    let sprite_full = job.cursor_sprite.as_ref().map(|a| a.as_slice());
    let (mw, mh) = match args.frame_save_size {
        Some((tw, th)) if tw < cap_w || th < cap_h => (tw, th),
        _ => (cap_w, cap_h),
    };
    let down = (mw, mh) != (cap_w, cap_h);
    let sx = mw as f64 / cap_w as f64;
    let sy = mh as f64 / cap_h as f64;

    let scaled: Vec<WindowLayer>;
    let wins: &[WindowLayer] = if down {
        scaled = job.windows.iter().map(|w| {
            let mut w = w.clone();
            w.bounds_pixels = scale_rect(w.bounds_pixels, sx, sy);
            w
        }).collect();
        &scaled
    } else {
        &job.windows
    };
    let cursor_bbox_m = if down { job.cursor_bbox.map(|b| scale_bbox(b, sx, sy)) } else { job.cursor_bbox };
    let sprite = job.cursor_sprite.as_ref().map(|a| a.as_slice());

    let (instance_map, class_mask_data, window_masks) = build_instance_mask(wins, fi, mw, mh);
    let windows_label = paint_windows_label_mask(wins, &job.labels, mw, mh);
    let combined = build_combined_label_mask(
        &windows_label, cursor_bbox_m, sprite, job.cursor_sprite_w, job.cursor_sprite_h, mw, mh);
    let cursor_mask = build_cursor_mask(
        cursor_bbox_m, sprite, job.cursor_sprite_w, job.cursor_sprite_h, mw, mh);

    let stem = format!("frame_{:06}", fi);
    let _ = write_label_png(&masks_dir.join(format!("{}_windows_label.png", stem)), &windows_label, mw, mh);
    let _ = write_cursor_png(&masks_dir.join(format!("{}_cursor_mask.png", stem)), &cursor_mask, mw, mh);
    let _ = write_label_png(&masks_dir.join(format!("{}_combined_label.png", stem)), &combined, mw, mh);
    let _ = write_label_png(&masks_dir.join(format!("{}_instance_map.png", stem)), &instance_map, mw, mh);
    let _ = write_class_mask_png(&masks_dir.join(format!("{}_class_mask.png", stem)), &class_mask_data, mw, mh);
    let _ = write_windows_json(&masks_dir.join(format!("{}_windows.json", stem)), &window_masks);

    // Restore the segmentation overlay in the .rrd, every RECORD_VIZ_EVERY frame
    // (matches the hot-path RGB cadence). Built at capture res so it aligns with
    // the capture-coord RGB + box overlays; this is the only capture-res mask
    // build, and only on logged frames, so the post pass stays cheap.
    if let Some(viz) = viz {
        if fi % RECORD_VIZ_EVERY == 0 {
            let (instance_full, class_full, windows_full) =
                build_instance_mask(&job.windows, fi, cap_w, cap_h);
            let cursor_full = build_cursor_mask(
                job.cursor_bbox, sprite_full, job.cursor_sprite_w, job.cursor_sprite_h, cap_w, cap_h);
            viz.log_segmentation(
                fi, job.timestamp_ns,
                &instance_full, &class_full, &windows_full, &cursor_full,
                job.cap_w, job.cap_h,
            );
        }
    }
}

fn scale_rect(r: crate::windows::model::RectI, sx: f64, sy: f64) -> crate::windows::model::RectI {
    crate::windows::model::RectI {
        x: (r.x as f64 * sx).round() as i32,
        y: (r.y as f64 * sy).round() as i32,
        w: (r.w as f64 * sx).round() as i32,
        h: (r.h as f64 * sy).round() as i32,
    }
}

fn scale_bbox(b: [i32; 4], sx: f64, sy: f64) -> [i32; 4] {
    [(b[0] as f64 * sx).round() as i32, (b[1] as f64 * sy).round() as i32,
     (b[2] as f64 * sx).round() as i32, (b[3] as f64 * sy).round() as i32]
}

fn rel_path(p: &Path) -> String {
    p.file_name().and_then(|n| n.to_str())
        .map(|n| format!("{}/{}",
            p.parent().and_then(|d| d.file_name()).and_then(|d| d.to_str()).unwrap_or(""), n))
        .unwrap_or_default()
}

fn mask_paths_for(masks_dir: &Path, stem: &str) -> MaskPaths {
    let rel = |suffix: &str| rel_path(&masks_dir.join(format!("{}_{}", stem, suffix)));
    MaskPaths {
        windows_label:  rel("windows_label.png"),
        cursor_mask:    rel("cursor_mask.png"),
        combined_label: rel("combined_label.png"),
        instance_map:   rel("instance_map.png"),
        class_mask:     rel("class_mask.png"),
        windows_json:   rel("windows.json"),
    }
}

fn window_at(pos: (i32, i32), wins: &[WindowLayer]) -> (Option<u32>, Option<usize>) {
    for w in wins.iter().filter(|w| w.include_in_segmentation) {
        let b = w.bounds_pixels;
        if pos.0 >= b.x && pos.0 < b.x + b.w && pos.1 >= b.y && pos.1 < b.y + b.h {
            return (Some(w.window_id), Some(w.z_index));
        }
    }
    (None, None)
}
