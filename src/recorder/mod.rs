pub mod masks;
pub mod record;
pub mod session;
pub mod writer;

use std::path::PathBuf;
use std::sync::mpsc;

use crate::cursor_action::CursorAction;

pub use writer::FrameWriteTask;

/// Raw key-down event forwarded from the input tap.
/// `(timestamp_ns, key_code)` — see `typing/tap.rs`.
pub type KeyTapEvent = (u64, u16);

// ── RecordArgs ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct RecordArgs {
    /// True with --record-session: write the dataset to disk + a debug.rrd.
    /// False: live-only, stream to a spawned Rerun viewer (nothing saved).
    pub enabled:           bool,
    /// Root directory under which session subdirs are created.
    pub session_dir:       String,
    pub save_frames:       bool,
    pub save_masks:        bool,
    pub dump_session_json: bool,
    /// 0 = unlimited.
    pub max_seconds:       u64,
    /// Downsample saved frames to this size (width, height). None = native res.
    /// --frame-save-size 1920x1080
    pub frame_save_size:   Option<(usize, usize)>,
    /// Only save frame PNGs every N capture frames. 1 = every frame (default).
    /// --save-frames-every 3
    pub save_frames_every: u64,
    /// Force-save this many extra frames after each action event (click, drag, etc.).
    /// Combined with is_mouse_down pre-saving, gives temporal context around events.
    /// --event-save-radius 2
    pub event_save_radius: u64,
}

impl RecordArgs {
    pub fn parse() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let has  = |flag: &str| args.iter().any(|a| a == flag);
        let val  = |flag: &str, default: &str| -> String {
            args.windows(2)
                .find(|w| w[0] == flag)
                .map(|w| w[1].clone())
                .unwrap_or_else(|| default.to_string())
        };
        let valu64 = |flag: &str, default: u64| -> u64 {
            args.windows(2)
                .find(|w| w[0] == flag)
                .and_then(|w| w[1].parse().ok())
                .unwrap_or(default)
        };

        // Default 960x540; override with --frame-save-size WxH or disable with --native-res.
        let frame_save_size = if has("--native-res") {
            None
        } else {
            args.windows(2)
                .find(|w| w[0] == "--frame-save-size")
                .and_then(|w| {
                    let parts: Vec<&str> = w[1].splitn(2, 'x').collect();
                    if parts.len() == 2 {
                        let pw = parts[0].parse::<usize>().ok()?;
                        let ph = parts[1].parse::<usize>().ok()?;
                        Some((pw, ph))
                    } else {
                        None
                    }
                })
                .or(Some((960, 540)))
        };

        Self {
            enabled:           has("--record-session"),
            session_dir:       val("--session-dir", "recordings"),
            save_frames:       !has("--no-save-frames"),
            save_masks:        !has("--no-save-masks"),
            dump_session_json: has("--dump-session-json"),
            max_seconds:       valu64("--max-seconds", 0),
            frame_save_size,
            save_frames_every: valu64("--save-frames-every", 1).max(1),
            event_save_radius: valu64("--event-save-radius", 2),
        }
    }
}

// ── Recorder ──────────────────────────────────────────────────────────────────

pub struct Recorder {
    pub frame_tx:    mpsc::SyncSender<FrameWriteTask>,
    pub event_tx:    mpsc::Sender<CursorAction>,
    pub key_tx:      mpsc::Sender<KeyTapEvent>,
    shutdown_tx:     mpsc::SyncSender<()>,
    writer_handle:   Option<std::thread::JoinHandle<()>>,
}

impl Recorder {
    /// Send a frame to the writer thread. Drops if the queue is full.
    pub fn send_frame(&self, task: FrameWriteTask) {
        let _ = self.frame_tx.try_send(task);
    }

    /// Clone the cursor-action event sender.
    pub fn event_sender(&self) -> mpsc::Sender<CursorAction> {
        self.event_tx.clone()
    }

    /// Clone the key-tap sender to pass to the typing detector.
    pub fn key_sender(&self) -> mpsc::Sender<KeyTapEvent> {
        self.key_tx.clone()
    }

    /// Signal the writer to flush and exit, then block until it has finished.
    /// Dropping frame_tx/event_tx/key_tx first lets the writer drain any queued
    /// work; the join guarantees NDJSON + .rrd are fully flushed before we return.
    pub fn shutdown(mut self) {
        let _ = self.shutdown_tx.try_send(());
        if let Some(h) = self.writer_handle.take() {
            let _ = h.join();
        }
    }
}

/// Start the capture→process pipeline. ALWAYS runs:
///   - recording (`--record-session`): writes the dataset + a debug.rrd, returns the dir.
///   - live: spawns a Rerun viewer and streams to it, writes nothing, returns None.
/// Either way Rerun is the single viewer; recording just adds disk writes.
pub fn start_recorder(
    args:        &RecordArgs,
    cap_w:       u32,
    cap_h:       u32,
    origin_x:    f64,
    origin_y:    f64,
) -> std::io::Result<(Recorder, Option<PathBuf>)> {
    let session = session::SessionMeta::new(cap_w, cap_h, origin_x, origin_y);

    let (session_dir, viz_sink) = if args.enabled {
        // Anchor a relative session_dir (default "recordings") to the project root,
        // NOT the current working directory — otherwise running from inside a
        // subfolder writes recordings/<id> nested under wherever you happened to be.
        let root = PathBuf::from(&args.session_dir);
        let root = if root.is_absolute() {
            root
        } else {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(root)
        };
        let dir = root.join(&session.session_id);
        std::fs::create_dir_all(&dir)?;
        println!("[recorder] writing session → {}", dir.display());
        let sink = crate::debug_viz::VizSink::Save(dir.join("debug.rrd"));
        (Some(dir), sink)
    } else {
        (None, crate::debug_viz::VizSink::Spawn)
    };

    // Small bound: each task holds a full-res frame Arc (~tens of MB). 8 frames
    // (~1.6s at 5fps) is plenty of slack; send_frame try_sends and drops if full,
    // so the capture loop never blocks and app memory stays bounded.
    let (frame_tx, frame_rx) = mpsc::sync_channel::<FrameWriteTask>(8);
    let (event_tx, event_rx) = mpsc::channel::<CursorAction>();
    let (key_tx,   key_rx)   = mpsc::channel::<KeyTapEvent>();
    let (shutdown_tx, shutdown_rx) = mpsc::sync_channel::<()>(1);

    let writer_handle = writer::start_writer(
        session_dir.clone(),
        session,
        frame_rx,
        event_rx,
        key_rx,
        shutdown_rx,
        args.clone(),
        viz_sink,
    );

    Ok((
        Recorder { frame_tx, event_tx, key_tx, shutdown_tx, writer_handle: Some(writer_handle) },
        session_dir,
    ))
}
