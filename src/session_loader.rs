// ── Session Loader — reads saved session data from disk ──────────────────────
//
// Used by replay_viz and model_input_viz to reload canonical session files.
// Does NOT use any live system APIs (no AX, no CGWindowList, no CGEvent, etc.).
// The source of truth is the saved session data.
#![allow(dead_code)]

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::recorder::record::{
    EventRecord, FocusEventRecord, FrameRecord, KeyEventRecord, OwnedWindowRecord,
};
use crate::recorder::session::SessionMeta;

// ── Loaded session ────────────────────────────────────────────────────────────

pub struct LoadedSession {
    pub session_dir: PathBuf,
    pub meta: SessionMeta,
    pub frames: Vec<FrameRecord>,
    /// All events (cursor actions, focus changes, key events) interleaved.
    /// Each entry is a JSON Value — callers can deserialize based on the "kind" or "class" field.
    pub events_raw: Vec<serde_json::Value>,
}

/// A single row from windows.ndjson: a per-frame window state snapshot.
#[derive(serde::Deserialize)]
pub struct WindowsEntry {
    pub frame_index: u64,
    #[serde(default)]
    pub timestamp_ns: u64,
    pub windows: Vec<OwnedWindowRecord>,
}

// ── Public API ────────────────────────────────────────────────────────────────

impl LoadedSession {
    /// Load a session from its directory.
    pub fn load(session_dir: &Path) -> Result<Self, String> {
        if !session_dir.is_dir() {
            return Err(format!("not a directory: {}", session_dir.display()));
        }

        let meta_path = session_dir.join("session.json");
        let meta: SessionMeta = if meta_path.exists() {
            let data = fs::read_to_string(&meta_path)
                .map_err(|e| format!("read session.json: {e}"))?;
            serde_json::from_str(&data)
                .map_err(|e| format!("parse session.json: {e}"))?
        } else {
            return Err("no session.json found".into());
        };

        let frames = Self::load_frames(session_dir)?;
        let events_raw = Self::load_events_raw(session_dir)?;

        Ok(Self {
            session_dir: session_dir.to_path_buf(),
            meta,
            frames,
            events_raw,
        })
    }

    /// Load parsed cursor-action + focus + key events from events.ndjson.
    pub fn load_events(&self) -> Result<Vec<EventRecord>, String> {
        let events_path = self.session_dir.join("events.ndjson");
        Self::read_ndjson(&events_path)
    }

    /// Load focus-change events from events.ndjson.
    pub fn load_focus_events(&self) -> Result<Vec<FocusEventRecord>, String> {
        let events_path = self.session_dir.join("events.ndjson");
        Self::read_ndjson(&events_path)
    }

    /// Load key events from events.ndjson.
    pub fn load_key_events(&self) -> Result<Vec<KeyEventRecord>, String> {
        let events_path = self.session_dir.join("events.ndjson");
        Self::read_ndjson(&events_path)
    }

    /// Load windows.ndjson entries.
    pub fn load_windows_entries(&self) -> Result<Vec<WindowsEntry>, String> {
        let windows_path = self.session_dir.join("windows.ndjson");
        Self::read_ndjson(&windows_path)
    }

    /// Load a frame image (PNG or JPEG) from the session's frames/ directory.
    /// Returns (pixels_rgba, width, height) or None if the image_path is None or file missing.
    pub fn load_frame_png(
        &self,
        frame: &FrameRecord,
    ) -> Option<(Vec<u8>, u32, u32)> {
        let image_path = frame.image_path.as_ref()?;
        let full_path = self.session_dir.join(image_path);
        if !full_path.exists() {
            return None;
        }
        let ext = full_path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext == "jpg" || ext == "jpeg" {
            Self::read_jpeg(&full_path).ok()
        } else {
            Self::read_png(&full_path).ok()
        }
    }

    /// Load a mask PNG from the session's masks/ directory.
    pub fn load_mask_png(&self, rel_path: &str) -> Option<(Vec<u8>, u32, u32)> {
        let full_path = self.session_dir.join(rel_path);
        if !full_path.exists() {
            return None;
        }
        Self::read_png(&full_path).ok()
    }

    /// Load the cursor mask for a specific frame.
    pub fn load_cursor_mask(&self, frame: &FrameRecord) -> Option<Vec<u8>> {
        let rel = &frame.mask_paths.cursor_mask;
        if rel.is_empty() { return None; }
        let full = self.session_dir.join(rel);
        if !full.exists() { return None; }
        // cursor masks are 8-bit grayscale
        let decoder = png::Decoder::new(fs::File::open(&full).ok()?);
        let mut reader = decoder.read_info().ok()?;
        let mut buf = vec![0u8; reader.output_buffer_size()];
        reader.next_frame(&mut buf).ok()?;
        Some(buf)
    }

    /// Load the windows label mask for a specific frame.
    pub fn load_windows_label_mask(&self, frame: &FrameRecord) -> Option<Vec<u8>> {
        let rel = &frame.mask_paths.windows_label;
        if rel.is_empty() { return None; }
        let full = self.session_dir.join(rel);
        if !full.exists() { return None; }
        // windows label masks are 16-bit (2 bytes per pixel)
        let decoder = png::Decoder::new(fs::File::open(&full).ok()?);
        let mut reader = decoder.read_info().ok()?;
        let mut buf = vec![0u8; reader.output_buffer_size()];
        reader.next_frame(&mut buf).ok()?;
        Some(buf)
    }

    /// Load the combined label mask for a specific frame.
    pub fn load_combined_label_mask(&self, frame: &FrameRecord) -> Option<Vec<u8>> {
        let rel = &frame.mask_paths.combined_label;
        if rel.is_empty() { return None; }
        let full = self.session_dir.join(rel);
        if !full.exists() { return None; }
        let decoder = png::Decoder::new(fs::File::open(&full).ok()?);
        let mut reader = decoder.read_info().ok()?;
        let mut buf = vec![0u8; reader.output_buffer_size()];
        reader.next_frame(&mut buf).ok()?;
        Some(buf)
    }

    /// Return frames that have a saved PNG image.
    pub fn frames_with_images(&self) -> Vec<&FrameRecord> {
        self.frames.iter()
            .filter(|f| f.image_path.is_some())
            .collect()
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

impl LoadedSession {
    fn load_frames(session_dir: &Path) -> Result<Vec<FrameRecord>, String> {
        let path = session_dir.join("frames.ndjson");
        Self::read_ndjson(&path)
    }

    fn load_events_raw(session_dir: &Path) -> Result<Vec<serde_json::Value>, String> {
        let path = session_dir.join("events.ndjson");
        if !path.exists() {
            return Ok(Vec::new());
        }
        let f = fs::File::open(&path)
            .map_err(|e| format!("open {}: {e}", path.display()))?;
        let reader = BufReader::new(f);
        let mut out = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| format!("read line: {e}"))?;
            if line.trim().is_empty() { continue; }
            let v: serde_json::Value = serde_json::from_str(&line)
                .map_err(|e| format!("parse events.ndjson: {e}"))?;
            out.push(v);
        }
        Ok(out)
    }

    fn read_ndjson<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>, String> {
        if !path.exists() {
            return Ok(Vec::new());
        }
        let f = fs::File::open(path)
            .map_err(|e| format!("open {}: {e}", path.display()))?;
        let reader = BufReader::new(f);
        let mut out = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| format!("read line in {}: {e}", path.display()))?;
            if line.trim().is_empty() { continue; }
            let v: T = serde_json::from_str(&line)
                .map_err(|e| format!("parse {}: {e}", path.display()))?;
            out.push(v);
        }
        Ok(out)
    }

    fn read_jpeg(path: &Path) -> Result<(Vec<u8>, u32, u32), String> {
        let data = fs::read(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        let mut decoder = jpeg_decoder::Decoder::new(data.as_slice());
        let pixels = decoder.decode()
            .map_err(|e| format!("jpeg decode {}: {e}", path.display()))?;
        let info = decoder.info()
            .ok_or_else(|| format!("jpeg info missing: {}", path.display()))?;
        let (w, h) = (info.width as u32, info.height as u32);
        // jpeg_decoder returns RGB; expand to RGBA
        let rgba: Vec<u8> = pixels.chunks_exact(3)
            .flat_map(|px| [px[0], px[1], px[2], 255u8])
            .collect();
        Ok((rgba, w, h))
    }

    fn read_png(path: &Path) -> Result<(Vec<u8>, u32, u32), String> {
        let decoder = png::Decoder::new(
            fs::File::open(path)
                .map_err(|e| format!("open {}: {e}", path.display()))?
        );
        let mut reader = decoder
            .read_info()
            .map_err(|e| format!("png info {}: {e}", path.display()))?;

        let info = reader.info();
        let (width, height) = (info.width, info.height);
        let color_type = info.color_type;
        let bit_depth = info.bit_depth;

        let mut buf = vec![0u8; reader.output_buffer_size()];
        reader
            .next_frame(&mut buf)
            .map_err(|e| format!("png decode {}: {e}", path.display()))?;

        // Convert to RGBA8 for uniform handling
        match (color_type, bit_depth) {
            (png::ColorType::Rgba, png::BitDepth::Eight) => Ok((buf, width, height)),
            (png::ColorType::Rgb, png::BitDepth::Eight) => {
                let mut rgba = Vec::with_capacity((width * height * 4) as usize);
                for px in buf.chunks_exact(3) {
                    rgba.push(px[0]);
                    rgba.push(px[1]);
                    rgba.push(px[2]);
                    rgba.push(255);
                }
                Ok((rgba, width, height))
            }
            (png::ColorType::Grayscale, png::BitDepth::Eight) => {
                // Expand grayscale to RGBA for visualization
                let mut rgba = Vec::with_capacity((width * height * 4) as usize);
                for &v in &buf {
                    rgba.push(v);
                    rgba.push(v);
                    rgba.push(v);
                    rgba.push(255);
                }
                Ok((rgba, width, height))
            }
            (png::ColorType::GrayscaleAlpha, png::BitDepth::Eight) => {
                let mut rgba = Vec::with_capacity((width * height * 4) as usize);
                for px in buf.chunks_exact(2) {
                    rgba.push(px[0]);
                    rgba.push(px[0]);
                    rgba.push(px[0]);
                    rgba.push(px[1]);
                }
                Ok((rgba, width, height))
            }
            other => Err(format!(
                "unsupported PNG format {:?} in {}",
                other,
                path.display()
            )),
        }
    }
}
