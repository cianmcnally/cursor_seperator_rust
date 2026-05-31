use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::windows::model::RectI;
use crate::windows::WindowLayer;

pub struct RingEntry {
    pub captured_at:   Instant,
    pub _timestamp_ns:  u64,
    /// Shared with the capture loop's frame Arc — no extra copy at push time.
    pub pixels:        Arc<Vec<u8>>,
    pub width:         usize,
    pub height:        usize,
    pub bytes_per_row: usize,
    /// Cursor bounding box in capture-pixel coordinates.
    pub cursor_rect:   Option<RectI>,
    pub windows:       Vec<WindowLayer>,
}

struct Inner {
    entries: VecDeque<Arc<RingEntry>>,
    cap:     usize,
}

pub struct FrameRingBuffer(Mutex<Inner>);

impl FrameRingBuffer {
    pub fn new(cap: usize) -> Arc<Self> {
        Arc::new(Self(Mutex::new(Inner {
            entries: VecDeque::with_capacity(cap),
            cap,
        })))
    }

    pub fn push(&self, entry: RingEntry) {
        let mut inner = self.0.lock().unwrap();
        if inner.entries.len() >= inner.cap {
            inner.entries.pop_front();
        }
        inner.entries.push_back(Arc::new(entry));
    }

    /// Latest frame strictly before `t`.
    pub fn latest_before(&self, t: Instant) -> Option<Arc<RingEntry>> {
        let inner = self.0.lock().unwrap();
        inner.entries.iter()
            .filter(|e| e.captured_at < t)
            .last()
            .cloned()
    }

    /// Earliest frame that arrived at least `min_gap` after `t`.
    pub fn earliest_after(&self, t: Instant) -> Option<Arc<RingEntry>> {
        let inner = self.0.lock().unwrap();
        let thresh = t + Duration::from_millis(8);
        inner.entries.iter()
            .find(|e| e.captured_at >= thresh)
            .cloned()
    }
}
