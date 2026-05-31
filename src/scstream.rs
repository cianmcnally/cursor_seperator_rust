// ScreenCaptureKit streaming.
// Capture size is independent of preview size; callers downsample for display.
// Pixel format: BGRA (B=[0], G=[1], R=[2], A=[3]).

use std::ffi::c_void;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use block::ConcreteBlock;
use objc::declare::ClassDecl;
use objc::rc::autoreleasepool;
use objc::runtime::{Object, Sel};
use objc::{class, msg_send, sel, sel_impl};

// ── CoreMedia / CoreVideo FFI ───────────────────────────────────────────────

#[link(name = "CoreMedia", kind = "framework")]
extern "C" {
    fn CMSampleBufferGetImageBuffer(buf: *mut c_void) -> *mut c_void;
}

#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    fn CVPixelBufferLockBaseAddress(pb: *mut c_void, flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(pb: *mut c_void, flags: u64) -> i32;
    fn CVPixelBufferGetBaseAddress(pb: *mut c_void) -> *mut c_void;
    fn CVPixelBufferGetWidth(pb: *mut c_void) -> usize;
    fn CVPixelBufferGetHeight(pb: *mut c_void) -> usize;
    fn CVPixelBufferGetBytesPerRow(pb: *mut c_void) -> usize;
}

#[link(name = "ScreenCaptureKit", kind = "framework")]
extern "C" {}

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGMainDisplayID() -> u32;
}

// dispatch_* lives in libSystem, always linked on macOS
extern "C" {
    fn dispatch_semaphore_create(value: isize) -> *mut c_void;
    fn dispatch_semaphore_wait(dsema: *mut c_void, timeout: u64) -> isize;
    fn dispatch_semaphore_signal(dsema: *mut c_void) -> isize;
    fn dispatch_get_global_queue(identifier: isize, flags: usize) -> *mut c_void;
}
const DISPATCH_TIME_FOREVER: u64 = !0u64;

// ── Types ───────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone)]
struct CGPoint { x: f64, y: f64 }
#[repr(C)]
#[derive(Copy, Clone)]
struct CGSize { width: f64, height: f64 }
#[repr(C)]
#[derive(Copy, Clone)]
struct CGRect { origin: CGPoint, size: CGSize }

// kCMTimeFlags_Valid = 1
#[repr(C)]
#[derive(Copy, Clone)]
struct CMTime { value: i64, timescale: i32, flags: u32, epoch: i64 }

// kCVPixelFormatType_32BGRA ('BGRA')
const BGRA_PIXEL_FORMAT: u32 = 0x42475241;

// ── Public types ────────────────────────────────────────────────────────────

/// What region of the display to capture.
pub enum CaptureSource {
    /// Full primary display at native pixel resolution. No sourceRect set on stream.
    FullDisplay,
    /// Specific pixel rectangle. SCK receives that region and outputs it at native size.
    Region { x: u32, y: u32, w: u32, h: u32 },
}

pub struct FrameData {
    pub pixels:        Vec<u8>,
    pub width:         usize,
    pub height:        usize,
    pub bytes_per_row: usize,
    pub seq:           u64,
    pub captured_at:   std::time::Instant,
}

pub type SharedFrame = Arc<Mutex<Option<FrameData>>>;

/// Returned by `start_capture`. Contains the live frame handle plus metadata
/// about the actual capture geometry (so callers know how to map cursor coords).
pub struct CaptureHandle {
    pub frame:    SharedFrame,
    /// Actual capture width in pixels (= display native width, or region w).
    pub width:    usize,
    /// Actual capture height in pixels.
    pub height:   usize,
    /// Top-left of the captured region in display pixels (0,0 for FullDisplay).
    pub origin_x: u32,
    pub origin_y: u32,
}

// ── SCStreamOutput ObjC delegate ────────────────────────────────────────────

extern "C" fn sample_callback(
    this: &mut Object,
    _sel: Sel,
    _stream: *mut Object,
    sample_buffer: *mut c_void,
    output_type: isize,
) {
    if output_type != 0 { return; } // SCStreamOutputTypeScreen = 0

    unsafe {
        let pb = CMSampleBufferGetImageBuffer(sample_buffer);
        if pb.is_null() { return; }

        CVPixelBufferLockBaseAddress(pb, 1); // kCVPixelBufferLock_ReadOnly

        let base = CVPixelBufferGetBaseAddress(pb);
        let w    = CVPixelBufferGetWidth(pb);
        let h    = CVPixelBufferGetHeight(pb);
        let bpr  = CVPixelBufferGetBytesPerRow(pb);
        let len  = h * bpr;

        if !base.is_null() && len > 0 {
            let ptr: usize = *this.get_ivar("frame_ptr");
            let mtx = &*(ptr as *const Mutex<Option<FrameData>>);
            // try_lock: prefer dropping this frame over blocking SCK's queue
            if let Ok(mut guard) = mtx.try_lock() {
                let now = std::time::Instant::now();
                let frame = guard.get_or_insert_with(|| FrameData {
                    pixels: Vec::new(), width: 0, height: 0,
                    bytes_per_row: 0, seq: 0, captured_at: now,
                });
                if frame.pixels.len() != len { frame.pixels.resize(len, 0); }
                std::ptr::copy_nonoverlapping(base as *const u8, frame.pixels.as_mut_ptr(), len);
                frame.width        = w;
                frame.height       = h;
                frame.bytes_per_row = bpr;
                frame.seq          += 1;
                frame.captured_at   = now;
            }
        }

        CVPixelBufferUnlockBaseAddress(pb, 1);
    }
}

fn register_output_class() -> &'static objc::runtime::Class {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        let mut decl = ClassDecl::new("RustSCStreamOutput", class!(NSObject)).unwrap();
        decl.add_ivar::<usize>("frame_ptr");
        unsafe {
            decl.add_method(
                sel!(stream:didOutputSampleBuffer:ofType:),
                sample_callback as extern "C" fn(&mut Object, Sel, *mut Object, *mut c_void, isize),
            );
        }
        decl.register();
    });
    objc::runtime::Class::get("RustSCStreamOutput").unwrap()
}

// ── Display geometry helpers ─────────────────────────────────────────────────

/// Native pixel dimensions of the primary display via NSScreen.
/// NSScreen.frame is in points; multiply by backingScaleFactor → pixels.
pub fn main_display_pixels() -> (usize, usize) {
    autoreleasepool(|| unsafe {
        let screen: *mut Object = msg_send![class!(NSScreen), mainScreen];
        if screen.is_null() { return (2560, 1440); }
        let frame: CGRect = msg_send![screen, frame];
        let scale: f64    = msg_send![screen, backingScaleFactor];
        (
            (frame.size.width  * scale).round() as usize,
            (frame.size.height * scale).round() as usize,
        )
    })
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Start a ScreenCaptureKit stream. Returns a `CaptureHandle` containing the
/// live SharedFrame and the geometry of what was captured.
///
/// Capture size = native resolution of the chosen source (no SCK downscaling).
/// Callers are responsible for downscaling to their preview panel size.
pub fn start_capture(source: CaptureSource, target_fps: u32) -> CaptureHandle {
    let (capture_w, capture_h, origin_x, origin_y) = match &source {
        CaptureSource::FullDisplay => {
            let (w, h) = main_display_pixels();
            (w, h, 0u32, 0u32)
        }
        CaptureSource::Region { x, y, w, h } => (*w as usize, *h as usize, *x, *y),
    };

    let shared: SharedFrame = Arc::new(Mutex::new(None));
    let shared_bg = shared.clone();

    std::thread::spawn(move || {
        autoreleasepool(|| unsafe {
            setup_stream(shared_bg, source, capture_w, capture_h, target_fps);
        });
    });

    // Wait up to 3 s for first frame
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    while std::time::Instant::now() < deadline {
        if shared.lock().unwrap().is_some() { break; }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    CaptureHandle { frame: shared, width: capture_w, height: capture_h, origin_x, origin_y }
}

unsafe fn setup_stream(
    shared: SharedFrame,
    source: CaptureSource,
    output_w: usize,
    output_h: usize,
    target_fps: u32,
) {
    // Scale factor: converts pixel coords to display points for sourceRect
    let main_screen: *mut Object = msg_send![class!(NSScreen), mainScreen];
    let scale: f64 = if main_screen.is_null() { 1.0 } else {
        msg_send![main_screen, backingScaleFactor]
    };

    // ── SCShareableContent ────────────────────────────────────────────────
    let sema  = dispatch_semaphore_create(0);
    let saddr = sema as usize;
    let catom: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
    let catom2 = catom.clone();

    let sct_block = ConcreteBlock::new(move |content: *mut Object, _err: *mut Object| {
        if !content.is_null() {
            let _: () = msg_send![content, retain];
            catom2.store(content as usize, Ordering::SeqCst);
        }
        dispatch_semaphore_signal(saddr as *mut c_void);
    });
    let sct_block = sct_block.copy();
    let _: () = msg_send![
        class!(SCShareableContent),
        getShareableContentWithCompletionHandler: &*sct_block
    ];
    dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);

    let content = catom.load(Ordering::SeqCst) as *mut Object;
    assert!(!content.is_null(), "SCShareableContent nil — grant Screen Recording permission");

    // ── Pick the main display (the one whose dimensions we sized for) ──────
    // Capture dimensions come from main_display_pixels() == NSScreen.mainScreen.
    // The SCDisplay we filter on MUST be that same display, else we'd capture a
    // different monitor at the wrong size. `displays firstObject` is order-
    // dependent and not guaranteed to be the main display — match by displayID.
    let displays: *mut Object = msg_send![content, displays];
    let dcount:   usize       = msg_send![displays, count];
    let main_id: u32 = CGMainDisplayID();
    let mut display: *mut Object = std::ptr::null_mut();
    for i in 0..dcount {
        let d: *mut Object = msg_send![displays, objectAtIndex: i];
        if d.is_null() { continue; }
        let did: u32 = msg_send![d, displayID];
        if did == main_id { display = d; break; }
    }
    if display.is_null() {
        display = msg_send![displays, firstObject];
        eprintln!("[capture] WARNING: main display {main_id} not in SCShareableContent; \
                   falling back to first display");
    }
    assert!(!display.is_null(), "no SCDisplay");

    // Multi-display warning: this is a single-display capture. Cursor/mouse
    // events on other displays land OUTSIDE the captured frame and will not be
    // pixel-aligned to masks (the ledger flags these as out-of-bounds). Work on
    // the captured display only, or disconnect the others while recording.
    if dcount > 1 {
        let cap_id: u32 = msg_send![display, displayID];
        eprintln!("[capture] WARNING: {dcount} displays present — capturing only \
                   display {cap_id} ({output_w}x{output_h}px). Mouse activity on \
                   other displays will be OUT OF FRAME and unusable for training.");
    }

    // ── Content filter ────────────────────────────────────────────────────
    let filter: *mut Object = {
        let alloc: *mut Object = msg_send![class!(SCContentFilter), alloc];
        let empty: *mut Object = msg_send![class!(NSArray), array];
        msg_send![alloc, initWithDisplay: display excludingWindows: empty]
    };

    // ── Stream configuration ──────────────────────────────────────────────
    let config: *mut Object = {
        let alloc: *mut Object = msg_send![class!(SCStreamConfiguration), alloc];
        let cfg:   *mut Object = msg_send![alloc, init];

        let _: () = msg_send![cfg, setWidth:  output_w];
        let _: () = msg_send![cfg, setHeight: output_h];
        let _: () = msg_send![cfg, setPixelFormat: BGRA_PIXEL_FORMAT];

        let interval = CMTime { value: 1, timescale: target_fps as i32, flags: 1, epoch: 0 };
        let _: () = msg_send![cfg, setMinimumFrameInterval: interval];
        let _: () = msg_send![cfg, setQueueDepth: 3usize];

        // sourceRect only for Region mode; FullDisplay captures the whole display
        if let CaptureSource::Region { x, y, w, h } = &source {
            let src = CGRect {
                origin: CGPoint { x: *x as f64 / scale, y: *y as f64 / scale },
                size:   CGSize  { width: *w as f64 / scale, height: *h as f64 / scale },
            };
            let _: () = msg_send![cfg, setSourceRect: src];
        }

        // Don't bake cursor — compositor overlays its own (setShowsCursor: macOS 13+)
        let sel_sc = objc::runtime::Sel::register("setShowsCursor:");
        let responds: bool = msg_send![cfg, respondsToSelector: sel_sc];
        if responds {
            let _: () = msg_send![cfg, setShowsCursor: 0u8];
        }

        cfg
    };

    // ── Delegate ──────────────────────────────────────────────────────────
    let cls      = register_output_class();
    let delegate: *mut Object = msg_send![cls, alloc];
    let delegate: *mut Object = msg_send![delegate, init];
    (*delegate).set_ivar("frame_ptr", Arc::as_ptr(&shared) as usize);

    // ── SCStream ──────────────────────────────────────────────────────────
    let stream: *mut Object = {
        let alloc: *mut Object = msg_send![class!(SCStream), alloc];
        msg_send![alloc, initWithFilter: filter
                         configuration: config
                              delegate: std::ptr::null::<Object>()]
    };

    let bg_queue = dispatch_get_global_queue(0, 0);
    let mut add_err: *mut Object = std::ptr::null_mut();
    let ok: bool = msg_send![
        stream,
        addStreamOutput: delegate
        type: 0isize
        sampleHandlerQueue: bg_queue
        error: &mut add_err
    ];
    if !ok {
        let desc: *mut Object = msg_send![add_err, localizedDescription];
        let s: *const i8      = msg_send![desc, UTF8String];
        panic!("addStreamOutput: {}", std::ffi::CStr::from_ptr(s).to_str().unwrap_or("?"));
    }

    // ── Start capture ─────────────────────────────────────────────────────
    let start_sema = dispatch_semaphore_create(0);
    let saddr2     = start_sema as usize;
    let start_block = ConcreteBlock::new(move |_err: *mut Object| {
        dispatch_semaphore_signal(saddr2 as *mut c_void);
    });
    let start_block = start_block.copy();
    let _: () = msg_send![stream, startCaptureWithCompletionHandler: &*start_block];
    dispatch_semaphore_wait(start_sema, DISPATCH_TIME_FOREVER);

    let _keep = shared;
    loop { std::thread::sleep(std::time::Duration::from_secs(3600)); }
}
