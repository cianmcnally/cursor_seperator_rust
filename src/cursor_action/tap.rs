use std::ffi::c_void;
use std::sync::mpsc::Sender;
use std::time::Instant;

#[repr(C)]
#[derive(Copy, Clone)]
struct CGPoint { x: f64, y: f64 }

#[derive(Debug, Clone)]
pub enum MouseTapEvent {
    LeftMouseDown { at: Instant, ts_ns: u64, x_pts: f64, y_pts: f64 },
    LeftMouseUp   { at: Instant, ts_ns: u64, x_pts: f64, y_pts: f64 },
    LeftMouseDrag { at: Instant, ts_ns: u64, x_pts: f64, y_pts: f64 },
    MouseMoved    { at: Instant, ts_ns: u64, x_pts: f64, y_pts: f64 },
    ScrollWheel   { at: Instant, ts_ns: u64, x_pts: f64, y_pts: f64, delta_y: i64 },
}

const CG_HID_EVENT_TAP:            u32 = 0;
const CG_HEAD_INSERT_EVENT_TAP:    u32 = 0;
const CG_EVENT_TAP_LISTEN_ONLY:    u32 = 1;
const CG_EVENT_LEFT_MOUSE_DOWN:    u32 = 1;
const CG_EVENT_LEFT_MOUSE_UP:      u32 = 2;
const CG_EVENT_MOUSE_MOVED:        u32 = 5;
const CG_EVENT_LEFT_MOUSE_DRAGGED: u32 = 6;
const CG_EVENT_SCROLL_WHEEL:       u32 = 22;
const CG_SCROLL_DELTA_AXIS1:       u32 = 11;

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGEventTapCreate(
        tap:                u32,
        place:              u32,
        options:            u32,
        events_of_interest: u64,
        callback: unsafe extern "C" fn(*mut c_void, u32, *mut c_void, *mut c_void) -> *mut c_void,
        user_info:          *mut c_void,
    ) -> *mut c_void;
    fn CGEventGetLocation(event: *mut c_void) -> CGPoint;
    fn CGEventGetIntegerValueField(event: *mut c_void, field: u32) -> i64;
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    static kCFRunLoopCommonModes: *const c_void;
    fn CFMachPortCreateRunLoopSource(
        allocator: *const c_void,
        port:      *mut c_void,
        order:     isize,
    ) -> *mut c_void;
    fn CFRunLoopGetCurrent() -> *mut c_void;
    fn CFRunLoopAddSource(rl: *mut c_void, source: *mut c_void, mode: *const c_void);
    fn CFRunLoopRun();
    fn CFRelease(cf: *const c_void);
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

struct TapCtx { tx: Sender<MouseTapEvent> }

unsafe extern "C" fn tap_callback(
    _proxy:     *mut c_void,
    event_type: u32,
    event:      *mut c_void,
    user_info:  *mut c_void,
) -> *mut c_void {
    let ctx = &*(user_info as *const TapCtx);
    let now = Instant::now();
    let ts  = now_ns();
    let pt  = CGEventGetLocation(event);

    let ev = match event_type {
        CG_EVENT_LEFT_MOUSE_DOWN => Some(MouseTapEvent::LeftMouseDown {
            at: now, ts_ns: ts, x_pts: pt.x, y_pts: pt.y,
        }),
        CG_EVENT_LEFT_MOUSE_UP => Some(MouseTapEvent::LeftMouseUp {
            at: now, ts_ns: ts, x_pts: pt.x, y_pts: pt.y,
        }),
        CG_EVENT_MOUSE_MOVED => Some(MouseTapEvent::MouseMoved {
            at: now, ts_ns: ts, x_pts: pt.x, y_pts: pt.y,
        }),
        CG_EVENT_LEFT_MOUSE_DRAGGED => Some(MouseTapEvent::LeftMouseDrag {
            at: now, ts_ns: ts, x_pts: pt.x, y_pts: pt.y,
        }),
        CG_EVENT_SCROLL_WHEEL => {
            let delta_y = CGEventGetIntegerValueField(event, CG_SCROLL_DELTA_AXIS1);
            Some(MouseTapEvent::ScrollWheel {
                at: now, ts_ns: ts, x_pts: pt.x, y_pts: pt.y, delta_y,
            })
        }
        _ => None,
    };

    if let Some(ev) = ev {
        let _ = ctx.tx.send(ev);
    }
    event
}

pub fn start_mouse_tap(tx: Sender<MouseTapEvent>) {
    std::thread::spawn(move || unsafe {
        let mask = (1u64 << CG_EVENT_LEFT_MOUSE_DOWN)
            | (1u64 << CG_EVENT_LEFT_MOUSE_UP)
            | (1u64 << CG_EVENT_MOUSE_MOVED)
            | (1u64 << CG_EVENT_LEFT_MOUSE_DRAGGED)
            | (1u64 << CG_EVENT_SCROLL_WHEEL);

        let ctx = Box::into_raw(Box::new(TapCtx { tx }));
        let tap = CGEventTapCreate(
            CG_HID_EVENT_TAP, CG_HEAD_INSERT_EVENT_TAP, CG_EVENT_TAP_LISTEN_ONLY,
            mask, tap_callback, ctx as *mut c_void,
        );
        if tap.is_null() {
            eprintln!("[cursor_action] CGEventTapCreate failed — grant Input Monitoring in System Settings → Privacy & Security → Input Monitoring");
            return;
        }
        let source = CFMachPortCreateRunLoopSource(std::ptr::null(), tap, 0);
        let rl     = CFRunLoopGetCurrent();
        CFRunLoopAddSource(rl, source, kCFRunLoopCommonModes);
        CFRelease(source as *const c_void);
        CFRunLoopRun();
    });
}
