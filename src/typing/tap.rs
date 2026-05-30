use std::ffi::c_void;
use std::sync::mpsc::Sender;
use std::time::Instant;

#[repr(C)]
#[derive(Copy, Clone)]
struct CGPoint { x: f64, y: f64 }

pub enum TapEvent {
    KeyDown   { at: Instant },
    MouseDown { at: Instant, x_pts: f64, y_pts: f64 },
}

// kCGHIDEventTap = 0, kCGHeadInsertEventTap = 0, kCGEventTapOptionListenOnly = 1
const CG_HID_EVENT_TAP:              u32 = 0;
const CG_HEAD_INSERT_EVENT_TAP:      u32 = 0;
const CG_EVENT_TAP_LISTEN_ONLY:      u32 = 1;
const CG_EVENT_KEY_DOWN:             u32 = 10;
const CG_EVENT_LEFT_MOUSE_DOWN:      u32 = 1;

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

struct TapCtx {
    tx: Sender<TapEvent>,
}

unsafe extern "C" fn tap_callback(
    _proxy:     *mut c_void,
    event_type: u32,
    event:      *mut c_void,
    user_info:  *mut c_void,
) -> *mut c_void {
    let ctx = &*(user_info as *const TapCtx);
    let now = Instant::now();
    match event_type {
        CG_EVENT_KEY_DOWN => {
            let _ = ctx.tx.send(TapEvent::KeyDown { at: now });
        }
        CG_EVENT_LEFT_MOUSE_DOWN => {
            let pt = CGEventGetLocation(event);
            let _ = ctx.tx.send(TapEvent::MouseDown { at: now, x_pts: pt.x, y_pts: pt.y });
        }
        _ => {}
    }
    event
}

pub fn start_key_tap(tx: Sender<TapEvent>) {
    std::thread::spawn(move || {
        unsafe {
            let mask = (1u64 << CG_EVENT_KEY_DOWN) | (1u64 << CG_EVENT_LEFT_MOUSE_DOWN);
            let ctx  = Box::into_raw(Box::new(TapCtx { tx }));

            let tap = CGEventTapCreate(
                CG_HID_EVENT_TAP,
                CG_HEAD_INSERT_EVENT_TAP,
                CG_EVENT_TAP_LISTEN_ONLY,
                mask,
                tap_callback,
                ctx as *mut c_void,
            );
            if tap.is_null() {
                eprintln!("[typing] CGEventTapCreate failed — grant Input Monitoring in System Settings → Privacy & Security → Input Monitoring");
                return;
            }

            let source = CFMachPortCreateRunLoopSource(std::ptr::null(), tap, 0);
            let rl     = CFRunLoopGetCurrent();
            CFRunLoopAddSource(rl, source, kCFRunLoopCommonModes);
            CFRelease(source as *const c_void);
            CFRunLoopRun();
        }
    });
}
