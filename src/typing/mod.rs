mod diff;
mod ring;
mod tap;
mod detector;

pub use detector::{
    TypingArgs, TypingDetectorResult,
    start_typing_detector,
};
pub use ring::{FrameRingBuffer, RingEntry};
