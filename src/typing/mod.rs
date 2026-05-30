mod diff;
mod ring;
mod tap;
mod detector;

pub use detector::{
    InteractionRegion, SharedTypingState, TypingArgs, TypingDetectorResult,
    start_typing_detector,
};
pub use ring::{FrameRingBuffer, RingEntry};
