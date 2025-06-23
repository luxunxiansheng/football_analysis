#!/usr/bin/env python3

import sys
import numpy as np
from football_ai.domain.data_models import FrameData, VideoData, CameraMotion
from football_ai.motion.camera_motion_processor import CameraMotionProcessor


def test_camera_motion():
    print("🧪 Testing CameraMotionProcessor...")

    # Create test frames with simple patterns
    frames = []
    for i in range(3):
        # Create a simple test image
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add some features to track
        if i > 0:
            frame[40:60, 40:60] = 255  # White square that might move

        frame_data = FrameData(
            frame_number=i,
            timestamp=i * 0.033,
            raw_frame=frame,
        )
        frames.append(frame_data)

    video_data = VideoData(
        video_path="test.mp4",
        frame_rate=30.0,
        resolution=(100, 100),
        duration=0.1,
        frames=frames,
    )

    # Test the processor
    processor = CameraMotionProcessor()

    try:
        result = processor.process(video_data)
        print("✅ CameraMotionProcessor completed successfully!")

        for i, frame in enumerate(result.frames):
            cm = frame.camera_motion
            print(f"  Frame {i}: x_offset={cm.x_offset}, y_offset={cm.y_offset}")

        return True

    except Exception as e:
        print(f"❌ CameraMotionProcessor failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_camera_motion()
    sys.exit(0 if success else 1)
