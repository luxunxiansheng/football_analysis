#!/usr/bin/env python3

import sys

sys.path.insert(0, ".")

try:
    from football_ai.sources.video.video_pipeline import VideoPipeline

    print("Testing VideoPipeline constructor...")
    pipeline = VideoPipeline(
        model_path="models/detect/best.pt",
        confidence_threshold=0.5,
        iou_threshold=0.4,
        device="cpu",
    )
    print("✓ VideoPipeline created successfully")

except Exception as e:
    print(f"✗ CAUGHT API MISMATCH: {e}")
    import traceback

    traceback.print_exc()
