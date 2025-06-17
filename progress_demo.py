#!/usr/bin/env python3
"""
Progress Bar Demo for Football Analysis Pipeline

This script demonstrates the new progress bars added to the pipeline.
"""

if __name__ == "__main__":
    from football_ai import FootballAnalysisPipeline, get_default_config
    import os

    print("🏈 Football Analysis Pipeline - Progress Bar Demo")
    print("=" * 60)

    # Create configuration with progress bars enabled
    config = get_default_config()
    config.show_progress_bars = True  # Enable progress bars

    # Update paths
    config.update_paths(
        model_path="models/detect/best.pt",
        input_video_path="input_videos/08fd33_4.mp4",
        output_video_path="outputs/videos/progress_demo.mp4",
    )

    # Verify files exist
    if not os.path.exists(config.model.player_model_path):
        print(f"❌ Model not found: {config.model.player_model_path}")
        exit(1)

    if not os.path.exists(config.processing.input_video_path):
        print(f"❌ Video not found: {config.processing.input_video_path}")
        exit(1)

    # Initialize pipeline
    pipeline = FootballAnalysisPipeline(config=config)

    print(f"🚀 Pipeline ready with {len(pipeline.processors)} processors")
    print("📊 Progress bars enabled - you'll see detailed progress for:")
    print("   • Video loading")
    print("   • Object detection")
    print("   • Object tracking")
    print("   • Speed analysis")
    print("   • Team assignment")
    print("   • Frame rendering")
    print("   • Video writing")
    print()

    try:
        # Process video with progress bars
        print("🎬 Starting video analysis with progress bars...")
        results = pipeline.process_video(
            video_path=config.processing.input_video_path,
            output_path=config.processing.output_video_path,
        )

        # Get analysis summary
        summary = pipeline.get_analysis_summary(results)

        print()
        print("✅ Analysis Complete!")
        print("=" * 60)
        print(f"📹 Frames processed: {summary['video_info']['frames_processed']}")
        print(f"🎯 Unique tracks: {summary['tracking_summary']['unique_tracks']}")
        print(f"🔢 Detections: {summary['detection_summary']}")
        print(f"👥 Teams: {summary['team_summary']}")
        print(f"🎬 Output video: {config.processing.output_video_path}")

        # Save results
        pipeline.save_results(results, "outputs/data/progress_demo_results.pkl")
        print(f"📊 Results saved to: outputs/data/progress_demo_results.pkl")

    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n🎉 Progress bar demo complete!")
