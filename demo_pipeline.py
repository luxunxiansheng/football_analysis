"""
Simple Demo: Football Analysis Pipeline

This demonstrates the new main pipeline class in action.
"""

# Test the new pipeline
if __name__ == "__main__":
    from football_ai import FootballAnalysisPipeline, get_default_config

    # Create configuration
    config = get_default_config()
    config.show_progress_bars = True  # Enable progress bars
    config.update_paths(
        model_path="models/detect/best.pt",
        input_video_path="input_videos/08fd33_4.mp4",
        output_video_path="outputs/videos/pipeline_demo.mp4",
    )

    # Initialize pipeline
    pipeline = FootballAnalysisPipeline(config=config)

    print("🏈 Football Analysis Pipeline Demo")
    print("=" * 40)
    print(f"Configuration: {config.get_summary()}")
    print()

    try:
        # Process video
        results = pipeline.process_video(
            video_path=config.processing.input_video_path,
            output_path=config.processing.output_video_path,
        )

        # Get analysis summary
        summary = pipeline.get_analysis_summary(results)

        print("✅ Analysis Complete!")
        print("=" * 40)
        print(f"Frames processed: {summary['video_info']['frames_processed']}")
        print(f"Unique tracks: {summary['tracking_summary']['unique_tracks']}")
        print(f"Detections: {summary['detection_summary']}")
        print(f"Teams: {summary['team_summary']}")

        # Save results
        pipeline.save_results(results, "outputs/data/pipeline_results.pkl")
        print(f"📊 Results saved to: outputs/data/pipeline_results.pkl")

    except Exception as e:
        print(f"❌ Error: {e}")
