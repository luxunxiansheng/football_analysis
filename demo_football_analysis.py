"""
Football AI - Game-Centric Analysis Demo

Demonstrates modern football analysis using the game-centric architecture.
Shows how to analyze matches through multiple sources with Games as central entities.
"""

if __name__ == "__main__":
    from football_ai.sources.video.video_analysis_factory import (
        create_demo_processor,
        create_high_accuracy_processor,
        create_fast_processor,
    )
    from football_ai.game.game_factory import GameFactory

    print("🏈 Football AI - Game-Centric Analysis Demo")
    print("=" * 50)
    print("Modern Architecture: Game → Sources → Comprehensive Analysis")
    print("✨ Now using explicit parameters instead of config objects!")
    print()

    # Input parameters (no more config objects!)
    model_path = "models/detect/best.pt"
    input_video = "input_videos/08fd33_4.mp4"
    output_video = "outputs/videos/game_demo.mp4"

    try:
        # Method 1: Demo processor with explicit parameters
        print("📹 Method 1: Demo Video Analysis (Explicit Parameters)")
        video_processor = create_demo_processor(
            model_path=model_path,
            input_video_path=input_video,
            output_video_path=output_video,
        )

        game = video_processor.create_game_from_video(
            video_path=input_video,
            home_team="Home Team",
            away_team="Away Team",
            competition="Demo League",
            venue="Demo Stadium",
        )

        print(f"✅ Game created: {game.game_id}")
        print(f"   Home: {game.home_team.name}")
        print(f"   Away: {game.away_team.name}")
        print(f"   Analysis sources: {len(game.analysis_sources)}")
        print()

        # Get analysis summary
        summary = video_processor.get_analysis_summary(game)
        print("📊 Analysis Summary:")
        print("=" * 30)
        print(f"Players tracked: {summary['analysis_summary']['players_tracked']}")
        print(f"Events detected: {summary['analysis_summary']['total_events']}")
        print(
            f"Ball tracking points: {summary['analysis_summary']['ball_tracking_points']}"
        )
        print(
            f"Video duration: {summary['video_summary'].get('duration', 'N/A')} seconds"
        )
        print()

        # Method 2: High accuracy processor example
        print("📹 Method 2: High Accuracy Analysis (Explicit Parameters)")

        # Create a high-accuracy processor with explicit parameters
        high_accuracy_processor = create_high_accuracy_processor(
            model_path=model_path,
            team_model_path="models/embed/siglip-base-patch16-224",
        )

        # Create a new game using GameFactory
        game2 = GameFactory.create_from_video(
            video_path=input_video,
            home_team_name="Team A",
            away_team_name="Team B",
            competition="Demo Cup",
            venue="Demo Arena",
        )

        print(f"✅ Game created: {game2.game_id}")

        # Now analyze the video for this game
        game2 = high_accuracy_processor.analyze_video_for_game(
            game=game2,
            video_path=input_video,
            output_path="outputs/videos/game2_analysis.mp4",
        )

        print(f"✅ Video analysis completed for {game2.game_id}")
        print(f"   Analysis sources: {len(game2.analysis_sources)}")
        print()

        # Show game statistics
        print("🎯 Game Statistics:")
        print("=" * 30)
        print(f"Game 1 - Players: {len(game.get_all_players())}")
        print(f"Game 1 - Events: {len(game.events)}")
        print(f"Game 2 - Players: {len(game2.get_all_players())}")
        print(f"Game 2 - Events: {len(game2.events)}")
        print()

        # Show possession analysis (if available)
        try:
            possession = game.calculate_possession()
            print("⚽ Ball Possession:")
            for team_id, percentage in possession.items():
                team = game.get_team_by_id(team_id)
                team_name = team.name if team else f"Team {team_id}"
                print(f"   {team_name}: {percentage:.1f}%")
        except Exception as e:
            print(f"   Possession analysis not available: {e}")

        print()
        print("🎉 Demo completed successfully!")
        print("✨ Benefits of explicit parameters:")
        print("• Clear interface - you see exactly what each component needs")
        print("• Easy testing - just pass the parameters you want to test")
        print("• No hidden dependencies in massive config objects")
        print("• IDE auto-completion shows available options")
        print("• Flexible - mix and match parameters without config overhead")
        print()
        print("🏗️ Architecture highlights:")
        print("• Games as central entities")
        print("• Multiple analysis sources (video, GPS, manual, etc.)")
        print("• Flexible integration of different data types")
        print("• Better separation of concerns")
        print("• Professional-grade explicit interfaces")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
