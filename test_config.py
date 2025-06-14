#!/usr/bin/env python3
"""
Test script for Football AI configuration system
"""

import sys
import os

sys.path.append("/workspaces/football_analysis")

from football_ai.config import (
    FootballAIConfig,
    get_default_config,
    get_high_accuracy_config,
    get_fast_processing_config,
    get_broadcast_config,
)
from football_ai.pipeline import FootballAnalysisPipeline


def test_configuration_system():
    """Test the configuration system functionality."""
    print("=== Testing Football AI Configuration System ===\n")

    # Test 1: Default configuration
    print("1. Testing default configuration...")
    default_config = get_default_config()
    assert default_config.model.confidence_threshold == 0.5
    assert default_config.tracking.track_threshold == 0.6
    print("✓ Default configuration works")

    # Test 2: Predefined configurations
    print("\n2. Testing predefined configurations...")
    high_acc = get_high_accuracy_config()
    fast_proc = get_fast_processing_config()
    broadcast = get_broadcast_config()

    assert (
        high_acc.model.confidence_threshold > default_config.model.confidence_threshold
    )
    assert fast_proc.processing.process_every_nth_frame == 2
    assert broadcast.rendering.show_tracks == True
    print("✓ Predefined configurations work")

    # Test 3: Custom configuration
    print("\n3. Testing custom configuration...")
    custom_config = FootballAIConfig()
    custom_config.model.confidence_threshold = 0.8
    custom_config.rendering.show_speeds = True
    custom_config.processing.enable_caching = False

    assert custom_config.model.confidence_threshold == 0.8
    assert custom_config.rendering.show_speeds == True
    assert custom_config.processing.enable_caching == False
    print("✓ Custom configuration works")

    # Test 4: Configuration update paths
    print("\n4. Testing configuration path updates...")
    custom_config.update_paths(
        model_path="test_model.pt",
        input_video_path="test_input.mp4",
        output_video_path="test_output.mp4",
        output_directory="test_output_dir",
    )

    assert custom_config.model.model_path == "test_model.pt"
    assert custom_config.processing.input_video_path == "test_input.mp4"
    assert custom_config.processing.output_video_path == "test_output.mp4"
    assert custom_config.processing.output_directory == "test_output_dir"
    print("✓ Path updates work")

    # Test 5: Configuration save/load
    print("\n5. Testing configuration save/load...")
    test_config_path = "/tmp/test_config.json"
    custom_config.save_to_file(test_config_path)

    loaded_config = FootballAIConfig.load_from_file(test_config_path)
    assert loaded_config.model.confidence_threshold == 0.8
    assert loaded_config.model.model_path == "test_model.pt"
    print("✓ Save/load works")

    # Test 6: Configuration validation
    print("\n6. Testing configuration validation...")
    invalid_config = FootballAIConfig()
    invalid_config.model.confidence_threshold = 1.5  # Invalid value
    invalid_config.model.model_path = "nonexistent_model.pt"

    issues = invalid_config.validate()
    assert len(issues) > 0
    print(f"✓ Validation found {len(issues)} issues as expected")

    # Test 7: Pipeline integration
    print("\n7. Testing pipeline integration...")

    # Test with configuration (using existing model)
    test_config = get_default_config()
    test_config.model.model_path = "/workspaces/football_analysis/models/best.pt"
    pipeline_with_config = FootballAnalysisPipeline(config=test_config)
    assert pipeline_with_config.config.model.confidence_threshold == 0.5

    # Test with legacy parameters (using existing model)
    pipeline_legacy = FootballAnalysisPipeline(
        model_path="/workspaces/football_analysis/models/best.pt",
        output_dir="test_output",
        save_cache=False,
    )
    assert (
        pipeline_legacy.config.model.model_path
        == "/workspaces/football_analysis/models/best.pt"
    )
    assert pipeline_legacy.config.processing.output_directory == "test_output"
    assert pipeline_legacy.config.processing.save_to_cache == False

    print("✓ Pipeline integration works")

    # Test 8: Configuration summary
    print("\n8. Testing configuration summary...")
    summary = default_config.get_summary()
    assert "Football AI Configuration Summary" in summary
    assert "Detection Confidence" in summary
    print("✓ Configuration summary works")

    print("\n=== All tests passed! ===")
    print("✅ Configuration system is working correctly")

    # Cleanup
    if os.path.exists(test_config_path):
        os.remove(test_config_path)


if __name__ == "__main__":
    test_configuration_system()
