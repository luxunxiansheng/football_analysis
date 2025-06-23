"""
Configuration factory for creating common configurations.
"""

from ..config import (
    FootballAIConfig,
    get_default_config,
    get_fast_processing_config,
    get_high_accuracy_config,
)
from typing import Optional


class ConfigFactory:
    """Factory for creating common configuration setups."""

    @staticmethod
    def create_test_config(
        model_path: Optional[str] = None,
        input_video_path: Optional[str] = None,
        output_video_path: Optional[str] = None,
        strict_mode: bool = False,
    ) -> FootballAIConfig:
        """
        Create a configuration optimized for testing.

        Args:
            model_path: Path to model file
            input_video_path: Path to input video
            output_video_path: Path to output video
            strict_mode: Whether to enable strict mode

        Returns:
            Test-optimized configuration
        """
        config = get_fast_processing_config()
        config.strict_mode = strict_mode
        config.show_progress_bars = False  # Disable for cleaner test output

        if model_path:
            config.model.player_model_path = model_path
        if input_video_path:
            config.processing.input_video_path = input_video_path
        if output_video_path:
            config.processing.output_video_path = output_video_path

        return config

    @staticmethod
    def create_demo_config(
        model_path: str = "models/detect/best.pt",
        input_video_path: str = "input_videos/08fd33_4.mp4",
        output_video_path: str = "outputs/videos/demo_output.mp4",
    ) -> FootballAIConfig:
        """
        Create a configuration optimized for demos.

        Args:
            model_path: Path to model file
            input_video_path: Path to input video
            output_video_path: Path to output video

        Returns:
            Demo-optimized configuration
        """
        config = get_default_config()
        config.show_progress_bars = True
        config.update_paths(
            model_path=model_path,
            input_video_path=input_video_path,
            output_video_path=output_video_path,
        )
        return config

    @staticmethod
    def create_evaluation_config(
        input_video_path: str, model_path: str = "models/detect/best.pt"
    ) -> FootballAIConfig:
        """
        Create a configuration optimized for evaluation/benchmarking.

        Args:
            input_video_path: Path to input video
            model_path: Path to model file

        Returns:
            Evaluation-optimized configuration
        """
        config = get_high_accuracy_config()
        config.model.player_model_path = model_path
        config.processing.input_video_path = input_video_path
        config.show_progress_bars = True
        config.debug_mode = True  # Enable detailed logging

        return config
