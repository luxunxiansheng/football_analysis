from dataclasses import dataclass
from typing import Dict, List, Optional, Union

@dataclass
class TrackManagerConfig:
    """Configuration for the track manager."""
    
    # Model settings
    confidence_threshold: float = 0.5
    model_path: str = "/workspaces/football_analysis/models/best.pt"

    # Video source and output settings
    input_video_path: str = "/workspaces/football_analysis/input_videos/08fd33_4.mp4"
    output_video_path: str = "/workspaces/football_analysis/output_videos/output_video.mp4"

    # Analysis parameters
    fps: float = 25.0
    resolution: tuple[int, int] = (1280, 720)
    max_tracks: int = 100
    
    # Detection and tracking settings
    detection_threshold: float = 0.5
    iou_threshold: float = 0.5
    min_track_length: int = 10
    max_track_age: int = 30
    
    # Player identification
    team_colors: Optional[Dict[str, List[tuple[int, int, int]]]] = None
    
    # Player tracking settings
    max_player_ball_distance: int = 70

    # Processing options
    use_gpu: bool = True
    batch_size: int = 4
    num_workers: int = 4
    
    # Visualization options
    show_visualization: bool = False
    save_visualization: bool = True
    visualization_fps: float = 15.0