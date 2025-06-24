# Domain models and interfaces

# Import all domain classes and make them available
from .player import Player
from .referee import Referee
from .goalkeeper import Goalkeeper
from .ball import Ball
from .field import Field
from .frame import Frame, CameraMotion, ProcessingStatus
from .video import Video, VideoMetadata, MatchContext, ProcessingConfig
from .game import Game, Team, MatchEvent, AnalysisSource, MatchType, MatchStatus
from .constants import ObjectType
from .stats import TeamStats, MatchStats

# Backward compatibility aliases
VideoData = Video
FrameData = Frame

__all__ = [
    # Core game-centric classes
    "Game",
    "Team",
    "MatchEvent",
    "AnalysisSource",
    "MatchType",
    "MatchStatus",
    # Domain object classes
    "Player",
    "Referee",
    "Goalkeeper",
    "Ball",
    "Field",
    "Frame",
    "Video",
    "CameraMotion",
    "ProcessingStatus",
    "VideoMetadata",
    "MatchContext",
    "ProcessingConfig",
    "ObjectType",
    "TeamStats",
    "MatchStats",
    # Legacy aliases
    "VideoData",
    "FrameData",
]
