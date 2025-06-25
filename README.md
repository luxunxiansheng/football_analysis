# Football AI - Game-Centric Analysis System

## Overview

Football AI is a modern, extensible football analysis system built around a **game-centric architecture**. The system treats each football match as a central `Game` entity that can be analyzed through multiple sources including video, GPS data, manual annotations, and more.

## 🎯 Key Features

- **Game-Centric Design**: All analysis revolves around `Game` objects representing actual matches
- **Multi-Source Analysis**: Video, GPS, manual data, and future sources all contribute to games
- **Advanced Video Analysis**: State-of-the-art object detection, tracking, and team assignment
- **Real-time Capabilities**: Optimized for both offline analysis and live match processing
- **Extensible Architecture**: Easy to add new analysis sources and capabilities
- **Professional Quality**: Broadcast-ready analysis with customizable accuracy/speed trade-offs

## 🚀 Quick Start

### Basic Game Analysis

```python
from football_ai import VideoAnalysisProcessor, GameManager

# Create video processor
processor = VideoAnalysisProcessor()

# Analyze match video and create game
game = processor.create_game_from_video(
    video_path="match.mp4",
    home_team="Barcelona",
    away_team="Real Madrid"
)

# Get match insights
possession = game.calculate_possession()
summary = game.get_match_summary()

print(f"Home possession: {possession['home_percentage']:.1f}%")
print(f"Players tracked: {summary['total_players']}")
```

### Game Management

```python
from football_ai import GameManager

# Manage multiple games
manager = GameManager()

# Create games from different sources
game1 = manager.create_game_from_video("match1.mp4", "Team A", "Team B")
game2 = manager.create_game_from_info({
    "home_team": "Liverpool", 
    "away_team": "Arsenal",
    "date": "2025-06-24"
})

# Comprehensive analysis
analysis = manager.analyze_game(
    game1.game_id, 
    analysis_types=["possession", "heatmaps", "formations"]
)

# Generate reports
report = manager.get_game_report(game1.game_id)
```

## 🔧 Installation & Setup

### Requirements
- Python 3.8+
- CUDA-capable GPU (optional, for acceleration)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd football_analysis

# Install dependencies
pip install -r requirements.txt

# Download models (see documentation for model sources)
# - Place detection models in models/detect/
# - Place team assignment models in models/embed/
# - Place pose models in models/pose/
```

### Quick Test

```bash
python demo_football_analysis.py
```

## ⚙️ Configuration

The system uses a comprehensive configuration system optimized for different use cases:

```python
from football_ai import (
    get_default_config,
    get_high_accuracy_config,
    get_fast_processing_config,
    get_broadcast_config
)

# Balanced performance and accuracy
config = get_default_config()

# Maximum accuracy for detailed analysis
config = get_high_accuracy_config()

# Fast processing for live analysis
config = get_fast_processing_config()

# Broadcast-quality professional analysis
config = get_broadcast_config()
```

## 📈 Performance Profiles

| Configuration | Speed | Accuracy | Use Case |
|---------------|-------|----------|----------|
| Fast Processing | 2-3x real-time | Good | Live analysis, quick insights |
| Default | 1x real-time | High | Standard match analysis |
| High Accuracy | 0.5x real-time | Excellent | Detailed post-match analysis |
| Broadcast | 0.3x real-time | Maximum | Professional broadcast analysis |

## 🎯 Use Cases

### Match Analysis
```python
# Professional match analysis
processor = VideoAnalysisProcessor(get_broadcast_config())
game = processor.create_game_from_video("champions_league.mp4", "Team A", "Team B")

# Generate comprehensive report
manager = GameManager()
manager.games[game.game_id] = game
report = manager.get_game_report(game.game_id)
```

### Training Analysis
```python
# Detailed training session analysis
processor = VideoAnalysisProcessor(get_high_accuracy_config())
game = processor.create_game_from_video("training.mp4", "First Team", "Reserves")

# Focus on individual player performance
heatmaps = game.get_player_heatmap_data("player_7")
speeds = game.get_player_speed_data("player_7")
```

## 🧪 Testing

```bash
# Complete test suite
python -m pytest tests/

# Specific components
python -m pytest tests/test_game_architecture.py
python -m pytest tests/test_pipeline_integration.py
```

## 📚 Documentation

For comprehensive documentation, see the [`doc/`](doc/) folder:

- **[Technical Specification](doc/TECHNICAL_SPECIFICATION.md)** - Complete system architecture
- **[Analytics Guide](doc/ANALYTICS_PURPOSE.md)** - Understanding the analytics package
- **[File Naming Guide](doc/FILE_NAMING_GUIDE.md)** - Code organization standards
- **[API Documentation](doc/)** - Detailed module documentation

## 🤝 Contributing

We welcome contributions! The game-centric architecture makes it easy to:

- Add new analysis sources (GPS, wearables, manual annotation)
- Implement custom analytics and insights
- Optimize processing pipelines
- Extend domain models

See the documentation in [`doc/`](doc/) for detailed guidelines.

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

Built with cutting-edge technologies:
- **YOLO** for object detection
- **ByteTrack** for multi-object tracking
- **SigLIP** for team assignment
- **OpenCV** for video processing

---

*Football AI - Transforming match analysis through intelligent, game-centric design.*
