# Football AI System

## Introduction

This project provides a comprehensive football video analysis system built with modern software engineering practices. The system detects and tracks players, referees, and footballs using state-of-the-art AI models, assigns players to teams based on jersey colors, analyzes ball possession, tracks camera movement, and provides real-world measurements through perspective transformation.

The system has been completely rewritten with a clean, modular architecture that replaces the original notebook-based approach with maintainable, extensible, and production-ready code.

![Screenshot](output_videos/screenshot.png)

## Features

- **Object Detection & Tracking**: YOLO-based detection with ByteTrack multi-object tracking
- **Team Assignment**: K-means clustering for automatic team color detection and player assignment
- **Ball Possession Analysis**: Distance-based possession detection with temporal smoothing
- **Camera Motion Tracking**: Feature-based camera movement compensation
- **Coordinate Transformation**: Perspective transformation for real-world measurements
- **Speed & Distance Calculation**: Player movement analysis in meters and km/h
- **Professional Visualization**: High-quality video annotations and overlays

## Architecture

The modern system follows clean architecture principles with:

- **Domain Models**: Core data structures and business logic
- **Modular Components**: Separate modules for detection, tracking, analysis, etc.
- **Abstract Interfaces**: Clear contracts for extensibility
- **Type Safety**: Full type hints throughout the codebase
- **Error Handling**: Robust error recovery and logging
- **Performance Optimization**: Efficient processing pipeline with caching

## Quick Start

### Basic Usage

```python
from football_ai import FootballAnalysisPipeline

# Initialize the pipeline
pipeline = FootballAnalysisPipeline(
    model_path="models/best.pt",
    output_dir="output"
)

# Process a video
results = pipeline.process_video(
    video_path="input_videos/match.mp4",
    output_video_path="output_videos/annotated.mp4"
)

print(f"Players tracked: {len(results.player_tracks)}")
print(f"Team possession: {results.team_ball_control}")
```

### Using the Demo Notebook

1. Open `football_ai_demo.ipynb`
2. Update the configuration paths
3. Run all cells to see the complete analysis

## Requirements

- Python 3.8+
- ultralytics (YOLO)
- opencv-python
- numpy
- scikit-learn
- tqdm

Install all requirements:
```bash
pip install -r requirements.txt
```

## Project Structure

```
football_ai/                      # Football AI system
├── domain/                        # Core models and interfaces
├── detection/                     # Object detection
├── tracking/                      # Multi-object tracking  
├── analysis/                      # Team and possession analysis
├── motion/                        # Camera motion tracking
├── transformation/                # Coordinate transformation
├── rendering/                     # Video visualization
├── utils/                         # Utility functions
└── pipeline.py                    # Main orchestrator

notebook/                          # Original reference notebook
training/                          # Model training resources
models/                           # Trained YOLO models
input_videos/                     # Input video files
output_videos/                    # Generated output videos
```

## Models

- **Trained YOLO v11**: Included in `models/` directory
- **Original YOLO v5**: [Download](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)

## Sample Data

- **Sample Video**: [Download](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)

## Benefits of Modern System

| Aspect | Original | Modern |
|--------|----------|---------|
| **Code Organization** | Monolithic notebook | Modular components |
| **Maintainability** | Difficult to maintain | Easy to maintain and extend |
| **Error Handling** | Basic error handling | Robust error recovery |
| **Performance** | Variable performance | Optimized with caching |
| **Extensibility** | Hard to add features | Easy to add new components |
| **Code Quality** | Mixed patterns | Clean, typed, documented code |
| **Testing** | Limited testability | Fully testable components |
| **Production Ready** | Development only | Production-ready architecture |

## Documentation

- **System Documentation**: See `football_ai/README.md`
- **API Documentation**: Type hints and docstrings throughout
- **Demo Notebook**: `football_ai_demo.ipynb`

## License

This project maintains the same license as the original football analysis system while providing a completely modernized codebase.

## Contributing

The modern architecture makes it easy to contribute new features:

1. Define interfaces in `domain/interfaces.py`
2. Implement components following existing patterns
3. Integrate into the main pipeline
4. Add tests and documentation

The clean architecture ensures your contributions integrate seamlessly with the existing system.