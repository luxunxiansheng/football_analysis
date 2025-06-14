# Football AI System

A clean, modular, and extensible football video analysis system designed to replicate the functionality of the original notebook with modern software engineering practices.

## Architecture

The system is built with a clean architecture that separates concerns into distinct modules:

### Core Components

- **Domain Models** (`domain/models.py`): Core data structures and enums
- **Domain Interfaces** (`domain/interfaces.py`): Abstract interfaces for all components
- **Detection** (`detection/yolo_detector.py`): YOLO-based object detection
- **Tracking** (`tracking/byte_tracker.py`): ByteTrack-based multi-object tracking
- **Analysis** (`analysis/`): Team color analysis and ball possession analysis
- **Motion** (`motion/camera_motion_tracker.py`): Camera movement tracking
- **Transformation** (`transformation/coordinate_transformer.py`): Pixel-to-field coordinate transformation
- **Rendering** (`rendering/video_renderer.py`): Video annotation and visualization
- **Utils** (`utils/`): Utility functions for video processing and bounding box operations
- **Pipeline** (`pipeline.py`): Main orchestrator that integrates all components

### Key Features

1. **Clean Architecture**: Modular design with clear separation of concerns
2. **Type Safety**: Full type hints throughout the codebase
3. **Error Handling**: Robust error handling and recovery mechanisms
4. **Extensibility**: Easy to add new analysis features or replace components
5. **Performance**: Optimized processing pipeline with caching support
6. **Modern Code**: Following Python best practices and design patterns

## Usage

### Basic Usage

```python
from football_ai import FootballAnalysisPipeline

# Initialize the pipeline
pipeline = FootballAnalysisPipeline(
    model_path="path/to/yolo/model.pt",
    output_dir="output_directory",
    save_cache=True,
    load_cache=True
)

# Process a video
analysis_results = pipeline.process_video(
    video_path="input_video.mp4",
    output_video_path="annotated_output.mp4",
    field_keypoints=[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]  # Optional field corners
)

# Access results
print(f"Players tracked: {len(analysis_results.player_tracks)}")
print(f"Team possession: {analysis_results.team_ball_control}")
```

### Using the Demo Notebook

1. Open `football_ai_demo.ipynb`
2. Update the configuration paths
3. Run all cells to see the complete analysis

## Requirements

- Python 3.8+
- OpenCV (cv2)
- NumPy
- YOLO (ultralytics)
- scikit-learn
- tqdm

Install requirements:
```bash
pip install -r requirements.txt
```

## Input/Output

### Input
- **Video File**: MP4 or other supported video format
- **YOLO Model**: Trained model file (.pt)
- **Field Keypoints** (optional): Four corner coordinates for field transformation

### Output
- **Annotated Video**: Video with all analysis overlays
- **Analysis Results**: Complete match analysis data structure
- **Cache Files**: Processed results for faster re-runs

## Analysis Features

### Object Detection & Tracking
- Players, referees, and ball detection using YOLO
- Multi-object tracking with track ID assignment
- Robust tracking across occlusions and frame gaps

### Team Analysis
- Automatic team color detection using K-means clustering
- Player-to-team assignment based on jersey colors
- Temporal smoothing for stable team assignments

### Ball Possession Analysis
- Distance-based ball possession detection
- Team possession statistics and temporal analysis
- Confidence-based possession state management

### Camera Motion Tracking
- Feature-based camera movement detection
- Position adjustment for camera motion compensation
- Stable tracking despite camera movements

### Coordinate Transformation
- Perspective transformation from pixels to field coordinates
- Real-world distance and speed calculations
- Field zone analysis and position mapping

### Video Rendering
- Professional-quality annotations and overlays
- Player tracks, team colors, and possession indicators
- Statistics display and legend rendering

## Comparison with Original System

The modern system provides identical functionality to the original notebook but with:

| Aspect | Original | Modern |
|--------|----------|---------|
| **Architecture** | Monolithic notebook | Modular components |
| **Code Quality** | Mixed patterns | Clean, typed code |
| **Extensibility** | Hard to extend | Easy to add features |
| **Error Handling** | Basic | Robust with recovery |
| **Performance** | Variable | Optimized pipeline |
| **Maintainability** | Difficult | Easy to maintain |
| **Testing** | Limited | Testable components |

## Development

### Adding New Features

1. Define interfaces in `domain/interfaces.py`
2. Implement components following existing patterns
3. Integrate into the main pipeline
4. Add tests and documentation

### Component Structure

Each component follows a consistent pattern:
- Inherits from domain interface
- Implements required methods
- Handles errors gracefully
- Provides configuration options
- Includes comprehensive documentation

## License

This project follows the same license as the original football analysis system.

## Credits

Built as a modern refactoring of the original football analysis notebook, maintaining full functionality while improving code quality and maintainability.
