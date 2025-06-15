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
- **🆕 Configuration System**: Comprehensive dataclass-based configuration with presets

## Architecture

The modern system follows clean architecture principles with:

- **Domain Models**: Core data structures and business logic
- **Modular Components**: Separate modules for detection, tracking, analysis, etc.
- **Abstract Interfaces**: Clear contracts for extensibility
- **Type Safety**: Full type hints throughout the codebase
- **Error Handling**: Robust error recovery and logging
- **Performance Optimization**: Efficient processing pipeline with caching
- **🆕 Flexible Configuration**: Type-safe, hierarchical configuration system

## Quick Start

### Configuration-Based Usage (Recommended)

```python
from football_ai import FootballAnalysisPipeline
from football_ai.config import get_broadcast_config

# Use a predefined configuration
config = get_broadcast_config()
config.update_paths(
    model_path="models/best.pt",
    input_video_path="input_videos/match.mp4",
    output_video_path="output_videos/annotated.mp4"
)

# Initialize with configuration
pipeline = FootballAnalysisPipeline(config=config)

# Process the video
results = pipeline.process_video(
    video_path=config.processing.input_video_path,
    output_video_path=config.processing.output_video_path
)
```

### Basic Usage (Legacy Support)

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

The `football_ai_demo.ipynb` provides a clean, simplified workflow:

1. **Import and Setup** - Load the Football AI system
2. **Configuration** - Choose from predefined configs or customize
3. **Initialize Pipeline** - Create the analysis pipeline
4. **Process Video** - Run complete football analysis
5. **View Results** - See analysis summary and outputs

The notebook demonstrates all key features in under 10 cells with clear explanations.

For advanced configuration examples, see `football_ai_demo_complete.ipynb`.

## Configuration System

The Football AI system includes a comprehensive configuration system that makes it easy to customize behavior for different scenarios.

### Predefined Configurations

```python
from football_ai.config import (
    get_default_config,      # Balanced performance and accuracy
    get_high_accuracy_config,    # Maximum analysis quality  
    get_fast_processing_config,  # Optimized for speed
    get_broadcast_config        # Professional broadcast quality
)

# Use a predefined configuration
config = get_broadcast_config()
```

### Custom Configuration

```python
from football_ai.config import FootballAIConfig

# Create custom configuration
config = FootballAIConfig()

# Customize detection settings
config.model.confidence_threshold = 0.7
config.model.model_path = "models/custom_model.pt"

# Customize rendering
config.rendering.show_tracks = True
config.rendering.show_team_colors = True
config.rendering.bbox_thickness = 3

# Customize processing
config.processing.process_every_nth_frame = 2  # Process every 2nd frame
config.processing.enable_caching = True

# Save configuration for later use
config.save_to_file("my_config.json")

# Load configuration from file
config = FootballAIConfig.load_from_file("my_config.json")
```

### Configuration Validation

```python
# Validate configuration
issues = config.validate()
if issues:
    for issue in issues:
        print(f"Warning: {issue}")

# Get configuration summary
print(config.get_summary())
```

### Configuration Structure

The configuration is organized hierarchically:

- **`model`**: YOLO detection settings (confidence, model path, device)
- **`tracking`**: ByteTrack settings (thresholds, buffers, smoothing)
- **`team_analysis`**: Team color analysis (clustering, samples, confidence)
- **`possession`**: Ball possession analysis (distance thresholds, smoothing)
- **`camera`**: Camera motion tracking (features, quality levels)
- **`transformation`**: Coordinate transformation (field dimensions, calibration)
- **`rendering`**: Visualization settings (colors, overlays, annotations)
- **`processing`**: Performance and I/O settings (paths, caching, batching)

This makes the system highly configurable while maintaining type safety and validation.

## Field Dimension Presets

The system includes comprehensive support for different soccer field sizes through predefined dimension presets and flexible coordinate transformation.

### Available Field Presets

```python
from football_ai.constants import FieldDimensions

# List all available presets
FieldDimensions.list_presets()

# Get specific dimensions
fifa_width, fifa_height = FieldDimensions.get_dimensions('FIFA_STANDARD')
print(f"FIFA Standard: {fifa_width}m x {fifa_height}m")
```

**Available Presets:**
- **FIFA_STANDARD** (68m x 105m) - Official FIFA standard for international matches
- **FIFA_MINIMUM** (45m x 90m) - Minimum FIFA allowed dimensions
- **FIFA_MAXIMUM** (90m x 120m) - Maximum FIFA allowed dimensions
- **MLS** (70m x 110m) - Major League Soccer standard
- **PREMIER_LEAGUE**, **LA_LIGA**, **BUNDESLIGA**, **SERIE_A** - Professional league standards
- **YOUTH_U12** (45m x 64m), **YOUTH_U14** (55m x 75m), **YOUTH_U16** (64m x 91m) - Youth fields
- **FIVE_A_SIDE** (25m x 42m), **SEVEN_A_SIDE** (50m x 70m) - Small-sided games
- **HIGH_SCHOOL** (55m x 100m), **COLLEGE** (68m x 105m) - Educational institutions

### Creating Coordinate Transformers with Field Presets

```python
from football_ai.transformation.coordinate_transformer import PerspectiveCoordinateTransformer

# Method 1: Convenience class methods
fifa_transformer = PerspectiveCoordinateTransformer.for_fifa_standard()
youth_transformer = PerspectiveCoordinateTransformer.for_youth('U12')
mls_transformer = PerspectiveCoordinateTransformer.for_league('mls')
small_transformer = PerspectiveCoordinateTransformer.for_small_sided('5-a-side')

# Method 2: Direct preset specification
transformer = PerspectiveCoordinateTransformer(field_preset='BUNDESLIGA')

# Method 3: Custom dimensions (overrides any preset)
custom_transformer = PerspectiveCoordinateTransformer(
    field_width=75.0, 
    field_height=110.0
)

# All transformers work the same way regardless of field size
transformer.set_field_corners(detected_corners)
field_coord = transformer.transform_point(pixel_coord)
distance = transformer.calculate_distance(point1, point2)
```

### Benefits of Field Size Support

- **Accurate Measurements**: Distance and speed calculations automatically adjust to actual field size
- **Flexible Analysis**: Support for youth soccer, professional leagues, and small-sided games
- **Zone Detection**: Field zones (defensive, midfield, attacking) adapt to field dimensions
- **Real-world Coordinates**: All measurements in actual meters for the specific field type

### Usage in Pipeline

The pipeline automatically uses FIFA standard dimensions by default, but you can customize:

```python
# For MLS matches
pipeline = FootballAnalysisPipeline(config=config)
pipeline.coordinate_transformer = PerspectiveCoordinateTransformer.for_league('mls')

# For youth soccer
pipeline.coordinate_transformer = PerspectiveCoordinateTransformer.for_youth('U14')

# Then set field corners and process as normal
pipeline.set_field_keypoints(detected_corners)
results = pipeline.process_video(video_path, output_path)
```

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
├── constants.py                   # Field dimension presets and constants
├── config.py                      # Configuration system
└── pipeline.py                    # Main orchestrator

input_videos/                      # Input video files
outputs/                           # All output files
├── videos/                        # Annotated video outputs
├── data/                          # Analysis data and cache
└── configs/                       # Saved configurations

models/                           # Trained YOLO models
training/                         # Model training resources
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