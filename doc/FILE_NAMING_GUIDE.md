# Football AI - File Naming Convention

## 📁 Clear and Descriptive File Names

The Football AI system now uses clear, descriptive file names that immediately communicate their purpose:

### Core Components

| Category | File Name | Purpose |
|----------|-----------|---------|
| **Game Management** | `game_factory.py` | Pattern-based game creation |
| **Game Management** | `game_manager.py` | Game lifecycle and analysis management |
| **Video Analysis** | `video_analysis_processor.py` | Main video analysis coordinator |
| **Video Processing** | `video_loader.py` | Video file loading and preprocessing |
| **Video Processing** | `video_pipeline.py` | Video processing pipeline orchestration |

### Video Analysis Processors

| Category | File Name | Technology/Purpose |
|----------|-----------|-------------------|
| **Detection** | `yolo_detector.py` | YOLO-based object detection |
| **Tracking** | `byte_tracker.py` | ByteTrack multi-object tracking |
| **Team Classification** | `siglip_team_classifier.py` | SigLIP-based team assignment |
| **Possession Analysis** | `possession_analyzer.py` | Ball possession detection |
| **Motion Analysis** | `speed_calculator.py` | Player speed and motion analysis |
| **Camera Handling** | `camera_stabilizer.py` | Camera motion compensation |
| **Coordinate Mapping** | `coordinate_transformer.py` | Field coordinate transformation |
| **Video Output** | `video_annotator.py` | Video annotation and rendering |
| **Export** | `video_exporter.py` | Video export and saving |
| **Analytics** | `speed_analyzer.py` | Speed and movement analytics |
| **Analytics** | `possession_tracker.py` | Ball possession tracking |

### Demo and Examples

| File Name | Purpose |
|-----------|---------|
| `demo_football_analysis.py` | Main demonstration of the system |

## 🎯 Naming Principles

### 1. **Technology-Specific Names**
- `yolo_detector.py` instead of `object_detection_processor.py`
- `byte_tracker.py` instead of `track_processor.py`
- `siglip_team_classifier.py` instead of `team_assignment_processor.py`

### 2. **Function-Descriptive Names**
- `possession_analyzer.py` instead of `ball_assignment_processor.py`
- `speed_calculator.py` instead of `object_motion_processor.py`
- `camera_stabilizer.py` instead of `camera_motion_processor.py`

### 3. **Purpose-Clear Names**
- `video_analysis_processor.py` instead of `processor.py`
- `game_factory.py` instead of `factory.py`
- `game_manager.py` instead of `manager.py`

### 4. **Consistent Patterns**
- **Analyzers**: Files that perform analysis (`speed_analyzer.py`, `possession_analyzer.py`)
- **Processors**: Files that process data (`video_analysis_processor.py`)
- **Managers**: Files that manage lifecycle (`game_manager.py`)
- **Factories**: Files that create objects (`game_factory.py`)
- **Loaders**: Files that load data (`video_loader.py`)
- **Exporters**: Files that export data (`video_exporter.py`)

## 📂 Directory Structure

```
football_ai/
├── domain/                  # Core domain models
├── game/                    # Game-centric management
│   ├── game_factory.py     # Game creation patterns
│   └── game_manager.py     # Game lifecycle management
├── sources/                 # Analysis data sources
│   └── video/              # Video analysis components
│       ├── video_analysis_processor.py  # Main coordinator
│       ├── video_loader.py             # Video loading
│       ├── video_pipeline.py           # Processing pipeline
│       └── processors/                 # Specialized processors
│           ├── detection/
│           │   └── yolo_detector.py    # YOLO detection
│           ├── tracking/
│           │   └── byte_tracker.py     # ByteTrack tracking
│           ├── assignment/
│           │   ├── siglip_team_classifier.py  # Team classification
│           │   └── possession_analyzer.py     # Ball possession
│           ├── motion/
│           │   ├── speed_calculator.py        # Speed analysis
│           │   └── camera_stabilizer.py       # Camera motion
│           ├── transformation/
│           │   └── coordinate_transformer.py  # Field mapping
│           ├── analysis/
│           │   ├── speed_analyzer.py          # Speed analytics
│           │   └── possession_tracker.py      # Possession tracking
│           ├── rendering/
│           │   └── video_annotator.py         # Video annotation
│           └── storing/
│               └── video_exporter.py          # Video export
└── utils/                   # Shared utilities
```

## ✅ Benefits of Clear Naming

1. **Immediate Understanding**: File names clearly indicate their purpose
2. **Technology Transparency**: Easy to identify which technologies are used
3. **Better Organization**: Logical grouping and hierarchy
4. **Easier Maintenance**: Developers can quickly find relevant code
5. **Professional Appearance**: Clean, enterprise-ready codebase

The improved naming convention makes the Football AI system more accessible, maintainable, and professional.
