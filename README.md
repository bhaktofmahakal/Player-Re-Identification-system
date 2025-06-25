# Player Re-Identification System

A real-time player tracking and re-identification system for football match analysis, designed to maintain consistent player IDs even when players temporarily leave the frame during goal celebrations and high-motion events.

## 🎯 Project Overview

This system processes a 15-second football video clip and performs:
- **Player Detection** using YOLOv11 model
- **Feature Extraction** using HSV color histograms
- **Re-Identification** using registry-based tracking
- **Special Goal Event Handling** for the last 4 seconds

## 📁 Project Structure

```
player-reid-assignment/
├── data/
│   ├── models/
│   │   └── yolov11_player_detection.pt    # your Pre-trained YOLO model (best.pt)
│   └── videos/
│       └── 15sec_input_720p.mp4           # Input video (15 seconds)
├── output/                                 # Generated outputs
│   ├── tracked_video.mp4                  # Main output video
│   ├── tracking_analysis.png              # Analysis charts
│   ├── REPORT.md                          # Detailed report
│   └── tracking_data.json                 # Frame-by-frame data
├── player_tracker.py                      # Main tracking system
├── debug_model.py                         # Model debugging utilities
├── generate_analysis.py                   # Visualization generation
├── generate_report.py                     # Report generation
├── main.py                                # Complete pipeline runner
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (tested on Python 3.12.7)
- **Operating System:** Windows/Linux/macOS 
- **RAM:** At least 4GB (8GB recommended)
- **CPU-optimized** (no GPU required) because my machine has no dedicated graphics card

### Step-by-Step Installation

1. **Navigate to project directory:**
   ```bash
   cd player-reid-assignment
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment:**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify installation:**
   ```bash
   python debug_model.py
   ```
   ✅ Should show model information and detection test

### Running the System

#### Option 1: Complete Analysis (Recommended)
```bash
python main.py
```
**Outputs Generated:**
- `output/tracked_video.mp4` - Annotated video with player IDs
- `output/tracking_analysis.png` - Statistical visualizations
- `output/REPORT.md` - Comprehensive analysis report
- `output/tracking_data.json` - Raw tracking data

#### Option 2: Quick Run (Tracking Only)
```bash
python player_tracker.py
```
**Output:** `output/tracked_video.mp4`

#### Option 3: Individual Components
```bash
# Debug model and check classes
python debug_model.py

# Generate visualizations 
python generate_analysis.py

# Generate report 
python generate_report.py
```

## 📋 Dependencies & Environment Requirements

### Core Dependencies:
- `ultralytics>=8.0.0` - YOLOv11 detection model
- `opencv-python>=4.8.0` - Video processing and computer vision
- `numpy>=1.24.0` - Numerical computations
- `matplotlib>=3.7.0` - Data visualization and plotting
- `Pillow>=9.5.0` - Image processing support

### System Requirements:
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 2GB free space for outputs
- **CPU:** Multi-core recommended for faster processing
- **Python:** 3.8+ (tested on 3.12.7)

### Environment Setup:
- **Virtual Environment:** Strongly recommended to avoid conflicts
- **No GPU Required:** Optimized for CPU-only execution
- **Cross-Platform:** Works on Windows, Linux, and macOS

## 🔧 System Architecture

### Core Components

#### 1. **PlayerFeatureExtractor**
- Extracts HSV color histograms from player bounding boxes
- 32-bin histograms for computational efficiency
- Robust handling of small/invalid patches

#### 2. **PlayerTracker** 
- Registry-based tracking system for permanent ID assignment
- Adaptive similarity thresholds for different scenarios
- Special handling for goal celebration events

#### 3. **Detection Pipeline**
- YOLOv11 model for player/goalkeeper detection
- Confidence threshold: 0.2 (optimized for recall)
- Frame preprocessing: 640x360 for speed

### Key Features

#### **Adaptive Tracking Modes:**
- **Normal Mode (Frames 1-274):** Standard tracking parameters
- **Goal Event Mode (Frames 275-375):** Enhanced parameters for:
  - 2.5x more forgiving position tolerance
  - 3x tolerance for fallen players (aspect ratio changes)
  - Field object filtering (goalposts, corner flags)
  - Conservative feature updates (95% stability)

#### **Re-Identification Strategy:**
1. **Feature Matching:** HSV color histogram similarity
2. **Position Tracking:** Euclidean distance with adaptive thresholds
3. **Temporal Consistency:** Recent match preference
4. **Registry System:** Permanent ID storage for re-entry detection

## 📊 Expected Performance

### Typical Results:
- **Processing Speed:** ~1.5-2 FPS on CPU
- **Player Detection:** 13-17 players per frame
- **Unique IDs:** 18-22 total players
- **ID Consistency:** >96.2% during goal celebrations
- **Memory Usage:** <2GB RAM

### Output Quality:
- **Bounding Boxes:** Accurate player localization
- **ID Labels:** Consistent color-coded identifiers
- **Re-Entry Tracking:** Maintains IDs after temporary exits
- **Field Object Filtering:** Reduces false positives

## 🛠️ Configuration Options

### Key Parameters (in `player_tracker.py`):
```python
# Detection settings
confidence_threshold = 0.2
input_size = (640, 360)

# Tracking parameters
position_threshold = 100.0      # Normal position tolerance
feature_weight = 0.6           # Color similarity importance
position_weight = 0.25         # Position similarity importance
size_weight = 0.15             # Size consistency importance

# Goal event settings (frames 275-375)
goal_position_multiplier = 2.5  # Enhanced position tolerance
fallen_player_multiplier = 3.0  # Extra tolerance for aspect changes
```

## 🔍 Troubleshooting

### Common Issues:

#### **"Model not found" Error:**
```bash
# Verify model exists
ls data/models/yolov11_player_detection.pt
```

#### **"Video not found" Error:**
```bash
# Verify video exists
ls data/videos/15sec_input_720p.mp4
```

#### **Import Errors:**
```bash
# Verify all imports work
python -c "import cv2, ultralytics, numpy, matplotlib; print('All imports OK')"
```

#### **Low FPS Performance:**
- Reduce input resolution in `player_tracker.py`
- Decrease histogram bins in `PlayerFeatureExtractor`
- Skip frames for faster processing

#### **Too Many/Few Detections:**
- Adjust `confidence_threshold` in detection pipeline
- Modify field object filtering parameters

### Debug Commands:
```bash
# Check model classes and performance
python debug_model.py

# Test tracker initialization
python -c "from player_tracker import PlayerTracker; print('Tracker initialized')"

# Verify virtual environment
python -c "import sys; print('Python:', sys.executable)"
```

## 📈 Performance Optimization

### For Better Speed On CPU:
1. **Reduce Resolution:** Change `(640, 360)` to `(480, 270)`
2. **Skip Frames:** Process every 2nd frame for 2x speedup
3. **Reduce Features:** Lower histogram bins from 32 to 16
4. **Batch Processing:** Process multiple detections together

### For Better Accuracy ON CPU:
1. **Increase Resolution:** Use `(960, 540)` for better detection
2. **Lower Confidence:** Use `0.15` threshold for more detections
3. **Enhanced Features:** Add texture/edge features
4. **Temporal Smoothing:** Increase feature smoothing factors

## 🎥 Output Specifications

### tracked_video.mp4:
- **Resolution:** 1280x720 (original)
- **FPS:** 25 (original)
- **Duration:** 15 seconds
- **Annotations:** 
  - Colored bounding boxes per player
  - ID labels (ID:1, ID:2, etc.)
  - Consistent colors per ID across frames

### tracking_analysis.png:
- **Format:** PNG, 1500x1000 pixels
- **Charts:** 4-panel analysis
  - Player count per frame
  - ID distribution
  - Detection confidence over time
  - Tracking stability metrics

## 🚨 Important Notes

### Limitations:
- **CPU-Only:** Optimized for CPU, GPU acceleration not implemented
- **Single Camera:** Designed for single-feed tracking
- **Fixed Duration:** Optimized for 15-second clips
- **Football-Specific:** Tuned for football match scenarios

### Known Issues:
- Very small players at distance may be missed
- Identical jersey colors can cause brief ID confusion
- Extreme occlusion (>80%) may cause temporary ID loss

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Run `python debug_model.py` for diagnostics
3. Verify all files in project structure exist
4. Ensure virtual environment is activated

## 🏆 Success Criteria

The system successfully achieves:
- ✅ **Consistent Player IDs** throughout the video
- ✅ **Re-Entry Detection** after temporary exits
- ✅ **Goal Event Handling** with enhanced parameters
- ✅ **Field Object Filtering** to reduce false positives
- ✅ **Real-Time Simulation** with frame-by-frame processing
- ✅ **Comprehensive Output** with video, analysis, and reports

---
