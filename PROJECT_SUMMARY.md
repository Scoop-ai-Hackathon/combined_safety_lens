# SpoonOS - Safety Lens: Active Sentinel

A real-time factory safety monitoring system that uses computer vision to detect crush hazards and protect workers' hands from industrial machinery.

## Overview

Safety Lens is a Streamlit-based web application that monitors a video feed in real-time, detecting hands and analyzing motion patterns to identify potential crush hazards. When a hazard is detected, the system triggers an emergency stop, captures a screenshot of the incident, and plays an audio alarm.

## Project Structure

```
spoonOs/
├── safety_lens.py          # Main application (Streamlit web UI + detection logic)
├── generate_alarm.py       # Utility to generate alarm.wav audio file
├── hand_landmarker.task    # MediaPipe hand detection model
├── efficientdet_lite0.tflite # Object detection model (TensorFlow Lite)
├── alarm.wav               # Generated alarm audio for hazard alerts
├── screenshots/            # Captured incident screenshots
│   └── incident_YYYYMMDD_HHMMSS.jpg
├── venv/                   # Python virtual environment
└── .claude/                # Claude AI integration hooks
```

## Core Technologies

| Technology | Purpose |
|------------|---------|
| **Python 3.12** | Runtime environment |
| **Streamlit** | Web UI framework |
| **OpenCV (cv2)** | Video capture and image processing |
| **MediaPipe** | Hand landmark detection |
| **NumPy** | Numerical computing for optical flow analysis |

## Key Features

### 1. Real-time Hand Detection
Uses MediaPipe Hand Landmarker to detect up to 2 hands simultaneously with configurable confidence thresholds.

### 2. Optical Flow Analysis
Implements Farneback optical flow algorithm to detect motion patterns in regions surrounding detected hands.

### 3. Multi-Algorithm Hazard Detection

- **Compression Detection**: Monitors rapid shrinking of hand bounding box area (>5% reduction per frame)
- **Pinch Detection**: Analyzes motion vectors in 4 directional ROIs (left, right, top, bottom)
- **Hand Movement Suppression**: Filters false positives caused by the hand itself moving
- **Leaky Bucket Algorithm**: Accumulates hazard signals over time with decay, triggering when threshold reached

### 4. Emergency Response System
- Automatic system pause when crush hazard detected
- Screenshot capture of incident frame
- Audio alarm playback
- Incident logging with timestamps
- Integration hook for external systems (Spoon Agent/x402)

### 5. Dashboard UI
- Real-time video feed with overlaid annotations
- Machine status indicator with spinning gear animation
- Status badges (SAFE/DANGER/OFFLINE)
- Incident log display
- FPS and frame count metrics

## Detection Logic Flow

```
Camera Frame
    │
    ▼
Hand Detection (MediaPipe)
    │
    ▼
Optical Flow Calculation (Farneback)
    │
    ▼
ROI Analysis (4 directions)
    │
    ├─► Translation Cancellation (filter global motion)
    ├─► Hand Movement Suppression (filter self-induced flow)
    │
    ▼
Hazard Accumulation (Leaky Bucket)
    │
    ▼
Threshold Check (≥10 = DANGER)
    │
    ▼
Emergency Stop + Screenshot + Alarm
```

## Configuration

Configurable parameters in `safety_lens.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `camera_index` | 0 | Webcam index (configurable via sidebar) |
| `velocity_thresh` | 4.0 | Minimum optical flow velocity to register |
| `pixel_thresh` | 400 | Minimum moving pixels to trigger hazard |
| `hand_speed_thresh` | 15.0 | Hand movement speed to suppress false positives |
| `hazard_threshold` | 10 | Leaky bucket threshold for emergency stop |

## Running the Application

```bash
# Activate virtual environment
source venv/bin/activate

# Generate alarm audio (if needed)
python generate_alarm.py

# Run the application
streamlit run safety_lens.py
```

## Session State Variables

| Variable | Type | Purpose |
|----------|------|---------|
| `monitor_active` | bool | Main pause/play toggle |
| `current_status` | str | "SAFE" or "DANGER" |
| `incident_log` | list | Timestamped incident messages |
| `pinch_frame_count` | int | Sustained compression counter |
| `impact_frame_count` | int | Impact accumulation counter |
| `hazard_count_l/r/t/b` | int | Per-direction hazard counters |

## Screenshots

When a hazard triggers the emergency stop, a screenshot is automatically saved to the `screenshots/` folder with the naming format:
```
incident_YYYYMMDD_HHMMSS.jpg
```

## External Integration

The `trigger_spoon_agent_incident()` function serves as a hook for external system integration (e.g., blockchain logging via x402 protocol). Currently implemented as a stub that logs to console.
