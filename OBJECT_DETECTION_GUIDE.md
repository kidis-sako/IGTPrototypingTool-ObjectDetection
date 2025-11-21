# Ultrasound Object Detection Guide

## Overview

This feature detects geometric objects in ultrasound images: lines (needles, boundaries), circles (calibration spheres), and horizontal interfaces (tissue layers).

## Detection Algorithms

### Line Detection

**Hough Line Detection**
- Fast detection of straight lines
- Detects multiple lines at once
- Best for clean images

**RANSAC Line Detection**
- More accurate than Hough in noisy images
- Detects up to 20 lines, ordered by strength
- Handles all angles: horizontal, vertical, diagonal
- Filters noise automatically
- Slower but more reliable

**Horizontal Interface Detection**
- Fast detection of horizontal features only
- Ideal for tissue layers and water bath interfaces
- Uses histogram analysis
- Adjustable sensitivity via Peak Height Ratio slider (default: 0.15)

### Circle Detection

**Hough Circle Detection**
- Accurate for perfect circular shapes
- Best for calibration sphere detection
- Slower but precise

**Blob Sphere Detection**
- Works with imperfect circles
- Very fast
- Detects circles with circularity score > 0.6
- Size range: 100-10,000 pixels

## Image Preprocessing

All algorithms automatically apply:
- Grayscale conversion
- CLAHE contrast enhancement
- Bilateral noise filtering

## Auto-Threshold Feature

Automatically calculates optimal edge detection thresholds for your image.

**How it works:**
1. Analyzes image gradient statistics
2. Sets lower threshold based on mean - 0.5×std
3. Sets upper threshold based on mean + 1.5×std
4. Clamps to safe ranges (10-80 lower, 30-200 upper)

**When to use:**
- First time with a new image or video
- When default values (30/90) don't work well
- As a starting point before fine-tuning

## User Interface

### Layout
- **Left:** Original image and detection results side-by-side
- **Right:** Algorithm selection, parameters, and text results

### Controls
- **Load Current Frame:** Capture frame from video
- **Load Test Frame:** Load demo image (needle.png)
- **Algorithm Dropdown:** Choose detection method
- **Run Detection:** Start detection
- **Show Visualization:** Toggle result overlay

### Parameters

**Edge Detection**
- Lower Threshold (10-100, default: 30): Edge sensitivity
- Upper Threshold (30-200, default: 90): Strong edge cutoff
- **Auto-Calculate Button**: Find optimal thresholds automatically

**Line Detection**
- Hough Threshold (20-200): Votes needed for detection

**Circle Detection**
- Min Radius (5-100 pixels): Smallest circle size
- Max Radius (50-400 pixels): Largest circle size

**Interface Detection**
- Peak Height Ratio (0.05-0.5, default: 0.15): Minimum peak height

## How to Use

1. Load video in Video Input tab
2. Switch to Object Detection tab
3. Click "Load Current Frame"
4. Select algorithm from dropdown
5. Click "Auto-Calculate Thresholds" (recommended)
6. Adjust parameters if needed using sliders
7. Click "Run Detection"
8. View results in text area and visualization

## Detection Results

**Lines:**
- Start/end coordinates
- Angle (degrees)
- Length (pixels)
- RANSAC: inlier count and confidence

**Circles:**
- Center coordinates (x, y)
- Radius (pixels)
- Blob: circularity score (1.0 = perfect)

**Visualization:**
- Lines in different colors (red, green, blue, yellow, magenta, cyan)
- Circles with outline, center point, and radius label
- RANSAC lines labeled with number and angle

## Common Use Cases

**Water Bath Bottom**
- Use RANSAC Line Detection
- First line is usually the water bottom
- Extract Y-coordinate for water level

**Needle Tracking**
- Use Hough (fast) or RANSAC (accurate)
- Filter by length to find longest line
- Get angle and position from endpoints

**Calibration Spheres**
- Use Hough Circle Detection
- Set radius range to match sphere size
- Results sorted by size

## Parameter Tuning Tips

**Too many false detections?**
- Click "Auto-Calculate" then increase both thresholds by 10-20%
- Increase Hough threshold
- Raise Peak Height Ratio for interfaces

**Missing real objects?**
- Click "Auto-Calculate" then decrease both thresholds by 10-20%
- Decrease Hough threshold
- Lower Peak Height Ratio (try 0.10-0.12)
- Try a different algorithm (e.g., Blob instead of Hough)

**Unstable results between frames?**
- Increase thresholds for consistency
- Use RANSAC instead of Hough
- Process every Nth frame instead of every frame

## Speed Comparison

- **Fastest:** Blob, Horizontal Interfaces, Hough Lines (10-40ms)
- **Medium:** RANSAC Lines (15-40ms per line)
- **Slowest:** Hough Circles (50-150ms)

## Accuracy

- **Most accurate in noise:** RANSAC
- **Good:** Hough methods
- **Variable:** Blob (depends on contrast)

## Files

**Core:**
- `UltrasoundObjectDetector.java`: All detection algorithms
- `UltrasoundDetectionExample.java`: Usage examples

**UI:**
- `ObjectDetectionController.java`: UI logic
- `ObjectDetectionView.fxml`: UI layout
- `MainView.fxml`: Tab integration
- `MainController.java`: Tab switching

**Resources:**
- `needle.png`: Test image
