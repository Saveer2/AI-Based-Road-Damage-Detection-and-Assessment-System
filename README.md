# RDD_AI_P1

### Deep Learning-Based Road Damage Detection and Video Analysis System

## Overview

RDD_AI is a computer vision project designed to detect road damage from uploaded videos using a trained YOLO object detection model. The system processes videos frame by frame, identifies road damage, and generates an annotated output video containing detection results.

The project aims to support automated road inspection and road condition assessment.

## Features

* Video upload through a web interface
* Automated road damage detection using YOLO
* Frame-by-frame video processing
* Bounding boxes and labels for detected damage
* Detection statistics and confidence scores
* Downloadable processed video
* Model evaluation and visualization

## Technologies Used

* Python
* YOLO and Ultralytics
* OpenCV
* Flask
* HTML and CSS
* Matplotlib
* PyTorch

### Detect Road Damage

1. Open the detection page.
2. Upload a supported video file.
3. Start the detection process.
4. Wait for the video to be processed.
5. Download the annotated output video.

Evaluation metrics and visualization plots are saved in the configured evaluation output directory.

## Workflow

1. Collect and prepare the road damage dataset.
2. Organize images and annotations into training, validation, and testing sets.
3. Train the YOLO object detection model.
4. Evaluate the trained model using test data.
5. Load the trained weights into the Flask application.
6. Extract and process video frames for road damage detection.
7. Generate an annotated output video and detection statistics.
8. Allow users to download the processed video.

## Applications

* Road condition monitoring
* Pothole and pavement defect identification
* Automated road inspection
* Infrastructure maintenance support
* Road damage assessment

## Limitations

Detection performance depends on the quality of the input video, lighting conditions, camera angle, dataset coverage, and trained model performance. The system's detection counts may include the same physical defect across multiple video frames.

## Future Enhancements

* Real-time road damage detection
* GPS-based damage location tracking
* Road damage severity classification
* Database integration for historical records
* Automated maintenance prioritization

---

**RDD_AI_p1** — An AI-powered approach to automated road damage detection and video analysis.
