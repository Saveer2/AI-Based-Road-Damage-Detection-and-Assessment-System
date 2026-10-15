# Comparative Road Damage Detection Using YOLO11n and Faster R-CNN for Video-Based Inspection

## Overview

This repository houses the complete implementation and comparative evaluation framework for a research study analyzing road infrastructure defects. The project benchmarks a state-of-the-art single-stage object detector (**YOLO11n**) against a traditional two-stage architecture (**Faster R-CNN with a ResNet-50-FPN backbone**) using a custom road damage dataset (1,609 images). 

In addition to model training and rigorous test evaluation, the system features a video-processing pipeline integrated into a web interface to support automated road inspection and infrastructure assessment.

---

## Key Findings & Research Conclusions

Empirical evaluation on the test split yielded distinct architectural insights:
- **Accuracy (mAP@0.5):** YOLO11n achieved superior overall localization accuracy (**55.6%**) compared to Faster R-CNN (**36.5%**).
- **Class Performance:** Both architectures achieved peak precision on regular geometric boundaries like manholes ($75.2\%$ vs. $58.4\%$), while struggling with amorphous, linear fractures like cracks ($41.5\%$ vs. $24.4\%$).
- **Architectural Trade-offs:** While Faster R-CNN attained a strong harmonic F1 peak after tuning, its Region Proposal Network (RPN) exhibited proposal flooding over textured asphalt surfaces. YOLO11n’s anchor-free grid regression provided cleaner, consolidated bounding box outputs out-of-the-box, establishing it as the superior model for both video-based inspection accuracy and deployment efficiency.

---

## Features

- **Dual-Model Support:** Implementation scripts for training and evaluating both YOLO11n and Faster R-CNN (`rcnn1/` and YOLO configurations).
- **Comprehensive Evaluation Pipeline:** Generates class-wise AP metrics, F1-confidence curves, precision-recall curves, and confusion matrices.
- **Web-Based Video Analysis:** Flask application allowing users to upload inspection footage for frame-by-frame processing.
- **Automated Defect Detection:** Identifies road anomalies (potholes, cracks, manholes) with confidence scoring and bounding box visualization.
- **Exportable Results:** Downloadable annotated output videos and structured evaluation logs.

---

## Technologies Used

- **Deep Learning & Frameworks:** PyTorch, Ultralytics YOLO, Torchvision
- **Evaluation & Metrics:** COCO Evaluator (`pycocotools`), NumPy, Matplotlib
- **Backend & Interface:** Flask, OpenCV
- **Frontend:** HTML, CSS

