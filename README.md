# PPE Compliance Monitoring Dashboard

This application provides real-time monitoring of Personal Protective Equipment (PPE) compliance using computer vision. It detects and tracks:
- Helmets
- Safety Vests
- Face Masks

## Features
- Real-time webcam monitoring
- Video file upload support
- Live compliance metrics
- Historical compliance tracking
- Interactive dashboard interface

## Setup Instructions

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the following model files in your workspace:
- `yolov8m.pt` (Base YOLO model)
- `best.pt` (Your custom trained model)

3. Run the Streamlit application:
```bash
streamlit run app.py
```

## Usage

1. The dashboard will open in your default web browser
2. Select your preferred input source:
   - Webcam: For real-time monitoring
   - Upload Video: To analyze pre-recorded footage
3. Monitor the compliance metrics in real-time:
   - Green indicators show compliant PPE usage
   - Red indicators show non-compliance
4. View the compliance history chart to track trends

## Requirements
- Python 3.8+
- Webcam (for live monitoring)
- CUDA-capable GPU (recommended for better performance)

## Notes
- The application uses YOLO for object detection and MediaPipe for pose estimation
- Compliance thresholds can be adjusted in the code if needed
- For best results, ensure good lighting and clear camera visibility

