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

## Streamlit Cloud Deployment

To deploy this app on Streamlit Cloud:

1. **Fork this repository** to your GitHub account
2. **Go to [Streamlit Cloud](https://share.streamlit.io/)** and sign in with GitHub
3. **Click "New app"** and select your forked repository
4. **Configure the deployment:**
   - Main file path: `PPE_FinalYearProject/app.py`
   - Branch: `main`
   - Python version: `3.11` (automatically detected from runtime.txt)
5. **Deploy!** The app will be available at your unique Streamlit Cloud URL

### Important Notes for Cloud Deployment:
- **Webcam functionality is disabled** on Streamlit Cloud (use video upload instead)
- **Model files must be included** in your repository (`yolov8m.pt` and `best.pt`)
- **GPU is not available** on Streamlit Cloud (models will run on CPU)
- **File size limits** apply to uploaded videos (typically 200MB max)

## Notes
- The application uses YOLO for object detection and MediaPipe for pose estimation
- Compliance thresholds can be adjusted in the code if needed
- For best results, ensure good lighting and clear camera visibility
- **Local development**: Use webcam for real-time monitoring
- **Cloud deployment**: Use video upload for analysis

