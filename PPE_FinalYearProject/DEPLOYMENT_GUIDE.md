# PPE Detection App - Streamlit Cloud Deployment Guide

## Changes Made for Deployment

### 1. Fixed Python Version Compatibility
- **Problem**: Streamlit Cloud was using Python 3.13.6, but MediaPipe doesn't support Python 3.13
- **Solution**: Set `runtime.txt` to `python-3.11` which is the latest supported version

### 2. Optimized Requirements
- **Problem**: Conflicting and unnecessary dependencies causing installation failures
- **Solution**: Created a clean `requirements.txt` with only essential packages and compatible versions

### 3. Added Configuration Files
- **`.streamlit/config.toml`**: Optimized for deployment with proper server settings
- **`.gitignore`**: Excludes unnecessary files from deployment

## Deployment Steps

1. **Commit and push all changes to your GitHub repository**
   ```bash
   git add .
   git commit -m "Fix deployment issues for Streamlit Cloud"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file path: `PPE_FinalYearProject/app.py`
   - Deploy!

## File Structure for Deployment

```
PPE_FinalYearProject/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Optimized dependencies
├── runtime.txt              # Python 3.11
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── .gitignore               # Git ignore rules
├── yolov8m.pt              # YOLO model file
├── Model/
│   └── weights/
│       └── best.pt         # Custom trained model
└── test_requirements.py     # Test script (optional)
```

## Key Changes Summary

### runtime.txt
```
python-3.11
```

### requirements.txt (Optimized)
```
# Core dependencies for PPE Detection App
streamlit>=1.24.0,<1.46.0
plotly>=5.13.0

# Computer Vision and ML
opencv-python>=4.7.0,<5.0.0
mediapipe>=0.10.0,<0.11.0
ultralytics>=8.0.0
torch>=2.0.0,<3.0.0
torchvision>=0.15.0,<1.0.0

# Data processing
numpy>=1.24.0,<2.0.0
pandas>=1.5.3
Pillow>=9.5.0
scipy>=1.10.0

# Utilities
PyYAML>=6.0.2
tqdm>=4.67.1
requests>=2.32.3

# MediaPipe dependencies
protobuf>=4.21.0,<5.0.0
absl-py>=1.0.0
attrs>=21.0.0
```

## Troubleshooting

If you still encounter issues:

1. **Check the deployment logs** in Streamlit Cloud dashboard
2. **Verify model files** are present in the repository
3. **Test locally** using `python test_requirements.py`
4. **Clear cache** and redeploy if needed

## Notes

- The app will automatically download YOLO models if needed
- GPU acceleration is not available on Streamlit Cloud (CPU only)
- Video file size is limited to 200MB (configurable in config.toml)

