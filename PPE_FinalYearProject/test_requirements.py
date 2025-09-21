#!/usr/bin/env python3
"""
Test script to verify all requirements can be imported successfully.
Run this before deploying to Streamlit Cloud.
"""

def test_imports():
    """Test all required imports"""
    try:
        print("Testing imports...")
        
        # Core dependencies
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        import cv2
        print("✅ OpenCV imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        from ultralytics import YOLO
        print("✅ Ultralytics imported successfully")
        
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")
        
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import torch
        print("✅ PyTorch imported successfully")
        
        from PIL import Image
        print("✅ Pillow imported successfully")
        
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Plotly imported successfully")
        
        print("\n🎉 All imports successful! Ready for deployment.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)

