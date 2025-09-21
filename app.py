# streamlit_app.py
import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import time
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import os
import torch

# Initialize MediaPipe and YOLO models
@st.cache_resource
def load_models():
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    
    # Load models and move to GPU if available
    model1 = YOLO('yolov8m.pt')
    model_path = os.path.join(os.path.dirname(__file__), 'Model', 'weights', 'best.pt')
    model = YOLO(model_path)
    
    # Move models to GPU if available
    if torch.cuda.is_available():
        model1.to('cuda')
        model.to('cuda')
        st.success("🚀 Models loaded on GPU for faster inference!")
    else:
        st.info("💻 Models running on CPU")
    
    return mpDraw, mpPose, pose, model1, model

def process_frame(frame, pose, model):
    frame = cv2.resize(frame, (1020, 500))
    
    # Use GPU for inference if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = model.predict(frame, device=device)
    
    compliance_data = {
        'helmet': False,
        'vest': False,
        'mask': False
    }
    
    a = results[0].boxes.data.cpu()
    px = pd.DataFrame(a).astype("float")
    
    for index, row in px.iterrows():
        class_id = int(row[5])
        conf = row[4]
        if conf > 0.8 and class_id == 5:  # Worker detection
            x1, y1, x2, y2 = map(int, row[0:4])
            worker_frame = frame[y1:y2, x1:x2, :]
            imgRGB = cv2.cvtColor(worker_frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(imgRGB)
            
            worker_results = model.predict(worker_frame, device=device)
            worker_a = worker_results[0].boxes.data.cpu()
            worker_px = pd.DataFrame(worker_a).astype("float")
            
            if pose_results.pose_landmarks:
                for id, lm in enumerate(pose_results.pose_landmarks.landmark):
                    if id in [4, 12, 9, 1, 11, 24, 23, 10, 0]:
                        h, w, c = worker_frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        
                        for _, w_row in worker_px.iterrows():
                            w_class_id = int(w_row[5])
                            if cx > w_row[0] and cx < w_row[2] and cy > w_row[1] and cy < w_row[3]:
                                if w_class_id == 0:
                                    compliance_data['helmet'] = True
                                elif w_class_id == 7:
                                    compliance_data['vest'] = True
                                elif w_class_id == 1:
                                    compliance_data['mask'] = True
            
            # Draw bounding box
            color = (0, 255, 0) if all(compliance_data.values()) else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
    return frame, compliance_data

def create_compliance_chart(compliance_history):
    df = pd.DataFrame(compliance_history)
    compliance_rates = df.mean() * 100
    
    # Create gauge charts for each PPE item
    gauges = []
    colors = {'helmet': '#2ecc71', 'vest': '#3498db', 'mask': '#e74c3c'}
    
    for item, rate in compliance_rates.items():
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{item.title()} Compliance"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': colors[item]},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        gauge.update_layout(height=200)
        gauges.append(gauge)
    
    # Create trend chart
    trend_data = pd.DataFrame(compliance_history).rolling(window=5).mean() * 100
    trend_fig = go.Figure()
    
    for item in trend_data.columns:
        trend_fig.add_trace(go.Scatter(
            y=trend_data[item],
            name=item.title(),
            line=dict(color=colors[item], width=2)
        ))
    
    trend_fig.update_layout(
        title="PPE Compliance Trend",
        yaxis_title="Compliance Rate (%)",
        xaxis_title="Frame Number",
        height=300,
        showlegend=True,
        yaxis_range=[0, 100]
    )
    
    return gauges, trend_fig

def process_uploaded_video(video_file, pose, model):
    # Create a temporary file to store the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    try:
        tfile.write(video_file.read())
        tfile.close()  # Close the temp file after writing
        
        # Open the video file
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Error: Could not open video file")
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize progress bar
        progress_bar = st.progress(0)
        frame_placeholder = st.empty()
        
        compliance_history = []
        frame_count = 0
        
        # Process every nth frame (e.g., every 5th frame)
        frame_skip = 5
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue
                    
                # Process frame
                processed_frame, compliance_data = process_frame(frame, pose, model)
                compliance_history.append(compliance_data)
                
                # Update progress
                progress = int((frame_count / total_frames) * 100)
                progress_bar.progress(progress)
                
                # Display current frame
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb)
                
        finally:
            # Make sure to release the video capture
            cap.release()
            
        return compliance_history
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return []
        
    finally:
        try:
            # Clean up: close video capture if it exists
            if 'cap' in locals():
                cap.release()
            # Try to remove temporary file
            if os.path.exists(tfile.name):
                try:
                    os.unlink(tfile.name)
                except PermissionError:
                    # If file is still locked, schedule it for deletion on next reboot
                    import stat
                    os.chmod(tfile.name, stat.S_IWRITE)
                    try:
                        os.unlink(tfile.name)
                    except PermissionError:
                        pass  # If still can't delete, let OS clean it up later
        except Exception as e:
            st.warning(f"Warning: Could not clean up temporary files: {str(e)}")
            # Non-critical error, so we don't need to raise it

def run_webcam_detection(pose, model):
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam")
        return
    
    # Create placeholders for UI elements
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    # Initialize session state for webcam
    if 'webcam_compliance_history' not in st.session_state:
        st.session_state.webcam_compliance_history = []
        st.session_state.frame_count = 0
    
    # Control buttons in a single row
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        start_button = st.button('Start Monitoring')
    with col2:
        stop_button = st.button('Stop')
    with col3:
        clear_button = st.button('Clear Statistics')
    
    if clear_button:
        st.session_state.webcam_compliance_history = []
        st.session_state.frame_count = 0
    
    update_interval = 5  # Update statistics every 5 frames
    
    try:
        while start_button and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, compliance_data = process_frame(frame, pose, model)
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Update frame display
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update statistics at specified interval
            st.session_state.frame_count += 1
            if st.session_state.frame_count % update_interval == 0:
                st.session_state.webcam_compliance_history.append(compliance_data)
                if len(st.session_state.webcam_compliance_history) > 50:  # Keep last 50 data points
                    st.session_state.webcam_compliance_history.pop(0)
                
                # Update statistics display
                if len(st.session_state.webcam_compliance_history) > 0:
                    with stats_placeholder.container():
                        # Calculate current compliance rates
                        current_rates = pd.DataFrame([compliance_data]).mean() * 100
                        cols = st.columns(3)
                        for i, (item, rate) in enumerate(current_rates.items()):
                            cols[i].metric(
                                f"Current {item.title()} Compliance",
                                f"{rate:.1f}%",
                                f"{rate - 80:.1f}%" if rate >= 80 else f"{rate - 80:.1f}%"
                            )
                    
                    # Update charts
                    gauges, trend_fig = create_compliance_chart(st.session_state.webcam_compliance_history)
                    with chart_placeholder.container():
                        # Display gauge charts in a row
                        gauge_cols = st.columns(3)
                        for i, gauge in enumerate(gauges):
                            with gauge_cols[i]:
                                st.plotly_chart(
                                    gauge,
                                    use_container_width=True,
                                    key=f"gauge_{i}_{st.session_state.frame_count}"
                                )
                        # Display trend chart
                        st.plotly_chart(
                            trend_fig,
                            use_container_width=True,
                            key=f"trend_{st.session_state.frame_count}"
                        )
            
            time.sleep(0.1)  # Small delay to prevent overwhelming the UI
            
    finally:
        cap.release()

def main():
    st.set_page_config(page_title="PPE Compliance Monitor", layout="wide")
    
    st.title("PPE Compliance Monitoring Dashboard")
    
    # Display GPU status
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.sidebar.success(f"🚀 GPU: {gpu_name}")
    else:
        st.sidebar.info("💻 CPU Mode")
    
    # Load models
    mpDraw, mpPose, pose, model1, model = load_models()
    
    # Sidebar
    st.sidebar.title("Controls")
    source_option = st.sidebar.radio("Select Source", ["Webcam", "Upload Video"])
    
    if source_option == "Webcam":
        st.subheader("Live PPE Compliance Monitoring")
        run_webcam_detection(pose, model)
    else:
        st.subheader("Video File Analysis")
        uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi'])
        if uploaded_file is not None:
            with st.spinner('Processing video...'):
                compliance_history = process_uploaded_video(uploaded_file, pose, model)
                if compliance_history:
                    gauges, trend_fig = create_compliance_chart(compliance_history)
                    
                    # Display gauge charts
                    gauge_cols = st.columns(3)
                    for i, gauge in enumerate(gauges):
                        with gauge_cols[i]:
                            st.plotly_chart(
                                gauge,
                                use_container_width=True,
                                key=f"uploaded_gauge_{i}"
                            )
                    
                    # Display trend chart
                    st.plotly_chart(
                        trend_fig,
                        use_container_width=True,
                        key="uploaded_trend"
                    )
                    
                    # Display summary statistics
                    st.subheader("Summary Statistics")
                    df = pd.DataFrame(compliance_history)
                    mean_compliance = df.mean() * 100
                    
                    summary_cols = st.columns(3)
                    for i, (item, rate) in enumerate(mean_compliance.items()):
                        with summary_cols[i]:
                            st.metric(
                                f"Average {item.title()} Compliance",
                                f"{rate:.1f}%"
                            )

if __name__ == "__main__":
    main()
