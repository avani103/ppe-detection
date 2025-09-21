# app.py
import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import time
from PIL import Image
import plotly.graph_objects as go
import tempfile
import os

# ================================
# Load Models
# ================================
@st.cache_resource
def load_models():
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    # Load both YOLO models
    yolov8m_model = YOLO("yolov8m.pt")   # Pretrained YOLOv8m
    best_model = YOLO("best.pt")         # Custom PPE detection model

    return mpDraw, mpPose, pose, yolov8m_model, best_model


# ================================
# Process Single Frame
# ================================
def process_frame(frame, pose, yolov8m_model, best_model):
    frame = cv2.resize(frame, (1020, 500))

    # Step 1: Detect workers using YOLOv8m
    results_yolo8 = yolov8m_model.predict(frame)

    compliance_data = {"helmet": False, "vest": False, "mask": False}

    a = results_yolo8[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for _, row in px.iterrows():
        class_id = int(row[5])
        conf = row[4]

        # Assume class_id == 5 corresponds to "person/worker"
        if conf > 0.8 and class_id == 5:
            x1, y1, x2, y2 = map(int, row[0:4])
            worker_frame = frame[y1:y2, x1:x2, :]
            imgRGB = cv2.cvtColor(worker_frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(imgRGB)

            # Step 2: Detect PPE on workers using best.pt
            worker_results = best_model.predict(worker_frame)
            worker_a = worker_results[0].boxes.data
            worker_px = pd.DataFrame(worker_a).astype("float")

            if pose_results.pose_landmarks:
                for id, lm in enumerate(pose_results.pose_landmarks.landmark):
                    if id in [4, 12, 9, 1, 11, 24, 23, 10, 0]:
                        h, w, c = worker_frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)

                        for _, w_row in worker_px.iterrows():
                            w_class_id = int(w_row[5])
                            if (
                                cx > w_row[0]
                                and cx < w_row[2]
                                and cy > w_row[1]
                                and cy < w_row[3]
                            ):
                                if w_class_id == 0:
                                    compliance_data["helmet"] = True
                                elif w_class_id == 7:
                                    compliance_data["vest"] = True
                                elif w_class_id == 1:
                                    compliance_data["mask"] = True

            # Draw bounding box
            color = (0, 255, 0) if all(compliance_data.values()) else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    return frame, compliance_data


# ================================
# Compliance Charts
# ================================
def create_compliance_chart(compliance_history):
    df = pd.DataFrame(compliance_history)
    compliance_rates = df.mean() * 100

    gauges = []
    colors = {"helmet": "#2ecc71", "vest": "#3498db", "mask": "#e74c3c"}

    for item, rate in compliance_rates.items():
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=rate,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"{item.title()} Compliance"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": colors[item]},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "gray"},
                        {"range": [80, 100], "color": "darkgray"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 80,
                    },
                },
            )
        )
        gauge.update_layout(height=200)
        gauges.append(gauge)

    # Trend chart
    trend_data = pd.DataFrame(compliance_history).rolling(window=5).mean() * 100
    trend_fig = go.Figure()

    for item in trend_data.columns:
        trend_fig.add_trace(
            go.Scatter(y=trend_data[item], name=item.title(), line=dict(color=colors[item], width=2))
        )

    trend_fig.update_layout(
        title="PPE Compliance Trend",
        yaxis_title="Compliance Rate (%)",
        xaxis_title="Frame Number",
        height=300,
        showlegend=True,
        yaxis_range=[0, 100],
    )

    return gauges, trend_fig


# ================================
# Process Uploaded Video
# ================================
def process_uploaded_video(video_file, pose, yolov8m_model, best_model):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    try:
        tfile.write(video_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Error: Could not open video file")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        frame_placeholder = st.empty()

        compliance_history = []
        frame_count = 0
        frame_skip = 5

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                processed_frame, compliance_data = process_frame(frame, pose, yolov8m_model, best_model)
                compliance_history.append(compliance_data)

                progress = int((frame_count / total_frames) * 100)
                progress_bar.progress(progress)

                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb)

        finally:
            cap.release()

        return compliance_history

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return []

    finally:
        if os.path.exists(tfile.name):
            try:
                os.unlink(tfile.name)
            except Exception:
                pass


# ================================
# Webcam Detection
# ================================
def run_webcam_detection(pose, yolov8m_model, best_model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam")
        return

    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    chart_placeholder = st.empty()

    if "webcam_compliance_history" not in st.session_state:
        st.session_state.webcam_compliance_history = []
        st.session_state.frame_count = 0

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        start_button = st.button("Start Monitoring")
    with col2:
        stop_button = st.button("Stop")
    with col3:
        clear_button = st.button("Clear Statistics")

    if clear_button:
        st.session_state.webcam_compliance_history = []
        st.session_state.frame_count = 0

    update_interval = 5

    try:
        while start_button and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, compliance_data = process_frame(frame, pose, yolov8m_model, best_model)
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            st.session_state.frame_count += 1
            if st.session_state.frame_count % update_interval == 0:
                st.session_state.webcam_compliance_history.append(compliance_data)
                if len(st.session_state.webcam_compliance_history) > 50:
                    st.session_state.webcam_compliance_history.pop(0)

                if len(st.session_state.webcam_compliance_history) > 0:
                    with stats_placeholder.container():
                        current_rates = pd.DataFrame([compliance_data]).mean() * 100
                        cols = st.columns(3)
                        for i, (item, rate) in enumerate(current_rates.items()):
                            cols[i].metric(
                                f"Current {item.title()} Compliance",
                                f"{rate:.1f}%",
                                f"{rate - 80:.1f}%" if rate >= 80 else f"{rate - 80:.1f}%",
                            )

                    gauges, trend_fig = create_compliance_chart(st.session_state.webcam_compliance_history)
                    with chart_placeholder.container():
                        gauge_cols = st.columns(3)
                        for i, gauge in enumerate(gauges):
                            with gauge_cols[i]:
                                st.plotly_chart(gauge, use_container_width=True, key=f"gauge_{i}_{st.session_state.frame_count}")
                        st.plotly_chart(trend_fig, use_container_width=True, key=f"trend_{st.session_state.frame_count}")

            time.sleep(0.1)

    finally:
        cap.release()


# ================================
# Main App
# ================================
def main():
    st.set_page_config(page_title="PPE Compliance Monitor", layout="wide")
    st.title("PPE Compliance Monitoring Dashboard")

    mpDraw, mpPose, pose, yolov8m_model, best_model = load_models()

    st.sidebar.title("Controls")
    source_option = st.sidebar.radio("Select Source", ["Webcam", "Upload Video"])

    if source_option == "Webcam":
        st.subheader("Live PPE Compliance Monitoring")
        run_webcam_detection(pose, yolov8m_model, best_model)
    else:
        st.subheader("Video File Analysis")
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])
        if uploaded_file is not None:
            with st.spinner("Processing video..."):
                compliance_history = process_uploaded_video(uploaded_file, pose, yolov8m_model, best_model)
                if compliance_history:
                    gauges, trend_fig = create_compliance_chart(compliance_history)

                    gauge_cols = st.columns(3)
                    for i, gauge in enumerate(gauges):
                        with gauge_cols[i]:
                            st.plotly_chart(gauge, use_container_width=True, key=f"uploaded_gauge_{i}")

                    st.plotly_chart(trend_fig, use_container_width=True, key="uploaded_trend")

                    st.subheader("Summary Statistics")
                    df = pd.DataFrame(compliance_history)
                    mean_compliance = df.mean() * 100

                    summary_cols = st.columns(3)
                    for i, (item, rate) in enumerate(mean_compliance.items()):
                        with summary_cols[i]:
                            st.metric(f"Average {item.title()} Compliance", f"{rate:.1f}%")


if __name__ == "__main__":
    main()
