import streamlit as st
import cv2
import tempfile
import time
from ultralytics import YOLO
from utils.analytics import traffic_density

# ==========================================================
# 🌐 PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Smart City Traffic Surveillance",
    page_icon="🚦",
    layout="wide"
)

# ==========================================================
# 🎨 LIGHT UI CSS
# ==========================================================
st.markdown("""
<style>
body {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 1.2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
h1 {
    color: #1f2937;
    font-weight: 700;
}
h2, h3 {
    color: #374151;
}
.metric-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
}
.alert-high {
    background: #fee2e2;
    color: #991b1b;
    padding: 16px;
    border-radius: 14px;
    font-weight: 700;
    border: 2px solid #fecaca;
}
.alert-medium {
    background: #fef3c7;
    color: #92400e;
    padding: 16px;
    border-radius: 14px;
    font-weight: 700;
    border: 2px solid #fde68a;
}
.alert-low {
    background: #dcfce7;
    color: #166534;
    padding: 16px;
    border-radius: 14px;
    font-weight: 700;
    border: 2px solid #86efac;
}
.video-box {
    border-radius: 18px;
    overflow: hidden;
    background: white;
    box-shadow: 0 12px 30px rgba(0,0,0,0.08);
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# ⚙️ SIDEBAR
# ==========================================================
st.sidebar.title("🛠️ Controls")
st.sidebar.caption("Adjust detection sensitivity")

confidence = st.sidebar.slider("🎯 Confidence Threshold", 0.1, 0.9, 0.25)
iou_thres = st.sidebar.slider("🔗 IoU Threshold", 0.1, 0.9, 0.45)

st.sidebar.markdown("---")
st.sidebar.info("🚦 Smart City Traffic AI")

# ==========================================================
# 🤖 LOAD MODEL
# ==========================================================
@st.cache_resource
def load_model():
    return YOLO("models/yolov8s.pt")

model = load_model()

VEHICLES = ["car", "bus", "truck", "motorcycle"]
PERSON = "person"

# ==========================================================
# 🏙️ HEADER
# ==========================================================
st.markdown("""
<h1>🚦 Smart City Traffic Surveillance</h1>
<p style="color:#6b7280;">
AI-powered traffic monitoring with real-time congestion alerts
</p>
""", unsafe_allow_html=True)

# ==========================================================
# 🚨 ALERT LOGIC
# ==========================================================
def congestion_alert(vehicle_count):
    if vehicle_count >= 25:
        return "HIGH"
    elif vehicle_count >= 10:
        return "MEDIUM"
    return "LOW"

# ==========================================================
# 📊 DASHBOARD
# ==========================================================
def render_dashboard(persons, vehicles, density, alert):
    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"""
    <div class="metric-card">
        <h3>👤 Persons</h3>
        <h2>{persons}</h2>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="metric-card">
        <h3>🚗 Vehicles</h3>
        <h2>{vehicles}</h2>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="metric-card">
        <h3>🚦 Traffic Density</h3>
        <h2>{density}</h2>
    </div>
    """, unsafe_allow_html=True)

    if alert == "HIGH":
        col4.markdown('<div class="alert-high">🚨 CONGESTION</div>', unsafe_allow_html=True)
    elif alert == "MEDIUM":
        col4.markdown('<div class="alert-medium">⚠️ MODERATE</div>', unsafe_allow_html=True)
    else:
        col4.markdown('<div class="alert-low">✅ SMOOTH</div>', unsafe_allow_html=True)

# ==========================================================
# 🎥 VIDEO PROCESSING (FAST)
# ==========================================================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    dashboard = st.empty()
    frame_box = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 🔥 RESIZE FRAME (Option 1)
        frame_small = cv2.resize(frame, (640, 360))

        results = model.predict(
            frame_small,
            conf=confidence,
            iou=iou_thres,
            verbose=False
        )

        annotated = results[0].plot()

        vehicles = persons = 0
        for box in results[0].boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            if name in VEHICLES:
                vehicles += 1
            if name == PERSON:
                persons += 1

        density = traffic_density(vehicles)
        alert = congestion_alert(vehicles)

        with dashboard.container():
            render_dashboard(persons, vehicles, density, alert)

        with frame_box.container():
            st.markdown('<div class="video-box">', unsafe_allow_html=True)
            st.image(annotated, channels="BGR", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        time.sleep(0.03)

    cap.release()

# ==========================================================
# 📤 VIDEO UPLOAD
# ==========================================================
st.markdown("## 📹 Video Traffic Analysis")

video_file = st.file_uploader(
    "Upload road traffic footage",
    type=["mp4", "avi", "mov"]
)

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    st.success("🚀 Processing video...")
    process_video(tfile.name)

# ==========================================================
# 📸 WEBCAM
# ==========================================================
st.markdown("## 📸 Live Camera Monitoring")

start_cam = st.button("▶ Start Camera")
stop_cam = st.button("⏹ Stop Camera")

if start_cam:
    cap = cv2.VideoCapture(0)

    dashboard = st.empty()
    frame_box = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 🔥 RESIZE FRAME (Option 1)
        frame_small = cv2.resize(frame, (640, 360))

        results = model.predict(
            frame_small,
            conf=confidence,
            iou=iou_thres,
            verbose=False
        )

        annotated = results[0].plot()

        vehicles = persons = 0
        for box in results[0].boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            if name in VEHICLES:
                vehicles += 1
            if name == PERSON:
                persons += 1

        density = traffic_density(vehicles)
        alert = congestion_alert(vehicles)

        with dashboard.container():
            render_dashboard(persons, vehicles, density, alert)

        with frame_box.container():
            st.markdown('<div class="video-box">', unsafe_allow_html=True)
            st.image(annotated, channels="BGR", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if stop_cam:
            break

        time.sleep(0.03)

    cap.release()
