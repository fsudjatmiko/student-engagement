import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import tempfile
import os
from utils import preprocess_frame, load_model

# Page Configuration
st.set_page_config(
    page_title="Student Engagement Detection",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="auto"
)

# Dark Modern CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main app background */
    .stApp {
        background-color: #0a0a0a;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 1px solid #2a2a2a;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e5e5e5;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #0a0a0a;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    h3 {
        font-size: 1.2rem;
        font-weight: 500;
        color: #a0a0a0;
        margin-bottom: 1.5rem;
    }
    
    /* Sidebar title */
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #2a2a2a;
    }
    
    /* Sidebar labels */
    .sidebar-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #2a2a2a;
        border: 1px solid #3a3a3a;
        border-radius: 8px;
        color: #ffffff;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #4a4a4a;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background-color: #1a1a1a;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stRadio > div > label {
        color: #e5e5e5;
        background-color: #2a2a2a;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
        border: 1px solid #3a3a3a;
    }
    
    .stRadio > div > label:hover {
        background-color: #3a3a3a;
        border-color: #4a4a4a;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Checkbox */
    .stCheckbox {
        padding: 1rem;
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .stCheckbox label {
        color: #e5e5e5;
        font-weight: 500;
    }
    
    /* Disabled button */
    .stButton > button:disabled {
        background: #2a2a2a;
        color: #666;
        cursor: not-allowed;
    }
    
    .stButton > button:disabled:hover {
        transform: none;
        box-shadow: none;
    }
    
    /* Sidebar toggle button - make it visible */
    button[kind="header"] {
        color: #e5e5e5 !important;
        background-color: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px !important;
    }
    
    button[kind="header"]:hover {
        background-color: #2a2a2a !important;
        border-color: #3a3a3a !important;
    }
    
    [data-testid="collapsedControl"] {
        color: #e5e5e5 !important;
        background-color: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px !important;
    }
    
    [data-testid="collapsedControl"]:hover {
        background-color: #2a2a2a !important;
        border-color: #3a3a3a !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #1a1a1a;
        border: 2px dashed #3a3a3a;
        border-radius: 12px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Info, success, error boxes */
    .stAlert {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        color: #e5e5e5;
    }
    
    /* Video container */
    video {
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
        border: 1px solid #2a2a2a;
    }
    
    /* Image container */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #2a2a2a;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background-color: #2a2a2a;
        color: #ffffff;
        border: 1px solid #3a3a3a;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        width: 100%;
    }
    
    .stDownloadButton > button:hover {
        background-color: #3a3a3a;
        border-color: #4a4a4a;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea;
    }
    
    /* Hide Streamlit branding but keep header for sidebar button */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Configuration
with st.sidebar:
    st.markdown('<p class="sidebar-title">Configuration</p>', unsafe_allow_html=True)
    
    # Model Selection - dynamically load from models folder
    st.markdown('<p class="sidebar-label">Model</p>', unsafe_allow_html=True)
    
    # Get all model files from models folder
    models_folder = "models"
    if os.path.exists(models_folder):
        model_files = [f for f in os.listdir(models_folder) if f.endswith('.keras') or f.endswith('.h5')]
        model_names = [os.path.splitext(f)[0] for f in model_files]
    else:
        model_names = []
        st.error("Models folder not found!")
    
    if model_names:
        model_option = st.selectbox(
            "model",
            model_names,
            label_visibility="collapsed"
        )
        selected_model_file = [f for f in model_files if f.startswith(model_option)][0]
    else:
        st.error("No models found in the models folder!")
        model_option = None
        selected_model_file = None
    
    # Input Source
    st.markdown('<p class="sidebar-label">Input Source</p>', unsafe_allow_html=True)
    input_option = st.radio(
        "input",
        ["Live Webcam", "Upload Video"],
        label_visibility="collapsed"
    )
    
    # Webcam selection (only show if Live Webcam is selected)
    if input_option == "Live Webcam":
        st.markdown('<p class="sidebar-label">Webcam Device</p>', unsafe_allow_html=True)
        webcam_index = st.selectbox(
            "webcam",
            [0, 1, 2],
            format_func=lambda x: f"Camera {x}",
            label_visibility="collapsed"
        )
    
    st.markdown('<div style="margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #2a2a2a;">', unsafe_allow_html=True)
    st.markdown('<p style="color: #666; font-size: 0.75rem;">Student Engagement Detection</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: #444; font-size: 0.7rem;">Powered by TensorFlow</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Load Model
if model_option and selected_model_file:
    model_path = os.path.join("models", selected_model_file)
    with st.spinner('Loading model...'):
        model = load_model(model_path)
else:
    st.error("Please ensure models are available in the models folder")
    st.stop()

# Main Content
st.title("Student Engagement Detection")
st.markdown("### Real-time analysis system")

st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)

if input_option == "Upload Video":
    st.subheader("Video Analysis")
    uploaded_file = st.file_uploader("Upload your video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file:
        if st.button("Start Analysis"):
            st.info("Processing your video... This may take a few moments.")
            
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()
            
            # Process video
            cap = cv2.VideoCapture(tfile.name)
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create output video in outputs folder
            if not os.path.exists("outputs"):
                os.makedirs("outputs")
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"engagement_detection_{timestamp}.mp4"
            output_path = os.path.join("outputs", output_filename)
            
            # Use H.264 codec for better browser compatibility
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocessing Frame
                preprocessed_frame = preprocess_frame(frame)
                prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0), verbose=0)
                engagement_status = "Engaged" if prediction[0][0] > 0.5 else "Disengaged"
                confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
                
                # Add text to frame
                color = (0, 255, 0) if engagement_status == "Engaged" else (0, 0, 255)
                cv2.putText(frame, f"Status: {engagement_status}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Confidence: {confidence:.2%}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                out.write(frame)
                frame_count += 1
                progress_bar.progress(frame_count / total_frames)
            
            cap.release()
            out.release()
            os.unlink(tfile.name)
            
            st.success(f"Analysis complete! Video saved to: {output_path}")
            
            st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
            st.video(output_path)
            
            # Provide download button
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name=output_filename,
                    mime="video/mp4",
                    use_container_width=True
                )
            
elif input_option == "Live Webcam":
    st.subheader("Live Detection")
    st.markdown('<p style="color: #a0a0a0; margin-bottom: 1.5rem;">Real-time engagement analysis using your webcam</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'detection_running' not in st.session_state:
        st.session_state.detection_running = False
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Detection", disabled=st.session_state.detection_running, use_container_width=True):
            st.session_state.detection_running = True
            st.rerun()
    
    with col2:
        if st.button("Stop Detection", disabled=not st.session_state.detection_running, use_container_width=True):
            st.session_state.detection_running = False
            st.rerun()
    
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    
    FRAME_WINDOW = st.empty()
    
    # Only show webcam when detection is running
    if st.session_state.detection_running:
        camera = cv2.VideoCapture(webcam_index)
        
        if not camera.isOpened():
            st.error("Could not access the selected webcam.")
            st.session_state.detection_running = False
        else:
            import time
            while st.session_state.detection_running:
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to read from webcam")
                    break
                
                # Preprocessing Frame
                preprocessed_frame = preprocess_frame(frame)
                prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0), verbose=0)
                engagement_status = "Engaged" if prediction[0][0] > 0.5 else "Disengaged"
                confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
                
                # Add text to frame
                color = (0, 255, 0) if engagement_status == "Engaged" else (0, 0, 255)
                cv2.putText(frame, f"Status: {engagement_status}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Confidence: {confidence:.2%}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb, channels="RGB", use_container_width=True)
                
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
            
            camera.release()
    else:
        st.info("Click 'Start Detection' to begin live webcam analysis")