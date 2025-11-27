import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
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
        model_files = [f for f in os.listdir(models_folder) if f.endswith(('.keras', '.h5', '.tflite'))]
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
        model_type = 'tflite' if selected_model_file.endswith('.tflite') else 'keras'
    else:
        st.error("No models found in the models folder!")
        model_option = None
        selected_model_file = None
        model_type = None
    
    # Input Source
    st.markdown('<p class="sidebar-label">Input Source</p>', unsafe_allow_html=True)
    input_option = st.radio(
        "input",
        ["Live Webcam", "Upload Video", "Model Benchmark", "TensorBoard"],
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
        if model_type == 'tflite':
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            model = None  # Set to None, we'll use interpreter instead
        else:
            # Load Keras model
            model = load_model(model_path)
            interpreter = None
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
        # Benchmark option
        benchmark_mode = st.checkbox("Enable Benchmark Mode")
        
        if st.button("Start Analysis"):
            if benchmark_mode:
                st.info("Processing your video with benchmarking enabled...")
            else:
                st.info("Processing your video... This may take a few moments.")
            
            # Initialize benchmark variables
            import psutil
            import time as time_module
            
            process = psutil.Process()
            start_time = time_module.time()
            initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
            peak_memory = initial_memory
            
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
            
            # First save as AVI (which always works)
            temp_output_filename = f"engagement_detection_{timestamp}_temp.avi"
            temp_output_path = os.path.join("outputs", temp_output_filename)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
            
            output_filename = f"engagement_detection_{timestamp}.mp4"
            output_path = os.path.join("outputs", output_filename)
            
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocessing Frame
                preprocessed_frame = preprocess_frame(frame)
                
                # Run inference based on model type
                if model_type == 'tflite':
                    input_data = np.expand_dims(preprocessed_frame, axis=0).astype(np.float32)
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    prediction = interpreter.get_tensor(output_details[0]['index'])
                else:
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
                
                # Track peak memory if benchmarking
                if benchmark_mode:
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    peak_memory = max(peak_memory, current_memory)
            
            cap.release()
            out.release()
            os.unlink(tfile.name)
            
            # Convert AVI to MP4 for better browser compatibility
            try:
                cap_convert = cv2.VideoCapture(temp_output_path)
                fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
                out_mp4 = cv2.VideoWriter(output_path, fourcc_mp4, fps, (width, height))
                
                while True:
                    ret, frame = cap_convert.read()
                    if not ret:
                        break
                    out_mp4.write(frame)
                
                cap_convert.release()
                out_mp4.release()
                
                # Remove temp AVI file
                os.unlink(temp_output_path)
            except Exception as e:
                # If conversion fails, use the AVI file
                output_path = temp_output_path
                output_filename = temp_output_filename
            
            # Calculate benchmark results
            end_time = time_module.time()
            total_time = end_time - start_time
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = peak_memory - initial_memory
            
            st.success(f"Analysis complete! Video saved to: {output_path}")
            
            # Display benchmark results if enabled
            if benchmark_mode:
                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                st.markdown("### Benchmark Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Processing Time", f"{total_time:.2f}s")
                with col2:
                    st.metric("Peak RAM Usage", f"{peak_memory:.2f} MB")
                with col3:
                    st.metric("RAM Increase", f"{memory_used:.2f} MB")
                with col4:
                    st.metric("FPS", f"{total_frames / total_time:.2f}")
            
            st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
            
            # Display video using file data instead of path
            with open(output_path, 'rb') as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)
            
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
                
                # Run inference based on model type
                if model_type == 'tflite':
                    input_data = np.expand_dims(preprocessed_frame, axis=0).astype(np.float32)
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    prediction = interpreter.get_tensor(output_details[0]['index'])
                else:
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

elif input_option == "Model Benchmark":
    st.subheader("Model Benchmark")
    st.markdown('<p style="color: #a0a0a0; margin-bottom: 1.5rem;">Evaluate model performance with test data</p>', unsafe_allow_html=True)
    
    # Upload test data
    st.markdown("#### Upload Test Data")
    col1, col2 = st.columns(2)
    
    with col1:
        test_video = st.file_uploader("Upload test video", type=["mp4", "avi", "mov"], key="benchmark_video")
    
    with col2:
        test_labels = st.file_uploader("Upload ground truth labels (CSV)", type=["csv"], key="benchmark_labels")
        st.markdown('<p style="color: #666; font-size: 0.8rem; margin-top: 0.5rem;">CSV format: frame_number, label (0 or 1)</p>', unsafe_allow_html=True)
    
    if test_video and test_labels:
        if st.button("Run Benchmark", use_container_width=True):
            import pandas as pd
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score
            import seaborn as sns
            
            st.info("Running benchmark... This may take a few moments.")
            
            # Save uploaded files temporarily
            import tempfile
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(test_video.read())
            tfile.close()
            
            # Read labels
            labels_df = pd.read_csv(test_labels)
            if labels_df.shape[1] == 2:
                labels_df.columns = ['frame', 'label']
            else:
                st.error("CSV should have 2 columns: frame_number, label")
                st.stop()
            
            # Process video and collect predictions
            cap = cv2.VideoCapture(tfile.name)
            predictions = []
            ground_truth = []
            confidence_scores = []
            frame_count = 0
            
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Benchmark timing
            import time as time_module
            import psutil
            
            process = psutil.Process()
            start_time = time_module.time()
            initial_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = initial_memory
            
            inference_times = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Only process frames that have labels
                if frame_count in labels_df['frame'].values:
                    # Preprocessing Frame
                    preprocessed_frame = preprocess_frame(frame)
                    
                    # Time the inference
                    inference_start = time_module.time()
                    
                    # Run inference based on model type
                    if model_type == 'tflite':
                        input_data = np.expand_dims(preprocessed_frame, axis=0).astype(np.float32)
                        interpreter.set_tensor(input_details[0]['index'], input_data)
                        interpreter.invoke()
                        prediction = interpreter.get_tensor(output_details[0]['index'])
                    else:
                        prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0), verbose=0)
                    
                    inference_time = time_module.time() - inference_start
                    inference_times.append(inference_time)
                    
                    pred_class = 1 if prediction[0][0] > 0.5 else 0
                    confidence = prediction[0][0]
                    
                    predictions.append(pred_class)
                    confidence_scores.append(confidence)
                    ground_truth.append(labels_df[labels_df['frame'] == frame_count]['label'].values[0])
                    
                    # Track peak memory
                    current_memory = process.memory_info().rss / 1024 / 1024
                    peak_memory = max(peak_memory, current_memory)
                
                progress_bar.progress(frame_count / total_frames)
            
            cap.release()
            os.unlink(tfile.name)
            
            end_time = time_module.time()
            total_time = end_time - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(ground_truth, predictions)
            precision = precision_score(ground_truth, predictions)
            recall = recall_score(ground_truth, predictions)
            f1 = f1_score(ground_truth, predictions)
            cm = confusion_matrix(ground_truth, predictions)
            
            # Display results
            st.success("Benchmark complete!")
            
            st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
            st.markdown("### Performance Metrics")
            
            # Metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy:.2%}")
            with col2:
                st.metric("Precision", f"{precision:.2%}")
            with col3:
                st.metric("Recall", f"{recall:.2%}")
            with col4:
                st.metric("F1 Score", f"{f1:.2%}")
            
            st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
            
            # System Performance
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Time", f"{total_time:.2f}s")
            with col2:
                st.metric("Avg Inference", f"{np.mean(inference_times)*1000:.2f}ms")
            with col3:
                st.metric("Peak RAM", f"{peak_memory:.2f} MB")
            with col4:
                st.metric("FPS", f"{len(predictions)/total_time:.2f}")
            
            st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Confusion Matrix")
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Disengaged', 'Engaged'],
                           yticklabels=['Disengaged', 'Engaged'])
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### Confidence Distribution")
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.hist(confidence_scores, bins=30, color='#667eea', alpha=0.7, edgecolor='black')
                ax.set_xlabel('Confidence Score')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Confidence Scores')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
            
            # Inference Time Distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Inference Time Distribution")
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.hist([t*1000 for t in inference_times], bins=30, color='#764ba2', alpha=0.7, edgecolor='black')
                ax.set_xlabel('Inference Time (ms)')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Inference Times')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### Prediction Over Time")
                fig, ax = plt.subplots(figsize=(6, 5))
                frames = list(range(len(predictions)))
                ax.plot(frames, ground_truth, 'g-', label='Ground Truth', alpha=0.7, linewidth=2)
                ax.plot(frames, predictions, 'r--', label='Predictions', alpha=0.7, linewidth=2)
                ax.set_xlabel('Frame Number')
                ax.set_ylabel('Class (0=Disengaged, 1=Engaged)')
                ax.set_title('Predictions vs Ground Truth')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
            
            # Detailed Classification Report
            st.markdown("#### Detailed Classification Report")
            report = classification_report(ground_truth, predictions, 
                                          target_names=['Disengaged', 'Engaged'],
                                          output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
    else:
        st.info("Upload both test video and ground truth labels to run benchmark")

elif input_option == "TensorBoard":
    st.subheader("TensorBoard Visualization")
    st.markdown('<p style="color: #a0a0a0; margin-bottom: 1.5rem;">Create and view benchmark results in TensorBoard format</p>', unsafe_allow_html=True)
    
    # Option to create new logs or view existing
    tab1, tab2 = st.tabs(["Create Benchmark Logs", "View Existing Logs"])
    
    with tab1:
        st.markdown("#### Generate TensorBoard Logs from Benchmark")
        st.markdown('<p style="color: #a0a0a0; margin-bottom: 1rem;">Run inference on test data and generate TensorBoard logs</p>', unsafe_allow_html=True)
        
        # Upload test data
        col1, col2 = st.columns(2)
        
        with col1:
            test_video_tb = st.file_uploader("Upload test video", type=["mp4", "avi", "mov"], key="tensorboard_video")
        
        with col2:
            test_labels_tb = st.file_uploader("Upload ground truth labels (CSV)", type=["csv"], key="tensorboard_labels")
            st.markdown('<p style="color: #666; font-size: 0.8rem; margin-top: 0.5rem;">CSV format: frame_number, label (0 or 1)</p>', unsafe_allow_html=True)
        
        log_name = st.text_input("Log name", f"benchmark_{model_option}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
        
        if test_video_tb and test_labels_tb:
            if st.button("Generate TensorBoard Logs", use_container_width=True):
                import pandas as pd
                from datetime import datetime
                
                st.info("Generating TensorBoard logs... This may take a few moments.")
                
                # Create logs directory
                log_dir = os.path.join("logs", log_name)
                os.makedirs(log_dir, exist_ok=True)
                
                # Create TensorBoard writer
                writer = tf.summary.create_file_writer(log_dir)
                
                # Save uploaded files temporarily
                import tempfile
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(test_video_tb.read())
                tfile.close()
                
                # Read labels
                labels_df = pd.read_csv(test_labels_tb)
                if labels_df.shape[1] == 2:
                    labels_df.columns = ['frame', 'label']
                else:
                    st.error("CSV should have 2 columns: frame_number, label")
                    st.stop()
                
                # Process video
                cap = cv2.VideoCapture(tfile.name)
                predictions = []
                ground_truth = []
                frame_count = 0
                
                progress_bar = st.progress(0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                correct_predictions = 0
                total_predictions = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    if frame_count in labels_df['frame'].values:
                        # Preprocessing Frame
                        preprocessed_frame = preprocess_frame(frame)
                        
                        # Run inference
                        if model_type == 'tflite':
                            input_data = np.expand_dims(preprocessed_frame, axis=0).astype(np.float32)
                            interpreter.set_tensor(input_details[0]['index'], input_data)
                            interpreter.invoke()
                            prediction = interpreter.get_tensor(output_details[0]['index'])
                        else:
                            prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0), verbose=0)
                        
                        pred_class = 1 if prediction[0][0] > 0.5 else 0
                        true_label = labels_df[labels_df['frame'] == frame_count]['label'].values[0]
                        
                        predictions.append(pred_class)
                        ground_truth.append(true_label)
                        
                        # Update running metrics
                        if pred_class == true_label:
                            correct_predictions += 1
                        total_predictions += 1
                        
                        # Log to TensorBoard
                        with writer.as_default():
                            tf.summary.scalar('confidence', float(prediction[0][0]), step=frame_count)
                            tf.summary.scalar('accuracy', correct_predictions / total_predictions, step=frame_count)
                            tf.summary.scalar('prediction', float(pred_class), step=frame_count)
                            tf.summary.scalar('ground_truth', float(true_label), step=frame_count)
                            
                            # Log image with prediction
                            if frame_count % 50 == 0:  # Log every 50th frame
                                status = "Engaged" if pred_class == 1 else "Disengaged"
                                color = (0, 255, 0) if pred_class == true_label else (255, 0, 0)
                                cv2.putText(frame, f"Pred: {status}", (10, 30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                cv2.putText(frame, f"True: {'Engaged' if true_label == 1 else 'Disengaged'}", (10, 60),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frame_tensor = tf.expand_dims(frame_rgb, 0)
                                tf.summary.image('predictions', frame_tensor, step=frame_count)
                    
                    progress_bar.progress(frame_count / total_frames)
                
                # Final metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                
                accuracy = accuracy_score(ground_truth, predictions)
                precision = precision_score(ground_truth, predictions, zero_division=0)
                recall = recall_score(ground_truth, predictions, zero_division=0)
                f1 = f1_score(ground_truth, predictions, zero_division=0)
                cm = confusion_matrix(ground_truth, predictions)
                
                # Log final metrics
                with writer.as_default():
                    tf.summary.scalar('final/accuracy', accuracy, step=0)
                    tf.summary.scalar('final/precision', precision, step=0)
                    tf.summary.scalar('final/recall', recall, step=0)
                    tf.summary.scalar('final/f1_score', f1, step=0)
                    
                    # Log confusion matrix as image
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                               xticklabels=['Disengaged', 'Engaged'],
                               yticklabels=['Disengaged', 'Engaged'])
                    ax.set_ylabel('True Label')
                    ax.set_xlabel('Predicted Label')
                    ax.set_title('Confusion Matrix')
                    
                    # Convert plot to image
                    fig.canvas.draw()
                    cm_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    cm_image = cm_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                    cm_image = cm_image[:, :, :3]  # Remove alpha channel
                    cm_tensor = tf.expand_dims(cm_image, 0)
                    tf.summary.image('confusion_matrix', cm_tensor, step=0)
                    plt.close()
                
                writer.close()
                cap.release()
                os.unlink(tfile.name)
                
                st.success(f"TensorBoard logs generated successfully!")
                st.info(f"Logs saved to: {log_dir}")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                with col2:
                    st.metric("Precision", f"{precision:.2%}")
                with col3:
                    st.metric("Recall", f"{recall:.2%}")
                with col4:
                    st.metric("F1 Score", f"{f1:.2%}")
        else:
            st.info("Upload both test video and ground truth labels to generate TensorBoard logs")
    
    with tab2:
        st.markdown("#### View Existing TensorBoard Logs")
        
        # Log directory input
        # Check for common log directories
        common_log_dirs = []
        possible_dirs = ["logs", "tensorboard_logs", "training_logs", "runs"]
        for dir_name in possible_dirs:
            if os.path.exists(dir_name):
                common_log_dirs.append(dir_name)
        
        if common_log_dirs:
            log_dir = st.selectbox("Select log directory", common_log_dirs)
        else:
            log_dir = st.text_input("Enter log directory path", "logs")
        
        if os.path.exists(log_dir):
            # Get all subdirectories (training runs)
            subdirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
            
            if subdirs:
                st.markdown("#### Available Training Runs")
                selected_run = st.selectbox("Select training run", sorted(subdirs, reverse=True))
            
                full_log_path = os.path.join(log_dir, selected_run)
                
                # Start TensorBoard
                if st.button("Launch TensorBoard", use_container_width=True):
                    import subprocess
                    import socket
                    
                    # Find available port
                    def find_free_port():
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.bind(('', 0))
                            s.listen(1)
                            port = s.getsockname()[1]
                        return port
                    
                    port = find_free_port()
                    
                    # Start TensorBoard in background
                    try:
                        # Kill any existing TensorBoard process on this port
                        subprocess.run(['taskkill', '/F', '/IM', 'tensorboard.exe'], 
                                     stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                        
                        # Start new TensorBoard
                        tensorboard_process = subprocess.Popen(
                            ['tensorboard', '--logdir', full_log_path, '--port', str(port), '--host', 'localhost'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        
                        import time
                        time.sleep(3)  # Wait for TensorBoard to start
                        
                        st.success(f"TensorBoard started on port {port}")
                        st.markdown(f"### [Open TensorBoard in New Tab](http://localhost:{port})")
                        
                        # Embed TensorBoard using iframe
                        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
                        st.components.v1.iframe(f"http://localhost:{port}", height=800, scrolling=True)
                        
                    except Exception as e:
                        st.error(f"Failed to start TensorBoard: {str(e)}")
                        st.info("Make sure TensorBoard is installed: pip install tensorboard")
                
                # Display log directory info
                st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
                st.markdown("#### Log Directory Information")
                st.code(f"Path: {full_log_path}")
                
                # List files in the log directory
                try:
                    from tensorboard.backend.event_processing import event_file_loader
                    from tensorflow.python.summary.summary_iterator import summary_iterator
                    
                    st.markdown("#### Training Summary")
                    
                    # Find event files
                    event_files = []
                    for root, dirs, files in os.walk(full_log_path):
                        for file in files:
                            if file.startswith('events.out.tfevents'):
                                event_files.append(os.path.join(root, file))
                    
                    if event_files:
                        # Read metrics from event files
                        metrics_data = {}
                        
                        for event_file in event_files:
                            try:
                                for event in summary_iterator(event_file):
                                    for value in event.summary.value:
                                        if value.tag not in metrics_data:
                                            metrics_data[value.tag] = []
                                        
                                        if value.HasField('simple_value'):
                                            metrics_data[value.tag].append({
                                                'step': event.step,
                                                'value': value.simple_value
                                            })
                            except Exception:
                                continue
                        
                        if metrics_data:
                            # Display metrics
                            import pandas as pd
                            import matplotlib.pyplot as plt
                            
                            # Create plots for each metric
                            metric_names = list(metrics_data.keys())
                            
                            if len(metric_names) > 0:
                                col1, col2 = st.columns(2)
                                
                                for idx, metric_name in enumerate(metric_names[:4]):  # Show first 4 metrics
                                    data = metrics_data[metric_name]
                                    if len(data) > 0:
                                        df = pd.DataFrame(data)
                                        
                                        fig, ax = plt.subplots(figsize=(6, 4))
                                        ax.plot(df['step'], df['value'], color='#667eea', linewidth=2)
                                        ax.set_xlabel('Step')
                                        ax.set_ylabel(metric_name)
                                        ax.set_title(metric_name.replace('_', ' ').title())
                                        ax.grid(True, alpha=0.3)
                                        plt.tight_layout()
                                        
                                        if idx % 2 == 0:
                                            with col1:
                                                st.pyplot(fig)
                                        else:
                                            with col2:
                                                st.pyplot(fig)
                                        
                                        plt.close()
                                
                                # Show all available metrics
                                st.markdown("#### Available Metrics")
                                st.write(", ".join(metric_names))
                        else:
                            st.info("No metrics found in event files")
                    else:
                        st.info("No event files found in the selected directory")
                        
                except ImportError:
                    st.warning("TensorBoard not installed. Install it with: pip install tensorboard")
                except Exception as e:
                    st.warning(f"Could not read event files: {str(e)}")
            else:
                st.warning(f"No training runs found in '{log_dir}' directory")
        else:
            st.error(f"Log directory '{log_dir}' does not exist")
            st.info("Generate logs in the 'Create Benchmark Logs' tab first")