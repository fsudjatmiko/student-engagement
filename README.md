# Student Engagement Detection

A real-time student engagement detection system powered by deep learning models. This application provides live webcam analysis, video processing, model benchmarking, and TensorBoard visualization capabilities.

## Features

- **Live Webcam Detection** - Real-time engagement analysis with configurable webcam selection
- **Video Processing** - Batch processing with optional performance benchmarking
- **Model Support** - Compatible with Keras (.keras, .h5) and TensorFlow Lite (.tflite) models
- **Benchmarking Suite** - Comprehensive model evaluation with metrics, confusion matrices, and visualizations
- **TensorBoard Integration** - Generate and visualize performance logs from inference results

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

## Models

The trained models are stored externally due to file size constraints.

Link: https://drive.google.com/drive/folders/17tVPjGMNK1fOEHfQxHPh9UMcBGq8Z798?usp=sharing

Place downloaded models in the `models/` directory.

## Dataset

The models are trained on a customized Zoom dataset. Access available upon request.

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- OpenCV
- Streamlit
- Additional dependencies listed in `requirements.txt`
