import cv2
import tensorflow as tf

def preprocess_frame(frame):
    """Preprocessing untuk frame video."""
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    return frame

def load_model(model_path):
    """Memuat model berdasarkan path yang dipilih."""
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model berhasil dimuat: {model_path}")
    return model