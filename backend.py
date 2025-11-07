from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
import pandas as pd
from datetime import datetime
import socket
import signal
import sys
import threading
import time

app = Flask(__name__)

# Global variables to store model and metadata
current_model = None
feature_names = None
last_training_time = None

# Global variable to track shutdown
is_shutting_down = False

def load_model():
    global current_model, feature_names, last_training_time
    try:
        # IMPLEMENTATION: Changed to match the actual path used by trainer
        model_path = '/shared-volume/model.joblib' 
        if os.path.exists(model_path):
            model_info = joblib.load(model_path)
            current_model = model_info['model']
            feature_names = model_info['feature_names']
            last_training_time = model_info['training_time']
            print(f"[{datetime.now()}] Model loaded successfully from {model_path} last trained at {last_training_time}")
        else:
            print(f"[{datetime.now()}] No model file found at {model_path}")
    except Exception as e:
        print(f"[{datetime.now()}] Error loading model: {e}")
     
@app.route('/model-info')
def get_model_info():
    if is_shutting_down:
        return jsonify({"status": "shutting down", "host": socket.gethostname()}), 503
        
    if current_model is None:
        return jsonify({"status": "No model loaded"}), 503
    
    return jsonify({
        "status": "active",
        "last_training_time": last_training_time,
        "features": feature_names,
        "model_type": type(current_model).__name__,
        "host": socket.gethostname()
    })


@app.route('/predict', methods=['POST'])
def predict_engagement():
    if is_shutting_down:
        return jsonify({"error": "Service shutting down"}), 503
        
    if current_model is None:
        return jsonify({"error": "No model loaded"}), 503
    
    try:
        # Get user features from request
        user_data = request.get_json()
        
        # Validate all required features are present
        if not all(feature in user_data for feature in feature_names):
            return jsonify({
                "error": "Missing features",
                "required_features": feature_names
            }), 400
        
        # Create feature vector
        features = pd.DataFrame([user_data])[feature_names]
        
        # IMPLEMENTATION: Added prediction using the current_model
        engagement_score = current_model.predict(features)[0]
        
        return jsonify({
            "engagement_score": float(engagement_score),
            "features_used": user_data,
            "model_training_time": last_training_time,
            "host": socket.gethostname()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    if is_shutting_down:
        return jsonify({"status": "shutting down", "host": socket.gethostname()}), 503
    return jsonify({"status": "healthy", "host": socket.gethostname()})

def _handle_sigterm(signum, frame):
    global is_shutting_down
    try:
        host = socket.gethostname()
        is_shutting_down = True
        print(f"[{datetime.now()}] SIGTERM received. Host being terminated: {host}. Last model training time: {last_training_time}")
        print(f"[{datetime.now()}] Starting graceful shutdown...")
        # Simulate cleanup work
        time.sleep(2)
        print(f"[{datetime.now()}] Cleanup completed. Shutting down.")
    except Exception as err:
        print(f"[{datetime.now()}] SIGTERM handler error: {err}")
    finally:
        sys.exit(0)

signal.signal(signal.SIGTERM, _handle_sigterm)

def _periodic_model_reloader(interval_seconds=30):
    while True:
        try:
            if not is_shutting_down:
                load_model()
        except Exception as e:
            print(f"[{datetime.now()}] Error reloading model: {e}")
        time.sleep(interval_seconds)

_reloader_thread = threading.Thread(target=_periodic_model_reloader, args=(30,), daemon=True)
_reloader_thread.start()

if __name__ == '__main__':
    load_model()  # Load model at startup
    app.run(host='0.0.0.0', port=5001)