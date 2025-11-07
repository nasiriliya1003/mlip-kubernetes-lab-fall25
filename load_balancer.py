from flask import Flask, request, jsonify
import itertools
import requests
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IMPLEMENTATION: Added backend server URL for round-robin distribution
BACKEND_SERVERS = [
    "http://flask-backend-service:5001"
]

# Round-robin iterator for distributing requests
server_pool = itertools.cycle(BACKEND_SERVERS)

@app.route('/model-info')
def load_balance():
    backend_url = next(server_pool)
    logger.info(f"Routing GET /model-info to {backend_url}")
    try:
        response = requests.get(f"{backend_url}/model-info", timeout=5)
        try:
            data = response.json()
            return jsonify(data), response.status_code
        except ValueError:
            return jsonify({"error": "Invalid JSON from backend", "raw": response.text}), 502
    except requests.exceptions.RequestException as e:
        logger.error(f"Error contacting backend {backend_url}: {e}")
        return jsonify({"error": "Backend service unavailable"}), 503

@app.route('/predict', methods=['POST'])
def predict():
    backend_url = next(server_pool)
    url = f"{backend_url}/predict"
    
    logger.info(f"Routing POST /predict to {backend_url}")
    
    # IMPLEMENTATION: Added POST request implementation for predict endpoint
    try:
        data = request.get_json()
        response = requests.post(url, json=data, timeout=5)
        try:
            response_data = response.json()
            return jsonify(response_data), response.status_code
        except ValueError:
            return jsonify({"error": "Invalid JSON from backend", "raw": response.text}), 502
    except requests.exceptions.RequestException as e:
        logger.error(f"Error contacting backend {backend_url}: {e}")
        return jsonify({"error": "Backend service unavailable"}), 503

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "load-balancer"})

if __name__ == '__main__':
    # IMPLEMENTATION: Using default port 8080 as specified
    app.run(host='0.0.0.0', port=8080, debug=False)