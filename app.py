from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Replace with your Azure ML model endpoint and API key
MODEL_API_URL = "https://careerpathendpoint.eastus.inference.ml.azure.com/score"
API_KEY = "dB50aEy4HQrStTgmB6tcIJivGn9Hn7id"

@app.route('/')
def home():
    return "<h1>Welcome to the Career Path Recommender API!</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data from the request
        input_data = request.get_json()
        skills = input_data.get("Skills")
        interests = input_data.get("Interests")

        if not skills or not interests:
            return jsonify({"error": "Both 'Skills' and 'Interests' are required"}), 400

        # Prepare data for the model API
        payload = {
            "input_data": {
                "columns": ["Skills", "Interests"],
                "data": [[skills, interests]]
            }
        }

        # Set headers and send request to the Azure ML model API
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.post(MODEL_API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            # Return the prediction from the model API
            prediction = response.json()[0] if isinstance(response.json(), list) else response.json().get("predictions", ["No prediction"])[0]
            return jsonify({"predictions": prediction}), 200
        else:
            return jsonify({
                "error": "Failed to get prediction",
                "details": response.text
            }), response.status_code
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
