from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS  # Import CORS

app = Flask(__name__)

# Enable CORS for the specific frontend URL
CORS(app, resources={r"/stock": {"origins": "https://stock-frontend-x3t4.onrender.com"}})

# Load your model
model = joblib.load('stock_model.pkl')

@app.route('/stock', methods=['POST'])
def predict():
    if request.content_type != 'application/json':
        return jsonify({"error": "Content-Type must be application/json"}), 415

    try:
        # Get the input data (features)
        data = request.get_json()
        
        # Validate that the 'features' key is present in the data
        if 'features' not in data:
            return jsonify({"error": "'features' key is required"}), 400

        # Prepare the input for prediction (reshape to (1, 8, 8) for the model)
        # Example: features with shape (1, 8, 8)
        # Ensure the list has 64 elements (8 rows of 8 features)
        if len(data['features']) != 64:
            return jsonify({"error": "Features must have 64 elements (8x8 matrix)"}), 400
        
        features = np.array(data['features']).reshape(1, 8, 8)  # Shape must be (1, 8, 8)
        
        # Make prediction
        prediction = model.predict(features)

        return jsonify({"prediction": prediction.tolist()})  # Send prediction as response

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
