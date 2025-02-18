from flask import Flask, request, jsonify
import joblib
import logging

# Initialize Flask App
app = Flask(__name__)

# Configure Logging
logging.basicConfig(filename='api.log', level=logging.INFO)

# Load the trained model
model_path = "C:/Users/hp/kifiya_acadamy_week8&9/kifiya_week8-9_assignment/notebooks/credit_card_model.pkl"
model = joblib.load(model_path)

@app.route('/credit_fraud_detection', methods=['GET'])
def health():
    return jsonify({'status': 'API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Get JSON request data
        prediction = model.predict([data['features']])
        fraud_label = "Non_Fraud" if prediction[0] == 0 else "Fraud"
        logging.info(f"Prediction request: {data} -> {fraud_label}")
        return jsonify({'fraud_prediction': fraud_label})
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
