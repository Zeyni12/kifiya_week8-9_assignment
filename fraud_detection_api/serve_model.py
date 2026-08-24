# # from flask import Flask, request, jsonify
# # import joblib
# # import logging

# # # Initialize Flask App
# # app = Flask(__name__)

# # # Configure Logging
# # logging.basicConfig(filename='api.log', level=logging.INFO)

# # # Load the trained model
# # model_path = "C:/Users/hp/kifiya_acadamy_week8&9/kifiya_week8-9_assignment/notebooks/credit_card_model.pkl"
# # model = joblib.load(model_path)

# # @app.route('/credit_fraud_detection', methods=['GET'])
# # def health():
# #     return jsonify({'status': 'API is running'})

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     try:
# #         data = request.json  # Get JSON request data
# #         prediction = model.predict([data['features']])
# #         fraud_label = "Non_Fraud" if prediction[0] == 0 else "Fraud"
# #         logging.info(f"Prediction request: {data} -> {fraud_label}")
# #         return jsonify({'fraud_prediction': fraud_label})
# #     except Exception as e:
# #         logging.error(f"Error processing request: {str(e)}")
# #         return jsonify({'error': str(e)}), 400

# # if __name__ == '__main__':
# #     app.run(host='0.0.0.0', port=5000)


# from flask import Flask, render_template, request
# from model import FraudModel
# import logging

# app = Flask(__name__)

# logging.basicConfig(
#     filename="api.log",
#     level=logging.INFO
# )

# # Load model
# fraud_model = FraudModel(
#     "fraud_data_model.pkl"
# )


# @app.route("/")
# def home():
#     return render_template("home.html")


# @app.route("/predict", methods=["GET", "POST"])
# def predict():

#     if request.method == "GET":
#         return render_template("predict.html")

#     try:

#         data = {
#             "user_id": request.form["user_id"],
#             "signup_time": request.form["signup_time"],
#             "purchase_time": request.form["purchase_time"],
#             "purchase_value": float(
#                 request.form["purchase_value"]
#             ),
#             "device_id": request.form["device_id"],
#             "source": request.form["source"],
#             "browser": request.form["browser"],
#             "sex": request.form["sex"],
#             "age": int(
#                 request.form["age"]
#             ),
#             "country": request.form["country"]
#         }

#         prediction = fraud_model.predict(data)

#         probability = fraud_model.predict_proba(data)

#         if prediction == 1:
#             result = "Fraud"
#         else:
#             result = "Non-Fraud"

#         logging.info(
#             f"Prediction: {result}, "
#             f"Probability: {probability:.4f}"
#         )

#         return render_template(
#             "predict.html",
#             prediction=result,
#             probability=round(
#                 probability * 100,
#                 2
#             )
#         )

#     except Exception as e:

#         logging.error(
#             f"Prediction error: {str(e)}"
#         )

#         return render_template(
#             "predict.html",
#             error=str(e)
#         )


# if __name__ == "__main__":

#     app.run(
#         host="0.0.0.0",
#         port=5001,
#         debug=True
#     )


from flask import Flask, render_template, request
from model import FraudModel

app = Flask(__name__)

# Load model once when Flask starts
fraud_model = FraudModel()


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        try:

            # Get form data
            data = request.form.to_dict()

            print("\nReceived data:")
            print(data)

            # Make prediction
            result = fraud_model.predict(data)

            print("\nPrediction:")
            print(result)

            return render_template(
                "predict.html",
                prediction=result["label"],
                probability=result["probability"]
            )

        except Exception as e:

            print("\nERROR:")
            print(e)

            return render_template(
                "predict.html",
                error=str(e)
            )

    return render_template("predict.html")


@app.route("/health")
def health():

    return {
        "status": "API is running",
        "model": "fraud_data_model.pkl"
    }


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )