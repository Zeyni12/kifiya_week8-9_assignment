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