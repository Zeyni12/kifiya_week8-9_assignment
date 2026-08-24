import os
import joblib
import pandas as pd
import numpy as np


class FraudModel:

    def __init__(self, model_path=None):

        # --------------------------------------------------
        # Find project directory
        # --------------------------------------------------

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # --------------------------------------------------
        # Load trained model
        # --------------------------------------------------

        if model_path is None:
            model_path = os.path.join(
                BASE_DIR,
                "..",
                "notebooks",
                "fraud_data_model.pkl"
            )

        self.model = joblib.load(model_path)

        print("Model loaded successfully")

        print("Expected features:")
        print(self.model.feature_names_in_)

    def preprocess(self, data):

        # --------------------------------------------------
        # Convert incoming dictionary to DataFrame
        # --------------------------------------------------

        df = pd.DataFrame([data])

        # --------------------------------------------------
        # Numerical features
        # --------------------------------------------------

        numeric_columns = [
            "user_id",
            "purchase_value",
            "device_id",
            "age",
            "ip_address",
            "country",
            "source_Direct",
            "source_SEO",
            "sex_M",
            "browser_FireFox",
            "browser_IE",
            "browser_Opera",
            "browser_Safari"
        ]

        for column in numeric_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # --------------------------------------------------
        # Convert signup_time and purchase_time
        # to Unix timestamps
        # --------------------------------------------------

        signup_datetime = pd.to_datetime(
            df["signup_time"],
            errors="coerce"
        )

        purchase_datetime = pd.to_datetime(
            df["purchase_time"],
            errors="coerce"
        )

        df["signup_time"] = (
            signup_datetime.astype("int64") // 10**9
        )

        df["purchase_time"] = (
            purchase_datetime.astype("int64") // 10**9
        )

        # --------------------------------------------------
        # Date features
        #
        # IMPORTANT:
        # These are created from signup_time because
        # that is how your training preprocessing worked.
        # --------------------------------------------------

        df["year"] = signup_datetime.dt.year
        df["month"] = signup_datetime.dt.month
        df["day"] = signup_datetime.dt.day
        df["hour"] = signup_datetime.dt.hour

        # --------------------------------------------------
        # EXACT 19 MODEL FEATURES
        # --------------------------------------------------

        feature_columns = [
            "user_id",
            "signup_time",
            "purchase_time",
            "purchase_value",
            "device_id",
            "age",
            "ip_address",
            "country",
            "source_Direct",
            "source_SEO",
            "sex_M",
            "browser_FireFox",
            "browser_IE",
            "browser_Opera",
            "browser_Safari",
            "year",
            "month",
            "day",
            "hour"
        ]

        X = df[feature_columns].copy()

        # --------------------------------------------------
        # Ensure all features are numeric
        # --------------------------------------------------

        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # --------------------------------------------------
        # Replace missing values
        # --------------------------------------------------

        X = X.fillna(0)

        # --------------------------------------------------
        # Verify feature order
        # --------------------------------------------------

        print("\nFinal prediction features:")
        print(X)

        print("\nFeature names:")
        print(list(X.columns))

        print("\nNumber of features:", X.shape[1])

        return X

    def predict(self, data):

        # --------------------------------------------------
        # Preprocess input
        # --------------------------------------------------

        X = self.preprocess(data)

        # --------------------------------------------------
        # Make prediction
        # --------------------------------------------------

        prediction = self.model.predict(X)[0]

        # --------------------------------------------------
        # Probability
        # --------------------------------------------------

        probability = None

        if hasattr(self.model, "predict_proba"):

            probability = self.model.predict_proba(X)[0][1]

        # --------------------------------------------------
        # Convert prediction to label
        # --------------------------------------------------

        if prediction == 1:

            fraud_label = "Fraud"

        else:

            fraud_label = "Non-Fraud"

        # --------------------------------------------------
        # Return result
        # --------------------------------------------------

        return {
            "prediction": int(prediction),
            "label": fraud_label,
            "probability": (
                round(float(probability), 4)
                if probability is not None
                else None
            )
        }