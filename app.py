from flask import Flask, request, jsonify
import category_encoders as ce
import pandas as pd
import numpy as np
from joblib import load

app = Flask(__name__)

# Preprocess input data
def preprocess_input(input_data):
    # Ensure all required keys are present in the input data
    required_keys = [
        "loan_amnt", "term", "int_rate", "installment", "emp_length", "annual_inc",
        "fico_range_low", "fico_range_high", "dti", "revol_util", "total_pymnt",
        "total_rec_prncp", "total_rec_int", "last_pymnt_amnt", "inq_last_6mths",
        "mths_since_last_delinq", "purpose", "home_ownership", "verification_status"
    ]

    # Check if all keys are present, raise an error if missing
    missing_keys = [key for key in required_keys if key not in input_data]
    if missing_keys:
        raise ValueError(f"Missing keys in input data: {', '.join(missing_keys)}")

    input_sample = {
        "loan_amnt": input_data["loan_amnt"],
        "term": input_data["term"],  # Assuming term is numerical (e.g., 36 for '36 months', 60 for '60 months')
        "int_rate": input_data["int_rate"],
        "installment": input_data["installment"],
        "emp_length": input_data["emp_length"],  # Assuming numerical years
        "annual_inc": input_data["annual_inc"],
        "fico_range_low": input_data["fico_range_low"],
        "fico_range_high": input_data["fico_range_high"],
        "dti": input_data["dti"],
        "revol_util": input_data["revol_util"],
        "total_pymnt": input_data["total_pymnt"],
        "total_rec_prncp": input_data["total_rec_prncp"],
        "total_rec_int": input_data["total_rec_int"],
        "last_pymnt_amnt": input_data["last_pymnt_amnt"],
        "inq_last_6mths": input_data["inq_last_6mths"],
        "mths_since_last_delinq": input_data["mths_since_last_delinq"],

        # One-hot encode purpose
        **{f"purpose_{purpose}": 1 if input_data["purpose"] == purpose else 0 for purpose in [
            "credit_card", "debt_consolidation", "educational", "home_improvement", "house",
            "major_purchase", "medical", "moving", "other", "renewable_energy", "small_business",
            "vacation", "wedding"
        ]},

        # One-hot encode home_ownership
        **{f"home_ownership_{ownership}": 1 if input_data["home_ownership"] == ownership else 0 for ownership in [
            "MORTGAGE", "NONE", "OTHER", "OWN", "RENT"
        ]},

        # One-hot encode verification_status
        **{f"verification_status_{status}": 1 if input_data["verification_status"] == status else 0 for status in [
            "Source Verified", "Verified"
        ]}
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_sample])

    # Convert DataFrame to dictionary
    return input_df.to_dict(orient='records')[0]  # Convert to a dictionary

# Predict loan status
# Predict loan status
# Predict loan status
def predict_loan_status(input_data, model_type):
    # Load the model
    model = None
    if model_type == "lg":
        model = load("lg_model.joblib")  # Load the trained model
    elif model_type == "rf":
        model = load("lg_model.joblib")
    elif model_type == "catboost":
        model = load("catboost_model.joblib")
    else:
        model = load("catboost_model.joblib")

    # Preprocess the input data (matching training data preprocessing)
    input_df = preprocess_input(input_data)

    # Convert the input DataFrame to a 2D numpy array (as expected by the model)
    input_array = np.array([list(input_df.values())])

    # Make the prediction
    prediction = model.predict(input_array)  # Predict class (status)
    prediction_prob = model.predict_proba(input_array)  # Predict probabilities

    # Ensure the values are serializable (convert to int or float)
    predicted_status = int(prediction[0])  # Convert to an integer
    probability = float(prediction_prob[0][1])  # Convert to a float

    return predicted_status, probability



@app.route('/predict/lg', methods=['POST'])
def predict_lg():
    try:
        # Get JSON data from the request
        input_data = request.get_json()

        # Predict the loan status
        predicted_status, probability = predict_loan_status(input_data, "lg")

        # Return the prediction as a JSON response
        return jsonify({"predicted_status": predicted_status, "probability": probability}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/predict/rf', methods=['POST'])
def predict_rf():
    try:
        # Get JSON data from the request
        input_data = request.get_json()

        # Predict the loan status
        predicted_status, probability = predict_loan_status(input_data, "rf")

        # Return the prediction as a JSON response
        return jsonify({"predicted_status": predicted_status, "probability": probability}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/predict/catboost', methods=['POST'])
def predict_catboost():
    try:
        # Get JSON data from the request
        input_data = request.get_json()

        # Predict the loan status
        predicted_status, probability = predict_loan_status(input_data, "catboost")

        # Return the prediction as a JSON response
        return jsonify({"predicted_status": predicted_status, "probability": probability}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
