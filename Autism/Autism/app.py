from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model pipeline (which includes scaling)
with open(r'C:\Users\HP\OneDrive\Desktop\ML\Autism\Model\autism_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        try:
            # List of required fields – adjust names if needed.
            required_fields = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Age", "Sex", "Jaundice", "Family_ASD"]
            for field in required_fields:
                value = request.form.get(field, "")
                if not value or not value.strip():
                    return f"Error: Missing input for {field}"

            # A helper to convert yes/no responses.
            def parse_int_answer(answer):
                mapping = {"zero": 0, "0": 0, "one": 1, "1": 1}
                answer = answer.strip().lower()
                if answer in mapping:
                    return mapping[answer]
                else:
                    raise ValueError(f"Invalid answer: '{answer}'")
            
            A_features = []
            for i in range(1, 11):
                ans = request.form.get(f"A{i}", "")
                A_features.append(parse_int_answer(ans))
            
            Age = float(request.form.get("Age", "").strip())
            
            Sex_input = request.form.get("Sex", "").strip().lower()
            if Sex_input not in ['m', 'f']:
                return f"Error: Invalid input for Sex: '{Sex_input}'"
            Sex = 0 if Sex_input == "m" else 1

            # Adjust field names for consistency (e.g., Jaundice might be 'Jaundice' not 'Jauundice')
            jaundice_input = request.form.get("Jaundice", "").strip().lower()
            if jaundice_input not in ["yes", "no"]:
                return f"Error: Invalid input for Jaundice: '{jaundice_input}'"
            Jaundice = 1 if jaundice_input == "yes" else 0

            family_asd_input = request.form.get("Family_ASD", "").strip().lower()
            if family_asd_input not in ["yes", "no"]:
                return f"Error: Invalid input for Family_ASD: '{family_asd_input}'"
            Family_ASD = 1 if family_asd_input == "yes" else 0

            # Combine features in the order expected by your model.
            feature_order = A_features + [Age, Sex, Jaundice, Family_ASD]
            input_data = np.array(feature_order).reshape(1, -1)

            # Make prediction using your loaded model.
            prediction = model.predict(input_data)[0]
            probas = model.predict_proba(input_data)[0]
            probability = probas[1] if prediction == 1 else 1 - probas[1]

            if prediction == 1:
                result_text = f"The model predicts that the individual is likely on the autism spectrum. (Probability: {probability:.2f})"
            else:
                result_text = f"The model predicts that the individual is not likely on the autism spectrum. (Probability: {probability:.2f})"
            return result_text

        except Exception as e:
            return f"Error: {str(e)}"
    return render_template("prediction.html")

if __name__ == "__main__":
    app.run(debug=True)
