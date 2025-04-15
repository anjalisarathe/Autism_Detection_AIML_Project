import pickle
import numpy as np

# Load the trained model pipeline (which includes scaling)
with open(r'C:\Users\HP\OneDrive\Desktop\ML\Autism\Model\autism_model.pkl', 'rb') as f:
    model = pickle.load(f)

def get_user_input():
    print("Welcome to the Autism Screening Interface!")
    print("Please answer the following questions based on the screening questionnaire.")
    print("For each question, enter 0 for 'No' and 1 for 'Yes'.\n")
    
    # Sample descriptions for the autism screening questions (A1 to A10)
    # Note: Adjust these descriptions as per your actual questionnaire.
    questions = {
        "A1": "A1: Does your child have difficulties with maintaining eye contact?",
        "A2": "A2: Does your child show limited interest in playing with other children?",
        "A3": "A3: Does your child have difficulty understanding nonverbal cues (e.g., facial expressions)?",
        "A4": "A4: Does your child struggle with changes in routine or unexpected events?",
        "A5": "A5: Does your child exhibit repetitive behaviors or restricted interests?",
        "A6": "A6: Does your child have delays in speech or language development?",
        "A7": "A7: Does your child have challenges in understanding or expressing emotions?",
        "A8": "A8: Does your child show unusual sensory responses (e.g., overreaction to sounds)?",
        "A9": "A9: Does your child find it difficult to initiate social interactions?",
        "A10": "A10: Does your child have difficulty forming relationships with peers?"
    }
    
    features = {}
    for key, question in questions.items():
        while True:
            try:
                val = int(input(f"{question} (0 for No, 1 for Yes): "))
                if val in [0, 1]:
                    features[key] = val
                    break
                else:
                    print("Invalid entry. Please enter 0 for No or 1 for Yes.")
            except ValueError:
                print("Invalid input. Please enter 0 or 1.")
    
    # Additional questions for non-screening features
    # Age input
    while True:
        try:
            age = float(input("Enter Age (e.g., 15): "))
            features["Age"] = age
            break
        except ValueError:
            print("Invalid age. Please enter a numeric value.")
    
    # Sex input
    while True:
        sex = input("Enter Sex (m for male, f for female): ").lower()
        if sex in ['m', 'f']:
            features["Sex"] = 0 if sex == 'm' else 1
            break
        else:
            print("Invalid input. Please enter 'm' or 'f'.")
    
    # Jaundice input
    while True:
        jaundice = input("Has your child experienced jaundice? (yes/no): ").lower()
        if jaundice in ['yes', 'no']:
            features["Jauundice"] = 1 if jaundice == 'yes' else 0
            break
        else:
            print("Invalid input. Please answer 'yes' or 'no'.")
    
    # Family history of autism input
    while True:
        family_asd = input("Is there a family history of Autism Spectrum Disorder (ASD)? (yes/no): ").lower()
        if family_asd in ['yes', 'no']:
            features["Family_ASD"] = 1 if family_asd == 'yes' else 0
            break
        else:
            print("Invalid input. Please answer 'yes' or 'no'.")
    
    return features

if __name__ == "__main__":
    user_features = get_user_input()
    
    # Ensure that the feature order matches the training data order:
    feature_order = [f"A{i}" for i in range(1, 11)] + ["Age", "Sex", "Jauundice", "Family_ASD"]
    input_data = np.array([user_features[feat] for feat in feature_order]).reshape(1, -1)
    
    # Predict using the loaded model
    prediction = model.predict(input_data)[0]
    prediction_prob = model.predict_proba(input_data)[0][1]  # probability for class 1
    
    # Output the result to the user with probability information
    if prediction == 1:
        print(f"\nThe model predicts that the individual is likely on the autism spectrum (probability: {prediction_prob:.2f}).")
    else:
        print(f"\nThe model predicts that the individual is not likely on the autism spectrum (probability: {1 - prediction_prob:.2f}).")
