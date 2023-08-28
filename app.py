from flask import Flask, render_template, request
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)

# List of features used by the model
features = ['IsFemale', 'IsCaucasian', 'IsNonHispanic', 'IsAgeGroup1', 'IsAgeGroup2', 
            'IsAgeGroup3', 'IsGeneralPractitioner', 'IsNonSpecialist', 'IsOBGYNorPCP', 
            'IsLowRiskPrior', 'IsLowRiskDuring', 'IsChangeRiskUnknown', 'IsChangeTScoreUnknown', 'IsAdherent']

# Define routes
@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the trained model
        model_filename = 'persistency_model.pkl'
        with open(model_filename, 'rb') as model_file:
            loaded_model = pickle.load(model_file)

        # Get user input from the form
        user_input = []
        for feature in features:
            user_input.append(float(request.form[feature]))

        # Convert the user input into a numpy array
        user_input_array = np.array([user_input])

        # Make predictions using the loaded model
        prediction = loaded_model.predict(user_input_array)
        prediction_text = "Persistency: Yes" if prediction == 1 else "Persistency: No"

    except ValueError as e:
        # Handle error when input conversion to float fails
        prediction_text = "Invalid input. Please enter numeric values for all features."

    return render_template('index1.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
