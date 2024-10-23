from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load your model and encoders here
model = joblib.load('crop_model.pkl')  # This loads the crop prediction model
state_label_encoder = joblib.load('state_label_encoder.pkl')  # This loads the state label encoder
crop_label_encoder = joblib.load('crop_label_encoder.pkl')  # This loads the crop label encoder

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Gather numerical input features from form
        int_features = [float(request.form.get(feature)) for feature in ['N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL', 'CROP_PRICE']]
    except ValueError as e:
        return render_template('index.html', prediction_text=f'Error: Please enter valid numerical values. {str(e)}')

    # Get the state name from the form
    STATE = request.form.get('STATE')

    try:
        # Encode the state into a numerical value
        STATE_encoded = state_label_encoder.transform([STATE])
    except ValueError:
        return render_template('index.html', prediction_text='Error: Invalid state name. Please ensure it is correct.')

    # Combine all input features into an array
    final_features = np.array([[*int_features, STATE_encoded[0]]])

    # Make prediction
    prediction = model.predict(final_features)

    # Decode the predicted crop
    predicted_crop = crop_label_encoder.inverse_transform(prediction)

    # Return the result to the frontend
    return render_template('index.html', prediction_text=f'The predicted crop is: {predicted_crop[0]}')

@app.route('/states', methods=['GET'])
def states():
    # Retrieve the list of available states from the state label encoder
    states_list = state_label_encoder.classes_.tolist()  
    return jsonify(states_list)

if __name__ == "__main__":
    app.run(debug=True)