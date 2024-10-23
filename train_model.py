import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset (replace with the correct path)
df = pd.read_csv('crop_recommendation.csv')

# Create Label Encoders for categorical features
state_encoder = LabelEncoder()
crop_encoder = LabelEncoder()

# Fit the encoders and transform categorical features
df['Encoded_State'] = state_encoder.fit_transform(df['STATE'])
df['Encoded_Crop'] = crop_encoder.fit_transform(df['CROP'])

# Define features and target variable
X = df[['N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL', 'Encoded_State', 'CROP_PRICE']]
y = df['Encoded_Crop']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Train the RandomForest model
clf = RandomForestClassifier(n_estimators=100, random_state=123)
clf.fit(X_train, y_train)

# Save the model and encoders for later use in the Flask app
joblib.dump(clf, 'trained_crop_model.pkl')
joblib.dump(state_encoder, 'state_encoder.pkl')
joblib.dump(crop_encoder, 'crop_encoder.pkl')

print("Model and encoders have been saved successfully.")