import pandas as pd
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import urllib.request

def download_file(url, filename):
    if not os.path.exists(filename):
        try:
            print(f"Téléchargement de {filename} depuis {url}...")
            # Télécharge le fichier
            urllib.request.urlretrieve(url, filename)
            print(f"Le fichier {filename} a été téléchargé avec succès.")
        except Exception as e:
            print(f"Erreur lors du téléchargement : {e}")
    else:
        print(f"Le fichier {filename} existe déjà.")

# Download the csv files
download_file('https://www.admandev.fr/application_train.csv', 'application_train.csv')
download_file('https://www.admandev.fr/application_test.csv', 'application_test.csv')

app = Flask(__name__)

# Load both datasets (train and test)
train_df = pd.read_csv('application_train.csv')
test_df = pd.read_csv('application_test.csv')

# Concatenate train and test datasets
df = pd.concat([train_df, test_df], ignore_index=True)

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# List of model features (columns used for training)
model_features = [
    'ANNUITY_INCOME_PERCENT', 'CODE_GENDER_M', 'CODE_GENDER_XNA',
    'CREDIT_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT',
    'EMERGENCYSTATE_MODE_Yes', 'FLAG_OWN_CAR_Y', 'FLAG_OWN_REALTY_Y',
    'NAME_CONTRACT_TYPE_Cash loans', 'NAME_EDUCATION_TYPE_Secondary / secondary special',
    'OCCUPATION_TYPE_Laborers', 'ORGANIZATION_TYPE_Business Entity Type 3'
]

# Function to recreate features for prediction
def create_features(df):
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    
    df = pd.get_dummies(df, columns=['CODE_GENDER', 'EMERGENCYSTATE_MODE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
                                     'NAME_CONTRACT_TYPE', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE'],
                        drop_first=True)
    
    # Ensure all model features are present, fill missing columns with default values (e.g., 0)
    for feature in model_features:
        if feature not in df.columns:
            df[feature] = 0
    
    return df

def try_prediction(client_data):
    """
    Tries to make a prediction with client data by systematically handling missing features.
    """
    for attempt in range(10):  # Number of attempts
        try:
            # Create features
            client_data = create_features(client_data)
            
            # Reindex to ensure columns match
            client_data = client_data[model_features]
            
            # Fill missing values
            client_data.fillna(client_data.median(numeric_only=True), inplace=True)
            client_data.fillna(client_data.mode().iloc[0], inplace=True)

            # Apply scaling
            X = scaler.transform(client_data)
            
            # Make prediction
            prediction = model.predict(X)
            probability = model.predict_proba(X)

            return {
                'prediction': int(prediction[0]),
                'probability': probability[0].tolist()
            }

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            # Modify client data to try another combination
            for feature in model_features:
                if feature not in client_data.columns or client_data[feature].isnull().any():
                    client_data[feature] = 0  # Example adjustment: fill missing with 0

    return {'error': 'Unable to make a prediction after multiple attempts'}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input data (client ID)
        input_data = request.json
        client_id = input_data.get('SK_ID_CURR')

        if client_id is None:
            return jsonify({'error': 'Client ID not provided'}), 400

        # Find the client data in the concatenated dataset (train + test)
        client_data = df[df['SK_ID_CURR'] == client_id]

        if client_data.empty:
            return jsonify({'error': 'Client data not found'}), 404

        # Try to make a prediction using different combinations of data
        result = try_prediction(client_data)

        # Return the result
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
