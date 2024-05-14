from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Load the training and evaluation data
train_data = pd.read_csv('RPS_train.csv')
eval_data = pd.read_csv('RPS_eval.csv')

train_data = train_data.drop(columns=['Unnamed: 5', 'Unnamed: 6', ' '])

# Assume `train_data` and `eval_data` are DataFrames with columns 'player_move', 'opponent_move', 'outcome'
# Convert categorical data to numerical data
encoder = LabelEncoder()
encoder.fit(train_data['outcome'])
train_data_encoded = encoder.fit_transform(train_data['outcome'])
eval_data_encoded = encoder.transform(eval_data['outcome'])

# Split data into features (X) and target (y)
X_train = train_data_encoded[:-1]
y_train = train_data_encoded[1:]
X_test = eval_data_encoded[:-1]
y_test = eval_data_encoded[1:]

# Build a linear regression model
model = RandomForestClassifier()

# Train the model
model.fit(X_train.reshape(-1, 1), y_train)

# Evaluate the model
test_score = model.score(X_test.reshape(-1, 1), y_test)
print("Test score:", test_score)

# Use the model to predict the next move
def predict_next_move(history):
    # Flatten the history list
    history_flat = [move for sublist in history for move in sublist]

    # Encode the history
    history_encoded = encoder.transform(history_flat)

    # Predict the next move
    prediction = model.predict(history_encoded.reshape(-1, 1))

    # Return the predicted move
    return encoder.inverse_transform(prediction)