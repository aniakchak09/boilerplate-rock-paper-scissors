# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, concatenate
from tensorflow.keras.models import Model
# from tensorflow.keras.layers.experimental.preprocessing import StringLookup

# Load dataset.
dftrain = pd.read_csv("RPS_train.csv")
dfeval = pd.read_csv("RPS_eval.csv")
y_train = dftrain.pop("outcome")
y_eval = dfeval.pop("outcome")

CATEGORICAL_COLUMNS = ["player1", "player2"]

input_layers = []
input_dict = {}
for feature_name in CATEGORICAL_COLUMNS:
    input_layer = Input(shape=(1,), name=feature_name, dtype=tf.string)
    input_layers.append(input_layer)
    input_dict[feature_name] = input_layer

# StringLookup layers
# embedding_layers = []
# for feature_name, input_layer in zip(CATEGORICAL_COLUMNS, input_layers):
#     vocabulary = dftrain[feature_name].unique()
#     embedding_layer = StringLookup(vocabulary=vocabulary, mask_token=None)(input_layer)
#     embedding_layers.append(Embedding(len(vocabulary)+1, 8)(embedding_layer))
#
# # Concatenate all embedding layers
# concatenated = concatenate(embedding_layers)
#
# # Flatten concatenated layer
# flattened = Flatten()(concatenated)
#
# # Dense layers
# dense1 = Dense(128, activation='relu')(flattened)
# dense2 = Dense(64, activation='relu')(dense1)
#
# # Output layer
# output = Dense(1, activation='sigmoid')(dense2)
#
# # Model
# model = Model(inputs=input_layers, outputs=output)
#
# # Compile model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# # Prepare input data
# X_train = {feature_name: dftrain[feature_name] for feature_name in CATEGORICAL_COLUMNS}
# X_eval = {feature_name: dfeval[feature_name] for feature_name in CATEGORICAL_COLUMNS}
#
# # Train model
# model.fit(X_train, y_train, epochs=10, batch_size=32)
#
# # Evaluate model
# result = model.evaluate(X_eval, y_eval)
# print("Accuracy:", result[1])

def player(prev_play, opponent_history=[]):
    opponent_history.append(prev_play)

    guess = "R"
    if len(opponent_history) > 2:
        guess = opponent_history[-2]

    return guess
