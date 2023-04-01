import astar
from board_state import BoardState
from initialization import Initialization
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle

HEURISTIC_OPTIONS = ["Sliding", "Greedy"]
WEIGHTED_OPTIONS = ["True", "False"]

Assignment3Dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
BOARDS_DIR = f"{Assignment3Dir}\\documentation\\test_boards"
OUTPUT_DIR = f"{Assignment3Dir}\\documentation\\data"

# def main():
#     initial = Initialization(BOARDS_DIR + "\\3x3x2.csv")
#     board_state = BoardState(initial.board, initial.goal, HEURISTIC_OPTIONS[0], WEIGHTED_OPTIONS[0])
#     astar.a_star(board_state)

def load_n_puzzle_dataset(num_boards: int, board_size: int):
    # Make sure to swap Bs and 0s in the board
    pass


# Load the dataset (Replace this with your own dataset loading function)
# X: n-puzzle instances, y: optimal solution lengths
X, y = load_n_puzzle_dataset()

# Preprocess the dataset
X = X.astype(np.float32)
y = y.astype(np.float32)
X = X / (len(X[0]) - 1)  # normalize X
scaler = MinMaxScaler()
y = scaler.fit_transform(y.reshape(-1, 1))  # normalize y

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(len(X[0]),)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test)

# Save the model and scaler for future use
model.save('n_puzzle_model.h5')
pickle.dump(scaler, open('scaler.pkl', 'wb'))


# if __name__ == "__main__":
#     main()