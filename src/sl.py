
import os
import csv

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle

Assignment3Dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
BOARDS_DIR = f"{Assignment3Dir}\\documentation\\tagged_boards"
OUTPUT_DIR = f"{Assignment3Dir}\\documentation\\data"
PROCESSED_DIR = f"{Assignment3Dir}\\documentation\\processed_data"

def read_processed_data():
    flattened_boards = []
    manhattan_distances = []
    dimensions = []
    blanks = []
    with open(f"{PROCESSED_DIR}\\flattend_boards.csv", "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for row in csv_reader:
            board = []
            for item in row:
                board.append(int(item))
            flattened_boards.append(board)

    with open(f"{PROCESSED_DIR}\\manhattan_distances.csv", "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for row in csv_reader:
            for item in row:
                manhattan_distances.append(int(item))

    with open(f"{PROCESSED_DIR}\\dimensions.csv", "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for row in csv_reader:
            for item in row:
                dimensions.append(int(item))

    with open(f"{PROCESSED_DIR}\\blanks.csv", "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for row in csv_reader:
            for item in row:
                blanks.append(int(item))
    return flattened_boards, manhattan_distances, dimensions, blanks

boards, manhattan_distances, dimensions, blanks = read_processed_data()
print(f"Boards: {boards[-1]}\n")
print(f"Manhattan: {manhattan_distances[-1]}\n")
print(f"Dimensions: {dimensions[-1]}\n")
print(f"Blanks: {blanks[-1]}\n")

# Preprocess the dataset
# Make all arrays into numpy arrays
boards = np.array([np.array(board) for board in boards])
manhattan_distances = np.array(manhattan_distances)
dimensions = np.array(dimensions)
blanks = np.array(blanks)

# Normalize the boards
boards = boards.astype(np.float32) / (len(boards[0])-1)

scaler = MinMaxScaler()
manhattan_distances = scaler.fit_transform(manhattan_distances.reshape(-1, 1))
dimensions = scaler.fit_transform(dimensions.reshape(-1, 1))  # normalize y
blanks = scaler.fit_transform(blanks.reshape(-1, 1))  # normalize y


# Split the dataset into training, validation, and test sets
boards_train, boards_temp, manhattan_distances_train, manhattan_distances_temp, dimensions_train, dimensions_temp, blanks_train, blanks_temp = train_test_split(boards, manhattan_distances, dimensions, blanks, test_size=0.2, random_state=42)
boards_val, boards_test, manhattan_distances_val, manhattan_distances_test, dimensions_val, dimensions_test, blanks_val, blanks_test = train_test_split(boards_temp, manhattan_distances_temp, dimensions_temp, blanks_temp, test_size=0.5, random_state=42)

# Define the neural network architecture using the functional API
input_board = tf.keras.layers.Input(shape=(len(boards[0]),), name="input_board")
input_dimensions = tf.keras.layers.Input(shape=(1,), name="input_dimensions")
input_blanks = tf.keras.layers.Input(shape=(1,), name="input_blanks")

concat_inputs = tf.keras.layers.Concatenate()([input_board, input_dimensions, input_blanks])
dense1 = tf.keras.layers.Dense(128, activation='relu')(concat_inputs)
dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
dense3 = tf.keras.layers.Dense(32, activation='relu')(dense2)
output = tf.keras.layers.Dense(1, activation='linear')(dense3)

model = tf.keras.models.Model(inputs=[input_board, input_dimensions, input_blanks], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit([boards_train, dimensions_train, blanks_train], manhattan_distances_train, validation_data=([boards_val, dimensions_val, blanks_val], manhattan_distances_val), epochs=100, batch_size=32)

# Evaluate the model on the test set
test_loss = model.evaluate([boards_test, dimensions_test, blanks_test], manhattan_distances_test)

# Save the model and scaler for future use
model.save('n_puzzle_model.h5')
pickle.dump(scaler, open('scaler.pkl', 'wb'))
