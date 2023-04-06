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
    euclidean_distances = []
    greedy_distances = []
    wrong_tiles = []
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
    
    with open(f"{PROCESSED_DIR}\\euclidean_distances.csv", "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for row in csv_reader:
            for item in row:
                euclidean_distances.append(int(item))

    with open(f"{PROCESSED_DIR}\\greedy_distances.csv", "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for row in csv_reader:
            for item in row:
                greedy_distances.append(int(item))

    with open(f"{PROCESSED_DIR}\\wrong_tiles.csv", "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for row in csv_reader:
            for item in row:
                wrong_tiles.append(int(item))

    return flattened_boards, manhattan_distances, dimensions, blanks, euclidean_distances, greedy_distances, wrong_tiles


boards, manhattan_distances, dimensions, blanks, euclidean_distances, greedy_distances, wrong_tiles = read_processed_data()

# Preprocess the dataset
# Make all arrays into numpy arrays
boards = np.array(boards)
manhattan_distances = np.array(manhattan_distances)
dimensions = np.array(dimensions)
blanks = np.array(blanks)
euclidean_distances = np.array(euclidean_distances)
greedy_distances = np.array(greedy_distances)
wrong_tiles = np.array(wrong_tiles)

# Normalize the boards
boards = boards.astype(np.float32) / (len(boards[0])-1)

scaler = MinMaxScaler()
manhattan_distances = scaler.fit_transform(manhattan_distances.reshape(-1, 1))
dimensions = scaler.fit_transform(dimensions.reshape(-1, 1))
blanks = scaler.fit_transform(blanks.reshape(-1, 1))
euclidean_distances = scaler.fit_transform(euclidean_distances.reshape(-1, 1))
greedy_distances = scaler.fit_transform(greedy_distances.reshape(-1, 1))
wrong_tiles = scaler.fit_transform(wrong_tiles.reshape(-1, 1))

# Split the dataset into training, validation, and test sets
boards_train, boards_temp, manhattan_distances_train, manhattan_distances_temp, dimensions_train, dimensions_temp, blanks_train, blanks_temp, euclidean_distances_train, euclidean_distances_temp, greedy_distances_train, greedy_distances_temp, wrong_tiles_train, wrong_tiles_temp = train_test_split(
    boards, manhattan_distances, dimensions, blanks, euclidean_distances, greedy_distances, wrong_tiles,
    test_size=0.2, random_state=42)
boards_val, boards_test, manhattan_distances_val, manhattan_distances_test, dimensions_val, dimensions_test, blanks_val, blanks_test, euclidean_distances_val, euclidean_distances_test, greedy_distances_val, greedy_distances_test, wrong_tiles_val, wrong_tiles_test = train_test_split(
    boards_temp, manhattan_distances_temp, dimensions_temp, blanks_temp, euclidean_distances_temp, greedy_distances_temp, wrong_tiles_temp,
    test_size=0.5, random_state=42)

# Define the neural network architecture using the functional API
input_board = tf.keras.layers.Input(shape=(len(boards[0]),), name="input_board")
input_dimensions = tf.keras.layers.Input(shape=(1,), name="input_dimensions")
input_blanks = tf.keras.layers.Input(shape=(1,), name="input_blanks")
input_euclidean_distances = tf.keras.layers.Input(shape=(1,), name="input_euclidean_distances")
input_greedy_distances = tf.keras.layers.Input(shape=(1,), name="input_greedy_distances")
input_wrong_tiles = tf.keras.layers.Input(shape=(1,), name="input_wrong_tiles")

concat_inputs = tf.keras.layers.Concatenate()(
    [input_board, input_dimensions, input_blanks, input_euclidean_distances, input_greedy_distances, input_wrong_tiles])
dense1 = tf.keras.layers.Dense(64, activation='relu')(concat_inputs)
dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
dense3 = tf.keras.layers.Dense(16, activation='relu')(dense2)
output = tf.keras.layers.Dense(1, activation='linear')(dense3)

model = tf.keras.models.Model(inputs=[input_board, input_dimensions, input_blanks, input_euclidean_distances,
                                       input_greedy_distances, input_wrong_tiles], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit([boards_train, dimensions_train, blanks_train, euclidean_distances_train, greedy_distances_train, wrong_tiles_train], manhattan_distances_train,
                    validation_data=([boards_val, dimensions_val, blanks_val, euclidean_distances_val, greedy_distances_val, wrong_tiles_val], manhattan_distances_val),
                    epochs=100, batch_size=32)

# Evaluate the model on the test set
test_loss = model.evaluate([boards_test, dimensions_test, blanks_test, euclidean_distances_test, greedy_distances_test, wrong_tiles_test], manhattan_distances_test)

# Save the model and scaler for future use
model.save('n_puzzle_model.h5')
pickle.dump(scaler, open('scaler.pkl', 'wb'))