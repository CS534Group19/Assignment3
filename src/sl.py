
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

# def main():

def load_n_puzzle_dataset():
    # Make sure to swap Bs and 0s in the board
    boards = []
    manhattan_distances = []
    dimensions = []
    blanks = []
    for file in os.listdir(BOARDS_DIR):
        with open(f"{BOARDS_DIR}\\{file}", "r") as board_file:
            csv_reader = csv.reader(board_file, delimiter=",")
            data = list(csv_reader)
            dimension = len(data[0])
            board = []
            for i in range(dimension):
                for j in range(dimension):
                    if data[i][j] == "B":
                        data[i][j] = 0
                    else:
                        data[i][j] = int(data[i][j])
                board.append(data[i])
            boards.append(board)
            manhattan_distances.append(int(data[dimension][0]))
            dimensions.append(int(data[dimension+1][0]))
            blanks.append(int(data[dimension+2][0]))
    print(f"Boards: {boards[-1]}\n")
    print(f"Manhattan: {manhattan_distances[-1]}\n")
    print(f"Dimensions: {dimensions[-1]}\n")
    print(f"Blanks: {blanks[-1]}\n")
    return boards, manhattan_distances, dimensions, blanks

# Load the dataset (Replace this with your own dataset loading function)\
boards, manhattan_distances, dimensions, blanks = load_n_puzzle_dataset()

# Preprocess the dataset

# Normalize dataset
# Assuming X_array is your input data as a 3-dimensional array
# Normalize X_array
boards_array = np.array(boards)

boards_array = boards_array.astype(np.float32) / (3 * 3 - 1)

# Reshape the input data to have a single channel
boards_array = boards_array.reshape(boards_array.shape[0], 3, 3, 1)

# # for board in boards
# for i in range(len(boards)):
#     # for row in board
#     for j in range(len(boards[i])):
#         # for element in row
#         for k in range(len(boards[i][j])):
#             boards[i][j][k] = boards[i][j][k] / (len(boards[i])**2)
# print(boards[-1])

scaler = MinMaxScaler()
manhattan_distances = scaler.fit_transform(manhattan_distances.reshape(-1, 1))
dimensions = scaler.fit_transform(dimensions.reshape(-1, 1))  # normalize y
blanks = scaler.fit_transform(blanks.reshape(-1, 1))  # normalize y


# Split the dataset into training, validation, and test sets
boards_train, boards_temp, manhattan_distances_train, manhattan_distances_temp, dimensions_train, dimensions_temp, blanks_train, blanks_temp = train_test_split(boards, manhattan_distances, dimensions, blanks, test_size=0.2, random_state=42)
boards_val, boards_test, manhattan_distances_val, manhattan_distances_test, dimensions_val, dimensions_test, blanks_val, blanks_test = train_test_split(boards_temp, manhattan_distances_temp, dimensions_temp, blanks_temp, test_size=0.5, random_state=42)

# Define the neural network architecture
# 3x3 model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=(3, 3, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(boards_train, manhattan_distances_train, dimensions_train, blanks_train, validation_data=(boards_val, manhattan_distances_val, dimensions_val, blanks_val), epochs=100, batch_size=32)

# # Evaluate the model on the test set
test_loss = model.evaluate(boards_test, manhattan_distances_test, dimensions_test, blanks_test)

# Save the model and scaler for future use
model.save('n_puzzle_model.h5')
pickle.dump(scaler, open('scaler.pkl', 'wb'))


# # if __name__ == "__main__":
# #     main()