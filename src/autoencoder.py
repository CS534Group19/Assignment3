import os
import csv

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.metrics import r2_score

Assignment3Dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
BOARDS_DIR = f"{Assignment3Dir}\\documentation\\tagged_boards"
OUTPUT_DIR = f"{Assignment3Dir}\\documentation\\data"
PROCESSED_DIR = f"{Assignment3Dir}\\documentation\\processed_data"

def read_processed_data():
    num_tiles = []
    blanks = []
    manhattan_h_val = []
    euclidean_h_val = []
    displaced_tiles = []
    effort = []

    with open(f"{PROCESSED_DIR}\\blanks.csv", "r") as blanks_file:
        csv_reader = csv.reader(blanks_file, delimiter=",")
        for row in csv_reader:
            blanks.append(float(row[0]))

    with open(f"{PROCESSED_DIR}\\displaced_tiles.csv", "r") as displaced_tiles_file:
        csv_reader = csv.reader(displaced_tiles_file, delimiter=",")
        for row in csv_reader:
            displaced_tiles.append(float(row[0]))

    with open(f"{PROCESSED_DIR}\\effort.csv", "r") as effort_file:
        csv_reader = csv.reader(effort_file, delimiter=",")
        for row in csv_reader:
            effort.append(float(row[0]))

    with open(f"{PROCESSED_DIR}\\euclidean_h_val.csv", "r") as euclidean_h_val_file:
        csv_reader = csv.reader(euclidean_h_val_file, delimiter=",")
        for row in csv_reader:
            euclidean_h_val.append(float(row[0]))

    with open(f"{PROCESSED_DIR}\\manhattan_h_val.csv", "r") as manhattan_h_val_file:
        csv_reader = csv.reader(manhattan_h_val_file, delimiter=",")
        for row in csv_reader:
            manhattan_h_val.append(float(row[0]))

    with open(f"{PROCESSED_DIR}\\num_tiles.csv", "r") as num_tiles_file:
        csv_reader = csv.reader(num_tiles_file, delimiter=",")
        for row in csv_reader:
            num_tiles.append(float(row[0]))

    return num_tiles, blanks, manhattan_h_val, euclidean_h_val, displaced_tiles, effort

num_tiles, blanks, manhattan_h_val, euclidean_h_val, displaced_tiles, effort = read_processed_data()
print(f"Num Tiles: {num_tiles[-1]}")
print(f"Blanks: {blanks[-1]}")
print(f"Manhattan H Val: {manhattan_h_val[-1]}")
print(f"Euclidean H Val: {euclidean_h_val[-1]}")
print(f"Displaced Tiles: {displaced_tiles[-1]}")
print(f"Effort: {effort[-1]}")

# Preprocess the dataset
# Make all arrays into numpy arrays
manhattan_h_val = np.array(manhattan_h_val)
num_tiles = np.array(num_tiles)
blanks = np.array(blanks)
euclidean_h_val = np.array(euclidean_h_val)
displaced_tiles = np.array(displaced_tiles)
effort = np.array(effort)

# Normalize the boards
# boards = boards.astype(np.float32) / (len(boards[0])-1)

# scaler = MinMaxScaler()
# manhattan_h_val = scaler.fit_transform(manhattan_h_val.reshape(-1, 1))
# num_tiles = scaler.fit_transform(num_tiles.reshape(-1, 1))
# blanks = scaler.fit_transform(blanks.reshape(-1, 1))
# euclidean_h_val = scaler.fit_transform(euclidean_h_val.reshape(-1, 1))
# displaced_tiles = scaler.fit_transform(displaced_tiles.reshape(-1, 1))

# Split the dataset into training, validation, and test sets
# manhattan_h_val_train, manhattan_h_val_temp, num_tiles_train, num_tiles_temp, blanks_train, blanks_temp, euclidean_h_val_train, euclidean_h_val_temp, displaced_tiles_train, displaced_tiles_temp = train_test_split(
#     manhattan_h_val, manhattan_h_val, num_tiles, blanks, euclidean_h_val, displaced_tiles,
#     test_size=0.2, random_state=42)
# boards_val, boards_test, manhattan_h_val_val, manhattan_h_val_test, num_tiles_val, num_tiles_test, blanks_val, blanks_test, euclidean_h_val_val, euclidean_h_val_test, displaced_tiles_val, displaced_tiles_test = train_test_split(
#     manhattan_h_val_temp, num_tiles_temp, blanks_temp, euclidean_h_val_temp, displaced_tiles_temp,
#     test_size=0.5, random_state=42)

manhattan_h_val_train,  manhattan_h_val_temp,   \
num_tiles_train,        num_tiles_temp,         \
blanks_train,           blanks_temp,            \
euclidean_h_val_train,  euclidean_h_val_temp,   \
displaced_tiles_train,  displaced_tiles_temp,   \
effort_train,           effort_temp =           \
train_test_split(manhattan_h_val, num_tiles, blanks, euclidean_h_val, displaced_tiles, effort, test_size=0.2, random_state=42) 

manhattan_h_val_val,    manhattan_h_val_test,   \
num_tiles_val,          num_tiles_test,         \
blanks_val,             blanks_test,            \
euclidean_h_val_val,    euclidean_h_val_test,   \
displaced_tiles_val,    displaced_tiles_test,   \
effort_val,             effort_test =           \
train_test_split(manhattan_h_val_temp, num_tiles_temp, blanks_temp, euclidean_h_val_temp, displaced_tiles_temp, effort_temp, test_size=0.5, random_state=42)

# Define the neural network architecture using the functional API
input_manhattan_h_val = tf.keras.layers.Input(shape=(1,), name="input_manhattan_h_val")
input_num_tiles = tf.keras.layers.Input(shape=(1,), name="input_num_tiles")
input_blanks = tf.keras.layers.Input(shape=(1,), name="input_blanks")
input_euclidean_h_val = tf.keras.layers.Input(shape=(1,), name="input_euclidean_h_val")
input_displaced_tiles = tf.keras.layers.Input(shape=(1,), name="input_displaced_tiles")


concat_inputs = tf.keras.layers.Concatenate()(
    [input_manhattan_h_val, input_num_tiles, input_blanks, input_euclidean_h_val, input_displaced_tiles])

encoding_dim = 3

# Define the autoencoder model
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(concat_inputs)
decoded = tf.keras.layers.Dense(5, activation='linear')(encoded)

autoencoder = tf.keras.models.Model(concat_inputs, decoded)
encoder = tf.keras.models.Model(concat_inputs, encoded)

autoencoder.compile(optimizer='adam', loss='mse')

input_data = np.column_stack([manhattan_h_val, num_tiles, blanks, euclidean_h_val, displaced_tiles])
input_data_train, input_data_temp, effort_train, effort_temp = train_test_split(input_data, effort, test_size=0.2, random_state=42)
input_data_val, input_data_test, effort_val, effort_test = train_test_split(input_data_temp, effort_temp, test_size=0.5, random_state=42)

# Train the autoencoder
autoencoder_history = autoencoder.fit(input_data_train, input_data_train,
                                      epochs=250,
                                      batch_size=32,
                                      shuffle=True,
                                      validation_data=(input_data_val, input_data_val))

encoded_data_train = encoder.predict(input_data_train)
encoded_data_val = encoder.predict(input_data_val)
encoded_data_test = encoder.predict(input_data_test)


# Save the model and scaler for future use
autoencoder.save('n_puzzle_autoencoder.h5')
#pickle.dump(scaler, open('scaler.pkl', 'wb'))

encoding_dim = 3
input_encoded = tf.keras.layers.Input(shape=(encoding_dim,), name="input_encoded")


dense1 = tf.keras.layers.Dense(3, activation='linear')(input_encoded)
#tf.keras.layers.Dropout(rate = 0.33)
dense2 = tf.keras.layers.Dense(4, activation='linear')(dense1)
#tf.keras.layers.Dropout(rate = 0.33)
dense3 = tf.keras.layers.Dense(3, activation='linear')(dense1)
#tf.keras.layers.Dropout(rate = 0.33)
dense4 = tf.keras.layers.Dense(2, activation='linear')(dense1)
output = tf.keras.layers.Dense(1, activation='linear')(dense4)

model = tf.keras.models.Model(inputs=input_encoded, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics = ['mae', 'mse'])

# Train the model
history = model.fit(encoded_data_train, effort_train,
                    validation_data=(encoded_data_val, effort_val),
                    epochs=250, batch_size=32)

# Evaluate the model on the test set
test_loss = model.evaluate(encoded_data_test, effort_test)

prediction = model.predict(encoded_data_test)
print(f"R^2 Value: {r2_score(effort_test, prediction)}")

# Save the model and scaler for future use
model.save('n_puzzle_model.h5')