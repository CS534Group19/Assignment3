import os
import csv

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
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

def custom_loss(y_true, y_pred):
    # Calculate the difference between the true and predicted values
    diff = y_pred - y_true

    # Set a penalty factor for overestimation
    penalty_factor = 1.05

    # Apply the penalty factor to the positive differences (overestimation)
    diff_penalty = tf.where(diff > 0, diff * penalty_factor, diff)

    # Calculate the mean squared error with the penalized differences
    loss = tf.square(diff_penalty)
    return tf.reduce_mean(loss)

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

k_folds = 5

#Initialize the KFold object
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

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
dense1 = tf.keras.layers.Dense(5, activation='linear')(concat_inputs)
#tf.keras.layers.Dropout(rate = 0.33)
dense2 = tf.keras.layers.Dense(4, activation='linear')(dense1)
#tf.keras.layers.Dropout(rate = 0.33)
dense3 = tf.keras.layers.Dense(3, activation='linear')(dense2)
#tf.keras.layers.Dropout(rate = 0.33)
dense4 = tf.keras.layers.Dense(2, activation='linear')(dense3)
output = tf.keras.layers.Dense(1, activation='linear')(dense4)

model = tf.keras.models.Model(inputs=[input_manhattan_h_val, input_num_tiles, input_blanks, input_euclidean_h_val,
                                       input_displaced_tiles], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss= custom_loss, metrics = ['mae', 'mse'])

test_loss = []
r2_scores = []

for fold, (train_indices, val_indices) in enumerate(kf.split(manhattan_h_val)):
    print(f"Fold {fold+1}")
    
    # Split the data into training and validation sets for this fold
    manhattan_h_val_train, manhattan_h_val_val = manhattan_h_val[train_indices], manhattan_h_val[val_indices]
    num_tiles_train, num_tiles_val = num_tiles[train_indices], num_tiles[val_indices]
    blanks_train, blanks_val = blanks[train_indices], blanks[val_indices]
    euclidean_h_val_train, euclidean_h_val_val = euclidean_h_val[train_indices], euclidean_h_val[val_indices]
    displaced_tiles_train, displaced_tiles_val = displaced_tiles[train_indices], displaced_tiles[val_indices]
    effort_train, effort_val = effort[train_indices], effort[val_indices]
    
    # Train the model on the training set for this fold
    history = model.fit([manhattan_h_val_train, num_tiles_train, blanks_train, euclidean_h_val_train, displaced_tiles_train], effort_train, 
                        validation_data = ([manhattan_h_val_val, num_tiles_val, blanks_val, euclidean_h_val_val, displaced_tiles_val], effort_val),
                        epochs=200, batch_size=8, verbose=0)
    
    # Evaluate the model on the test set for this fold
    fold_test_loss = model.evaluate([manhattan_h_val_val, num_tiles_val, blanks_val, euclidean_h_val_val, displaced_tiles_val], effort_val, verbose=0)
    test_loss.append(fold_test_loss)
    prediction = model.predict([manhattan_h_val_val, num_tiles_val, blanks_val, euclidean_h_val_val, displaced_tiles_val])

    fold_r2_score = r2_score(effort_val, prediction)
    r2_scores.append(r2_score(effort_val, prediction))
    
    print(f"Fold {fold+1} - Test Loss: {fold_test_loss[0]:.4f} - R^2 Score: {fold_r2_score:.4f}")
    print()

# Print the mean and standard deviation of the evaluation results across all folds
print(f"Mean Test Loss: {np.mean(test_loss):.4f} - Std Dev Test Loss: {np.std(test_loss):.4f}")
print(f"Mean R^2 Score: {np.mean(r2_scores):.4f} - Std Dev R^2 Score: {np.std(r2_scores):.4f}")

# Save the model for future use
model.save('n_puzzle_model_cross.h5')
          

# Train the model
history = model.fit([manhattan_h_val_train, num_tiles_train, blanks_train, euclidean_h_val_train, displaced_tiles_train], effort_train, 
                    validation_data = ([manhattan_h_val_val, num_tiles_val, blanks_val, euclidean_h_val_val, displaced_tiles_val], effort_val),
                    epochs=200, batch_size=8)

# Evaluate the model on the test set
test_loss = model.evaluate([manhattan_h_val_test, num_tiles_test, blanks_test, euclidean_h_val_test, displaced_tiles_test], effort_test)

prediction = model.predict([manhattan_h_val_test, num_tiles_test, blanks_test, euclidean_h_val_test, displaced_tiles_test])
print(f"R^2 Value: {r2_score(effort_test, prediction)}")

# Save the model and scaler for future use
model.save('n_puzzle_model.h5')
#pickle.dump(scaler, open('scaler.pkl', 'wb'))