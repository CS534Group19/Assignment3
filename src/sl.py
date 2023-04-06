
import os
import csv

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error

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

# Organize the data into a single NumPy array
data = np.column_stack((num_tiles, blanks, manhattan_h_val, euclidean_h_val, displaced_tiles, effort))

# Split the data into training and testing sets
X = data[:, :-1]  # Inputs (all columns except the last one)
y = data[:, -1]  # Output (the last column)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Load the saved scaler
with open("scaler.pkl", "rb") as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# Apply the loaded scaler to the test data
X_test_scaled_loaded = loaded_scaler.transform(X_test)

# Create the linear regression model
lr_model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=10)

# Train the model
lr_model.fit(X_train_scaled, y_train)

# Print the best alpha value
best_alpha = lr_model.alpha_
print(f"Best alpha value: {best_alpha}")

# Evaluate the model
y_pred = lr_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Calculate R^2 score
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2}")

# Save the model
with open("linear_regression_model.pkl", "wb") as model_file:
    pickle.dump(lr_model, model_file)

# Load the saved model
with open("linear_regression_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

# Make predictions using the loaded model and loaded scaler
y_pred_loaded = loaded_model.predict(X_test_scaled_loaded)

# Create a scatter plot of true effort values vs predicted effort values
plt.scatter(y_test, y_pred_loaded, alpha=0.5)
plt.xlabel("True Effort Values")
plt.ylabel("Predicted Effort Values")
plt.title("True vs Predicted Effort Values (Linear Regression)")
plt.show()