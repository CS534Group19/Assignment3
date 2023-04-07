import csv
import os
from initialization import Initialization
import tensorflow as tf
import numpy as np
import pickle

Assignment3Dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
BOARDS_DIR = f"{Assignment3Dir}\\documentation\\4Boards"

model = tf.keras.models.load_model('n_puzzle_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# write code to loop over every file in BOARDS_DIR
for file in os.listdir(BOARDS_DIR):
    manhattan_h_val = 0
    euclidean_h_val = 0
    effort = 0
    displaced_tiles = 0

    rows = []
    row_counter = 0
    metadata_added = False
    print(file)
    with open(f"{BOARDS_DIR}\\{file}", "r") as board_file:
        board_init = Initialization(
            f"{BOARDS_DIR}\\{file}", 4, "Sliding", "True", model, scaler)
        # read the file
        csv_reader = csv.reader(board_file, delimiter=",")
        data = list(csv_reader)

        manhattan_h_val = board_init.getHVal("Sliding", board_init.goal)
        euclidean_h_val = board_init.getHVal("Greedy", board_init.goal)
        displaced_tiles = board_init.tiles_displaced

        # loop over each row in the file
        for row in data:
            if row_counter < board_init.side_length:
                rows.append(row)
            row_counter += 1
        effort = data[board_init.side_length]
    with open(f"{BOARDS_DIR}\\{file}", "w", newline="") as board_file:
        csv_writer = csv.writer(board_file)
        # Xs
        rows.append([board_init.num_tiles])
        rows.append([board_init.blanks])
        rows.append([manhattan_h_val])
        rows.append([euclidean_h_val])
        rows.append([displaced_tiles])

        # Y
        rows.append([int(effort[0])])

        csv_writer.writerows(rows)
