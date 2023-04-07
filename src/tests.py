import astar
from board_state import BoardState
from initialization import Initialization
import os
import csv
import random
from time import perf_counter

import numpy as np
import tensorflow as tf
import pickle

Assignment3Dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
BOARDS_DIR = f"{Assignment3Dir}\\documentation\\3Boards"
OUTPUT_DIR = f"{Assignment3Dir}\\documentation\\data"


def load_model(heuristic_type, model_type):
    if heuristic_type == "NN":
        if model_type == "Linear":
            with open("linear_regression_model.pkl", "rb") as f:
                model = pickle.load(f)
            with open('scaler_linear.pkl', 'rb') as f:
                scaler = pickle.load(f)
        elif model_type == "Dense":
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
            model = tf.keras.models.load_model('n_puzzle_model.h5', custom_objects={'custom_loss': custom_loss})
            scaler = None
        else:
            model = None
            scaler = None
    else:
        model = None
        scaler = None
    return model, scaler


def get_random_board_numbers(num_boards):
    random_board_numbers = []
    while len(random_board_numbers) < num_boards:
        rand_val = random.randint(0, 999)
        if rand_val not in random_board_numbers:
            random_board_numbers.append(rand_val)
    return random_board_numbers

with open(f"{OUTPUT_DIR}\\data_03x03_1000_tests_NN.csv", "w", newline="") as data_file:
    heuristic = "NN"
    weighted = "False"
    model_type = "Linear"
    model, scaler = load_model(heuristic, model_type)

    data_writer = csv.writer(data_file)
    data_writer.writerow(["File Name", "Heuristic", "Weighted", "Nodes Expanded",
                          "Moves Required", "Solution Cost", "Estimated Branching Factor", "Search Time"])
    for i in range(1):
        new_board = Initialization(
            f"{BOARDS_DIR}\\03x03_board_{i}.csv", 3, heuristic, weighted, model, scaler)
        board_state = BoardState(
            new_board.board, new_board.goal, new_board.heuristic_type, new_board.weighted, new_board.blanks, new_board.manhattan_h_val, new_board.euclidean_h_val, new_board.tiles_displaced, new_board.model, new_board.scaler)
        data_writer.writerow(
            [f"03x03_board_{i}.csv", heuristic, weighted] + astar.a_star(board_state))


# For a given board, run 1 sliding h val, 1 nn h val - log the times in a csv
# with open(f"{OUTPUT_DIR}\\sliding_vs_nn_h_val_times.csv", "w", newline="") as data_file:
#     heuristic = "NN"
#     weighted = "False"
#     model_type = "Dense"
#     model, scaler = load_model(heuristic, model_type)

#     data_writer = csv.writer(data_file)
#     data_writer.writerow(["File Name", "Sliding", "NN"])

#     board_size = 3
#     random_board_numbers = get_random_board_numbers(25)

#     for board_num in random_board_numbers:
#         new_board = Initialization(
#             f"{BOARDS_DIR}\\03x03_board_{board_num}.csv", board_size, heuristic, weighted, model, scaler)
#         board_state = BoardState(
#             new_board.board, new_board.goal, new_board.heuristic_type, new_board.weighted, new_board.blanks, new_board.manhattan_h_val, new_board.euclidean_h_val, new_board.tiles_displaced, new_board.model, new_board.scaler)

#         sliding_start = perf_counter()
#         board_state.calc_total_manhattan_for_board(board_state.board_array)
#         sliding_end = perf_counter()
#         sliding_time = sliding_end - sliding_start

#         board_state.model = model
#         nn_start = perf_counter()
#         board_state.calc_nn_heuristic_for_board(board_state.num_tiles, board_state.blanks,
#                                                 board_state.manhattan_h_val, board_state.euclidean_h_val, board_state.displaced_tiles)
#         nn_end = perf_counter()
#         nn_time = nn_end - nn_start

#         data_writer.writerow(
#             [f"03x03_board_{board_num}.csv", sliding_time, nn_time])
