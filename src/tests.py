import astar
from board_state import BoardState
from initialization import Initialization
import os
import csv

import numpy as np
import tensorflow as tf
import pickle

Assignment3Dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
BOARDS_DIR = f"{Assignment3Dir}\\documentation\\3Boards"
OUTPUT_DIR = f"{Assignment3Dir}\\documentation\\data"

# 10 runs of 3x3x2.csv with sliding heuristic and weighted
# with open(f"{OUTPUT_DIR}\\data_astar_3x3x2_sliding_weighted.csv", "w", newline="") as data_file:
#     data_writer = csv.writer(data_file)
#     data_writer.writerow(["File Name", "Heuristic", "Weighted", "Nodes Expanded",
#                          "Moves Required", "Solution Cost", "Estimated Branching Factor", "Search Time"])
#     for i in range(10):
#         new_board = Initialization(
#             f"{Assignment3Dir}\\documentation\\test_boards\\3x3x2.csv")
#         board_state = BoardState(
#             new_board.board, new_board.goal, "Sliding", "True")
#         data_writer.writerow(
#             ["3x3x2.csv", "Sliding", "True"] + astar.a_star(board_state))

with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(f"{OUTPUT_DIR}\\data_03x03_1000_tests_NN.csv", "w", newline="") as data_file:
    heur = "NN"
    weight = "False"
    data_writer = csv.writer(data_file)
    data_writer.writerow(["File Name", "Heuristic", "Weighted", "Nodes Expanded",
                        "Moves Required", "Solution Cost", "Estimated Branching Factor", "Search Time"])
    for i in range(1):
        new_board = Initialization(
            f"{BOARDS_DIR}\\03x03_board_{i}.csv", 3, heur, weight, model, scaler)
        # print(new_board.goal)
        board_state = BoardState(
            new_board.board, new_board.goal, new_board.heuristic_type, new_board.weighted, new_board.blanks, new_board.manhattan_h_val, new_board.euclidean_h_val, new_board.tiles_displaced, new_board.model, new_board.scaler)
        data_writer.writerow(
            [f"03x03_board_{i}.csv", heur, weight] + astar.a_star(board_state))