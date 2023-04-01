import astar
from board_state import BoardState
from initialization import Initialization
import os
import csv

Assignment3Dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
BOARDS_DIR = f"{Assignment3Dir}\\documentation\\tagged_boards"
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


with open(f"{OUTPUT_DIR}\\data_03x03_1000_tests.csv", "w", newline="") as data_file:
    heur = "Sliding"
    weight = "True"
    data_writer = csv.writer(data_file)
    data_writer.writerow(["File Name", "Heuristic", "Weighted", "Nodes Expanded",
                        "Moves Required", "Solution Cost", "Estimated Branching Factor", "Search Time"])
    for i in range(1000):
        new_board = Initialization(
            f"{BOARDS_DIR}\\03x03_board_{i}.csv", 3, heur, weight)
        board_state = BoardState(
            new_board.board, new_board.goal, new_board.heuristic_type, new_board.weighted)
        data_writer.writerow(
            [f"03x03_board_{i}.csv", heur, weight] + astar.a_star(board_state))