import os
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler

Assignment3Dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
PROCESSED_DIR = f"{Assignment3Dir}\\documentation\\processed_data"

def load_n_puzzle_dataset(dim):
    BOARDS_DIR = f"{Assignment3Dir}\\documentation\\{dim}Boards"
    # Make sure to swap Bs and 0s in the board
    num_tiles = []
    blanks = []
    manhattan_h_val = []
    euclidean_h_val = []
    displaced_tiles = []
    effort = []
    for file in os.listdir(BOARDS_DIR):
        with open(f"{BOARDS_DIR}\\{file}", "r") as board_file:
            csv_reader = csv.reader(board_file, delimiter=",")
            data = list(csv_reader)
            
            num_tiles.append(float(data[dim][0]))
            blanks.append(float(data[dim + 1][0]))
            manhattan_h_val.append(float(data[dim + 2][0]))
            euclidean_h_val.append(float(data[dim + 3][0]))
            displaced_tiles.append(float(data[dim + 4][0]))
            effort.append(float(data[dim + 5][0]))

    print(f"Number of Tiles: {num_tiles[-1]}")
    print(f"Number of Blanks: {blanks[-1]}")
    print(f"Manhttan H Val: {manhattan_h_val[-1]}")
    print(f"Euclidean H Val: {euclidean_h_val[-1]}")
    print(f"Displaced Tiles: {displaced_tiles[-1]}")
    print(f"Effort: {effort[-1]}")

    return num_tiles, blanks, manhattan_h_val, euclidean_h_val, displaced_tiles, effort

# Load the dataset (Replace this with your own dataset loading function)\
num_tiles3, blanks3, manhattan_h_val3, euclidean_h_val3, displaced_tiles3, effort3 = load_n_puzzle_dataset(3)
num_tiles4, blanks4, manhattan_h_val4, euclidean_h_val4, displaced_tiles4, effort4 = load_n_puzzle_dataset(3)

num_tiles = num_tiles3 + num_tiles4
blanks = blanks3 + blanks4
manhattan_h_val = manhattan_h_val3 + manhattan_h_val4
euclidean_h_val = euclidean_h_val3 + euclidean_h_val4
displaced_tiles = displaced_tiles3 + displaced_tiles4
effort = effort3 + effort4

with open(f"{PROCESSED_DIR}\\num_tiles.csv", "w", newline="") as file:
    csv_writer = csv.writer(file, delimiter=",")
    for item in num_tiles:
        csv_writer.writerow([item])

with open(f"{PROCESSED_DIR}\\blanks.csv", "w", newline="") as file:
    csv_writer = csv.writer(file, delimiter=",")
    for item in blanks:
        csv_writer.writerow([item])

with open(f"{PROCESSED_DIR}\\manhattan_h_val.csv", "w", newline="") as file:
    csv_writer = csv.writer(file, delimiter=",")
    for item in manhattan_h_val:
        csv_writer.writerow([item])

with open(f"{PROCESSED_DIR}\\euclidean_h_val.csv", "w", newline="") as file:
    csv_writer = csv.writer(file, delimiter=",")
    for item in euclidean_h_val:
        csv_writer.writerow([item])

with open(f"{PROCESSED_DIR}\\displaced_tiles.csv", "w", newline="") as file:
    csv_writer = csv.writer(file, delimiter=",")
    for item in displaced_tiles:
        csv_writer.writerow([item])

with open(f"{PROCESSED_DIR}\\effort.csv", "w", newline="") as file:
    csv_writer = csv.writer(file, delimiter=",")
    for item in effort:
        csv_writer.writerow([item])