import os
import csv

Assignment3Dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
PROCESSED_DIR = f"{Assignment3Dir}\\documentation\\processed_data"

def load_n_puzzle_dataset(dim):
    BOARDS_DIR = f"{Assignment3Dir}\\documentation\\{dim}Boards"
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
boards, manhattan_distances, dimensions, blanks = load_n_puzzle_dataset(3)

def flatten():
    # Flatten the boards
    boards_array = []
    for board in boards:
        boards_array.append([item for sublist in board for item in sublist])
    return boards_array

flattened_boards = flatten()
print(flattened_boards[-1])

with open(f"{PROCESSED_DIR}\\flattend_boards.csv", "w", newline="") as file:
    csv_writer = csv.writer(file, delimiter=",")
    csv_writer.writerows(flattened_boards)

with open(f"{PROCESSED_DIR}\\manhattan_distances.csv", "w", newline="") as file:
    csv_writer = csv.writer(file, delimiter=",")
    for item in manhattan_distances:
        csv_writer.writerow([item])

with open(f"{PROCESSED_DIR}\\dimensions.csv", "w", newline="") as file:
    csv_writer = csv.writer(file, delimiter=",")
    for item in dimensions:
        csv_writer.writerow([item])

with open(f"{PROCESSED_DIR}\\blanks.csv", "w", newline="") as file:
    csv_writer = csv.writer(file, delimiter=",")
    for item in blanks:
        csv_writer.writerow([item])