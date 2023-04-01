import csv
import os

Assignment3Dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
BOARDS_DIR = f"{Assignment3Dir}\\documentation\\tagged_boards"

# write code to loop over every file in BOARDS_DIR
for file in os.listdir(BOARDS_DIR):
    # open the file
    dimension = 0
    blank_counter = 0
    rows = []
    row_counter = 0
    metadata_added = False
    with open(f"{BOARDS_DIR}\\{file}", "r") as board_file:
        # read the file
        csv_reader = csv.reader(board_file, delimiter=",")
        data = list(csv_reader)
        dimension = len(data[0])
        # loop over each row in the file
        for row in data:
            if row_counter < dimension:
                for value in row:
                    if value == "B":
                        blank_counter += 1
                rows.append(row)
            row_counter += 1
        best_manhattan = data[dimension]
        rows.append(best_manhattan)
    with open(f"{BOARDS_DIR}\\{file}", "w", newline="") as board_file:
        csv_writer = csv.writer(board_file)
        rows.append([dimension])
        rows.append([blank_counter])
        csv_writer.writerows(rows)


