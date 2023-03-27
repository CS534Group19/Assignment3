import astar
from board_state import BoardState
from initialization import Initialization
import os

HEURISTIC_OPTIONS = ["Sliding", "Greedy"]
WEIGHTED_OPTIONS = ["True", "False"]

Assignment3Dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
BOARDS_DIR = f"{Assignment3Dir}\\documentation\\test_boards"
OUTPUT_DIR = f"{Assignment3Dir}\\documentation\\data"

def main():
    initial = Initialization(BOARDS_DIR + "\\3x3x2.csv")
    board_state = BoardState(initial.board, initial.goal, HEURISTIC_OPTIONS[0], WEIGHTED_OPTIONS[0])
    astar.a_star(board_state)

if __name__ == "__main__":
    main()