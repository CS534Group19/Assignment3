from initialization import Initialization
from board_state import BoardState
import time
import sys


def a_star(board_state):
    open = [board_state]
    closed = []
    current_state: BoardState

    print("Running A*...\n")

    start_time = time.perf_counter()
    while True:
        current_state = open.pop(0)
        if current_state.board_array == board_state.goal_array:
            output_data = []
            moves = []
            effort_total = 0
            final_depth = current_state.node_depth
            while current_state.parent is not None:
                effort_total += current_state.effort
                moves.append(current_state.move_title)
                current_state = current_state.parent
            moves.reverse()
            for move in moves:
                print(move)

            print(f"\nNodes expanded: {len(closed)}")
            output_data.append(len(closed))
            print(f"Moves required: {len(moves)}")
            output_data.append(len(moves))
            print(f"Solution Cost: {effort_total}")
            output_data.append(effort_total)
            if final_depth != 0:
                print(
                    f"Estimated branching factor {len(closed)**(1/final_depth):0.3f}")
                output_data.append(len(closed)**(1/final_depth))
            else:
                print(f"Estimated branching factor undefined")
                output_data.append("undefined")
            end_time = time.perf_counter()
            print(f"\nSearch took {end_time - start_time:0.4f} seconds")
            output_data.append(end_time - start_time)
            return output_data
        children = current_state.get_children()
        for child in children:
            # Speeds up processing by not checking the child if it's already been checked
            if child.board_array in [board.board_array for board in closed]:
                continue
            if child.board_array not in [board.board_array for board in open]:
                open.append(child)
        closed.append(current_state)
        open.sort(key=lambda x: x.f)


if __name__ == "__main__":
    FILE_NAME = sys.argv[1]
    HEURISTIC = sys.argv[2]
    WEIGHTED = sys.argv[3]

    new_board = Initialization(FILE_NAME)
    board_state = BoardState(new_board.board, new_board.goal, HEURISTIC, WEIGHTED)

    a_star(board_state)
