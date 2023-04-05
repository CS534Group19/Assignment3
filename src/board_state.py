import numpy as np


class BoardState():
    def __init__(self, board_array, goal, heuristic_type: str, weighted: str, blanks: int, model, scaler, move_cost: int = 0, parent=None, move_title="", effort=0, node_depth=0):
        """
        ### Parameters
        - board_array: 2D representation of the board
        - goal_array: 2D representation of the goal state
        - heuristic_type: the type of heuristic to use
        - weighted: whether or not to use weighted heuristics

        ### Represents
        - board_array: 2D representation of the board
        """
        self.board_array = board_array
        self.side_length = len(self.board_array)
        self.heuristic_type = heuristic_type
        self.weighted = weighted
        self.goal_array = goal
        self.blanks = blanks
        self.model = model
        self.scaler = scaler

        self.g = move_cost  # g
        self.h = self.getHVal(self.heuristic_type)  # h
        self.f = self.g + self.h  # f

        self.parent = parent
        self.move_title = move_title

        self.effort = effort
        self.node_depth = node_depth

    def __str__(self):
        return str(self.board_array)

    def calc_manhattan_distance_for_value(self, value: int, value_x, value_y) -> int:
        if value == 0:
            return 0
        else:
            for row in range(self.side_length):
                for col in range(self.side_length):
                    if self.goal_array[row][col] == value:
                        # Do Manhattan Calculation
                        if self.weighted == "True":
                            return (abs(value_x - row) + abs(value_y - col)) * value
                        else:
                            return (abs(value_x - row) + abs(value_y - col))

    def calc_total_manhattan_for_board(self, board_array) -> int:
        total = 0
        for row in range(self.side_length):
            for col in range(self.side_length):
                total += self.calc_manhattan_distance_for_value(
                    board_array[row][col], row, col)
        return total

    # def calc_euclidean_distance_for_value(self, value: int, value_x, value_y) -> int:
    #     if value == 0:
    #         return 0
    #     else:
    #         for row in range(self.side_length):
    #             for col in range(self.side_length):
    #                 if self.goal_array[row][col] == value:
    #                     if self.weighted == "True":
    #                         return (abs(value_x - row)**2 + abs(value_y - col)**2)**(1/2) * value
    #                     else:
    #                         return (abs(value_x - row)**2 + abs(value_y - col)**2)**(1/2)

    # def calc_total_euclidean_for_board(self, board_array) -> int:
    #     total = 0
    #     for row in range(self.side_length):
    #         for col in range(self.side_length):
    #             total += self.calc_euclidean_distance_for_value(
    #                 board_array[row][col], row, col)
    #     return total

    def flatten(self, board):
        # Flatten the boards
        board_array = [item for row in board for item in row]
        return board_array

    def calc_nn_heuristic_for_board(self, board, dimensions, blanks, model, scaler):
        # Preprocess the board
        # Normalize
        board = self.flatten(board)
        board_array = np.array([board]).astype(np.float32) / (len(board)-1)

        # Preprocess dimensions and blanks
        dimensions_array = scaler.transform(
            np.array([dimensions]).reshape(-1, 1))
        blanks_array = scaler.transform(np.array([blanks]).reshape(-1, 1))

        # Predict the Manhattan distance using the model
        manhattan_distance_estimate = model.predict(
            [board_array, dimensions_array, blanks_array])[0][0]
        print(manhattan_distance_estimate)

        return manhattan_distance_estimate

    def getHVal(self, heuristic_type: str) -> int:
        if heuristic_type == "Sliding":
            return self.calc_total_manhattan_for_board(self.board_array)
        elif heuristic_type == "NN":
            # TODO: Use NN to determine heuristic
            return self.calc_nn_heuristic_for_board(self.board_array, self.side_length, self.blanks, self.model, self.scaler)

    def get_children(self):
        if self.parent is None:
            g = 0
            node_depth = 0
        else:
            g = self.parent.g
            node_depth = self.parent.node_depth
        states = []
        for row in range(self.side_length):
            for col in range(self.side_length):
                if self.board_array[row][col] == 0:
                    # Up
                    delta_y = row+1
                    if delta_y >= 0 and delta_y < self.side_length:
                        if self.board_array[delta_y][col] != 0:
                            current_copy = [x[:] for x in self.board_array]
                            # # Swap the zero and the value
                            current_copy[row][col] = current_copy[delta_y][col]
                            current_copy[delta_y][col] = 0
                            states.append(BoardState(current_copy, self.goal_array,
                                                     self.heuristic_type, self.weighted, self.blanks, self.model, self.scaler, g+1, self, move_title=f"Move {self.board_array[delta_y][col]} up", effort=self.board_array[delta_y][col], node_depth=node_depth+1))
                    # Down
                    delta_y = row-1
                    if delta_y >= 0 and delta_y < self.side_length:
                        if self.board_array[delta_y][col] != 0:
                            current_copy = [x[:] for x in self.board_array]
                            # Swap the zero and the value
                            current_copy[row][col] = current_copy[delta_y][col]
                            current_copy[delta_y][col] = 0
                            states.append(BoardState(current_copy, self.goal_array,
                                                     self.heuristic_type, self.weighted, self.blanks, self.model, self.scaler, g+1, self, move_title=f"Move {self.board_array[delta_y][col]} down", effort=self.board_array[delta_y][col], node_depth=node_depth+1))
                    # Left
                    delta_x = col-1
                    if delta_x >= 0 and delta_x < self.side_length:
                        if self.board_array[row][delta_x] != 0:
                            current_copy = [x[:] for x in self.board_array]
                            # Swap the zero and the value
                            current_copy[row][col] = current_copy[row][delta_x]
                            current_copy[row][delta_x] = 0
                            states.append(BoardState(current_copy, self.goal_array,
                                                     self.heuristic_type, self.weighted, self.blanks, self.model, self.scaler, g+1, self, move_title=f"Move {self.board_array[row][delta_x]} left", effort=self.board_array[row][delta_x], node_depth=node_depth+1))
                    # Right
                    delta_x = col+1
                    if delta_x >= 0 and delta_x < self.side_length:
                        if self.board_array[row][delta_x] != 0:
                            current_copy = [x[:] for x in self.board_array]
                            # Swap the zero and the value
                            current_copy[row][col] = current_copy[row][delta_x]
                            current_copy[row][delta_x] = 0
                            states.append(BoardState(current_copy, self.goal_array,
                                                     self.heuristic_type, self.weighted, self.blanks, self.model, self.scaler, g+1, self, move_title=f"Move {self.board_array[row][delta_x]} right", effort=self.board_array[row][delta_x], node_depth=node_depth+1))
        return states
