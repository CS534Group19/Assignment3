import csv

class Initialization():
    def __init__(self, csv_file, sides, heuristic, weighted):
        """
        ### Parameters
        - csv_file: the file path to the csv file containing the board

        ### Represents
        - board: a 2D array of the board
        - goal: a 2D array of the goal state
        """
        self.csv_file = csv_file
        self.sides = sides
        self.board = self.create_2D_board()
        self.side_length = len(self.board)
        self.heuristic_type = heuristic
        self.weighted = weighted
        self.front_goal = self.find_goal_state_front()
        self.back_goal = self.find_goal_state_back()
        self.goal = self.calc_best_goal(self.heuristic_type)

    def create_2D_board(self):
        board = []
        # TODO is this encoding proper?
        with open(self.csv_file, 'r', encoding='utf-8-sig') as f:
            csv_reader = csv.reader(f, delimiter=',')
            row_counter = 0
            for row in csv_reader:
                if row_counter < self.sides:
                    temp = []
                    for value in row:
                        if value == "B":
                            temp.append(0)
                        else:
                            temp.append(int(value))
                    board.append(temp)
                row_counter += 1
        return board

    def create_1D_board(self):
        board_1D = []
        for row in self.board:
            for value in row:
                board_1D.append(value)
        return board_1D

    def make_2D(self, sorted_1D: list[int]):
        """
        ### Parameters
        - sorted_1D: a sorted 1D representation of a board

        ### Returns
        - 2D array of the sorted board
        """
        goal_1D = sorted_1D
        goal_2D = []
        index = 0
        for i in range(self.side_length):
            temp_row = []
            for j in range(self.side_length):
                temp_row.append(goal_1D[index])
                index += 1
            goal_2D.append(temp_row)
        return goal_2D

    def find_goal_state_back(self):
        """Returns a 2D array created from a re-arranged sorted 1D array with all zeros in the bottom right
        """
        sorted_board = sorted(self.create_1D_board())
        end_zeroes = 0
        for i in range(len(sorted_board)):
            if sorted_board[i] == 0:
                end_zeroes = i
        zero_list = sorted_board[0: end_zeroes + 1]
        sorted_board = sorted_board[end_zeroes + 1:] + zero_list
        return self.make_2D(sorted_board)
    
    def find_goal_state_front(self):
        """Returns a 2D array created from a re-arranged sorted 1D array with all zeros in the top left
        """
        sorted_board = sorted(self.create_1D_board())
        return self.make_2D(sorted_board)
    


    def calc_manhattan_distance_for_value(self, value: int, value_x, value_y, goal_board) -> int:
        if value == 0:
            return 0
        else:
            for row in range(self.side_length):
                for col in range(self.side_length):
                    if goal_board[row][col] == value:
                        # Do Manhattan Calculation
                        if self.weighted == "True":
                            return (abs(value_x - row) + abs(value_y - col)) * value
                        else:
                            return (abs(value_x - row) + abs(value_y - col))

    def calc_total_manhattan_for_board(self, board_array, goal_board) -> int:
        total = 0
        for row in range(self.side_length):
            for col in range(self.side_length):
                total += self.calc_manhattan_distance_for_value(
                    board_array[row][col], row, col, goal_board)
        return total


    def calc_euclidean_distance_for_value(self, value: int, value_x, value_y) -> int:
        if value == 0:
            return 0
        else:
            for row in range(self.side_length):
                for col in range(self.side_length):
                    if self.goal_array[row][col] == value:
                        if self.weighted == "True":
                            return (abs(value_x - row)**2 + abs(value_y - col)**2)**(1/2) * value
                        else:
                            return (abs(value_x - row)**2 + abs(value_y - col)**2)**(1/2)

    def calc_total_euclidean_for_board(self, board_array) -> int:
        total = 0
        for row in range(self.side_length):
            for col in range(self.side_length):
                total += self.calc_euclidean_distance_for_value(
                    board_array[row][col], row, col)
        return total

    def getHVal(self, heuristic_type: str, goal_board) -> int:
        if heuristic_type == "Sliding":
            return self.calc_total_manhattan_for_board(self.board, goal_board)
        elif heuristic_type == "Greedy":
            # TODO: Use NN to determine heuristic
            return self.calc_total_euclidean_for_board(self.board_array)

    
    def calc_best_goal(self, heuristic_type: str):
        back_h = self.getHVal(heuristic_type, self.back_goal)
        front_h = self.getHVal(heuristic_type, self.front_goal)

        if back_h < front_h:
            return self.back_goal
        else:
            return self.front_goal