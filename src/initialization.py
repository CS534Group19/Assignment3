import csv

class Initialization():
    def __init__(self, csv_file):
        """
        ### Parameters
        - csv_file: the file path to the csv file containing the board

        ### Represents
        - board: a 2D array of the board
        - goal: a 2D array of the goal state
        """
        self.csv_file = csv_file
        self.board = self.create_2D_board()
        self.side_length = len(self.board)
        self.goal = self.find_goal_state()

    def create_2D_board(self):
        board = []
        # TODO is this encoding proper?
        with open(self.csv_file, 'r', encoding='utf-8-sig') as f:
            csv_reader = csv.reader(f, delimiter=',')
            for row in csv_reader:
                temp = []
                for value in row:
                    if value == "B":
                        temp.append(0)
                    else:
                        temp.append(int(value))
                board.append(temp)
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

    def find_goal_state(self):
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