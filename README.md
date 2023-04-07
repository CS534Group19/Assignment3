# Assignment3
Supervised Learning

## Group 19
- Michael Alicea, malicea2@wpi.edu
- Cutter Beck, cjbeck@wpi.edu
- Jeffrey Davis, jrdavis2@wpi.edu
- Oliver Shulman, ohshulman@wpi.edu
- Edward Smith, essmith@wpi.edu

## Dependencies
- Python 3.11.2 or higher
- From the Assignment3 directory, run `pip install -r requirements.txt`

## Run Instructions
- From the src directory, run `python astar.py path_to_board.csv heuristic tile_weight?`
    - path_to_board.csv: the path location of the board CSV file the algorithm will search
    - heuristic: one of `Sliding`, `Greedy`, or `Learned`
        - **`Learned` will run the learned Neural Network heuristic**
    - tile_weight?: one of `True` or `False`
        - This will run the A* Search with either a weighted heuristic (`True`) or an unweighted heuristic (`False`)

## Output
1. Moves
    1. The exact moves necessary to solve the board state
2. Nodes Expanded
    1. The total number of nodes expanded during the search for the goal
3. Moves Required
    1. total number of moves required to reach the goal
4. Solution Cost
    1. The total cost of the moves adjusted for the tile weights
5. Estimated Branching Factor
    1. The average number of branches from a parent node to a child node during the search. This is computed with the formula (total # of expanded nodes) ^ (1 / solution node depth)