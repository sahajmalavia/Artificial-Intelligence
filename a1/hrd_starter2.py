import argparse
import sys
import pdb
import heapq

# ====================================================================================

char_single = '2'


class Piece:
    """
    This represents a piece on the Hua Rong Dao puzzle.
    """

    def __init__(self, is_2_by_2, is_single, coord_x, coord_y, orientation):
        """
        :param is_2_by_2: True if the piece is a 2x2 piece and False otherwise.
        :type is_2_by_2: bool
        :param is_single: True if this piece is a 1x1 piece and False otherwise.
        :type is_single: bool
        :param coord_x: The x coordinate of the top left corner of the piece.
        :type coord_x: int
        :param coord_y: The y coordinate of the top left corner of the piece.
        :type coord_y: int
        :param orientation: The orientation of the piece (one of 'h' or 'v') 
            if the piece is a 1x2 piece. Otherwise, this is None
        :type orientation: str
        """

        self.is_2_by_2 = is_2_by_2
        self.is_single = is_single
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.orientation = orientation

    def set_coords(self, coord_x, coord_y):
        """
        Move the piece to the new coordinates. 

        :param coord: The new coordinates after moving.
        :type coord: int
        """

        self.coord_x = coord_x
        self.coord_y = coord_y

    def __repr__(self):
        return '2by2:{} single:{} x:{} y:{} orientation:{}'.format(self.is_2_by_2, self.is_single, \
                                                                   self.coord_x, self.coord_y, self.orientation)


class Board:
    """
    Board class for setting up the playing board.
    """

    def __init__(self, height, pieces):
        """
        :param pieces: The list of Pieces
        :type pieces: List[Piece]
        """

        self.width = 4
        self.height = height
        self.pieces = pieces

        # self.grid is a 2-d (size * size) array automatically generated
        # using the information on the pieces when a board is being created.
        # A grid contains the symbol for representing the pieces on the board.
        self.grid = []
        self.__construct_grid()

        self.blanks = []

    # customized eq for object comparison.
    def __eq__(self, other):
        if isinstance(other, Board):
            return self.grid == other.grid
        return False

    def __construct_grid(self):
        """
        Called in __init__ to set up a 2-d grid based on the piece location information.

        """

        for i in range(self.height):
            line = []
            for j in range(self.width):
                line.append('.')
            self.grid.append(line)

        for piece in self.pieces:
            if piece.is_2_by_2:
                self.grid[piece.coord_y][piece.coord_x] = '1'
                self.grid[piece.coord_y][piece.coord_x + 1] = '1'
                self.grid[piece.coord_y + 1][piece.coord_x] = '1'
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = '1'
            elif piece.is_single:
                self.grid[piece.coord_y][piece.coord_x] = char_single
            else:
                if piece.orientation == 'h':
                    self.grid[piece.coord_y][piece.coord_x] = '<'
                    self.grid[piece.coord_y][piece.coord_x + 1] = '>'
                elif piece.orientation == 'v':
                    self.grid[piece.coord_y][piece.coord_x] = '^'
                    self.grid[piece.coord_y + 1][piece.coord_x] = 'v'

    def display(self):
        """
        Print out the current board.

        """
        for i, line in enumerate(self.grid):
            for ch in line:
                print(ch, end='')
            print()


class State:
    """
    State class wrapping a Board with some extra current state information.
    Note that State and Board are different. Board has the locations of the pieces.
    State has a Board and some extra information that is relevant to the search:
    heuristic function, f value, current depth and parent.
    """

    def __init__(self, board, hfn, f, depth, g, parent=None):
        """
        :param board: The board of the state.
        :type board: Board
        :param hfn: The heuristic function.
        :type hfn: Optional[Heuristic]
        :param f: The f value of current state.
        :type f: int
        :param depth: The depth of current state in the search tree.
        :type depth: int
        :param g: The g value of current state..
        :type g: int
        :param parent: The parent of current state.
        :type parent: Optional[State]
        """
        self.board = board
        self.hfn = hfn
        self.f = f
        self.depth = depth
        self.g = g
        self.parent = parent
        self.path = []

    def __lt__(self, other):
        return self.f < other.f


class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        else:
            raise IndexError("pop from empty stack")


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.count = 0

    def insert(self, item):
        # Use a tuple (item.f, count, item) to ensure the heap is ordered by item.f
        heapq.heappush(self.heap, (item.f, self.count, item))
        self.count += 1

    def extract(self):
        # Pop the smallest item from the heap
        return heapq.heappop(self.heap)[2]

    def is_empty(self):
        return len(self.heap) == 0


def read_from_file(filename):
    """
    Load initial board from a given file.

    :param filename: The name of the given file.
    :type filename: str
    :return: A loaded board
    :rtype: Board
    """

    puzzle_file = open(filename, "r")

    line_index = 0
    pieces = []
    final_pieces = []
    final = False
    found_2by2 = False
    finalfound_2by2 = False
    height_ = 0

    for line in puzzle_file:
        height_ += 1
        if line == '\n':
            if not final:
                height_ = 0
                final = True
                line_index = 0
            continue
        if not final:  # initial board
            for x, ch in enumerate(line):
                if ch == '^':  # found vertical piece
                    pieces.append(Piece(False, False, x, line_index, 'v'))
                elif ch == '<':  # found horizontal piece
                    pieces.append(Piece(False, False, x, line_index, 'h'))
                elif ch == char_single:
                    pieces.append(Piece(False, True, x, line_index, None))
                elif ch == '1':
                    if found_2by2 == False:
                        pieces.append(Piece(True, False, x, line_index, None))
                        found_2by2 = True
        else:  # goal board
            for x, ch in enumerate(line):
                if ch == '^':  # found vertical piece
                    final_pieces.append(Piece(False, False, x, line_index, 'v'))
                elif ch == '<':  # found horizontal piece
                    final_pieces.append(Piece(False, False, x, line_index, 'h'))
                elif ch == char_single:
                    final_pieces.append(Piece(False, True, x, line_index, None))
                elif ch == '1':
                    if finalfound_2by2 == False:
                        final_pieces.append(Piece(True, False, x, line_index, None))
                        finalfound_2by2 = True
        line_index += 1

    puzzle_file.close()
    board = Board(height_, pieces)
    goal_board = Board(height_, final_pieces)
    return board, goal_board


def find_legal(piece, empty_squares) -> list:
    """
        find legal moves for <piece> given a list of coordinates for empty squares
        returns a list of directions <piece> can move in. Directions include "down", "up", "right", and "left".
        """
    directions = []
    if piece.is_2_by_2:
        if ((piece.coord_x, piece.coord_y + 2) in empty_squares) and ((piece.coord_x + 1, piece.coord_y + 2) in
                                                                      empty_squares):
            directions.append("down")
        if ((piece.coord_x, piece.coord_y - 1) in empty_squares) and ((piece.coord_x + 1, piece.coord_y - 1) in
                                                                      empty_squares):
            directions.append("up")
        if ((piece.coord_x + 2, piece.coord_y) in empty_squares) and ((piece.coord_x + 2, piece.coord_y + 1) in
                                                                      empty_squares):
            directions.append("right")
        if ((piece.coord_x - 1, piece.coord_y) in empty_squares) and ((piece.coord_x - 1, piece.coord_y + 1) in
                                                                      empty_squares):
            directions.append("left")
    if piece.orientation == "h":
        if ((piece.coord_x, piece.coord_y + 1) in empty_squares) and ((piece.coord_x + 1, piece.coord_y + 1) in
                                                                      empty_squares):
            directions.append("down")
        if ((piece.coord_x, piece.coord_y - 1) in empty_squares) and ((piece.coord_x + 1, piece.coord_y - 1) in
                                                                      empty_squares):
            directions.append("up")
        if (piece.coord_x + 2, piece.coord_y) in empty_squares:
            directions.append("right")
        if (piece.coord_x - 1, piece.coord_y) in empty_squares:
            directions.append("left")
    if piece.is_single:
        if (piece.coord_x, piece.coord_y + 1) in empty_squares:
            directions.append("down")
        if (piece.coord_x, piece.coord_y - 1) in empty_squares:
            directions.append("up")
        if (piece.coord_x + 1, piece.coord_y) in empty_squares:
            directions.append("right")
        if (piece.coord_x - 1, piece.coord_y) in empty_squares:
            directions.append("left")
    if piece.orientation == "v":
        if (piece.coord_x, piece.coord_y + 2) in empty_squares:
            directions.append("down")
        if (piece.coord_x, piece.coord_y - 1) in empty_squares:
            directions.append("up")
        if ((piece.coord_x + 1, piece.coord_y) in empty_squares) and ((piece.coord_x + 1, piece.coord_y + 1) in
                                                                      empty_squares):
            directions.append("right")
        if ((piece.coord_x - 1, piece.coord_y) in empty_squares) and ((piece.coord_x - 1, piece.coord_y + 1) in
                                                                      empty_squares):
            directions.append("left")
    return directions


def find_successors(state, goal_board):
    """
    returns a list of successor states for a given <state>
    """
    lst = []
    board = state.board
    pieces = board.pieces
    empty_squares = find_blanks(state)
    for piece in pieces:
        legal_directions = find_legal(piece, empty_squares)
        for direction in legal_directions:
            lst.append(find_state(piece, state, direction, goal_board))
    return lst


def find_state(piece, state, direction, goal_board) -> State:
    """
        returns a State when <piece> is moved in <direction> from current <state>
        """
    board = state.board
    new_pieces = []
    for p in board.pieces:
        if p == piece:
            if direction == "down":
                new_p = Piece(p.is_2_by_2, p.is_single, p.coord_x, p.coord_y + 1, p.orientation)
                new_pieces.append(new_p)
            elif direction == "up":
                new_p = Piece(p.is_2_by_2, p.is_single, p.coord_x, p.coord_y - 1, p.orientation)
                new_pieces.append(new_p)
            elif direction == "right":
                new_p = Piece(p.is_2_by_2, p.is_single, p.coord_x + 1, p.coord_y, p.orientation)
                new_pieces.append(new_p)
            elif direction == "left":
                new_p = Piece(p.is_2_by_2, p.is_single, p.coord_x - 1, p.coord_y, p.orientation)
                new_pieces.append(new_p)
        else:
            new_pieces.append(p)
    new_board = Board(height=board.height, pieces=new_pieces)
    f = heuristic_function(new_board, goal_board) + state.g + 1
    new_state = State(board=new_board, hfn=heuristic_function(new_board, goal_board), f=f, depth=state.depth + 1,
                      g=state.g + 1,
                      parent=state)
    return new_state


def find_blanks(state):
    """
    returns coordinates of the blank sqares of a board
    """
    lst = []
    grid = state.board.grid
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == '.':
                lst.append((j, i))
    return lst


def grid_to_string(grid):
    """
        Returns a string version of Board.grid
        """
    return '\n'.join([''.join(row) for row in grid])


def heuristic_function(board: Board, goal_board: Board) -> int:
    """
        returns a heuristic given <board> and <goal_board>
        """
    distance = 0
    goal_set = {"single": [], "2by2": [], "h": [], "v": []}
    for p in goal_board.pieces:
        if p.is_single:
            goal_set["single"].append(p)
        elif p.is_2_by_2:
            goal_set["2by2"].append(p)
        elif p.orientation == "h":
            goal_set["h"].append(p)
        elif p.orientation == "v":
            goal_set["v"].append(p)

    for piece in board.pieces:
        min = float('inf')
        min_p = None
        if piece.is_single:
            for p2 in goal_set["single"]:
                d = find_distance(piece.coord_x, piece.coord_y, p2.coord_x, p2.coord_y)
                if d < min:
                    min = d
                    min_p = p2
            goal_set["single"].remove(min_p)
        elif piece.is_2_by_2:
            for p2 in goal_set["2by2"]:
                d = find_distance(piece.coord_x, piece.coord_y, p2.coord_x, p2.coord_y)
                if d < min:
                    min = d
                    min_p = p2
            goal_set["2by2"].remove(min_p)
        elif piece.orientation == "h":
            for p2 in goal_set["h"]:
                d = find_distance(piece.coord_x, piece.coord_y, p2.coord_x, p2.coord_y)
                if d < min:
                    min = d
                    min_p = p2
            goal_set["h"].remove(min_p)
        elif piece.orientation == "v":
            for p2 in goal_set["v"]:
                d = find_distance(piece.coord_x, piece.coord_y, p2.coord_x, p2.coord_y)
                if d < min:
                    min = d
                    min_p = p2
            goal_set["v"].remove(min_p)

        distance += min

    return distance


def find_distance(curr_x, curr_y, goal_x, goal_y) -> tuple:
    """
    Returns the manhattan distance between two coordinates.
    """
    return abs(curr_x - goal_x) + abs(curr_y - goal_y)


def astar(board, goal_board) -> str:
    """
        returns path from <board> to <goal_board> using a* search
        """
    f = heuristic_function(board, goal_board)
    start = State(board, heuristic_function(board, goal_board), f, 0, 0, None)
    start.path = [start.board.grid]

    frontier = PriorityQueue()
    frontier.insert(start)
    visited_grids = set()

    while not frontier.is_empty():
        curr = frontier.extract()
        curr_grid_tuple = tuple(map(tuple, curr.board.grid))

        if curr_grid_tuple in visited_grids:
            continue
        else:
            visited_grids.add(curr_grid_tuple)
            if curr.board == goal_board:
                return "\n\n".join(grid_to_string(path) for path in curr.path)
            else:
                successors = find_successors(curr, goal_board)
                for successor in successors:
                    successor.path = curr.path + [successor.board.grid]
                    frontier.insert(successor)

    return "No solution"


def dfs(board, goal_board) -> str:
    """
    returns path from <board> to <goal_board> using depth-first-search (DFS)
    """
    path = ""
    f = heuristic_function(board, goal_board)
    start = State(board, heuristic_function(board, goal_board), f, 0, 0, None)
    start.path = [start.board.grid]
    frontier = Stack()
    frontier.push(start)
    visited_grids = set()
    while not frontier.is_empty():
        curr = frontier.pop()
        curr_grid_tuple = tuple(map(tuple, curr.board.grid))
        # path += grid_to_string(curr.board.grid) + "\n\n" #add to the path here
        if curr_grid_tuple in visited_grids:
            pass
        else:
            visited_grids.add(curr_grid_tuple)
            # path += grid_to_string(curr.board.grid) #moved above
            if curr.board == goal_board:
                return "\n\n".join(grid_to_string(path) for path in curr.path) #path #return the path
            else:
                successors = find_successors(curr, goal_board)
                for successor in successors:
                    successor.path = curr.path + [successor.board.grid]
                    frontier.push(successor)

    return "No solution"
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzles."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=['astar', 'dfs'],
        help="The searching algorithm."
    )
    args = parser.parse_args()

    # read the board from the file
    board, goal_board = read_from_file(args.inputfile)

    if args.algo == "astar":
        solution = astar(board, goal_board)
    elif args.algo == "dfs":
        solution = dfs(board, goal_board)

    with open(args.outputfile, "w") as f:
        f.write(solution + "\n\n")

    # An example of how to write solutions to the outputfile. (This is not a correct solution, of course).
    # with open(args.outputfile, 'w') as sys.stdout:
    #    board.display()
    #    print("")
    #    goal_board.display()