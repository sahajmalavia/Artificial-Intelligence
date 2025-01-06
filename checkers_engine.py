import argparse
import copy
import sys
import time

cache = {}  # you can use this to implement state caching


class State:
    # This class is used to represent a state.
    # board : a list of lists that represents the 8*8 board
    def __init__(self, board):

        self.board = board
        self.evaluation = None

        self.width = 8
        self.height = 8

    def display(self):
        for i in self.board:
            for j in i:
                print(j, end="")
            print("")
        print("")

    def return_display(self):
        s = ""
        for i in self.board:
            for j in i:
                s += str(j)
            s += "\n"
        return s


def get_opp_char(player):
    if player in ['b', 'B']:
        return ['r', 'R']
    else:
        return ['b', 'B']


def get_next_turn(curr_turn):
    if curr_turn == 'red':
        return 'black'
    else:
        return 'red'


def find_multiple_captures(board, tile, j, i):
    # Return all possible capture sequences
    capture_sequences = []
    if tile == "r":
        if (i > 1 and j > 1 and (board[j - 1][i - 1] in ["b", "B"]) and
                board[j - 2][i - 2] == "."):
            new_board = [row[:] for row in board]
            new_board[j][i] = '.'
            new_board[j - 1][i - 1] = '.'
            if (j - 2) == 0:
                new_board[j - 2][i - 2] = "R"
            else:
                new_board[j - 2][i - 2] = "r"

            capture_state = State(new_board)
            sequence = [capture_state]
            future_sequences = find_multiple_captures(new_board, tile, j - 2, i - 2)

            if future_sequences:
                for future in future_sequences:
                    capture_sequences.append(sequence + future)
            else:
                capture_sequences.append(sequence)

        if (i < 6 and j > 1 and (board[j - 1][i + 1] in ["b", "B"]) and
                board[j - 2][i + 2] == "."):
            new_board = [row[:] for row in board]
            new_board[j][i] = '.'
            new_board[j - 1][i + 1] = '.'
            if (j - 2) == 0:
                new_board[j - 2][i + 2] = "R"
            else:
                new_board[j - 2][i + 2] = "r"

            capture_state = State(new_board)
            sequence = [capture_state]
            future_sequences = find_multiple_captures(new_board, tile, j - 2, i + 2)

            if future_sequences:
                for future in future_sequences:
                    capture_sequences.append(sequence + future)
            else:
                capture_sequences.append(sequence)

    elif tile == "R":
        if (i > 1 and j < 6 and (board[j + 1][i - 1] in ["b", "B"]) and
                board[j + 2][i - 2] == "."):
            new_board = [row[:] for row in board]
            new_board[j][i] = '.'
            new_board[j + 1][i - 1] = '.'
            new_board[j + 2][i - 2] = "R"

            capture_state = State(new_board)
            sequence = [capture_state]
            future_sequences = find_multiple_captures(new_board, tile, j + 2, i - 2)

            if future_sequences:
                for future in future_sequences:
                    capture_sequences.append(sequence + future)
            else:
                capture_sequences.append(sequence)

        if (i < 6 and j < 6 and (board[j + 1][i + 1] in ["b", "B"]) and
                board[j + 2][i + 2] == "."):
            new_board = [row[:] for row in board]
            new_board[j][i] = '.'
            new_board[j + 1][i + 1] = '.'
            new_board[j + 2][i + 2] = "R"

            capture_state = State(new_board)
            sequence = [capture_state]
            future_sequences = find_multiple_captures(new_board, tile, j + 2, i + 2)

            if future_sequences:
                for future in future_sequences:
                    capture_sequences.append(sequence + future)
            else:
                capture_sequences.append(sequence)

        if (i > 1 and j > 1 and (board[j - 1][i - 1] in ["b", "B"]) and
                board[j - 2][i - 2] == "."):
            new_board = [row[:] for row in board]
            new_board[j][i] = '.'
            new_board[j - 1][i - 1] = '.'
            new_board[j - 2][i - 2] = "R"

            capture_state = State(new_board)
            sequence = [capture_state]
            future_sequences = find_multiple_captures(new_board, tile, j - 2, i - 2)

            if future_sequences:
                for future in future_sequences:
                    capture_sequences.append(sequence + future)
            else:
                capture_sequences.append(sequence)

        if (i < 6 and j > 1 and (board[j - 1][i + 1] in ["b", "B"]) and
                board[j - 2][i + 2] == "."):
            new_board = [row[:] for row in board]
            new_board[j][i] = '.'
            new_board[j - 1][i + 1] = '.'
            new_board[j - 2][i + 2] = "R"

            capture_state = State(new_board)
            sequence = [capture_state]
            future_sequences = find_multiple_captures(new_board, tile, j - 2, i + 2)

            if future_sequences:
                for future in future_sequences:
                    capture_sequences.append(sequence + future)
            else:
                capture_sequences.append(sequence)

    elif tile == "b":
        if (i > 1 and j < 6 and (board[j + 1][i - 1] in ["r", "R"]) and
                board[j + 2][i - 2] == "."):

            new_board = [row[:] for row in board]
            new_board[j][i] = '.'
            new_board[j + 1][i - 1] = '.'
            if (j + 2) == 7:
                new_board[j + 2][i - 2] = "B"
            else:
                new_board[j + 2][i - 2] = "b"

            capture_state = State(new_board)
            sequence = [capture_state]
            future_sequences = find_multiple_captures(new_board, tile, j + 2, i - 2)

            if future_sequences:
                for future in future_sequences:
                    capture_sequences.append(sequence + future)
            else:
                capture_sequences.append(sequence)

        if (i < 6 and j < 6 and (board[j + 1][i + 1] in ["r", "R"]) and
                board[j + 2][i + 2] == "."):

            new_board = [row[:] for row in board]
            new_board[j][i] = '.'
            new_board[j + 1][i + 1] = '.'
            if (j + 2) == 7:
                new_board[j + 2][i + 2] = "B"
            else:
                new_board[j + 2][i + 2] = "b"

            capture_state = State(new_board)
            sequence = [capture_state]
            future_sequences = find_multiple_captures(new_board, tile, j + 2, i + 2)

            if future_sequences:
                for future in future_sequences:
                    capture_sequences.append(sequence + future)
            else:
                capture_sequences.append(sequence)

    elif tile == "B":
        if (i > 1 and j < 6 and (board[j + 1][i - 1] in ["r", "R"]) and
                board[j + 2][i - 2] == "."):

            new_board = [row[:] for row in board]
            new_board[j][i] = '.'
            new_board[j + 1][i - 1] = '.'
            new_board[j + 2][i - 2] = "B"

            capture_state = State(new_board)
            sequence = [capture_state]
            future_sequences = find_multiple_captures(new_board, tile, j + 2, i - 2)

            if future_sequences:
                for future in future_sequences:
                    capture_sequences.append(sequence + future)
            else:
                capture_sequences.append(sequence)

        if (i < 6 and j < 6 and (board[j + 1][i + 1] in ["r", "R"]) and
                board[j + 2][i + 2] == "."):

            new_board = [row[:] for row in board]
            new_board[j][i] = '.'
            new_board[j + 1][i + 1] = '.'
            new_board[j + 2][i + 2] = "B"

            capture_state = State(new_board)
            sequence = [capture_state]
            future_sequences = find_multiple_captures(new_board, tile, j + 2, i + 2)

            if future_sequences:
                for future in future_sequences:
                    capture_sequences.append(sequence + future)
            else:
                capture_sequences.append(sequence)

        if (i > 1 and j > 1 and (board[j - 1][i - 1] in ["r", "R"]) and
                board[j - 2][i - 2] == "."):

            new_board = [row[:] for row in board]
            new_board[j][i] = '.'
            new_board[j - 1][i - 1] = '.'
            new_board[j - 2][i - 2] = "B"

            capture_state = State(new_board)
            sequence = [capture_state]
            future_sequences = find_multiple_captures(new_board, tile, j - 2, i - 2)

            if future_sequences:
                for future in future_sequences:
                    capture_sequences.append(sequence + future)
            else:
                capture_sequences.append(sequence)

        if (i < 6 and j > 1 and (board[j - 1][i + 1] in ["r", "R"]) and
                board[j - 2][i + 2] == "."):
            new_board = [row[:] for row in board]
            new_board[j][i] = '.'
            new_board[j - 1][i + 1] = '.'
            new_board[j - 2][i + 2] = "B"

            capture_state = State(new_board)
            sequence = [capture_state]
            future_sequences = find_multiple_captures(new_board, tile, j - 2, i + 2)

            if future_sequences:
                for future in future_sequences:
                    capture_sequences.append(sequence + future)
            else:
                capture_sequences.append(sequence)

    return capture_sequences


def find_possible_moves(state, turn):
    # Return all possible moves for the current player's turn
    if turn == "black":
        player = ["b", "B"]
    else:
        player = ["r", "R"]
    possible_moves = []
    possible_captures = []

    for i in range(state.width):
        for j in range(state.height):
            if state.board[j][i] in player:
                tile = state.board[j][i]
                if tile == "r":
                    capture_sequences = find_multiple_captures(state.board, tile, j, i)
                    if capture_sequences:
                        for capture_state in capture_sequences:
                            possible_captures.append(capture_state[-1])
                    else:
                        if i > 0 and j > 0 and state.board[j - 1][i - 1] == ".":
                            new_board = [row[:] for row in state.board]
                            new_board[j][i] = "."
                            if j - 1 == 0:
                                new_board[j - 1][i - 1] = "R"
                            else:
                                new_board[j - 1][i - 1] = "r"

                            possible_moves.append(State(new_board))

                        if i < 7 and j > 0 and state.board[j - 1][i + 1] == ".":
                            new_board = [row[:] for row in state.board]
                            new_board[j][i] = "."
                            if j - 1 == 0:
                                new_board[j - 1][i + 1] = "R"
                            else:
                                new_board[j - 1][i + 1] = "r"

                            possible_moves.append(State(new_board))
                elif tile == "R":
                    capture_sequences = find_multiple_captures(state.board, tile, j, i)
                    if capture_sequences:
                        for capture_state in capture_sequences:
                            possible_captures.append(capture_state[-1])
                    else:
                        if i > 0 and j < 7 and state.board[j + 1][i - 1] == ".":
                            new_board = [row[:] for row in state.board]
                            new_board[j][i] = "."
                            new_board[j + 1][i - 1] = "R"

                            possible_moves.append(State(new_board))
                        if i < 7 and j < 7 and state.board[j + 1][i + 1] == ".":
                            new_board = [row[:] for row in state.board]
                            new_board[j][i] = "."
                            new_board[j + 1][i + 1] = "R"

                            possible_moves.append(State(new_board))

                        if i > 0 and j > 0 and state.board[j - 1][i - 1] == ".":
                            new_board = [row[:] for row in state.board]
                            new_board[j][i] = "."
                            new_board[j - 1][i - 1] = "R"

                            possible_moves.append(State(new_board))
                        if i < 7 and j > 0 and state.board[j - 1][i + 1] == ".":
                            new_board = [row[:] for row in state.board]
                            new_board[j][i] = "."
                            new_board[j - 1][i + 1] = "R"

                            possible_moves.append(State(new_board))

                elif tile == "b":
                    capture_sequences = find_multiple_captures(state.board, tile, j, i)
                    if capture_sequences:
                        for capture_state in capture_sequences:
                            possible_captures.append(capture_state[-1])
                    else:
                        if i > 0 and j < 7 and state.board[j + 1][i - 1] == ".":
                            new_board = [row[:] for row in state.board]
                            new_board[j][i] = "."
                            if j + 1 == 7:
                                new_board[j + 1][i - 1] = "B"
                            else:
                                new_board[j + 1][i - 1] = "b"

                            possible_moves.append(State(new_board))
                        if i < 7 and j < 7 and state.board[j + 1][i + 1] == ".":
                            new_board = [row[:] for row in state.board]
                            new_board[j][i] = "."
                            if j + 1 == 7:
                                new_board[j + 1][i + 1] = "B"
                            else:
                                new_board[j + 1][i + 1] = "b"

                            possible_moves.append(State(new_board))

                elif tile == "B":
                    capture_sequences = find_multiple_captures(state.board, tile, j, i)
                    if capture_sequences:
                        for capture_state in capture_sequences:
                            possible_captures.append(capture_state[-1])
                    else:
                        if i > 0 and j < 7 and state.board[j + 1][i - 1] == ".":
                            new_board = [row[:] for row in state.board]
                            new_board[j][i] = "."
                            new_board[j + 1][i - 1] = "B"

                            possible_moves.append(State(new_board))
                        if i < 7 and j < 7 and state.board[j + 1][i + 1] == ".":
                            new_board = [row[:] for row in state.board]
                            new_board[j][i] = "."
                            new_board[j + 1][i + 1] = "B"

                            possible_moves.append(State(new_board))

                        if i > 0 and j > 0 and state.board[j - 1][i - 1] == ".":
                            new_board = [row[:] for row in state.board]
                            new_board[j][i] = "."
                            new_board[j - 1][i - 1] = "B"

                            possible_moves.append(State(new_board))
                        if i < 7 and j > 0 and state.board[j - 1][i + 1] == ".":
                            new_board = [row[:] for row in state.board]
                            new_board[j][i] = "."
                            new_board[j - 1][i + 1] = "B"

                            possible_moves.append(State(new_board))

    if possible_captures:
        return possible_captures

    return possible_moves


def node_order(successors, turn):
    for successor in successors:
        successor.evaluation = evaluate(successor, turn)

    if turn == "red":
        return sorted(successors, key=lambda k: k.evaluation, reverse=True)
    elif turn == "black":
        return sorted(successors, key=lambda k: k.evaluation)


def alpha_beta(state, MIN, MAX, player, alpha, beta, depth, path=[]):
    # Return the best move for current turn
    if terminal(state, player):
        return utility(depth, get_next_turn(player)), state, path + [state]
    if depth == 0:
        return evaluate(state, player), state, path + [state]

    if MAX:
        evalMAX = -float('inf')
        best_move = None
        best_path = []

        successors = find_possible_moves(state, "red")
        successors_sorted = node_order(successors, "red")
        for successor in successors_sorted:
            value, state2, path2 = alpha_beta(successor, True, False, get_next_turn(player), alpha, beta, depth - 1, path+[state])
            if value > evalMAX:
                evalMAX = value
                best_move = successor
                best_path = path2

            alpha = max(alpha, value)

            if beta <= alpha:
                break

        return evalMAX, best_move, best_path

    elif MIN:
        evalMIN = float('inf')
        best_move = None
        best_path = []

        successors = find_possible_moves(state, "black")
        successors_sorted = node_order(successors, "black")
        for successor in successors_sorted:
            value, state2, path2 = alpha_beta(successor, False, True, get_next_turn(player), alpha, beta, depth - 1, path+[state])
            if value < evalMIN:
                evalMIN = value
                best_move = successor
                best_path = path2

            beta = min(beta, value)

            if beta <= alpha:
                break

        return evalMIN, best_move, best_path


def terminal(state, turn):
    # This function is incorrect but has a general idea of what my tired head remembers atm
    if not find_possible_moves(state, turn):
        return True
    return False


def evaluate(state, turn):
    # It's just a start I'll build on it
    if turn == "black":
        player = ["b", "B"]
    else:
        player = ["r", "R"]

    opp_player = get_opp_char(turn[0])

    kings_num = 0
    normal_num = 0
    opp_kings = 0
    opp_normal = 0

    for i in range(8):
        for j in range(8):
            if state.board[j][i] in player and state.board[j][i].isupper():
                kings_num += 1
            elif state.board[j][i] in player and state.board[j][i].islower():
                normal_num += 1
            elif state.board[j][i] in opp_player and state.board[j][i].isupper():
                opp_kings += 1
            elif state.board[j][i] in opp_player and state.board[j][i].islower():
                opp_normal += 1

    total_points = (2 * kings_num + normal_num) - (2 * opp_kings + opp_normal)

    if turn == "red":
        return total_points
    return -total_points


def utility(depth, winner):
    if winner == "red":
        return 9999999999 + depth
    elif winner == "black":
        return -9999999999 - depth


def gts(state, turn, c):

    depth = 9
    alpha = -float('inf')
    beta = float('inf')

    if turn == "r":
        player = "red"
        MAX = True
        MIN = False
    elif turn == "b":
        player = "black"
        MAX = False
        MIN = True

    evaluation, best_state, path = alpha_beta(state, MIN, MAX, player, alpha, beta, depth)
    string = ""
    for k in path:
        string += k.return_display() + "\n"
    return string


def read_from_file(filename):

    f = open(filename)
    lines = f.readlines()
    board = [[str(x) for x in l.rstrip()] for l in lines]
    f.close()

    return board


if __name__ == '__main__':

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
    args = parser.parse_args()

    initial_board = read_from_file(args.inputfile)
    state = State(initial_board)
    turn = 'r'
    ctr = 0

    s = gts(state, turn, ctr)

    with open(args.outputfile, "w") as f:
        f.write(s + "\n\n")