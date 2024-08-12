from copy import deepcopy
import time

# Defining Global Variables.
EMPTY = '-'  # Symbol representing an empty cell on the game board
X = 'X'      # Symbol representing Player X
O = "O"      # Symbol representing Player O
T = "T"      # Symbol representing a Tie (when the game ends with no winner)
SIZE = 9      # Total number of mini-boards in the game
SIZEMINI = 3  # Size of each mini-board
WINNING_POSITIONS = [  # List of tuples representing winning positions on the mini-boards
    ((0, 0), (0, 1), (0, 2)),
    ((1, 0), (1, 1), (1, 2)),
    ((2, 0), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 1), (1, 1), (2, 1)),
    ((0, 2), (1, 2), (2, 2)),
    ((0, 0), (1, 1), (2, 2)),
    ((0, 2), (1, 1), (2, 0))
]
MAXSCORE = 10000  # Maximum score value for the game
print_player_moves = False  # Flag to control whether player moves should be printed

# Global variables list
global_vars = [EMPTY, X, O, T, SIZE, SIZEMINI, WINNING_POSITIONS, MAXSCORE]

import players  # Importing a module named 'players'

class UtttState:
    '''A class to represent a state in Ultimate Tic-Tac-Toe game.'''
    # Constructor method to initialize the state
    def __init__(self, currentplayer, otherplayer, last_move=None, board_array=None, num_moves=0):
        if board_array != None:
            self.board_array = board_array
        else:
            # Creating an empty 3D array to represent the game board
            self.board_array = [[[EMPTY] * SIZEMINI for _ in range(SIZEMINI)] for _ in range(SIZE)]
        self.current = currentplayer
        self.other = otherplayer
        self.last_move = last_move
        self.master_board = build_master(self.board_array)  # Building the master board
        self.num_moves = num_moves

    # String representation of the state
    def __repr__(self) -> str:
        new_string = "Last Action Chosen: " + str(self.last_move) + "     Player Turn: " + str(
            self.current.get_sign()) + "\n"
        new_string += "-------------------------------------\n"
        for m in range(SIZEMINI):
            for j in range(SIZEMINI):
                new_string += "|  "
                for i in range(SIZEMINI):
                    for k in range(SIZEMINI):
                        new_string += self.board_array[i + (3 * m)][j][k] + "  "
                    new_string += "|  "
                new_string += "\n"
            new_string += "-------------------------------------\n"

        new_string += " ## MASTER BOARD ##\n"
        new_string += "-------------\n"
        for i in range(SIZEMINI):
            new_string += "|  "
            for j in range(SIZEMINI):
                new_string += self.master_board[i][j] + "  "
            new_string += "|  "
            new_string += "\n"
        new_string += "-------------\n"
        # new_string += "Current Player: " + str(self.current.get_sign()) + "    Heuristic: " + str(self.heuristic) + "\n"
        return new_string
    


def terminal_test(state):
    # INPUT: 3X3 GAME BOARD
    # OUTPUT: BOOLEAN IF GAME IS WON OR NOT, CHARACTER OF WINNING PLAYER
    return terminal_test_mini(state.master_board)


def terminal_test_mini(mini_board):
    """Terminal test for Mini Boards. Input is the 3*3 array and outputs whether the game has ended and the character who won or 'T' if tied."""
    for positions in WINNING_POSITIONS:
        a, b, c = positions
        if (mini_board[a[0]][a[1]] != EMPTY) and (mini_board[a[0]][a[1]] != T):
            if (mini_board[a[0]][a[1]] == mini_board[b[0]][b[1]] == mini_board[c[0]][c[1]]):
                return True, mini_board[a[0]][a[1]]
    ## It might be the case that a mini board has tied.
    ## Now check for a tie. In our game, a tie only happens if all the board pieces have been filled, and there is yet no winner.
    for i in range(SIZEMINI):
        for j in range(SIZEMINI):
            if mini_board[i][j] == EMPTY:
                # If any empty place is found, that means we have not Tied.
                return False, None
    ## We reached here means the board is not won yet, and it is a Tie.
    return True, T


def build_master(game):
    """This is a time-consuming function should be used with caution."""
    # INPUT: 9X3X3 MASTER GAME BOARD
    # OUTPUT: SMALLER 3X3 GAME BOARD REPRESENTING LARGER BOARD
    return_game = [[EMPTY] * SIZEMINI for _ in range(SIZEMINI)]
    for i in range(SIZE):
        mini_board = game[i]
        game_won, winner = terminal_test_mini(mini_board)
        if game_won:
            return_game[i // 3][i % 3] = winner
    return return_game

## Change Legal actions to account for when the selected board is full,
#  and hence the player can move anywhere on the board.
def actions(state):
    """Return a list of possible actions possible from this state (Instance of UtttState).
    Actions look like (i, x, y), where i is the mini board number and x, y
    represent the row and column of the place in the mini-board."""
    legal_actions = []

    ## If this is the first move.
    if state.last_move == None:
        # All actions are legal.
        # 'mb' stands for mini-board,
        for mb in range(SIZE):
            for i in range(SIZEMINI):
                for j in range(SIZEMINI):
                    legal_actions.append((mb, i, j))
    else:
        # There was some last move.
        last_mb, last_i, last_j = state.last_move
        # Find the new legal mb from the last i, j.
        mb = 3 * last_i + last_j
        # We need to check that this mb is empty in the master board, or else we need to give all other open boards as legal moves.
        if state.master_board[mb // 3][mb % 3] == EMPTY:
            for i in range(3):
                for j in range(3):
                    if state.board_array[mb][i][j] == EMPTY:
                        legal_actions.append((mb, i, j))
        else:
            # This is when the master_board has a Tie or Has been won.
            # Add all mb to the possible boards.
            possible_mb = []
            for i in range(SIZEMINI):
                for j in range(SIZEMINI):
                    if state.master_board[i][j] == EMPTY:
                        possible_mb.append(3 * i + j)
            for mb in possible_mb:
                for i in range(SIZEMINI):
                    for j in range(SIZEMINI):
                        if state.board_array[mb][i][j] == EMPTY:
                            legal_actions.append((mb, i, j))
    return legal_actions

def result(state, action):
    """Returns the resulting state after taking the given action.
    Returns None if the action is not legal."""
    # Get the position of move from action.
    mb, i, j = action
    ## Check for legal action, by checking that the position is empty.
    if state.board_array[mb][i][j] != EMPTY:
        return None
    else:
        # New state has to be formed here.
        # Make the new Board array.
        new_board = deepcopy(state.board_array)
        new_board[mb][i][j] = state.current.sign
        new_state = UtttState(state.other, state.current, last_move=action, board_array=new_board, num_moves=state.num_moves + 1)
        return new_state


def search_test(player1, player2, num_iterations=1, printouts=True):
    # Function to conduct a series of game simulations and evaluate player performance
    score = [0, 0, 0]  # [player1, player2, tie]
    ts = time.time()
    for i in range(num_iterations):
        print("Iteration: ", i + 1, "/", num_iterations, "  Score: ", score, "[P1, P2, Tie]")
        winner = play_game(player1, player2, printouts)
        if winner == player1.get_sign():
            score[0] += 1
        elif winner == player2.get_sign():
            score[1] += 1
        elif winner == T:
            score[2] += 1
        else:
            raise Exception("Winner not recognized")
    t_stop = time.time()
    print("Final score: [P1, P2, Tie]", score)
    print("Computational time: ", t_stop - ts, "s    Average: ", (t_stop - ts) / num_iterations, "s")
    return score

def play_game(p1=None, p2=None, printouts=True):
    """Play the game with two players. Default use two humans."""
    # Function to simulate a game between two players
    if p1 == None:
        p1 = players.HumanPlayer(X)
    if p2 == None:
        p2 = players.HumanPlayer(O)

    s = UtttState(p1, p2)
    while True:
        action = p1.make_move(s)
        if action not in actions(s):
            print("Illegal move made by X")
            print("O wins!")
            return None
        s = result(s, action)
        if printouts: print(s)
        game_over, winner = terminal_test(s)
        if game_over:
            print("Game Over")
            print("Player " + winner + " wins!")
            if not printouts: print(s)
            return winner
        action = p2.make_move(s)
        if action not in actions(s):
            print("Illegal move made by O")
            print("X wins!")
            return None
        s = result(s, action)
        if printouts: print(s)
        game_over, winner = terminal_test(s)
        if game_over:
            print("Game Over")
            print("Player " + winner + " wins!")
            if not printouts: print(s)
            return winner


def main():
    # HEURISTIC OPTIONS
    # Name - Max_tested_working_depth - designer
    # players.heu.heuristic1 - Designed and made by Mohit/Alex
    # players.heu.pulkit_github - Designed and made by Pulkit (From uttt github python repo)
    # players.heu.heuristic2 - Designed and made by Mohit/Alex, gained inspiration from fulkit_github

    # PLAYER OPTIONS
    # Name - designer
    # players.HumanPlayer - Designed and made by Mohit/Alex
    # players.AlphaBetaPlayer(PlayerSymbol (char), depth_limit (int), heuristic) - Designed and made by Mohit/Alex
    # players.MinimaxPlayer(PlayerSymbol (char), depth_limit (int), heuristic) - Designed and made by Mohit/Alex
    # players.RandomPlayer - Designed and made by Mohit/Alex

    # RUNNING THE PROGRAM :
    # TO RUN THE PROGRAM, 2 PLAYERS NEED TO BE DEFINED. AN EXAMPLE IS SHOWN BELOW
    # BOTH PLAYERS NEED TO BE PASSED TO THE PLAY_GAME FUNCTION. THE PLAY_GAME FUNCTION WILL HANDLE
    # ALL GAMEPLAY UNTIL THE GAME IS TERMINATED IN WIN, LOSS OR TIE.
    # IN THE PLAY_GAME() FUNCTION, THE BOOL VARIABLE 'PRINTOUTS' CAN BE MARKED TRUE OR FALSE
    # THIS CORRESPONDS TO WHETHER OR NOT EACH GAME WILL BE PRINTED AFTER EACH MOVE. ITS DEFAULT IS TRUE
    # NOTE: IN  PRINTING THE MASTER BOARD, THE SYMBOL 'T' CORRESPONDS TO A SUB-BOARD THAT ENDED IN A TIE
    # HumanPlayer and RandomPlyer take one positional argument i.e., player_symbol which is char (p1 : X, p2 : O).
    # AlphaBetaPlayer() and MinimaxPlayer() also take two named arguments i.e., depth_limit(defaults to 4) and heuristic(defauts to huiristic2).

    p1 = players.AlphaBetaPlayer(X, depth_limit=6, heuristic=players.heu.heuristic2)
    p2 = players.HumanPlayer(O)
    play_game(p1, p2, printouts=True)


    # CODE USED IN THE FINAL TESTING OF THE ALGORITHMS AND HEURISTICS
    # THE FOLLOWING IS A FUNCTION THAT ALLOWS THE USER TO RUN MULTIPLE AGENT MATCHUPS.
    # 2 PLAYERS NEED TO BE DEFINED. GENERALLY THE HUMAN AGENT IS NOT GOOD FOR THIS MODE
    # THE NUMBER OF GAMES TO BE PLAYED IS DEFINED AS THE 'ITERS' VARIABLE
    # THE PROGRAM WILL THEN KEEP TRACK OF TIME STATS AND SCORE STATS
    # TO FIND THE SCORE RESULTS, RUN THE FUNCTION SEARCH_TEST(), WITH THE ARGUMENTS: PLAYER1, PLAYER2, PRINTOUTS(BOOL)
    # SETTING PRINOUTS TO TRUE WILL PRINT OUT THE GAMEPLAY AS THE GAMES ARE RUN
    # TIME WILL BE TALLIED AND PRINTED FROM WITHIN THE SEARCH_TEST() FUCNTION

    # p1 = players.AlphaBetaPlayer(O, 6, heu.homemadeV2)
    # p2 = players.AlphaBetaPlayer(X, 6, heu.pulkit_github)
    # iters = 1
    # print("Running homemadeV2 vs PulkitGithub test")
    # score = search_test(p1, p2, iters, printouts=True)
    # print("Score: [random_wins, homemade_wins, ties", score)


if __name__ == '__main__':
    main()



