# 6.034 Fall 2010 Lab 3: Games
# Name: <Your Name>
# Email: <Your Email>

from util import INFINITY

### 1. Multiple choice

# 1.1. Two computerized players are playing a game. Player MM does minimax
#      search to depth 6 to decide on a move. Player AB does alpha-beta
#      search to depth 6.
#      The game is played without a time limit. Which player will play better?
#
#      1. MM will play better than AB.
#      2. AB will play better than MM.
#      3. They will play with the same level of skill.
ANSWER1 = 3

# 1.2. Two computerized players are playing a game with a time limit. Player MM
# does minimax search with iterative deepening, and player AB does alpha-beta
# search with iterative deepening. Each one returns a result after it has used
# 1/3 of its remaining time. Which player will play better?
#
#   1. MM will play better than AB.
#   2. AB will play better than MM.
#   3. They will play with the same level of skill.
ANSWER2 = 2

### 2. Connect Four
from connectfour import *
from basicplayer import *
from util import *
import tree_searcher

## This section will contain occasional lines that you can uncomment to play
## the game interactively. Be sure to re-comment them when you're done with
## them.  Please don't turn in a problem set that sits there asking the
## grader-bot to play a game!
## 
## Uncomment this line to play a game as white:
# run_game(human_player, basic_player)

## Uncomment this line to play a game as black:
#run_game(basic_player, human_player)

## Or watch the computer play against itself:
# run_game(basic_player, basic_player)

## Change this evaluation function so that it tries to win as soon as possible,
## or lose as late as possible, when it decides that one side is certain to win.
## You don't have to change how it evaluates non-winning positions.

def focused_evaluate(board):
    """
    Given a board, return a numeric rating of how good
    that board is for the current player.
    A return value >= 1000 means that the current player has won;
    a return value <= -1000 means that the current player has lost
    """    
    score = 0
    
    if board.is_game_over():
    
        # If the game has been won, we know that it must have been
        # won or ended by the previous move.
        # The previous move was made by our opponent.
        # Therefore, we can't have won, so return -1000.
        # (note that this causes a tie to be treated like a loss)
        score = -1000
    else:

        for move, next_board in get_all_next_moves(board):
            if next_board.longest_chain(next_board.get_other_player_id()) >= 4:
                score = 1000
                break
        score  += board.longest_chain(board.get_current_player_id()) * 10
        
    # Prefer having your pieces in the center of the board.
        for row in range(6):
            for col in range(7):
                if board.get_cell(row, col) == board.get_current_player_id():
                    score -= abs(3-col)
                elif board.get_cell(row, col) == board.get_other_player_id():
                    score += abs(3-col)

    return score

# print "focused score", focused_evaluate(BARELY_WINNING_BOARD)

## Create a "player" function that uses the focused_evaluate function
quick_to_win_player = lambda board: minimax(board, depth=4,
                                            eval_fn=focused_evaluate)

## You can try out your new evaluation function by uncommenting this line:
# run_game(basic_player, quick_to_win_player)
# run_game(human_player, quick_to_win_player)

## Write an alpha-beta-search procedure that acts like the minimax-search
## procedure, but uses alpha-beta pruning to avoid searching bad ideas
## that can't improve the result. The tester will check your pruning by
## counting the number of static evaluations you make.
##
## You can use minimax() in basicplayer.py as an example.

# def alpha_beta_search(board, depth,
#                       eval_fn,
#                       # NOTE: You should use get_next_moves_fn when generating
#                       # next board configurations, and is_terminal_fn when
#                       # checking game termination.
#                       # The default functions set here will work
#                       # for connect_four.
#                       get_next_moves_fn=get_all_next_moves,
# 		      is_terminal_fn=is_terminal, verbose=True):
#     best_val = (NEG_INFINITY, None, None)
    
#     for move, new_board in get_next_moves_fn(board):
#         val = -1 * alpha_beta_find_board_value(new_board, depth-1, eval_fn,
#                                             get_next_moves_fn,
#                                             is_terminal_fn, NEG_INFINITY, INFINITY)
#         if val > best_val[0]:
#             best_val = (val, move, new_board)
            
#     if verbose:
#         print "ALPHA_BETA: Depth %d-Decided on column %s with rating %d" % (depth, best_val[1], best_val[0])

#     return best_val[1]

total_count = 0
def alpha_beta_search(board, depth,
                    eval_fn,
                    # NOTE: You should use get_next_moves_fn when generating
                    # next board configurations, and is_terminal_fn when
                    # checking game termination.
                    # The default functions set here will work
                    # for connect_four.
                    get_next_moves_fn=get_all_next_moves,
		            is_terminal_fn=is_terminal, verbose=True):
    max_player = True
    try:
        if board.get_current_player_id() == 2:
            max_player = False
    except:
        if board.node_type == "MIN":
            max_player = False
    global total_count
    total_count = 0
    # print "at", board, "max", max_player
    alpha = NEG_INFINITY
    beta = INFINITY
    if max_player:
        best_val, best_move = NEG_INFINITY, None
        for move, new_board in get_next_moves_fn(board):
            val = max(best_val, min_value(new_board, depth-1, alpha, beta, eval_fn, get_next_moves_fn, is_terminal_fn))
            if val > best_val:
                best_val = val
                best_move = move
                alpha = val
                # print "best value for", board, best_val, "alpha", alpha, "beta", beta
        if verbose:
            print "ALPHA_BETA: Depth %d-Decided on column %s with rating %d" % (depth, best_move, best_val)   
            print "total evals", total_count   
        return best_move
    else:
        best_val, best_move = INFINITY, None
        for move, new_board in get_next_moves_fn(board):
            # print "min:", "depth", depth, "move", move, "score", best_val
            val = min(best_val, max_value(new_board, depth-1, alpha, beta, eval_fn, get_next_moves_fn, is_terminal_fn))
            if val < best_val:
                best_val = val
                best_move = move
                beta = val
        if verbose:
            print "ALPHA_BETA: Depth %d-Decided on column %s with rating %d" % (depth, best_move, best_val)  
            print "total evals", total_count  
        return best_move


def max_value(board, depth, alpha, beta, eval_fn, get_next_moves_fn=get_all_next_moves,
            is_terminal_fn=is_terminal):
    global total_count
    total_count += 1
    # print "evaluating max:", board, "at depth", depth, "with alpha", alpha, "beta", beta, "..."
    if is_terminal_fn(depth, board):
        return eval_fn(board)
    best_val = NEG_INFINITY
    for move, new_board in get_next_moves_fn(board):
        # print "min value from ", new_board, "value", min_value(new_board, depth-1, alpha, beta, eval_fn,
                                            # get_next_moves_fn, is_terminal_fn)
        # print "max:", ' ' * (4-depth), "depth", depth, "move", move, "score", best_val
        best_val = max(best_val, min_value(new_board, depth-1, alpha, beta, eval_fn,
                                            get_next_moves_fn, is_terminal_fn))
        alpha = max(alpha, best_val)
        if alpha >= beta:
            # Prune!
            # print "pruning ", alpha, beta
            best_val = alpha
            break
    # print "best value for", board, best_val, "alpha", alpha, "beta", beta
    return best_val

def min_value(board, depth, alpha, beta, eval_fn, get_next_moves_fn=get_all_next_moves,
            is_terminal_fn=is_terminal):
    global total_count
    total_count += 1
    # print "evaluating min:", board, "at depth", depth, "with alpha", alpha, "beta", beta, "..."
    if is_terminal_fn(depth, board):
        return -1 * eval_fn(board)
    best_val = INFINITY
    for move, new_board in get_next_moves_fn(board):
        # print "max value from ", new_board, "value", max_value(new_board, depth-1, alpha, beta, eval_fn,
        #                                     get_next_moves_fn, is_terminal_fn)
        best_val = min(best_val, max_value(new_board, depth-1, alpha, beta, eval_fn,
                                            get_next_moves_fn, is_terminal_fn))
        # print "min:", ' ' * (4-depth), "depth", depth, "move", move, "score", best_val
        beta = min(beta, best_val)
        if alpha >= beta:
            # Prune!
            # print "pruning ", alpha, beta
            best_val = beta
            break
    # print "best value for", board, best_val, "alpha", alpha, "beta", beta
    return best_val      

# print "search result:", alpha_beta_search(BARELY_WINNING_BOARD, 6, focused_evaluate)

# print "search result:", minimax(BARELY_WINNING_BOARD, 6, focused_evaluate)
# @count_runs
# def alpha_beta_find_board_value(board, depth, eval_fn,
#                              get_next_moves_fn=get_all_next_moves,
#                              is_terminal_fn=is_terminal, alpha = NEG_INFINITY, beta = INFINITY):
#     """
#     Alpha-beta search helper function: Return the minimax value of a particular board,
#     given a particular depth to estimate to
#     """
#     if is_terminal_fn(depth, board):
#         return eval_fn(board)

#     best_val = NEG_INFINITY
#     for move, new_board in get_next_moves_fn(board):
#         print "evaluating:", new_board, "with value", best_val, "alpha", alpha, "beta", beta
#         best_val = max(best_val, -1 * alpha_beta_find_board_value(new_board, depth-1, eval_fn,
#                                             get_next_moves_fn, is_terminal_fn, -beta, -alpha))
#         alpha = max(alpha, best_val)                                    
        
#         # alpha = max(alpha, best_val)
#         if alpha >= beta:
#             # Prune!
#             print "Pruned!", alpha, beta
#             return alpha
#     return best_val

## Now you should be able to search twice as deep in the same amount of time.
## (Of course, this alpha-beta-player won't work until you've defined
## alpha-beta-search.)
alphabeta_player = lambda board: alpha_beta_search(board,
                                                   depth=5,
                                                   eval_fn=focused_evaluate)

## This player uses progressive deepening, so it can kick your ass while
## making efficient use of time:
quick_to_win_player2 = lambda board: \
    run_search_function(board,
                        search_fn=minimax,
                        eval_fn=focused_evaluate, timeout=5)
ab_iterative_player = lambda board: \
    run_search_function(board,
                        search_fn=alpha_beta_search,
                        eval_fn=focused_evaluate, timeout=5)
#run_game(human_player, alphabeta_player)
# run_game(quick_to_win_player2, ab_iterative_player)
# run_game(quick_to_win_player, alphabeta_player)

## Finally, come up with a better evaluation function than focused-evaluate.
## By providing a different function, you should be able to beat
## simple-evaluate (or focused-evaluate) while searching to the
## same depth.

def better_evaluate(board):
    """
    * play center column
    * odd/even row strategy
    """
    score = 0
    longest = 0
    
    if board.is_game_over():
    
        # If the game has been won, we know that it must have been
        # won or ended by the previous move.
        # The previous move was made by our opponent.
        # Therefore, we can't have won, so return -1000.
        # (note that this causes a tie to be treated like a loss)
        # score = -1000
        
        if board.longest_chain(board.get_current_player_id()) >=4:
            return 2000
        elif board.longest_chain(board.get_other_player_id()) >= 4:
            return -2000
        else:
            return 0
    else:

        # for move, next_board in get_all_next_moves(board):
            
        #     if next_board.longest_chain(next_board.get_other_player_id()) >= 4:
        #         score = 1000
        #         break
        
        # score  += board.longest_chain(board.get_current_player_id()) * 10
        my_big_chains = board.chain_cells(board.get_current_player_id())
        enemy_big_chains =  board.chain_cells(board.get_other_player_id())
        for big_chain in my_big_chains:
            # Give bonus to longer chains, 3 chains deserves 6 points
            if len(big_chain) > 1:
                score += 10 ** (len(big_chain) -1)
        for big_chain in enemy_big_chains:
            if len(big_chain) > 1:
                score -= 10 ** (len(big_chain) -1)
        
        # Prefer having your pieces in the center of the board.
        for row in range(6):
            for col in range(7):
                if board.get_cell(row, col) == board.get_current_player_id():
                    score -= abs(3-col) * 10
                    # odd/even rule
                    if row % 2 == 1 and board.get_current_player_id() == 1:
                        score += 3 * (row + 1)
                elif board.get_cell(row, col) == board.get_other_player_id():
                    score += abs(3-col) * 10
                    # Player 2 prefers even rows
                    if row % 2 == 0 and board.get_current_player_id() == 2:
                        score += 3 * (row + 1)

    return score

def better_evaluate2(board):
    """
    * play center column
    * create a trap with 3 on any row
    * odd/even row strategy
    * establish 7 formation
    """
    score = 0
    SEVEN_BONUS = 50
    CHAIN_BASE = 10
    ODD_EVEN = 10
    CENTER_BONUS = 10
    
    if board.is_game_over():
    
        # If the game has been won, we know that it must have been
        # won or ended by the previous move.
        # The previous move was made by our opponent.
        # Therefore, we can't have won, so return -1000.
        # (note that this causes a tie to be treated like a loss)
        # score = -1000
        
        if board.longest_chain(board.get_current_player_id()) >=4:
            return 2000
        elif board.longest_chain(board.get_other_player_id()) >= 4:
            return -2000
        else:
            # Going for win!
            return 0
    else:
        # Tally up all the chains, 2-chain = 10, 3-chain=100
        my_big_chains = board.chain_cells(board.get_current_player_id())
        enemy_big_chains =  board.chain_cells(board.get_other_player_id())
        for big_chain in my_big_chains:
            # Give bonus to longer chains
            if len(big_chain) > 1:
                score += CHAIN_BASE ** (len(big_chain) -1)
                # Favor open ended chains

                # Huge favor if a chain forms a 7
                if len(big_chain) == 3 and big_chain[0][0] == big_chain[1][0] == big_chain[2][0]:
                    
                    if board.valid_cell(big_chain[1][0]+1, big_chain[1][1]) and \
                        board.valid_cell(big_chain[0][0]+2, big_chain[0][1]) and \
                        board.get_cell(big_chain[1][0]+1, big_chain[1][1]) == \
                        board.get_cell(big_chain[0][0]+2, big_chain[0][1]) == \
                        board.get_current_player_id():
                        score += SEVEN_BONUS
                    if board.valid_cell(big_chain[1][0]-1, big_chain[1][1]) and \
                        board.valid_cell(big_chain[0][0]-2, big_chain[0][1]) and \
                        board.get_cell(big_chain[1][0]-1, big_chain[1][1]) == \
                        board.get_cell(big_chain[0][0]-2, big_chain[0][1]) == \
                        board.get_current_player_id():
                        score += SEVEN_BONUS

                    if board.valid_cell(big_chain[1][0]+1, big_chain[1][1]) and \
                        board.valid_cell(big_chain[0][0]+2, big_chain[2][1]) and \
                        board.get_cell(big_chain[1][0]+1, big_chain[1][1]) == \
                        board.get_cell(big_chain[2][0]+2, big_chain[2][1]) == \
                        board.get_current_player_id():
                        score += SEVEN_BONUS
                    
                    if board.valid_cell(big_chain[1][0]-1, big_chain[1][1]) and \
                        board.valid_cell(big_chain[2][0]-2, big_chain[2][1]) and \
                        board.get_cell(big_chain[1][0]-1, big_chain[1][1]) == \
                        board.get_cell(big_chain[2][0]-2, big_chain[2][1]) == \
                        board.get_current_player_id():
                        score += SEVEN_BONUS

                
        for big_chain in enemy_big_chains:
            if len(big_chain) > 1:
                # Block enemy from building up chains
                score -= CHAIN_BASE ** (len(big_chain) -1)
                # Huge favor if a chain forms a 7
                if len(big_chain) == 3 and big_chain[0][0] == big_chain[1][0] == big_chain[2][0]:
                    
                    
                    if board.valid_cell(big_chain[1][0]+1, big_chain[1][1]) and \
                        board.valid_cell(big_chain[0][0]+2, big_chain[0][1]) and \
                        board.get_cell(big_chain[1][0]+1, big_chain[1][1]) == \
                        board.get_cell(big_chain[0][0]+2, big_chain[0][1]) == \
                        board.get_other_player_id():
                        score -= SEVEN_BONUS
                    if board.valid_cell(big_chain[1][0]-1, big_chain[1][1]) and \
                        board.valid_cell(big_chain[0][0]-2, big_chain[0][1]) and \
                        board.get_cell(big_chain[1][0]-1, big_chain[1][1]) == \
                        board.get_cell(big_chain[0][0]-2, big_chain[0][1]) == \
                        board.get_other_player_id():
                        score -= SEVEN_BONUS

                    if board.valid_cell(big_chain[1][0]+1, big_chain[1][1]) and \
                        board.valid_cell(big_chain[0][0]+2, big_chain[2][1]) and \
                        board.get_cell(big_chain[1][0]+1, big_chain[1][1]) == \
                        board.get_cell(big_chain[2][0]+2, big_chain[2][1]) == \
                        board.get_other_player_id():
                        score -= SEVEN_BONUS
                    
                    if board.valid_cell(big_chain[1][0]-1, big_chain[1][1]) and \
                        board.valid_cell(big_chain[2][0]-2, big_chain[2][1]) and \
                        board.get_cell(big_chain[1][0]-1, big_chain[1][1]) == \
                        board.get_cell(big_chain[2][0]-2, big_chain[2][1]) == \
                        board.get_other_player_id():
                        score -= SEVEN_BONUS
        
        # Prefer having your pieces in the center of the board.
        for row in range(6):
            for col in range(7):
                if board.get_cell(row, col) == board.get_current_player_id():
                    score -= abs(3-col) * CENTER_BONUS
                    # odd/even rule
                    if row % 2 == 1 and board.get_current_player_id() == 1:
                        score += ODD_EVEN * (row + 1)
                elif board.get_cell(row, col) == board.get_other_player_id():
                    score += abs(3-col) * CENTER_BONUS
                    # Player 2 prefers even rows
                    if row % 2 == 0 and board.get_current_player_id() == 2:
                        score += ODD_EVEN * (row + 1)

    return score

# Comment this line after you've fully implemented better_evaluate
# better_evaluate = memoize(basic_evaluate)
# better_evaluate = memoize(focused_evaluate)

# Uncomment this line to make your better_evaluate run faster.
better_evaluate = memoize(better_evaluate)
better_evaluate2 = memoize(better_evaluate2)

# For debugging: Change this if-guard to True, to unit-test
# your better_evaluate function.
if False:
    board_tuples = (( 0,0,0,0,0,0,0 ),
                    ( 0,0,0,0,0,0,0 ),
                    ( 0,0,0,0,0,0,0 ),
                    ( 0,2,2,1,1,2,0 ),
                    ( 0,2,1,2,1,2,0 ),
                    ( 2,1,2,1,1,1,0 ),
                    )
    test_board_1 = ConnectFourBoard(board_array = board_tuples,
                                    current_player = 1)
    test_board_2 = ConnectFourBoard(board_array = board_tuples,
                                    current_player = 2)
    # better evaluate from player 1
    print "%s => %s" %(test_board_1, better_evaluate(test_board_1))
    # better evaluate from player 2
    print "%s => %s" %(test_board_2, better_evaluate(test_board_2))
    print "%s => %s" %(BASIC_STARTING_BOARD_2, better_evaluate(BASIC_STARTING_BOARD_2))

# print "search result:", alpha_beta_search(BASIC_STARTING_BOARD_2, 6, better_evaluate)

## A player that uses alpha-beta and better_evaluate:
your_player = lambda board: run_search_function(board,
                                                search_fn=alpha_beta_search,
                                                eval_fn=better_evaluate,
                                                timeout=5)
your_player2 = lambda board: run_search_function(board,
                                                search_fn=alpha_beta_search,
                                                eval_fn=better_evaluate2,
                                                timeout=5)

#your_player = lambda board: alpha_beta_search(board, depth=4,
#                                              eval_fn=better_evaluate)

## Uncomment to watch your player play a game:
#run_game(your_player, your_player)


## Uncomment this (or run it in the command window) to see how you do
## on the tournament that will be graded.


## These three functions are used by the tester; please don't modify them!
def run_test_game(player1, player2, board):
    assert isinstance(globals()[board], ConnectFourBoard), "Error: can't run a game using a non-Board object!"
    return run_game(globals()[player1], globals()[player2], globals()[board])
    
def run_test_search(search, board, depth, eval_fn):
    assert isinstance(globals()[board], ConnectFourBoard), "Error: can't run a game using a non-Board object!"
    return globals()[search](globals()[board], depth=depth,
                             eval_fn=globals()[eval_fn])

## This function runs your alpha-beta implementation using a tree as the search
## rather than a live connect four game.   This will be easier to debug.
def run_test_tree_search(search, board, depth):
    return globals()[search](globals()[board], depth=depth,
                             eval_fn=tree_searcher.tree_eval,
                             get_next_moves_fn=tree_searcher.tree_get_next_move,
                             is_terminal_fn=tree_searcher.is_leaf)

# run_test_game("your_player", "basic_player", "BASIC_STARTING_BOARD_2")
## Do you want us to use your code in a tournament against other students? See
## the description in the problem set. The tournament is completely optional
## and has no effect on your grade.

def print_results(result_list):
    retVal = [ "\t" + '\t'.join([str(x) for x in range(len(result_list)) ]) + '\t' + "score" ]
    retVal += [ "" + str(i) + '\t' + '\t'.join([str(x) for x in row]) for i, row in enumerate(result_list) ]
    return '\n' + '\n'.join(retVal) + '\n'

def tournament(total_rounds, player_list):

    # Keeping track of games, win, loss, tie, and score for each player
    result_list = []
    for i in range(len(player_list)):
        result_list.append([0] * (len(player_list)+1)) # last item keeps total score for the player
    
    game_num = 1
    # Play round-robin
    for round in range(total_rounds):
        for i in range(len(player_list)):
            for j in range(len(player_list)):
                if player_list[i] != player_list[j]:
                    for game in ["BASIC_STARTING_BOARD", "BASIC_STARTING_BOARD_1", "BASIC_STARTING_BOARD_2"]:
                        winner = run_test_game(player_list[i], player_list[j], game)
                        if winner == 1:
                            result_list[i][j] += 1
                            result_list[j][i] += 0
                        elif winner == 2:
                            result_list[i][j] += 0
                            result_list[j][i] += 1
                        else:
                            result_list[i][j] += 0.5
                            result_list[j][i] += 0.5
                        # Update scores    
                        for row in result_list:
                            row[len(player_list)] = sum(row[:-1])
                        
                        print "Game " + str(game_num) + ":", player_list[i], "vs", player_list[j], "winner:", winner
                        print print_results(result_list)
                        game_num += 1

    print "Final Result after", total_round*len(player_list)*(len(player_list)-1) * 3, "games:"
    print print_results(result_list)
                        
    
tournament(1, ["basic_player", "quick_to_win_player2", "your_player", "your_player2"])
# tournament(1, ["your_player", "your_player2"])


COMPETE = (False)

## The standard survey questions.
HOW_MANY_HOURS_THIS_PSET_TOOK = "8"
WHAT_I_FOUND_INTERESTING = "come up with a good evaluate method"
WHAT_I_FOUND_BORING = "nothing"
NAME = "Mark Young"
EMAIL = "cobase2010@live.com"

"""
Game log:

12/15/2021
Basic vs Better_Evaluate: Games 8 Win: 5 Loss: 0 Tie: 3 Score: 3.5
Basic vs Better_Evaluate2: Games 8 Win: 3 loss: 3 Tie: 2 Score: 1.0
Better_Evaluate2 vs Better_Evaluate: Games 8 Win: 6 Loss: 2 Tie: 0 Score: 4.0
Better_Evaluate2 vs Basic using all three boards:
    Games 12 Win: 5 Loss: 4 Ties: 3 Final score 2.5


Final Result after 18 games between basic, evaluate, evaluate2 players:

SEVEN_BONUS = 100
CHAIN_BASE = 10
ODD_EVEN = 5
CENTER_BONUS = 3

        0       1       2       score
0       0       1.5     2.0     3.5
1       4.5     0       3.5     8.0
2       4.0     2.5     0       6.5

Round 2:
Final Result after 18 games:

        0       1       2       score
0       0       0.5     2.0     2.5
1       5.5     0       3.5     9.0
2       4.0     2.5     0       6.5

Tweak to evaluate2:
SEVEN_BONUS = 50
CHAIN_BASE = 10
ODD_EVEN = 10
CENTER_BONUS = 10

Final Result after 6 games: evaluate, evaluate2

        0       1       score
0       0       2.5     2.5
1       3.5     0       3.5


Final Result after 18 games: basic, evaluate, evaluate2 (on pc)

        0       1       2       score
0       0       2.5     2.0     4.5
1       3.5     0       4       7.5
2       4.0     2       0       6.0


Final Result after 0 games: basic, basic2, evaluate, evaluate2 (on pc)

        0       1       2       3       score
0       0       4       0       3.0     7.0
1       2       0       4       3       9
2       6       2       0       2.0     10.0
3       3.0     3       4.0     0       10.0

"""