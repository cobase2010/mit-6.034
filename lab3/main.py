from collections import OrderedDict
import numpy as np
from functools import lru_cache
import pickle
import random
import time
from kaggle_environments import make
import sys
sys.path.append("/kaggle_simulations/agent")

# Global vairable initialization for cached_positions
opening_book = None

class LRUCache:
 
    # initialising capacity
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
 
    # we return the value of the key
    # that is queried in O(1) and return -1 if we
    # don't find the key in out dict / cache.
    # And also move the key to the end
    # to show that it was recently used.
    def get(self, key: int) -> object:
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]
 
    # first, we add / update the key by conventional methods.
    # And also move the key to the end to show that it was recently used.
    # But here we will also check whether the length of our
    # ordered dictionary has exceeded our capacity,
    # If so we remove the first key (least recently used)
    def put(self, key: int, value: object) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last = False)

transposition_table = LRUCache(1000000)
# cache.put(1, (-1, 0, 7))
# cache.put(2, (-2, 5, 5))
# print(cache.get(1))

# print(cache.cache)

def ternary (n):
    if n == 0:
        return '0'
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums))

# return a bitmask containg a single 1 corresponding to the top cel of a given column
@lru_cache(maxsize=None)
def top_mask_col(col):
    return (1 << 5) << col * 7

# return a bitmask containg a single 1 corresponding to the bottom cell of a given column
@lru_cache(maxsize=None)
def bottom_mask_col(col):
    return 1 << col * 7

# return a bitmask 1 on all the cells of a given column
@lru_cache(maxsize=None)
def column_mask(col):
    return ((1 << 6) - 1) << col * 7

@lru_cache(maxsize=None)
def has_won(position):
    # Horizontal check
    m = position & (position >> 7)
    if m & (m >> 14):
        return True
    # Diagonal \
    m = position & (position >> 6)
    if m & (m >> 12):
        return True
    # Diagonal /
    m = position & (position >> 8)
    if m & (m >> 16):
        return True
    # Vertical
    m = position & (position >> 1)
    if m & (m >> 2):
        return True
    # Nothing found
    return False

@lru_cache(maxsize=None)
def evaluate3(oppBoard, myBoard):
    """
    Returns the number of possible 3 in a rows in bitboard format.
    Running time: O(1)
    http://www.gamedev.net/topic/596955-trying-bit-boards-for-connect-4/
    """
    inverseBoard = ~(myBoard | oppBoard)
    rShift7MyBoard = myBoard >> 7
    lShift7MyBoard = myBoard << 7
    rShift14MyBoard = myBoard >> 14
    lShit14MyBoard = myBoard << 14
    rShift16MyBoard = myBoard >> 16
    lShift16MyBoard = myBoard << 16
    rShift8MyBoard = myBoard >> 8
    lShift8MyBoard = myBoard << 8
    rShift6MyBoard = myBoard >> 6
    lShift6MyBoard = myBoard << 6
    rShift12MyBoard = myBoard >> 12
    lShift12MyBoard = myBoard << 12

    # check _XXX and XXX_ horizontal
    result = inverseBoard & rShift7MyBoard & rShift14MyBoard\
        & (myBoard >> 21)

    result |= inverseBoard & rShift7MyBoard & rShift14MyBoard\
        & lShift7MyBoard

    result |= inverseBoard & rShift7MyBoard & lShift7MyBoard\
        & lShit14MyBoard

    result |= inverseBoard & lShift7MyBoard & lShit14MyBoard\
        & (myBoard << 21)

    # check XXX_ diagonal /
    result |= inverseBoard & rShift8MyBoard & rShift16MyBoard\
        & (myBoard >> 24)

    result |= inverseBoard & rShift8MyBoard & rShift16MyBoard\
        & lShift8MyBoard

    result |= inverseBoard & rShift8MyBoard & lShift8MyBoard\
        & lShift16MyBoard

    result |= inverseBoard & lShift8MyBoard & lShift16MyBoard\
        & (myBoard << 24)

    # check _XXX diagonal \
    result |= inverseBoard & rShift6MyBoard & rShift12MyBoard\
        & (myBoard >> 18)

    result |= inverseBoard & rShift6MyBoard & rShift12MyBoard\
        & lShift6MyBoard

    result |= inverseBoard & rShift6MyBoard & lShift6MyBoard\
        & lShift12MyBoard

    result |= inverseBoard & lShift6MyBoard & lShift12MyBoard\
        & (myBoard << 18)

    # check for _XXX vertical
    result |= inverseBoard & (myBoard << 1) & (myBoard << 2)\
        & (myBoard << 3)

    return result

@lru_cache(maxsize=None)
def evaluate2(myBoard, oppBoard):
    """
    Returns the number of possible 2 in a rows in bitboard format.
    Running time: O(1)
    """
    inverseBoard = ~(myBoard | oppBoard)
    rShift7MyBoard = myBoard >> 7
    rShift14MyBoard = myBoard >> 14
    lShift7MyBoard = myBoard << 7
    lShift14MyBoard = myBoard << 14
    rShift8MyBoard = myBoard >> 8
    lShift8MyBoard = myBoard << 8
    lShift16MyBoard = myBoard << 16
    rShift16MyBoard = myBoard >> 16
    rShift6MyBoard = myBoard >> 6
    lShift6MyBoard = myBoard << 6
    rShift12MyBoard = myBoard >> 12
    lShift12MyBoard = myBoard << 12

    # check for _XX
    result = inverseBoard & rShift7MyBoard & rShift14MyBoard
    result |= inverseBoard & rShift7MyBoard & rShift14MyBoard
    result |= inverseBoard & rShift7MyBoard & lShift7MyBoard

    # check for XX_
    result |= inverseBoard & lShift7MyBoard & lShift14MyBoard

    # check for XX / diagonal
    result |= inverseBoard & lShift8MyBoard & lShift16MyBoard

    result |= inverseBoard & rShift8MyBoard & rShift16MyBoard
    result |= inverseBoard & rShift8MyBoard & rShift16MyBoard
    result |= inverseBoard & rShift8MyBoard & lShift8MyBoard

    # check for XX \ diagonal
    result |= inverseBoard & rShift6MyBoard & rShift12MyBoard
    result |= inverseBoard & rShift6MyBoard & rShift12MyBoard
    result |= inverseBoard & rShift6MyBoard & lShift6MyBoard
    result |= inverseBoard & lShift6MyBoard & lShift12MyBoard

    # check for _XX vertical
    result |= inverseBoard & (myBoard << 1) & (myBoard << 2) \
        & (myBoard << 2)

    return result

@lru_cache(maxsize=None)
def evaluate1(oppBoard, myBoard):
    """
    Returns the number of possible 1 in a rows in bitboard format.
    Running time: O(1)
    Diagonals are skipped since they are worthless.
    """
    inverseBoard = ~(myBoard | oppBoard)
    # check for _X
    result = inverseBoard & (myBoard >> 7)

    # check for X_
    result |= inverseBoard & (myBoard << 7)

    # check for _X vertical
    result |= inverseBoard & (myBoard << 1)

    return result

@lru_cache(maxsize=None)
def bitboardBits(i):
    """"
    Returns the number of bits in a bitboard (7x6).
    Running time: O(1)
    Help from: http://stackoverflow.com/q/9829578/1524592
    """
    i = i & 0xFDFBF7EFDFBF  # magic number to mask to only legal bitboard
    # positions (bits 0-5, 7-12, 14-19, 21-26, 28-33, 35-40, 42-47)
    i = (i & 0x5555555555555555) + ((i & 0xAAAAAAAAAAAAAAAA) >> 1)
    i = (i & 0x3333333333333333) + ((i & 0xCCCCCCCCCCCCCCCC) >> 2)
    i = (i & 0x0F0F0F0F0F0F0F0F) + ((i & 0xF0F0F0F0F0F0F0F0) >> 4)
    i = (i & 0x00FF00FF00FF00FF) + ((i & 0xFF00FF00FF00FF00) >> 8)
    i = (i & 0x0000FFFF0000FFFF) + ((i & 0xFFFF0000FFFF0000) >> 16)
    i = (i & 0x00000000FFFFFFFF) + ((i & 0xFFFFFFFF00000000) >> 32)

    return i

@lru_cache(maxsize=None)
def evaluate_position(n_moves, oppBoard, myBoard):
    """
    Returns cost of each board configuration.
    winning is a winning move
    blocking is a blocking move
    Running time: O(7n)
    """
    winReward = 9999999
    OppCost3Row = 1000
    MyCost3Row = 3000
    OppCost2Row = 500
    MyCost2Row = 500
    OppCost1Row = 100
    MyCost1Row = 100


    if has_won(oppBoard):
        return -winReward - (42 - n_moves) // 2
    elif has_won(myBoard):
        return winReward + (42 - n_moves) // 2
    # elif self.has_drawn():
    #     return 0  # draw score

    get3Win = evaluate3(oppBoard, myBoard)
    winning3 = bitboardBits(get3Win) * MyCost3Row

    get3Block = evaluate3(myBoard, oppBoard)
    blocking3 = bitboardBits(get3Block) * -OppCost3Row

    get2Win = evaluate2(oppBoard, myBoard)
    winning2 = bitboardBits(get2Win) * MyCost2Row

    get2Block = evaluate2(myBoard, oppBoard)
    blocking2 = bitboardBits(get2Block) * -OppCost2Row

    get1Win = evaluate1(oppBoard, myBoard)
    winning1 = bitboardBits(get1Win) * MyCost1Row

    get1Block = evaluate1(myBoard, oppBoard)
    blocking1 = bitboardBits(get1Block) * -OppCost1Row

    return winning3 + blocking3 + winning2 + blocking2\
        + winning1 + blocking1

@lru_cache(maxsize=None)
def compute_winning_position(position, mask):
    HEIGHT = 6
    # vertical
    r = (position << 1) & (position << 2) & (position << 3)

    # horizontal
    p = (position << (6+1)) & (position << 2*(6+1))
    r |= p & (position << 3*(6+1))
    r |= p & (position >> (6+1))
    p = (position >> (6+1)) & (position >> 2*(6+1))
    r |= p & (position << (6+1))
    r |= p & (position >> 3*(6+1))

    # diagonal 1
    p = (position << HEIGHT) & (position << 2*HEIGHT)
    r |= p & (position << 3*HEIGHT)
    r |= p & (position >> HEIGHT)
    p = (position >> HEIGHT) & (position >> 2*HEIGHT)
    r |= p & (position << HEIGHT)
    r |= p & (position >> 3*HEIGHT)

    # diagonal 2
    p = (position << (HEIGHT+2)) & (position << 2*(HEIGHT+2))
    r |= p & (position << 3*(HEIGHT+2))
    r |= p & (position >> (HEIGHT+2))
    p = (position >> (HEIGHT+2)) & (position >> 2*(HEIGHT+2))
    r |= p & (position << (HEIGHT+2))
    r |= p & (position >> 3*(HEIGHT+2))

    return r & (279258638311359 ^ mask)

class Position(object):
    WIDTH = 7
    HEIGHT = 6
    bottom_mask = 4432676798593
    board_mask = 279258638311359

    def __init__(self, position, mask):
        self.position = position
        self.mask = mask
        self.moves = bitboardBits(self.mask)   #TODO: do this only once, not when creating child positions as we already have moves by then

    @classmethod
    def get_position_mask_bitmap(self, grid, mark):
        # print("grid", grid)
        # print("mark", mark)
        position, mask = '', ''
        # Start with right-most column
        for j in range(6, -1, -1):
            # Add 0-bits to sentinel
            mask += '0'
            position += '0'
            # Start with bottom row
            for i in range(0, 6):
                mask += ['0', '1'][int(grid[i][j] != 0)]
                position += ['0', '1'][int(grid[i][j] == mark)]
        return int(position, 2), int(mask, 2)

    def key(self):
        return self.position + self.mask

    def partial_key3(self, key, col):
        pos = 1 << (col * 7)
        while pos & self.mask > 0:
            key *= 3
            if pos & self.position > 0:
                key += 1
            else:
                key += 2
            pos <<= 1

        key *= 3

        return key

    def key3(self):
        key_forward = 0
        for i in range(7):
            key_forward = self.partial_key3(key_forward, i)
        
        key_reverse = 0
        for i in range(6, -1, -1):
            key_reverse = self.partial_key3(key_reverse, i)

        return key_forward // 3 if key_forward < key_reverse else key_reverse // 3


    def can_play(self, col):
        return (self.mask & top_mask_col(col)) == 0

    def play(self, col):
        self.position ^= self.mask
        self.mask |= self.mask + bottom_mask_col(col)
        self.moves += 1

    # Plays a sequence of successive played columns, mainly used to initilize a board.
    def play_seq(self, seq):
        for i in range(len(seq)):
            col = int(seq[i]) - 1
            if col < 0 or col >= 7 or not self.can_play(col) or self.is_winning_move2(col):
                return i
            self.play(col)
        return len(seq)


    def is_winning_move(self, col):
        pos = self.position
        pos |= (self.mask + bottom_mask_col(col)) & column_mask(col)
        return self.has_won(pos)

    def has_drawn(self):
        """
        If the board has all of its valid slots filled, then it is a draw.
        We mask the board to a bitboard with all positions filled
        (0xFDFBF7EFDFBF) and if all the bits are active, it is a draw.
        """
        # return (self.position & 0xFDFBF7EFDFBF) == 0xFDFBF7EFDFBF
        return self.moves == 42

    def pretty_print(self):
        print("position", self.position)
        print("mask", self.mask)
        opp_position = self.position ^ self.mask
        print("board     position  mask      key       bottom")
        print("          0000000   0000000")
        for i in range(5, -1, -1):  # iterate backwards from 5 to 0
            board_row = "".join("x" if (self.position >> i+k*7) & 1 == 1
                                else "o" if (opp_position >> i+k*7) & 1 == 1 else "." for k in range(7))
            pos_row = "".join(str((self.position >> i+j*7) & 1)
                            for j in range(7))
            mask_row = "".join(str((self.mask >> i+j*7) & 1) for j in range(7))
            print(board_row + "   " + pos_row + "   " + mask_row)

    

    def possible(self):
        return (self.mask + self.bottom_mask) & self.board_mask

    # Return a bitmask of the possible winning positions for the opponent
    def opponent_winning_position(self):
        return compute_winning_position(self.position ^ self.mask, self.mask)

    # Return a bitmask of the possible winning positions for the current player
    def winning_position(self):
        return compute_winning_position(self.position, self.mask)

    

    # Indicates whether the current player wins by playing a given column.
    # This function should never be called on a non-playable column.
    def is_winning_move2(self, col):
        return self.winning_position() & self.possible() & column_mask(col)


    def can_win_next(self):
        return self.winning_position() & self.possible()

    # Return a bitmap of all the possible next moves the do not lose in one turn.
    # A losing move is a move leaving the possibility for the opponent to win directly.
    def possible_non_losing_moves(self):
        possible_mask = self.possible()
        opponent_win = self.opponent_winning_position()
        forced_moves = possible_mask & opponent_win
        if forced_moves:
            if forced_moves & (forced_moves - 1):
                return 0
            else:
                possible_mask = forced_moves
        return possible_mask & ~(opponent_win >> 1)


class Solver(object):
    
    cache = None
    def __init__(self):
        self.node_count = 0
        self.column_order = []
        for i in range(7):
            self.column_order.append(7 // 2 + (1 - 2 * (i % 2)) * (i + 1) // 2)

        # global cached_positions
        # if cached_positions == None:
        #     print("initializing cached_positions...")
        #     cached_positions = dict()
        #     with open("./connect-4.data") as f:
        #         count = 0
        #         for line in f:
        #             l = line.split(",")
        #             s = l[-1].strip("\n")
        #             l = [1 if x == 'x' else 2 if x == 'o' else 0 for x in l[:-1]]
        #             grid = np.asarray(l).reshape(7, 6)
        #             grid = np.rot90(grid)
        #             position, mask = Position.get_position_mask_bitmap(grid, 1)
        #             p = Position(position, mask)
        #             # p.pretty_print()
        #             # print(s)
        #             count += 1
        #             cached_positions[position + mask] = 1 if s == 'win' else -1 if s == "loss" else 0  
        # self.cache = cached_positions   

        global opening_book
        if opening_book == None:
            print("Loading openbing book...")
            opening_book = dict()
            with open("/kaggle_simulations/agent/opening_book.12", 'rb') as f:
                opening_book = pickle.load(f)
    
    # def negamax(self, p, alpha, beta):
    #     self.node_count += 1
    #     if p.has_drawn(): # check for draw game
    #         return 0

    #     for x in range(7): # check if current player can win next move
    #         if p.can_play(x) and p.is_winning_move(x):
    #             return
    #     return alpha

    def negamax(self, p: Position, depth, mark, config, alpha=-np.Inf, beta=np.Inf):
        """
        function negamax(node, depth, α, β, color) is
            alphaOrig := α

            (* Transposition Table Lookup; node is the lookup key for ttEntry *)
            ttEntry := transpositionTableLookup(node)
            if ttEntry is valid and ttEntry.depth ≥ depth then
                if ttEntry.flag = EXACT then
                    return ttEntry.value
                else if ttEntry.flag = LOWERBOUND then
                    α := max(α, ttEntry.value)
                else if ttEntry.flag = UPPERBOUND then
                    β := min(β, ttEntry.value)

                if α ≥ β then
                    return ttEntry.value

            if depth = 0 or node is a terminal node then
                return color × the heuristic value of node

            childNodes := generateMoves(node)
            childNodes := orderMoves(childNodes)
            value := −∞
            for each child in childNodes do
                value := max(value, −negamax(child, depth − 1, −β, −α, −color))
                α := max(α, value)
                if α ≥ β then
                    break

        """
        # assert(alpha < beta)
        # assert(not p.can_win_next())
        # p.pretty_print()
        self.node_count += 1
        alpha_orig = alpha

        # Check 8 ply cache first
        # if p.moves == 8 and p.position + p.mask in self.cache:
        #     if self.cache[p.position + p.mask] == 1:
        #         # print("at depth", depth, "matched to a winning position for player 1")
        #         return 9999999 + (42 - p.moves) // 2
        #     elif self.cache[p.position + p.mask] == -1:
        #         # print("at depth", depth, "matched to a losing position for player 1")
        #         return -9999999 - (42 - p.moves) // 2
        #     else:
        #         # print("at depth", depth, "matched to a draw position for player 1")
        #         return 0

        # Transposition table lookup
        key3 = p.key3()
        tt_entry = transposition_table.get(key3)
        if tt_entry != None and tt_entry[2] >= depth:
            if tt_entry[1] == 0: # Exact
                return tt_entry[0]
            elif tt_entry[1] == 1: # lowerbound
                alpha = max(alpha, tt_entry[0])
            elif tt_entry[1] == 2: # upperbound
                beta = min(beta, tt_entry[0])

            if alpha >= beta:
                return tt_entry[0]

    
        # Opponent must've won, as the current position was provided from the previous move
        is_terminal = (p.has_drawn() or has_won(p.position))

        if depth == 0 or is_terminal:
            return -evaluate_position(p.moves, p.position, p.position ^ p.mask)

        if p.can_win_next():
            return 9999999 + (42 - p.moves) // 2

        next = p.possible_non_losing_moves()
        if next == 0: # no possible non losing move, opponent wins next move
            # p.pretty_print()
            # print("opponent wins next")
            return -9999999 - (42 - p.moves) // 2

        if p.moves >= 40: # check for draw game
            return 0

        value = -np.Inf
        # for col in valid_moves:
        for col in range(7):
            if next & column_mask(self.column_order[col]) > 0:
                child = Position(p.position, p.mask)
                # print("Playing ", col, "with mark", mark)
                child.play(self.column_order[col])
                
                value = max(value, -self.negamax(child, depth -
                            1, mark % 2+1, config, -beta, -alpha))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break

        # Transposition table store
        if value <= alpha_orig:
            transposition_table.put(key3, (value, 2, depth))
        elif value >= beta:
            transposition_table.put(key3, (value, 1, depth))
        else:
            transposition_table.put(key3, (value, 0, depth))
                
        return value

    def negamax2(self, p: Position, depth, mark, config, alpha=-np.Inf, beta=np.Inf):
        """
        function negamax(node, depth, α, β, color) is
        if depth = 0 or node is a terminal node then
            return color × the heuristic value of node

        childNodes := generateMoves(node)
        childNodes := orderMoves(childNodes)
        value := −∞
        foreach child in childNodes do
            value := max(value, −negamax(child, depth − 1, −β, −α, −color))
            α := max(α, value)
            if α ≥ β then
                break (* cut-off *)
        return alpha
        """
        # assert(alpha < beta)
        # assert(not p.can_win_next())
        # p.pretty_print()
        self.node_count += 1
        # Opponent must've won, as the current position was provided from the previous move
        is_terminal = (p.has_drawn() or has_won(p.position))

        if depth == 0 or is_terminal:
            return -evaluate_position(p.position, p.position ^ p.mask)

        if p.can_win_next():
            return 9999999

        next = p.possible_non_losing_moves()
        if next == 0: # no possible non losing move, opponent wins next move
            # p.pretty_print()
            # print("opponent wins next")
            return -9999999

        if p.moves >= 40: # check for draw game
            return 0

        # if p.moves == 8 and p.position + p.mask in self.cache:

        #     if self.cache[p.position + p.mask] == 1:
        #         # print("at depth", depth, "matched to a winning position for player 1")
        #         return 9999999
        #     elif self.cache[p.position + p.mask] == -1:
        #         # print("at depth", depth, "matched to a losing position for player 1")
        #         return -9999999
        #     else:
        #         # print("at depth", depth, "matched to a draw position for player 1")
        #         return 0
        # valid_moves = [self.column_order[x]
        #                for x in range(7) if p.can_play(self.column_order[x])]
        # min = -(40 - p.moves) / 2	# lower bound of score as opponent cannot win next move
        # if alpha < min:
        #     alpha = min                   # there is no need to keep beta above our max possible score.
        #     if(alpha >= beta):
        #          return alpha   #  prune the exploration if the [alpha;beta] window is empty.
        

        # max = (41 - p.moves)/2	   # upper bound of our score as we cannot win immediately
        # # if(int val = transTable.get(P.key()))
        # # max = val + Position::MIN_SCORE - 1;

        # if beta > max: 
        #     beta = max                     # there is no need to keep beta above our max possible score.
        # if alpha >= beta:
        #      return beta;  # prune the exploration if the [alpha;beta] window is empty.
      
        value = -np.Inf
        # for col in valid_moves:
        for col in range(7):
            if next & column_mask(self.column_order[col]) > 0:
                child = Position(p.position, p.mask)
                # print("Playing ", col, "with mark", mark)
                child.play(self.column_order[col])
                # child.pretty_print()

                # value = max(value, -self.negamax(child, depth -
                #             1, mark % 2+1, config, -beta, -alpha))
                # alpha = max(alpha, value)
                # if alpha >= beta:
                #     break

                score = -self.negamax(child, depth -
                            1, mark % 2+1, config, -beta, -alpha)
                if score >= beta:
                    return score
                if score >= alpha:
                    alpha = score
                
        return alpha

    # Uses minimax to calculate value of dropping piece in selected column
    def score_move(self, p: Position, col, mark, config, nsteps):
        
        next_grid = Position(p.position, p.mask)
        next_grid.play(col)
        score = -self.negamax(next_grid, nsteps - 1, mark % 2+1, config)
        return score

def my_agent(obs, config):
    
    # import cProfile, pstats, io
    # from pstats import SortKey
    
    
    # with cProfile.Profile() as pr:

    if obs.step > 900:
        raise Exception("Don't Submit To Competition")

    start = time.time()
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # print(grid)
    position, mask = Position.get_position_mask_bitmap(grid, obs.mark)
    # print(position, mask)
    p = Position(position, mask)
    solver = Solver()
    valid_moves = [solver.column_order[x] for x in range(7) if p.can_play(solver.column_order[x])]

    # Very first move, always play the middle
    if sum(obs.board) == 0:
        return 3
    elif sum(obs.board) == 1 and grid[0, 3] == 0:
        return 3 # Play middle whether opponent places there or not
    
    # If there is a winning move, return it now!
    if p.can_win_next():
        # print("can win next", p.position, p.mask)
        # p.pretty_print()
        for col in valid_moves:
            if p.is_winning_move2(col) > 0:
                print("my_agent_new winning move", col)
                return col

    if p.moves < 12:  # we should have entries in the opening book
        best_score, best_move = -100, None
        for col in valid_moves:
            p2 = Position(position, mask)
            p2.play(col)
            key3 = p2.key3()
            if key3 in opening_book:
                # if obs.mark == 2:
                #     score = -opening_book[key3]
                # else: 
                #     score = opening_book[key3]
                score = -opening_book[key3]
                # print("found", key3, score, "for", col)
                if score > best_score:
                    best_score = score
                    best_move = col
            else:
                print("key", key3, "not found for col", col)
        if best_move != None:
            print("Playing best move from the book", best_move)
            return best_move

    # Use the heuristic to assign a score to each possible board in the next step
    if len(valid_moves) >= 6:
        N_STEPS=11
    if len(valid_moves) ==5:
        N_STEPS=13
    if len(valid_moves) ==4:
        N_STEPS= 16
    if len(valid_moves) <=3:
        N_STEPS= 20

    # if p.moves == 29:
    #     p.pretty_print()
    scores = dict(zip(valid_moves, [solver.score_move(p, col, obs.mark, config, N_STEPS) for col in valid_moves]))   
    
    #Get the highest score value    
    
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]    
    # col = random.choice(max_cols) # Taking the center columns first
    col = max_cols[0]
    end = time.time()
    print("my_agent_new move #", p.moves+1, "time", (end-start), "move", col, "score", scores[col], "at depth", N_STEPS, "pos count", solver.node_count)
        # print(evaluate_position.cache_info())
    # s = io.StringIO()
    # # sortby = SortKey.CUMULATIVE, SortKey.PCALLS
    # ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.PCALLS)
    # ps.dump_stats("./profile.stats")
    # # pr.print_stats()
    # ps.print_stats(50)
    # print(s.getvalue())
    
    return col