from collections import OrderedDict
import numpy as np
from functools import lru_cache
import pickle
import random
import time
import zlib
import base64 as b64
# from kaggle_environments import make


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

transposition_table = LRUCache(1048576)
# cache.put(1, (-1, 0, 7))
# cache.put(2, (-2, 5, 5))
# print(cache.get(1))

# print(cache.cache)

def serializeAndCompress(value, verbose=True):
    serializedValue = pickle.dumps(value)
    if verbose:
        print('Lenght of serialized object:', len(serializedValue))
    c_data =  zlib.compress(serializedValue, 9)
    if verbose:
        print('Lenght of compressed and serialized object:', len(c_data))
    #   return b64.b64encode(c_data)
    return c_data

def decompressAndDeserialize(compresseData):
    # d_data_byte = b64.b64decode(compresseData)
    # data_byte = zlib.decompress(d_data_byte)
    # d_data_byte = b64.b64decode(compresseData)
    data_byte = zlib.decompress(compresseData)
    value = pickle.loads(data_byte)
    return value

def ternary(n):
    if n == 0:
        return '0'
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums))

# return a bitmask containg a single 1 corresponding to the top cel of a given column


@lru_cache(maxsize=7)
def top_mask_col(col):
    return (1 << 5) << col * 7

# return a bitmask containg a single 1 corresponding to the bottom cell of a given column


@lru_cache(maxsize=7)
def bottom_mask_col(col):
    return 1 << col * 7

# return a bitmask 1 on all the cells of a given column


@lru_cache(maxsize=7)
def column_mask(col):
    return ((1 << 6) - 1) << col * 7



@lru_cache(maxsize=128)
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


@lru_cache(maxsize=1048576)
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

def move_to_col(move):
    """
    Convert move bitmap to column number (zero indexed)
    """
    if move == None:
        return None
    for i in range(7):
        if move >> 7 * (i+1) == 0:
            return i




class Position(object):
    WIDTH = 7
    HEIGHT = 6
    bottom_mask = 4432676798593
    board_mask = 279258638311359

    def __init__(self, position, mask):
        self.position = position
        self.mask = mask
        # TODO: do this only once, not when creating child positions as we already have moves by then
        self.moves = bitboardBits(self.mask)
        self.best_move = None

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

    # Plays a possible move given by its bitmap representation
    # move: a possible move given by its bitmap representation
    # only one bit of the bitmap should be set to 1
    # the move should be a valid possible move for the current player
    def play_move(self, move):
        self.position ^= self.mask
        self.mask |= move
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

    # Bitmap of the next possible valid moves for the current player
    # Including losing moves.

    def possible(self):
        return (self.mask + self.bottom_mask) & self.board_mask

    # counts number of bit set to one in a 64bits integer
    def popcount(self, m):
        c = 0
        while m > 0:
            m &= m - 1
            c += 1
        return c

    # Score a possible move.
    # The score we are using is the number of winning spots
    # the current player has after playing the move.
    def move_score(self, move):
        return self.popcount(compute_winning_position(self.position | move, self.mask))

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

    # mirrored bitboard
    # 6666666 5555555 4444444 3333333 2222222 1111111 0000000
    # 0000000 0000000 0000000 0000000 0000000 0000000 1111111 column_mask[0]
    # 0000000 0000000 0000000 0000000 0000000 1111111 0000000 column_mask[1]

    @classmethod
    def mirror(self, field):
        mirrored = 0x0
        mirrored |= ((field & column_mask(0)) << 42)
        mirrored |= ((field & column_mask(1)) << 28)
        mirrored |= ((field & column_mask(2)) << 14)
        mirrored |= (field & column_mask(3))
        mirrored |= ((field & column_mask(4)) >> 14)
        mirrored |= ((field & column_mask(5)) >> 28)
        mirrored |= ((field & column_mask(6)) >> 42)
        return mirrored

        # k1 = 0x5555555555555555
        # k2 = 0x3333333333333333
        # k4 = 0x0f0f0f0f0f0f0f0f
        # x = ((x >> 1) & k1) | ((x & k1) << 1)
        # x = ((x >> 2) & k2) | ((x & k2) << 2)
        # x = ((x >> 4) & k4) | ((x & k4) << 4)
        # x = x >> 1
        # return x
        
class Solver(object):

    cache = None

    def __init__(self):
        self.node_count = 0
        self.MIN_SCORE = -18
        self.MAX_SCORE = 18

        self.column_order = []
        for i in range(7):
            self.column_order.append(7 // 2 + (1 - 2 * (i % 2)) * (i + 1) // 2)

        global opening_book
        if opening_book == None:
            print("Loading opening book...")
            opening_book = dict()
            with open("./opening_book.24", 'rb') as f:
                opening_book_data = f.read()
                # opening_book = pickle.load(f)
                opening_book = decompressAndDeserialize(opening_book_data)

    def negamax(self, p: Position, alpha, beta):
        """
        Alpha-beta with Memory to solve a position without depth
        """
        # assert(alpha < beta)
        # assert(not p.can_win_next())
        alpha_orig = alpha
        best_move = p.best_move

        self.node_count += 1    # increment counter of explored nodes

        possible = p.possible_non_losing_moves()
        if possible == 0:   # if no possible non losing moves, opponent wins next move
            return int(-(42 - p.moves) / 2)

        if p.moves >= 40:  # check for draw game
            # print("draw game")
            if p.best_move == None:
                for i in range(7):
                    move = possible & column_mask(self.column_order[i])
                    if move != 0:   # don't add none move
                        p.best_move = move
                        break
            return 0

        # lower bound of score as opponent cannot win next move
        min = int(-(40 - p.moves) / 2)
        if alpha < min:
            alpha = min             # there is no need to keep alpha below our min possible score
            # prune the exploration if the [alpha:beta] window is empty
            if alpha >= beta:
                return alpha

        max = (41 - p.moves) // 2
        if beta > max:
            beta = max              # there is no need to keep beta above our max possible score
            # prune the exploration if [alpha:beta] window is empty
            if alpha >= beta:
                return beta

        key = p.key()
        entry = transposition_table.get(key)
        if entry != None:
            val, best_move2 = entry
            if val > self.MAX_SCORE - self.MIN_SCORE + 1:   # we have a lower bound
                min = val + 2 * self.MIN_SCORE - self.MAX_SCORE - 2
                if alpha < min:
                    # here is no need to keep alpha below our min possible score.
                    alpha = min
                    best_move = best_move2    # Setting the current best_move from previous table entry
                    # prune the exploration if the [alpha;beta] window is empty.
                    if alpha >= beta:
                        if best_move != None:
                            p.best_move = best_move # Set best_move before cutting off
                        return alpha

            else:       # we have an upper bound
                max = val + self.MIN_SCORE - 1
                if beta > max:
                    # there is no need to keep beta above our max possible score.
                    beta = max
                    # prune the exploration if the [alpha;beta] window is empty.
                    if alpha >= beta:
                        return beta

        moves = []
        for i in range(7):
            move = possible & column_mask(self.column_order[i])
            if move != 0 and move != p.best_move:   # don't add none move
            # if move != 0:   # don't add none move
                moves.append(move)

        moves.sort(reverse=True, key=lambda move: p.move_score(move))

        if p.best_move != None:     # Add previous best_move to the front
            moves = [p.best_move] + moves
        # This is super important!!! Initializing best_move to the first in moves.
        # It handles the cases where aggressive min/max prunning might have best_move uninitialized otherwise.
        if best_move == None:   
            best_move = moves[0]
        best_score = alpha
        for move in moves:
            p2 = Position(p.position, p.mask)
            # It's opponent turn in P2 position after current player plays x column.
            p2.play_move(move)

            # explore opponent's score within [-beta;-alpha] windows:
            # no need to have good precision for score better than beta (opponent's score worse than -beta)
            # no need to check for score worse than alpha (opponent's score worse better than -alpha)
            score = -self.negamax(p2, -beta, -alpha)
            if score > best_score:
                best_score = score
                best_move = move

            if score >= beta:
                # prune the exploration if we find a possible move better than what we were looking for.
                break

            if score > alpha:
                # reduce the [alpha;beta] window for next exploration, as we only
                # need to search for a position that is better than the best so far.
                alpha = score

        if best_score >= beta:
            # save the lower bound of the position, as well as best_move
            transposition_table.put(
                    key, ((score + self.MAX_SCORE - 2 * self.MIN_SCORE + 2), best_move))
            if best_move != None:
                p.best_move = best_move
            
        elif best_score < alpha_orig:
        # save the upper bound of the position, no need to cache best_move
            transposition_table.put(key, (alpha - self.MIN_SCORE + 1, None))
        if best_move != None:
            p.best_move = best_move
        return best_score

    def solve(self, p: Position, weak=False):

        best_move = None
        # check if win in one move as the Negamax function does not support this case.
        if p.can_win_next():
            for col in range(7):
                if p.is_winning_move2(col) > 0:
                    # print("my_agent_new winning move", col)
                    return int((43 - p.moves + 1) / 2)

        min = int(-(42 - p.moves) / 2)
        max = int((43 - p.moves) / 2)
        if weak:
            min = -1
            max = 1

        while min < max:
            med = min + int((max - min) / 2)
            if med <= 0 and int(min / 2) < med:
                med = int(min / 2)
            elif med >= 0 and int(max / 2) > med:
                med = int(max / 2)
            r= self.negamax(p, med, med + 1)
            
            if r <= med:
                max = r    
            else:
                min = r
                
        return min

    def analyze(self, p: Position, weak=False):
        scores = dict()
        for col in range(7):
            if p.can_play(col):
                # print("Getting score for col", col)
                if p.is_winning_move2(col):
                    scores[col] = (43 - p.moves) // 2
                else:
                    p2 = Position(p.position, p.mask)
                    p2.play(col)
                    scores[col] = -self.solve(p2, weak)
        return scores
                    
def my_agent(obs, config):
    import numpy as np
    import random
    import time
    # import cProfile, pstats, io
    # from pstats import SortKey
    
    
    # with cProfile.Profile() as pr:
    weak = True

    start = time.time()
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # print(grid)
    position, mask = Position.get_position_mask_bitmap(grid, obs.mark)
    # print(position, mask)
    p = Position(position, mask)
    solver = Solver()

    
    # Very first move, always play the middle
    if sum(obs.board) == 0:
        return 3

    # If there is a winning move, return it now!
    if p.can_win_next():
        # print("can win next", p.position, p.mask)
        # p.pretty_print()
        for col in range(7):
            if p.is_winning_move2(col) > 0:
                print("my_agent_new winning move", col)
                return col
    
    # If we only store mirrored positions, then if we find something with the same key3
    # We need to check if it's mirrored from the current position. 
    # book storage: key -> move
    # during generation, we use key3 to not store transposed positions
    # at lookup time, if key doesn't exist, mirror the current position and check again if found, return mirrored col
    if p.moves < 27:
        p2 = Position(position, mask)
        key = p2.key()
        if key in opening_book:
            best_move = opening_book[key]
            print("Move", p.moves + 1, "Playing best move from the book", best_move)
            return best_move
        else:
            p3 = Position(Position.mirror(position), Position.mirror(mask))
            key2 = p3.key()
            if key2 in opening_book:
                best_move = opening_book[key2]
                print("Move", p.moves + 1, "Playing mirrored best move from the book", 6 - best_move)
                return 6 - best_move
    
    # scores = solver.analyze(p, weak)
    # max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]    
    # best_move = random.choice(max_cols) # Taking the center columns first
    # if p.moves >= 39:
    #     print("here")
    score = solver.solve(p, weak)
    # print("score", score)
    if p.best_move == None:   # we lost next move, just play any move
        for i in range(7):
            if p.can_play(i):
                best_col = i
                break
    else:
        best_col = move_to_col(p.best_move)

    end = time.time()
    print("my_agent_new move #", p.moves+1, "time", (end-start), "move", best_col, "score", score, "pos count", solver.node_count)
    return best_col

import json
from kaggle_environments.utils import structify

def win_loss_draw(score):
    if score>0: 
        return 'win'
    if score<0: 
        return 'loss'
    return 'draw'

def score(agent, max_lines = 1000):
    #Scores a connect-x agent with the dataset
    print("scoring ",agent)
    count = 0
    good_move_count = 0
    perfect_move_count = 0
    observation = structify({'mark': None, 'board': None})
    with open("./connect4/refmoves1k_kaggle") as f:
        for line in f:
            
            data = json.loads(line)
            moves = data["move score"]
            perfect_score = max(moves)
            
            observation.board = data["board"]
            num_moves = sum(observation.board)
            # if abs(perfect_score) <= (43 - num_moves) // 2 - 10:
            #     continue
            # find out how many moves are played to set the correct mark.
            ply = len([x for x in data["board"] if x>0])
            if ply&1:
                observation.mark = 2
            else:
                observation.mark = 1
            
            #call the agent
            agent_move = agent(observation,env.configuration)
            count += 1
            
            perfect_moves = [ i for i in range(7) if moves[i]==perfect_score]

            if(agent_move in perfect_moves):
                perfect_move_count += 1
                print("Perfect!")
            else:
                print("Not perfect")
                print(observation)

            if win_loss_draw(moves[agent_move]) == win_loss_draw(perfect_score):
                good_move_count += 1
                print("Good!")
            else:
                print("Not Good")

            if count == max_lines:
                break

        print("perfect move percentage: ",perfect_move_count/count)
        print("good moves percentage: ",good_move_count/count)
        print("Total analyzed", count)

def test():
    import cProfile, pstats, io
    from pstats import SortKey

    grid = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]]

    position, mask = Position.get_position_mask_bitmap(grid, 1)
    print(position, mask)
    p = Position(position, mask)
    # p.play_seq("4444443")
    p.play_seq("444444326555553233322676235266611177")
    p.pretty_print()
    # position, mask = 35254972727296, 136583988658304
    # p = Position(position, mask)

    # board = [0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0]
    # board = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 1, 0, 0, 2, 2, 0, 0, 2, 0, 1, 2, 1, 0, 0, 1, 1, 2, 1, 1]
    # board = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 2, 1, 1, 0, 0, 2, 1, 2, 2, 1, 1]
    # board = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 1, 0, 0, 2, 2, 0, 0, 2, 0, 1, 2, 1, 0, 0, 1, 1, 2, 1, 1]
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 1, 0, 0, 2, 2, 0, 0, 2, 0, 1, 2, 1, 0, 0, 1, 1, 2, 1, 1]
    grid = np.asarray(board).reshape(6, 7)
    print(grid)
    # print(grid)
    position, mask = Position.get_position_mask_bitmap(grid, 1)
    # print(position, mask)
    p = Position(position, mask)
    solver = Solver()


    print("moves made so far", p.moves)
    # count = p.popcount(p.mask)
    # print("count", count)
    # constant = 4
    # p2 = Position(Position.mirror(p.position), Position.mirror(p.mask))
    # p2.pretty_print()
    # 
    # with cProfile.Profile() as pr:
    start = time.time()
    score = solver.solve(p, False)
    print("solved", score, "move", move_to_col(p.best_move), (time.time()-start), "seconds with", solver.node_count, "nodes explored.") #format(move, "0b"))
    
    score = solver.solve(p, False)
    print("solved", score, "move", move_to_col(p.best_move), (time.time()-start), "seconds with", solver.node_count, "nodes explored.") #format(move, "0b"))
    start = time.time()
    scores = solver.analyze(p, False)
    print(scores, time.time()-start, "seconds with", solver.node_count, "nodes explored.")
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE  #, SortKey.PCALLS
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # # ps.dump_stats("./profile.stats")
    # # pr.print_stats()
    # ps.print_stats(50)
    # print(s.getvalue())

# Score the 2 built in agents
from kaggle_environments import make
env = make("connectx")
# the built in agents are remarkably slow so only evaluating on 100 moves here
# score(env.agents["random"],100)  
# score(env.agents["negamax"],100)
# score(my_agent, 100)
start = time.time()
score(my_agent, 1000)
print("total time", (time.time() - start), "seconds.")

# test()


