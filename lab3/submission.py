
def my_agent(obs, config):
    import numpy as np
    from functools import lru_cache

    class Position(object):
        WIDTH = 7
        HEIGHT = 6
        def __init__(self, position, mask):
            self.position = position
            self.mask = mask

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
        
        @lru_cache
        def has_won(self, position):
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

        def can_play(self, col):
            return (self.mask & self.top_mask(col)) == 0

        def play(self, col):
            self.position ^= self.mask
            self.mask |= self.mask + self.bottom_mask(col)

        def is_winning_move(self, col):
            pos = self.position
            pos |= (self.mask + self.bottom_mask(col)) & self.column_mask(col)
            return self.has_won(pos)
        
        def has_drawn(self):
            """
            If the board has all of its valid slots filled, then it is a draw.
            We mask the board to a bitboard with all positions filled
            (0xFDFBF7EFDFBF) and if all the bits are active, it is a draw.
            """
            return (self.position & 0xFDFBF7EFDFBF) == 0xFDFBF7EFDFBF

        
        # return a bitmask containg a single 1 corresponding to the top cel of a given column
        @lru_cache
        def top_mask(self, col):
            return (1 << (self.HEIGHT - 1)) << col * (self.HEIGHT + 1)

        # return a bitmask containg a single 1 corresponding to the bottom cell of a given column
        @lru_cache
        def bottom_mask(self, col):
            return 1 << col * (self.HEIGHT + 1)

        #return a bitmask 1 on all the cells of a given column
        @lru_cache
        def column_mask(self, col):
            return ((1 << self.HEIGHT) - 1) << col * (self.HEIGHT + 1)

        def pretty_print(self):
            opp_position = self.position ^ self.mask
            print("board     position  mask      key       bottom")
            print("          0000000   0000000")
            for i in range(5, -1, -1): # iterate backwards from 5 to 0
                board_row = "".join("x" if (self.position >> i+k*7) & 1 == 1 \
                    else "o" if (opp_position >> i+k*7) & 1 == 1 else "." for k in range(8))
                pos_row = "".join(str((self.position >> i+j*7) & 1) for j in range(7))
                mask_row = "".join(str((self.mask >> i+j*7) & 1) for j in range(7))
                print(board_row + "   " + pos_row + "   " + mask_row)
        
        @lru_cache
        def evaluate3(self, oppBoard, myBoard):
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

        @lru_cache
        def evaluate2(self, myBoard, oppBoard):
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

        @lru_cache
        def evaluate1(self, oppBoard, myBoard):
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

        def bitboardBits(self, i):
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

        @lru_cache
        def evaluate(self, oppBoard, myBoard):
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

            if self.has_won(oppBoard):
                return -winReward
            elif self.has_won(myBoard):
                return winReward
            elif self.has_drawn():
                return 0 # draw score

            get3Win = self.evaluate3(oppBoard, myBoard)
            winning3 = self.bitboardBits(get3Win) * MyCost3Row

            get3Block = self.evaluate3(myBoard, oppBoard)
            blocking3 = self.bitboardBits(get3Block) * -OppCost3Row

            get2Win = self.evaluate2(oppBoard, myBoard)
            winning2 = self.bitboardBits(get2Win) * MyCost2Row

            get2Block = self.evaluate2(myBoard, oppBoard)
            blocking2 = self.bitboardBits(get2Block) * -OppCost2Row

            get1Win = self.evaluate1(oppBoard, myBoard)
            winning1 = self.bitboardBits(get1Win) * MyCost1Row

            get1Block = self.evaluate1(myBoard, oppBoard)
            blocking1 = self.bitboardBits(get1Block) * -OppCost1Row

            return winning3 + blocking3 + winning2 + blocking2\
                + winning1 + blocking1

        

    class Solver(object):
        def __init__(self):
            self.node_count = 0
            self.column_order = []
            for i in range(7):
                self.column_order.append(7 // 2 + (1 - 2 * (i % 2)) * (i + 1) //2)

        # def negamax(self, p, alpha, beta):
        #     self.node_count += 1
        #     if p.has_drawn(): # check for draw game
        #         return 0
            
        #     for x in range(7): # check if current player can win next move
        #         if p.can_play(x) and p.is_winning_move(x):
        #             return 
        #     return alpha

        
        def negamax(self, p:Position, depth, mark, config, alpha=-np.Inf, beta=np.Inf):
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
            # p.pretty_print()
            self.node_count += 1

            # Opponent must've won, as the current position was provided from the previous move
            is_terminal = (p.has_drawn() or p.has_won(p.position))
            
            if depth == 0 or is_terminal:
                return -p.evaluate(p.position, p.position ^ p.mask)

            # Can look up 8 ply book here and return result based on book
            
            # if p.bitboardBits(p.mask) == 8 and p.position + p.mask in cached_positions:

            #     if cached_positions[p.position + p.mask] == 1:
            #         # print("at depth", depth, "matched to a winning position for player 1")
            #         return 9999999
            #     elif cached_positions[p.position + p.mask] == -1:
            #         # print("at depth", depth, "matched to a losing position for player 1")
            #         return -9999999
            #     else:
            #         # print("at depth", depth, "matched to a draw position for player 1")
            #         return 0
            
            valid_moves = [self.column_order[x] for x in range(7) if p.can_play(self.column_order[x])]

            value = -np.Inf
            for col in valid_moves:
                child = Position(p.position, p.mask)
                # print("Playing ", col, "with mark", mark)
                child.play(col)
                # child.pretty_print()

                value = max(value, -self.negamax(child, depth - 1, mark%2+1, config, -beta, -alpha))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return alpha

        # Uses minimax to calculate value of dropping piece in selected column
        def score_move(self, p:Position, col, mark, config, nsteps):
            next_grid = Position(p.position, p.mask)
            next_grid.play(col)        
            score = -self.negamax(next_grid, nsteps - 1, mark%2+1, config)
            return score


        
        # print(len(cached_positions))
        # count = 0
        # for key in cached_positions:
        #     print(key, cached_positions[key])
        #     count += 1
        #     if count > 100:
        #         break
            

        
        # def solve(self, p, weak=False):
        #     self.node_count = 0
        #     if weak:
        #         return self.negamax(p, -1, 1)
        #     else:
        #         return self.negamax(p, -7 * 6 //2, 7*6//2)

    # grid = [
    #  [0,0,0,0,0,0,0],
    #  [0,0,0,0,0,0,0],
    #  [0,0,0,0,0,0,0],
    #  [0,0,0,0,0,0,0],
    #  [0,0,0,0,0,0,0],
    #  [0,0,0,2,1,1,2]
    #  ]
        
    # position, mask = Position.get_position_mask_bitmap(grid, 1)
    # p = Position(position, mask)
    # # p = Position(0b0000000000000100000010000000000000000000000000000, 0b0000001000000100000010000001000000000000000000000)
    # print("top_mask", format(p.top_mask(1), "064b"))
    # print("bottom_mask", format(p.bottom_mask(1), "064b"))
    # print("column_mask", format(p.column_mask(1), "064b"))
    # print("can play", p.can_play(3))
    # p.pretty_print()
    # p.play(3)
    # print("position after play", format(p.position, "049b"))
    # p.pretty_print()
    # p.play(3)
    # p.play(3)
    # p.play(3)
    # p.play(3)
    # print("can now play", p.can_play(3))
    # p.play(2)
    # p.pretty_print()
    # print("has_drawn", p.has_drawn())


    import numpy as np
    import random
    import time

    
    start = time.time()
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # print(grid)
    
    # Very first move, always play the middle
    if sum(obs.board) == 0:
        return 3
    elif sum(obs.board) == 1 and grid[0, 3] == 0:
        return 3 # Play middle whether opponent places there or not

    position, mask = Position.get_position_mask_bitmap(grid, obs.mark)
    # print(position, mask)
    p = Position(position, mask)
    solver = Solver()
    valid_moves = [solver.column_order[x] for x in range(7) if p.can_play(solver.column_order[x])]
    # If there is a winning move, return it now!
    for col in valid_moves:
        if p.is_winning_move(col):
            return col

    # Use the heuristic to assign a score to each possible board in the next step
    if len(valid_moves) >= 6:
        N_STEPS=4
    if len(valid_moves) ==5:
        N_STEPS=6
    if len(valid_moves) ==4:
        N_STEPS= 10
    if len(valid_moves) <=3:
        N_STEPS= 16

    scores = dict(zip(valid_moves, [solver.score_move(p, col, obs.mark, config, N_STEPS) for col in valid_moves]))   
    
    #Get the highest score value    
    
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]    
    # col = random.choice(max_cols) # Taking the center columns first
    col = max_cols[0]
    end = time.time()
    # print("my_agent excecution time", (end-start), "move", col, "score", scores[col], "at depth", N_STEPS, "pos count", solver.node_count)
    
    return col
