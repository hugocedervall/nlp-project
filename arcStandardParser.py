import copy
class ArcStandardParser(Parser):

    MOVES = tuple(range(3))

    SH, LA, RA = MOVES  # Parser moves are specified as integers.

    @staticmethod
    def initial_config(num_words):
        return (0, [], [0] * num_words)

    @staticmethod
    def valid_moves(config):
        moves = []
        if len(config[1]) >= 2: moves += [ArcStandardParser.LA, ArcStandardParser.RA]
        if config[0] < len(config[2]): moves.append(ArcStandardParser.SH)
        return moves

    @staticmethod
    def next_config(config, move):
        new_config = list(copy.deepcopy(config))
        if move == ArcStandardParser.SH: 
            new_config[1].append(new_config[0])
            new_config[0]+=1
            
        elif move == ArcStandardParser.LA: 
            dependent = new_config[1].pop(-2) # remove second last elem from stack
            new_config[2][dependent] = new_config[1][-1] # set top of stack as parent of dependent
            
        else: 
            dependent = new_config[1].pop(-1) # remove last elem from stack
            new_config[2][dependent] = new_config[1][-1] # set second top-most as parent of dependent (top already poped, so using -1)
            
        return tuple(new_config)

    @staticmethod
    def is_final_config(config):
        return not ArcStandardParser.valid_moves(config) # no valid moves means final config