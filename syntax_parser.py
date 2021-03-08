import copy
import torch
import torch.nn as nn

from window_models import FixedWindowModel, LSTMParserModel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


PAD = '<pad>'
UNK = '<unk>'

class Parser(object):

    def predict(self, words, tags):
        raise NotImplementedError

class ArcStandardParser(Parser):
    MOVES = tuple(range(3))
    SH, LA, RA = MOVES  # Parser moves are specified as integers.
    error_class = 3 # error class representing a error state in beam search training

    @staticmethod
    def initial_config(num_words):
        return (0, [], [0] * num_words)

    @staticmethod
    def valid_moves(config):
        moves = []
        if len(config[1]) >= 2:
            moves.append(ArcStandardParser.RA)
            if (config[1][-2] != 0): moves.append(ArcStandardParser.LA)

        if config[0] < len(config[2]): moves.append(ArcStandardParser.SH)
        return moves

    @staticmethod
    def next_config(config, move):
        new_config = list(copy.deepcopy(config))
        if move == ArcStandardParser.SH:
            new_config[1].append(new_config[0])
            new_config[0] += 1

        elif move == ArcStandardParser.LA:
            dependent = new_config[1].pop(-2)  # remove second last elem from stack
            new_config[2][dependent] = new_config[1][-1]  # set top of stack as parent of dependent

        else:
            dependent = new_config[1].pop(-1)  # remove last elem from stack
            # set second top-most as parent of dependent (top already poped, so using -1)
            new_config[2][dependent] = new_config[1][-1]

        return tuple(new_config)

    @staticmethod
    def is_final_config(config):
        return not ArcStandardParser.valid_moves(config)  # no valid moves means final config

class FixedWindowParser(ArcStandardParser):

    def __init__(self,vocab_words,vocab_tags, word_dim=100, tag_dim=25, lstm_dim = 180, hidden_dim=180, dropout = 0.3):
        output_dim = 4  # nr of possible moves + error class
        self.vocab_words = vocab_words
        self.vocab_tags = vocab_tags
        nr_feats = 5
        self.model = LSTMParserModel(word_dim, len(vocab_words), tag_dim, len(vocab_tags), nr_feats, lstm_dim, 
                                     hidden_dim, output_dim, dropout).to(device)

    def featurize(self, words, tags, config):
        feats = []
        buffer, stack, heads = config

        ###### SURROUNDING WORDS #######
        # 1st word in stack
        if len(stack) > 0: feats.append(stack[-1])
        else: feats.append(-1)

        # 2nd word in stack
        if len(stack) > 1: feats.append(stack[-2])
        else: feats.append(-1)

        # 3rd word in stack
        if len(stack) > 2: feats.append(stack[-3])
        else: feats.append(-1)

        # 1st word in buffer
        if buffer < len(heads): feats.append(buffer)
        else: feats.append(-1)

        # 2nd word in buffer
        if buffer+1 < len(heads): feats.append(buffer)
        else: feats.append(-1)

        return torch.tensor(feats)

    def beam_argmax(self, config, pred, split_width):
        moves = self.valid_moves(config)
        pred = torch.nn.functional.log_softmax(pred)
        temp = []
        for i, p in enumerate(pred[0]):
            if i in moves: # will remove invalid moves and error class
                temp.append((p.item(), i))

        temp.sort(key = lambda x: x[0], reverse=True)
        # return the split_width top scoring moves (or less if not enough valid moves)
        return temp[0:min(len(temp), split_width)]

    def valid_argmax(self, config, pred):
        best_score = None
        best_move = -1
        moves = self.valid_moves(config)
        for i, p in enumerate(pred[0]):
            if i in moves and (best_score is None or p.item() > best_score):
                best_score = p.item()
                best_move = i
        return best_move

    def beam_search(self, config, word_ids, tag_ids, split_width, beam_width):
        # branches is a list of (config, score) tuples
        branches = [(config, 0)]  # init to 0 if beam greedy search. Init to 1 if best-first search

        while not self.is_final_config(branches[0][0]):
            new_branches = []
            for branch_config, branch_score in branches:
                feats = self.featurize(word_ids, tag_ids, branch_config)
                pred_moves = self.beam_argmax(branch_config,
                                              self.model.forward(torch.tensor(word_ids).unsqueeze(0).to(device),
                                                                 torch.tensor(tag_ids).unsqueeze(0).to(device),
                                                                 feats.unsqueeze(dim=0)),
                                              split_width=split_width)

                # TODO kan effektiviseras genom att först kolla score sedan räkna ut config
                for pred in pred_moves:
                    score = pred[0]
                    move = pred[1]
                    new_config = self.next_config(branch_config, move)
                    new_branches.append((new_config, branch_score + score))

            new_branches.sort(key=lambda x: x[1], reverse=True)  # sort on score
            branches = new_branches[0:min(len(new_branches), beam_width)]

        # return best branch (already sorted)
        best_heads = branches[0][0][2]
        return best_heads

    def predict(self, words, tags, split_width=2, beam_width=2, beam_search=True):
        self.model.eval()
        config = self.initial_config(len(words))
        word_ids = [self.vocab_words[word] if word in self.vocab_words else self.vocab_words[UNK] for word in words]
        tag_ids = [self.vocab_tags[tag] if tag in self.vocab_tags else self.vocab_tags[UNK] for tag in tags]

        return self.beam_search(config, word_ids, tag_ids, split_width, beam_width)
