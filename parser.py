import copy
import torch
import torch.nn as nn

from window_models import FixedWindowModel

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
        if len(config[1]) >= 2: moves += [ArcStandardParser.LA, ArcStandardParser.RA]
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

    def __init__(self, vocab_words, vocab_tags, word_dim=50, tag_dim=10, hidden_dim=400):
        embedding_specs = [(3, len(vocab_words), word_dim), (3, len(vocab_tags), tag_dim)]
        output_dim = 4  # nr of possible moves + error class
        self.vocab_words = vocab_words
        self.vocab_tags = vocab_tags
        self.model = FixedWindowModel(embedding_specs, hidden_dim, output_dim)

    def featurize(self, words, tags, config):
        feats = []

        if config[0] < len(config[2]):
            feats.append(words[config[0]])
        else:
            feats.append(self.vocab_words[PAD])

        if len(config[1]) > 0:
            feats.append(words[config[1][-1]])
        else:
            feats.append(self.vocab_words[PAD])

        if len(config[1]) > 1:
            feats.append(words[config[1][-2]])
        else:
            feats.append(self.vocab_words[PAD])

        if config[0] < len(config[2]):
            feats.append(tags[config[0]])
        else:
            feats.append(self.vocab_tags[PAD])

        if len(config[1]) > 0:
            feats.append(tags[config[1][-1]])
        else:
            feats.append(self.vocab_tags[PAD])

        if len(config[1]) > 1:
            feats.append(tags[config[1][-2]])
        else:
            feats.append(self.vocab_tags[PAD])

        return torch.tensor(feats)

    def valid_argmax(self, config, pred):
        best_score = None
        best_move = -1
        moves = self.valid_moves(config)
        for i, p in enumerate(pred[0]):
            if i in moves and (best_score is None or p.item() > best_score):
                best_score = p.item()
                best_move = i
        return best_move

    def predict(self, words, tags):

        config = self.initial_config(len(words))
        word_ids = [self.vocab_words[word] if word in self.vocab_words else self.vocab_words[UNK] for word in words]
        tag_ids = [self.vocab_tags[tag] if tag in self.vocab_tags else self.vocab_tags[UNK] for tag in tags]

        while not self.is_final_config(config):
            feats = self.featurize(word_ids, tag_ids, config)
            pred_move = self.valid_argmax(config, self.model.forward(feats.unsqueeze(dim=0)))
            # print("Doing move: ", pred_move)
            config = self.next_config(config, pred_move)

        return config[2]