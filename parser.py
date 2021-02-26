import copy
import torch
import torch.nn as nn

from window_models import FixedWindowModel


class Parser(object):

    def predict(self, words, tags):
        raise NotImplementedError


class TrivialParser(Parser):

    def predict(self, words, tags):
        return [0] + list(range(len(words) - 1))


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


class FixedWindowParser2(ArcStandardParser):

    def __init__(self, vocab_words, vocab_tags, word_dim=50, tag_dim=10, hidden_dim=180):
        self.vocab_words = vocab_words
        self.vocab_tags = vocab_tags
        embedding_specs = [(3, len(vocab_words), word_dim), (3, len(vocab_tags), tag_dim)]
        self.model = FixedWindowModel(embedding_specs, hidden_dim, 3)
        self.parser = ArcStandardParser()
        super().__init__()

    def featurize(self, words, tags, config):
        try:
            f1 = words[config[0]] if config[0] < len(words) else 0
            f2 = words[config[1][0]] if len(config[1]) > 0 else 0
            f3 = words[config[1][1]] if len(config[1]) > 1 else 0
            f4 = tags[config[0]] if config[0] < len(words) else 0
            f5 = tags[config[1][0]] if len(config[1]) > 0 else 0
            f6 = tags[config[1][1]] if len(config[1]) > 1 else 0
        except Exception as e:
            print(words)
            print(tags)
            print(config)
            raise e
        return torch.tensor([f1, f2, f3, f4, f5, f6])

    def argmax(self, res, valid_moves):
        best_index = valid_moves[0]
        best_val = res[best_index]
        for i in valid_moves:
            if res[i] > best_val:
                best_index = i
                best_val = res[i]
        return best_index

    def predict(self, words, tags):
        words = [self.vocab_words[word] if word in self.vocab_words else 1 for word in words]
        tags = [self.vocab_tags[tag] if tag in self.vocab_tags else 1 for tag in tags]
        config = self.parser.initial_config(len(words))
        while (not self.parser.is_final_config(config)):
            f = self.featurize(words, tags, config)
            res = self.model.softmax(self.model.forward(f.unsqueeze(0)))
            valid_moves = self.parser.valid_moves(config)
            res = self.argmax(list(res[0]), valid_moves)

            config = self.parser.next_config(config, res)
        return config[2]


class FixedWindowModel(nn.Module):

    def __init__(self, embedding_specs, hidden_dim, output_dim, pretrained=False, frozen=False):
        super().__init__()
        # list to keep track of nr of features that map to each embedding layer

        self.concat_emb_len = 0
        self.index_list = []
        self.emb_len_list = []

        for i, _, emb in embedding_specs:
            self.index_list.append(i)
            self.concat_emb_len += i * emb
            self.emb_len_list.append(i * emb)

        if pretrained:  # hard coded for this case

            word_embedding = embedding = nn.Embedding.from_pretrained(glove, freeze=frozen)
            tag_embedding = nn.Embedding(embedding_specs[1][1], embedding_specs[1][2], padding_idx=0)

            # store embeddings in ModuleList
            self.embeddings = nn.ModuleList([word_embedding, tag_embedding])

            # init weights with std 10^-2
            nn.init.normal_(self.embeddings[1].weight, std=1e-2)

        else:
            # store embeddings in ModuleList
            self.embeddings = nn.ModuleList(
                [nn.Embedding(num_words, embedding_dim, padding_idx=0) for (i, num_words, embedding_dim) in
                 embedding_specs])

            # init weights with std 10^-2
            for emb in self.embeddings:
                nn.init.normal_(emb.weight, std=1e-2)

        # calc dimensions of concat embeddings
        concat_dim = 0
        for i, num_words, embedding_dim in embedding_specs:
            concat_dim += i * embedding_dim

        # feed forward
        self.linear1 = nn.Linear(concat_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):
        batch_size = len(features)

        concat_embeddings = torch.zeros((batch_size, 0))
        curr = 0
        for i, emb in enumerate(self.embeddings):
            concat_embeddings = torch.cat((concat_embeddings,
                                           emb(features[:, curr:curr + self.index_list[i]]).view(batch_size,
                                                                                                 self.emb_len_list[i])),
                                          dim=1)
            # temp.append(emb(features[:, curr:curr + self.index_list[i]]).view(batch_size, self.emb_len_list[i]))
            curr += self.index_list[i]

        # concat_embeddings = torch.cat(temp, dim=1).view(batch_size, self.concat_emb_len)

        res = self.linear1(concat_embeddings)
        res = self.relu(res)
        res = self.linear2(res)

        return res