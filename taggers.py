import torch

from window_models import FixedWindowModel, FixedWindowModelLstm, FixedWindowModelLstm2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Tagger(object):

    def predict(self, sentence):
        return [(w[0], "NOUN") for w in sentence]


class FixedWindowTagger(Tagger):

    def __init__(self, vocab_words, vocab_tags, output_dim, word_dim=50, tag_dim=10, hidden_dim=100):
        self.vocab_words = vocab_words
        self.vocab_tags = vocab_tags
        #embedding_specs = [(3, len(vocab_words), word_dim), (1, len(vocab_tags), tag_dim)]
        embedding_specs = [(3, len(vocab_words), word_dim, True), (1, len(vocab_tags), word_dim, False)]
        
        #self.model = FixedWindowModel(embedding_specs, hidden_dim, output_dim)
        self.model = FixedWindowModelLstm2(embedding_specs, hidden_dim, output_dim)

    def featurize(self, words, i, pred_tags):
        if i == 0:
            f1 = 0
            f4 = 0
        else:
            f1 = words[i - 1]
            f4 = pred_tags[i - 1]

        f2 = words[i]
        if i == len(words) - 1:
            f3 = 0
        else:
            f3 = words[i + 1]

        return torch.tensor([f1, f2, f3, f4])


    def predict(self, words):
        words = [self.vocab_words[word[0]] if word[0] in self.vocab_words else 1 for word in words]
        pred_tags = []
        out = []
        for i in range(len(words)):
            f = self.featurize(words, i, pred_tags).to(device)
            res = self.model.forward(f.unsqueeze(0))

            res = torch.argmax(res)
            pred_tags.append(res)
            out.append((words[i], list(self.vocab_tags.keys())[res]))
        return out
    
    def predict_sentence(self, words):
        #print(words)
        # mapping from id to tag string
        id_to_tag = list(self.vocab_tags.keys())
        
        sentence = [self.vocab_words[word[0]] if word[0] in self.vocab_words else 1 for word in words]
        input_sent = torch.tensor(sentence).unsqueeze(0).to(device)
        print(input_sent.shape)
        # size (batch_size, seq_len, classes)
        pred = self.model.forward(input_sent).squeeze(0)
        pred_tags = torch.argmax(pred, dim=1)
        print(pred_tags.shape)
        
        pred_tags_list = pred_tags.tolist()
        return list(map(lambda tag : id_to_tag[tag], pred_tags_list))