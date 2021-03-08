import torch.nn as nn
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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


class LstmTaggerModel(nn.Module):

    def __init__(self, embedding_specs, hidden_dim, output_dim, pretrained=False, frozen=False):
        super().__init__()

        self.embeddings = nn.ModuleList()
        self.embeddings2 = []
        self.e_tot = 0
        self.feature_count = 0
        for m, n, e in embedding_specs:
            self.e_tot += e * m
            self.e = e
            self.feature_count += m
            embedding = nn.Embedding(n, e)
            nn.init.normal_(embedding.weight, 0, 0.01)
            self.embeddings.append(embedding)
            self.embeddings2.append(m)

        self.lstm1 = nn.LSTM(self.e, hidden_dim, bidirectional=True, batch_first=True, num_layers=2)

        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, bidirectional=False, batch_first=True)
        self.lin1 = nn.Linear(2 * hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, features):
        res = torch.zeros((features.shape[0], features.shape[1], self.e)).to(device)
        index = 0
        res_index = 0
        for e, count in zip(self.embeddings, self.embeddings2):
            a = e(features[:, index:index + count])

            # a = a.reshape((a.shape[0], a.shape[1] * a.shape[2]))

            res[:, res_index:res_index + a.shape[1], :] = a

            res_index += a.shape[1]
            index += count

        out, _ = self.lstm1(res)
        # out, _ = self.lstm2(out)

        res = self.lin1(out[:, -1, :])

        return res


class LSTMParserModel(nn.Module):

    def __init__(self, word_emd_dim, vocab_word_size, tag_emb_dim, vocab_tag_size, nr_feats, lstm_dim, hidden_dim, 
                 output_dim, dropout, pretrained=False, frozen=False):
        super().__init__()
        self.embeddings = nn.ModuleList()
        self.lstm_dim = lstm_dim

        self.word_embs = nn.Embedding(vocab_word_size, word_emd_dim, padding_idx=0)
        self.tag_embs = nn.Embedding(vocab_tag_size, tag_emb_dim, padding_idx=0)
        nn.init.normal_(self.word_embs.weight, 0, 0.01)
        nn.init.normal_(self.tag_embs.weight, 0, 0.01)

        self.lstm1 = nn.LSTM(word_emd_dim + tag_emb_dim, lstm_dim, bidirectional=True, batch_first=True, 
                             num_layers=2, dropout=dropout)
        self.lin1 = nn.Linear(2 * lstm_dim * nr_feats, hidden_dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, word_ids, tag_ids, feature_ids):
        # TODO Lstm ignore padding: https://galhever.medium.com/sentiment-analysis-with-pytorch-part-4-lstm-bilstm-model-84447f6c4525

        word_embs = self.word_embs(word_ids)
        tag_embs = self.tag_embs(tag_ids)

        # Pair word embs with corresponding tag embs by using view and cat:
        # [word_emb1, word_emb2, word_emb3] &  [tag_emb1, tag_emb2, tag_emb3] ->   
        # [[word_emb1, tag_emb1], [word_emb2, tag_emb2], [word_emb3, tag_emb3]]
        batch_size, sentence_len, word_emb_dim = word_embs.shape
        tag_emb_dim = tag_embs.shape[2]
        tempw = word_embs.view(batch_size*sentence_len, word_emb_dim)
        tempt = tag_embs.view(batch_size * sentence_len, tag_emb_dim)
        word_tag_embs = torch.cat((tempw, tempt), dim=1)
        word_tag_embs = word_tag_embs.view(batch_size, sentence_len, word_emb_dim + tag_emb_dim).to(device)

        # Give entire sentence to bi-lstm 
        out, _ = self.lstm1(word_tag_embs)

        # Filter out only the features specified in feature_ids from the lstm output
        inx_batch = torch.repeat_interleave(torch.tensor(range(feature_ids.shape[0])),feature_ids.shape[1]).to(device)
        inx_emb = feature_ids.reshape(feature_ids.shape[0]*feature_ids.shape[1]).to(device)
        lstm_embs = out[inx_batch, inx_emb].reshape(feature_ids.shape[0], feature_ids.shape[1], out.shape[2]).to(device)

        # Any features that don't exist (indicated by -1) are padded to zeros
        indices = (feature_ids == -1)
        lstm_embs[indices] = torch.FloatTensor([0] * 2 * self.lstm_dim).to(device)
        lstm_embs = lstm_embs.view(lstm_embs.shape[0], lstm_embs.shape[1] * lstm_embs.shape[2])

        # Pass through FFN
        res = self.lin1(lstm_embs)
        res = self.relu(res)
        res = self.lin2(res)

        return res
