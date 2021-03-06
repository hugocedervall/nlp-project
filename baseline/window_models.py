import torch.nn as nn
import torch

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