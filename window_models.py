import torch.nn as nn

class FixedWindowModel(nn.Module):

    def __init__(self, embedding_specs, hidden_dim, output_dim):
        super().__init__()
        self.embeddings = nn.ModuleList()
        self.embeddings2 = []
        self.e_tot = 0
        for m, n, e in embedding_specs:
            self.e_tot += e * m
            embedding = nn.Embedding(n, e)
            nn.init.normal_(embedding.weight, 0, 0.01)
            self.embeddings.append(embedding)
            self.embeddings2.append(m)
        self.linear1 = nn.Linear(self.e_tot, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, features):
        res = torch.zeros((features.shape[0], self.e_tot))
        index = 0
        res_index = 0
        for e, count in zip(self.embeddings, self.embeddings2):
            a = e(features[:, index:index + count])

            a = a.reshape((a.shape[0], a.shape[1] * a.shape[2]))

            res[:, res_index:res_index + a.shape[1]] = a

            res_index += a.shape[1]
            index += count
        res = self.linear2(self.relu(self.linear1(res)))
        return res