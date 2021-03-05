import torch.nn as nn
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class FixedWindowModelBra(nn.Module):

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
    
    

from gensim.models import KeyedVectors

    
class FixedWindowModelLstm(nn.Module):

    def __init__(self, embedding_specs, hidden_dim, output_dim, pretrained=False, frozen=False):
        super().__init__()
        
        #Load pre-trained
        filename='GoogleNews-vectors-negative300.bin'
        word2vec_model = KeyedVectors.load_word2vec_format(filename, binary=True)
        word2vec_weights = torch.FloatTensor(word2vec_model.wv.vectors)
        word2vec_embedding = nn.Embedding.from_pretrained(word2vec_weights)
        
        self.embeddings = nn.ModuleList()
        self.embeddings2 = []
        self.e_tot = 0
        self.feature_count = 0
        self.e = 300
        for m, n, e, pre_trained in embedding_specs:
            self.feature_count += m
            if pre_trained:
                #len(word2vec_model.wv.vectors)
                self.embeddings.append(word2vec_embedding)
                self.embeddings2.append(m)
            else:
                self.e_tot += e * m
                embedding = nn.Embedding(n, e)
                nn.init.normal_(embedding.weight, 0, 0.01)
                self.embeddings.append(embedding)
                self.embeddings2.append(m)
        
        num_layers=1
        nr_directions = 2
        self.lstm1 = nn.LSTM(self.e, hidden_dim, bidirectional=(nr_directions==2), batch_first=True, num_layers=num_layers)
        self.dropout = nn.Dropout(p=0.2)
        
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, bidirectional=False, batch_first=True)
        self.lin1 = nn.Linear(self.feature_count*nr_directions*hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        

    def forward(self, features):
        res = torch.zeros((features.shape[0], features.shape[1], self.e)).to(device)
        index = 0
        res_index = 0
        for e, count in zip(self.embeddings, self.embeddings2):
            a = e(features[:, index:index + count])

            #a = a.reshape((a.shape[0], a.shape[1] * a.shape[2]))

            res[:, res_index:res_index + a.shape[1], :] = a

            res_index += a.shape[1]
            index += count
            
        out, _ = self.lstm1(res)
        out = self.dropout(out)
        out = out.reshape((out.shape[0],out.shape[1]*out.shape[2]))
        #out, _ = self.lstm2(out)
        
        #res = self.lin1(out[:,-1,:])
        res = self.lin1(out)

        return res
    
    
class FixedWindowModelLstm2(nn.Module):

    def __init__(self, embedding_specs, hidden_dim, output_dim, pretrained=False, frozen=False):
        super().__init__()
        
        #self.embedding = nn.Embedding() 
        #self.embeddings2 = []
        #self.e_tot = 0
        #self.feature_count = 0
        
        self.word_emb = nn.Embedding(embedding_specs[0][1], hidden_dim)    
        #self.tag_emb = nn.Embedding()
        
        num_layers=1
        nr_directions = 2
        
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, bidirectional=(nr_directions==2), batch_first=True, num_layers=num_layers)
        self.dropout = nn.Dropout(p=0.2)
        self.lin1 = nn.Linear(nr_directions*hidden_dim, output_dim)

        
        #self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, bidirectional=False, batch_first=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        

    def forward(self, features):
        res = self.word_emb(features)
       
        out, _ = self.lstm1(res)
        out = self.dropout(out)

        #out = out.reshape((out.shape[0],out.shape[1]*out.shape[2]))
        #out, _ = self.lstm2(out)
        
        #print(out.shape)
        
        #res = self.lin1(out[:,-1,:])
        res = self.lin1(out)
        return res