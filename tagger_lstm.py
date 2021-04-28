import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import KeyedVectors
from torchtext import data
from torchtext import datasets
import torchtext
import spacy
import numpy as np

import time
import random
from data_handling import *

from create_vocab import *

filename = 'glove.840B.300d.txt' ; binary = False ; no_header = True

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
train_data = Dataset('data/en_gum-ud-train-projectivized.conllu')
dev_data = Dataset('data/en_gum-ud-dev-projectivized.conllu')
test_data = Dataset('data/en_gum-ud-test-projectivized.conllu')
 
#word2vec_model = KeyedVectors.load_word2vec_format(filename, binary=binary, no_header=no_header)
word2vec_model = KeyedVectors.load('embedding_weights')
word2vec_weights = torch.FloatTensor(word2vec_model.vectors)


vocab_words, vocab_tags = make_vocabs(train_data)
BATCH_SIZE = 32
N_EPOCHS = 200
INPUT_DIM = word2vec_weights.shape[0]
EMBEDDING_DIM = 300
HIDDEN_DIM = 200
OUTPUT_DIM = len(vocab_tags)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25
PAD_IDX = 2196016
TAG_PAD_IDX = 0
LEARNING_RATE = 1e-4

criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)
criterion = criterion.to(device)



def training_examples_tagger2(vocab_words, vocab_tags, gold_data, batch_size=100, max_len=40):
    assert batch_size > 0 and max_len > 0
        
    # max sequence length
    x = torch.zeros((batch_size, max_len)).long()
    x[:,:] = 2196016 
    y = torch.zeros((batch_size, max_len)).long()
    count = 0
    for sentence in gold_data:
        words = torch.Tensor(list(map(lambda x: word2vec_model.get_index(x[0]) if x[0] in word2vec_model else 2196015 , sentence))).long()
        
        labels = torch.Tensor(list(map(lambda x: vocab_tags[x[1]], sentence))).long()
        # pad to max len
        x[count,:len(words)] = words[:max_len]
        
        y[count,:len(labels)] = labels[:max_len]
        
        count += 1
        
        if(count == batch_size):
            yield x.long() ,y.long()
            x = torch.zeros((batch_size, max_len))
            y = torch.zeros((batch_size, max_len))
            count = 0
    if(count):
        yield x[:count,:].long(), y[:count,:].long()
        
        
def sent2id(sent):
    return [word[0] if word[0] in word2vec_model else 2196015 for word in sent]
        
class LSTMTagger(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(word2vec_weights, padding_idx=2196016)#, freeze=True)
        #self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers = n_layers, 
                            bidirectional = bidirectional,
                            dropout = dropout
                           )
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)


        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        res = self.linear1(self.dropout(lstm_out))
        
        
        return res
    

# accurayc per batch
def tagger_accuracy(preds, y, tag_pad_idx):
    max_preds = preds.argmax(dim = 1, keepdim = True) 
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]).to(device)


def train_tagger():
    

    model = LSTMTagger(INPUT_DIM, 
                            EMBEDDING_DIM, 
                            HIDDEN_DIM, 
                            OUTPUT_DIM, 
                            N_LAYERS, 
                            BIDIRECTIONAL, 
                            DROPOUT, 
                            PAD_IDX)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

    
    
    best_valid_acc = 0

    for epoch in range(N_EPOCHS):
        model.train()

        start_time = time.time()

        train_iterator = training_examples_tagger2(vocab_words, vocab_tags, train_data, BATCH_SIZE)
        #valid_iterator= training_examples_tagger2(vocab_words, vocab_tags, dev_data, BATCH_SIZE)

        train_loss = 0
        train_acc = 0
        model.train()
        count = 0
        for x, y in train_iterator:
            count += 1
            text = x.reshape(x.shape[1], x.shape[0]).to(device)
            tags = y.reshape(x.shape[1], x.shape[0]).to(device)

            optimizer.zero_grad()

            predictions = model.forward(text)

            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)

            loss = criterion(predictions, tags)

            acc = tagger_accuracy(predictions.to(device), tags, TAG_PAD_IDX)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_acc += acc.item()

        train_loss /= count
        train_acc /= count
        valid_loss, valid_acc = evaluate(model, criterion, TAG_PAD_IDX, dev_data)

        if valid_acc > best_valid_acc and valid_acc > 0.91:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), 'best-tagger-model.pt')
            
        print(f'Epoch: {epoch+1}')
        print(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.3f}%')
        print(f'Val. Loss: {valid_loss:.3f},  Val. Acc: {valid_acc*100:.3f}%')
    
    return model


def get_saved_tagger(path="best-tagger-model.pt"):
    #model = LSTMTagger(*args, **kwargs)
    model = LSTMTagger(INPUT_DIM, 
                            EMBEDDING_DIM, 
                            HIDDEN_DIM, 
                            OUTPUT_DIM, 
                            N_LAYERS, 
                            BIDIRECTIONAL, 
                            DROPOUT, 
                            PAD_IDX)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
    

def evaluate(model, criterion, tag_pad_idx, data):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    count = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index = tag_pad_idx)

    criterion = criterion.to(device)
    
    data = training_examples_tagger2(vocab_words, vocab_tags, data, BATCH_SIZE)

    with torch.no_grad():
    
        for x, y in data:
            count += 1
            text = x.reshape(x.shape[1], x.shape[0]).to(device)
            tags = y.reshape(x.shape[1], x.shape[0]).to(device)
            
            predictions = model(text)
            
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            
            loss = criterion(predictions, tags)
            
            acc = tagger_accuracy(predictions, tags, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / count, epoch_acc / count

def predict_tags(model, sentence, vocab_tags):
    id2tag = list(vocab_tags.keys())
    words = [word for word in sentence]
    encoded = torch.LongTensor([word2vec_model.get_index(word) for word in words]).to(device)
    res = torch.argmax(model(encoded.unsqueeze(dim=0)), dim=2)
    return [id2tag[x] for x in res.squeeze()]