def make_vocabs(gold_data):
    word_vocab = {}
    label_vocab = {}
    
    word_vocab[UNK] = 1

    word_vocab[PAD] = 0
    label_vocab[PAD] = 0
    for sentence in gold_data:
        for word, label, _ in sentence:
            if word not in word_vocab:
                word_vocab[word] = len(word_vocab)
            if label not in label_vocab:
                label_vocab[label] = len(label_vocab)
            
    return word_vocab, label_vocab