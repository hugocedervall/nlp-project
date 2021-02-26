def uas(parser, gold_data):
    acc = 0
    nr_words = 0
    for sentence in gold_data:
        words = []
        labels = []
        heads = []
        for word, label, head in sentence:
            words.append(word)
            labels.append(label)
            heads.append(head)
            
        nr_words += len(words)-1 # skip psuedo root
        preds = parser.predict(words, labels)
        #print("predicted:", preds)
        #print("actual:", heads)

        acc += sum(pred == head for pred, head in zip(preds[1:], heads[1:]))  # skip psuedo root
    return acc/nr_words

def accuracy(tagger, gold_data):
    acc = 0
    nr_words = 0
    for sentence in gold_data: 
        words = [word for word, _ in sentence]
        labels = [label for _, label in sentence]
        preds = tagger.predict(words)
        acc += sum(pred == label for pred, label in zip(preds, labels)) 
        nr_words += len(words)
    
    return acc/nr_words