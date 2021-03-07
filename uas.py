
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
        acc += sum(pred == head for pred, head in zip(preds[1:], heads[1:]))  # skip psuedo root
    return acc/nr_words

def accuracy(tagger, gold_data):
    correct = 0
    count = 0
    for sentence in gold_data:
        predicted_sentence = tagger.predict(sentence)
        for i in range(len(predicted_sentence)):
            if sentence[i][1] == predicted_sentence[i][1]:
                correct += 1
            count += 1
        
    return correct/count

def accuracy_sentences(tagger, gold_data):
    correct = count = 0
    for sentence in gold_data:
        pred = tagger.predict_sentence(sentence)
        #print(pred)
        for i in range(len(pred)):
            #print(pred[i], sentence[i][1])
            if sentence[i][1] == pred[i]:
                correct += 1
            count += 1
    return correct/count