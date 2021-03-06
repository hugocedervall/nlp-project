import torch 

from parser import ArcStandardParser

# ===========================
# Functions for parser (L3)
# ===========================

def training_examples_tagger(vocab_words, vocab_tags, gold_data, tagger, batch_size=100):
    feats = []
    ys = []

    for sentence in gold_data:
        words = list(map(lambda x: vocab_words[x[0]], sentence))
        labels = list(map(lambda x: vocab_tags[x[1]], sentence))
    
        for j in range(len(words)):
            if len(feats) == batch_size:
                yield torch.stack(feats), torch.tensor(ys)
                ys = []
                feats = []
                
            feats.append(tagger.featurize(words, j, labels))
            ys.append(labels[j])
            
    # yield last batch if not full       
    if len(feats) > 0:
        yield torch.stack(feats), torch.tensor(ys)

def training_examples_tagger2(vocab_words, vocab_tags, gold_data, tagger, batch_size=101, max_len=20):
    assert batch_size > 0 and max_len > 0
    # max sequence length
    x = torch.zeros((batch_size, max_len)).long()
    y = torch.zeros((batch_size, max_len)).long()
    count = 0
    for sentence in gold_data:
        words = torch.Tensor(list(map(lambda x: vocab_words[x[0]] if x[0] in vocab_words else 1, sentence))).long()
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
        
# ====================================
# Functions for syntactic parser (L4)
# ====================================

# TODO: byt denna mot räkna antal occurances av varje head i gold_heads en gång.
def check_all_arcs_from_t_have_been_assigned(t, heads, gold_heads):
    for tt in range(len(gold_heads)):
        if gold_heads[tt] == t and heads[tt] != t: return False
    return True
        

def oracle_moves(gold_heads):
    parser = ArcStandardParser()
    config = parser.initial_config(len(gold_heads))

    while not parser.is_final_config(config):
        moves = parser.valid_moves(config)
        
        i, stack, heads = config 
        
        if parser.LA in moves:
            
            if gold_heads[stack[-2]] == stack[-1] and check_all_arcs_from_t_have_been_assigned(stack[-2], heads, gold_heads):
                yield config, parser.LA
                config = parser.next_config(config, parser.LA) 
                continue
                

        if parser.RA in moves:
             if gold_heads[stack[-1]] == stack[-2] and check_all_arcs_from_t_have_been_assigned(stack[-1], heads, gold_heads):
                yield config, parser.RA
                config = parser.next_config(config, parser.RA) 
                continue
        
        yield config, parser.SH
        config = parser.next_config(config, parser.SH)

def training_examples_parser(vocab_words, vocab_tags, gold_data, parser, batch_size=100):

    feats = []
    ys = []

    for sentence in gold_data:
    
        words = list(map(lambda x: vocab_words[x[0]], sentence))
        tags = list(map(lambda x: vocab_tags[x[1]], sentence))
        heads = [head for _, _, head in sentence]

        for config, move in oracle_moves(heads):
            if len(feats) == batch_size:
                yield torch.stack(feats), torch.tensor(ys)
                ys = []
                feats = []
                
            feats.append(parser.featurize(words, tags, config))
            ys.append(move)
            
    # yield last batch if not full       
    if len(feats) > 0:
        yield torch.stack(feats), torch.tensor(ys)

