import dynet as dy
from collections import Counter
import random
import sys
import numpy as np

POS_PATH = 'data/pos/'
NER_PATH = 'data/ner/'
MODEL_PATH = 'data/models/'
UNKNOWN_WORD = 'UNKNOWN'

WORD_EMBED_SIZE = 128#50
CHAR_EMBED_SIZE = 30
LSTM_LAYERS_SIZE = 2
EPOCHS = 5

vocab = []
tags = []

# also creates vocab for words and labels
def read_data_into_batches(fname):
    batches = []
    current_batch = []
    for line in file(fname):
        if line == '\n':
            batches.append(current_batch)
            current_batch = []
        else:
            text, label = line.strip().lower().split()
            vocab.append(text)
            tags.append(label)
            current_batch.append((text, label))
    return batches

def get_subword_vocab(words):
    vocab = []
    for word in words:
        vocab.append(word)
        if len(word) > 3:
            vocab.append(word[:3])
            vocab.append(word[-3:])
    complete_vocab = set(vocab)
    return complete_vocab


def validate_args():
    if input_representation != 'a' and input_representation != 'b' and  input_representation != 'c' and  input_representation != 'd':
        raise ValueError
    if tagging_type != 'pos' and tagging_type != 'ner':
        raise ValueError


def word_rep(w):
    w_index = word2index[w]
    return word_embedding[w_index]


def build_tagging_graph(words):
     dy.renew_cg()
     # initialize the RNNs
     fl1_init = fwdRNN_layer1.initial_state()
     bl1_init = bwdRNN_layer1.initial_state()
     fl2_init = fwdRNN_layer2.initial_state()
     bl2_init = bwdRNN_layer2.initial_state()

     wembs = [word_rep(w) for w in words]
     fws = fl1_init.transduce(wembs)
     bws = bl1_init.transduce(reversed(wembs))

     # biLSTM states
     bi = [dy.concatenate([f,b]) for f,b in zip(fws, reversed(bws))]

     fws2 = fl2_init.transduce(bi)
     bws2 = bl2_init.transduce(reversed(bi))

     b_tag = [dy.concatenate([f,b]) for f,b in zip(fws2, reversed(bws2))]


     # check how should it be (the linear layer)
     # MLPs
     H = dy.parameter(pH)
     O = dy.parameter(pO)
     outs = [O*(dy.tanh(H * x)) for x in b_tag]
     return outs


def sent_loss(words, tags):
     vecs = build_tagging_graph(words)
     losses = []
     for v,t in zip(vecs, tags):
         tid = tag2index[t]
         loss = dy.pickneglogsoftmax(v, tid)
         losses.append(loss)
     return dy.esum(losses)

def tag_sent(words):
     vecs = build_tagging_graph(words)
     vecs = [dy.softmax(v) for v in vecs]
     probs = [v.npvalue() for v in vecs]
     tags = []
     for prb in probs:
        tag = np.argmax(prb)
        tags.append(index2tag[tag])
     return zip(words, tags)

input_representation = sys.argv[1]
train_file = sys.argv[2]
model_file = sys.argv[3]
tagging_type = sys.argv[4]

validate_args()

train_batches = read_data_into_batches((POS_PATH if tagging_type == 'pos' else NER_PATH) + 'train')
dev_batches = read_data_into_batches((POS_PATH if tagging_type == 'pos' else NER_PATH) + 'dev')

#temp
# train_batches = train_batches[1:2]

vocab.append(UNKNOWN_WORD)
vocab = set(vocab)
tags = set(tags)

tag2index = {c:i for i,c in enumerate(tags)}
index2tag = {v: k for k, v in tag2index.iteritems()}
word2index = {c:i for i,c in enumerate(vocab)}
vocab_size = len(vocab)

uniqe_chars = []
for word in vocab:
    uniqe_chars.extend(list(word))
uniqe_chars = set(uniqe_chars)
num_of_chars = len(uniqe_chars)

model = dy.Model()
trainer = dy.AdamTrainer(model)

word_embedding = model.add_lookup_parameters((vocab_size, WORD_EMBED_SIZE))
char_embedding = model.add_lookup_parameters((num_of_chars, CHAR_EMBED_SIZE))

pH = model.add_parameters((32, 100))
pO = model.add_parameters((len(tags), 32))

# here we'll need to change the dim by the selected way of representation
fwdRNN_layer1 = dy.LSTMBuilder(layers=1, input_dim=128, hidden_dim=50, model=model)
bwdRNN_layer1 = dy.LSTMBuilder(layers=1, input_dim=128, hidden_dim=50, model=model)

fwdRNN_layer2 = dy.LSTMBuilder(layers=1, input_dim=100, hidden_dim=50, model=model)
bwdRNN_layer2 = dy.LSTMBuilder(layers=1, input_dim=100, hidden_dim=50, model=model)


num_tagged = cum_loss = 0
for iter in xrange(EPOCHS):
    random.shuffle(train_batches)
    for i,s in enumerate(train_batches, 1):

        if i % 5000 == 0:
            good = bad = 0.0
            for sent in dev_batches:
                words = [w for w, t in sent]
                golds = [t for w, t in sent]
                tags = [t for w, t in tag_sent(words)]
                for go, gu in zip(golds, tags):
                    if go == gu:
                        good += 1
                    else:
                        bad += 1
            print good / (good + bad)

        words = [w for w, t in s]
        golds = [t for w, t in s]
        loss_exp = sent_loss(words, golds)
        cum_loss += loss_exp.scalar_value()
        num_tagged += len(golds)
        loss_exp.backward()
        trainer.update()



c = 2