import dynet as dy
from collections import Counter
import random
import sys
import numpy as np
from datetime import datetime
import os

POS_PATH = 'data/pos/'
NER_PATH = 'data/ner/'
UNKNOWN_WORD = 'UNKNOWN'

WORD_EMBED_SIZE = 128  # 50
CHAR_EMBED_SIZE = 20
EPOCHS = 5  # 5
EVALUATE_ITERATION = 10000
# replace rare words with the unknown char


def read_data_into_batches(file_name, is_training_file=False):
    vocab = []
    tags = []
    batches = []
    current_batch = []
    for line in file(file_name):
        if line == '\n':
            batches.append(current_batch)
            current_batch = []
        else:
            text, label = line.strip().lower().split()
            current_batch.append((text, label))
            if is_training_file:
                vocab.append(text)
                tags.append(label)
    if is_training_file:
        return batches, vocab, tags
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


def validate_args(input_representation, tagging_type):
    if input_representation != 'a' and input_representation != 'b' and input_representation != 'c' and input_representation != 'd':
        raise Exception("representation type not supported ")
    if tagging_type != 'pos' and tagging_type != 'ner':
        raise Exception("tagging type not supported ")


def get_word_rep(word, holder):
    if holder.input_representation == 'a':
        w_index = holder.word2index[word] if word in holder.word2index else holder.word2index[UNKNOWN_WORD]
        return holder.word_embedding[w_index]
    elif holder.input_representation == 'b':
        c_init = holder.char_rnn.initial_state()
        char_indexes = [holder.char2index[c] for c in list(word)]
        char_embeddings = [holder.char_embedding[c_index] for c_index in char_indexes]
        return c_init.transduce(char_embeddings)[-1]
    elif holder.input_representation == 'c':
        if word not in holder.word2index:
            return holder.word_embedding[holder.word2index[UNKNOWN_WORD]] * 3
        elif len(word) < 4:
            return holder.word_embedding[holder.word2index[word]] * 3
        else:
            return holder.word_embedding[holder.word2index[word]] + holder.word_embedding[holder.word2index[word[:3]]] + holder.word_embedding[holder.word2index[word[-3:]]]
    else:
        d_W = dy.parameter(holder.d_W)
        d_b = dy.parameter(holder.d_b)
        w_index = holder.word2index[word] if word in holder.word2index else holder.word2index[UNKNOWN_WORD]
        a = holder.word_embedding[w_index]
        c_init = holder.char_rnn.initial_state()
        char_indexes = [holder.char2index[c] for c in list(word)]
        char_embeddings = [holder.char_embedding[c_index] for c_index in char_indexes]
        b = c_init.transduce(char_embeddings)[-1]
        ab = dy.concatenate([a, b])
        result = dy.tanh(d_W * ab) + d_b
        return result


def build_graph(words, holder):
    dy.renew_cg()
    # initialize the RNNs
    fl1_init = holder.fwdRNN_layer1.initial_state()
    bl1_init = holder.bwdRNN_layer1.initial_state()
    fl2_init = holder.fwdRNN_layer2.initial_state()
    bl2_init = holder.bwdRNN_layer2.initial_state()

    wembs = [get_word_rep(w, holder) for w in words]
    fws = fl1_init.transduce(wembs)
    bws = bl1_init.transduce(reversed(wembs))

    # biLSTM states
    bi = [dy.concatenate([f,b]) for f,b in zip(fws, reversed(bws))]
    fws2 = fl2_init.transduce(bi)
    bws2 = bl2_init.transduce(reversed(bi))
    b_tag = [dy.concatenate([f,b]) for f,b in zip(fws2, reversed(bws2))]

    # check how should it be (the linear layer)
    # MLPs
    H = dy.parameter(holder.pH)
    O = dy.parameter(holder.pO)
    outs = [O*(dy.tanh(H * x)) for x in b_tag]
    return outs


def calc_loss(words, tags, holder):
    vecs = build_graph(words, holder)
    losses = []
    for v, t in zip(vecs, tags):
        tid = holder.tag2index[t]
        loss = dy.pickneglogsoftmax(v, tid)
        losses.append(loss)
    return dy.esum(losses)


def predict_tags(words, holder):
    vecs = build_graph(words, holder)
    vecs = [dy.softmax(v) for v in vecs]
    probs = [v.npvalue() for v in vecs]
    tags = []
    for prb in probs:
       tag = np.argmax(prb)
       tags.append(holder.index2tag[tag])
    return zip(words, tags)

# change to support ner
def evaluate_set(dev_batches, holder):
    good = 0
    bad = 0
    for sentence in dev_batches:
        words = [word for word, tag in sentence]
        real_tags = [tag for word, tag in sentence]
        tags = [tag for word, tag in predict_tags(words, holder)]
        for go, gu in zip(real_tags, tags):
            if go == gu:
                good += 1
            else:
                bad += 1
    return float(good) / (good + bad)


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def save_results_and_model(evaluation_results, current_date, current_time, tagging_type, input_representation, total_training_time, model):
    path = 'results/{}_{}'.format(current_date, current_time)
    ensure_directory_exists(path)
    with open(path + '/results.txt', 'w') as f:
        f.write('tagging type: {}\n'.format(tagging_type))
        f.write('representation type: {}\n'.format(input_representation))
        f.write('total training time: {}\n'.format(total_training_time))
        f.write('num of epochs: {}\n'.format(EPOCHS))
        f.write('evaluated every: {}\n'.format(EVALUATE_ITERATION))
    with open(path +'/evaluation.txt', 'w') as f:
        f.write(''.join([str(x) + ' ' for x in evaluation_results]))
    # model.save('model', [word_embedding, char_embedding, pH, pO, fwdRNN_layer1, bwdRNN_layer1, fwdRNN_layer2, bwdRNN_layer2])
    model.save(path + '/model')


class ComponentHolder:
    def __init__(self, input_representation, word2index, index2tag, tag2index, char2index, fwdRNN_layer1, bwdRNN_layer1, fwdRNN_layer2,
                 bwdRNN_layer2, pH, pO, word_embedding, char_embedding, char_rnn, d_W, d_b):
        self.input_representation = input_representation
        self.word2index = word2index
        self.index2tag = index2tag
        self.tag2index = tag2index
        self.char2index = char2index
        self.fwdRNN_layer1 = fwdRNN_layer1
        self.bwdRNN_layer1 = bwdRNN_layer1
        self.fwdRNN_layer2 = fwdRNN_layer2
        self.bwdRNN_layer2 = bwdRNN_layer2
        self.pH = pH
        self.pO = pO
        self.word_embedding = word_embedding
        self.char_embedding = char_embedding
        self.char_rnn = char_rnn
        self.d_W = d_W
        self.d_b = d_b


def main():
    vocab = []
    tags = []
    input_representation = sys.argv[1]
    train_file = sys.argv[2]
    model_file = sys.argv[3]
    tagging_type = sys.argv[4]

    validate_args(input_representation, tagging_type)

    train_batches, vocab, tags = read_data_into_batches((POS_PATH if tagging_type == 'pos' else NER_PATH) + 'train', is_training_file=True)
    dev_batches = read_data_into_batches((POS_PATH if tagging_type == 'pos' else NER_PATH) + 'dev')

    # train_batches = train_batches[:100] # remove later


    vocab = set(vocab)
    vocab = list(vocab)
    if input_representation == 'c':
        vocab = list(get_subword_vocab(vocab))
    vocab.append(UNKNOWN_WORD)
    tags = set(tags)



    tag2index = {c: i for i, c in enumerate(tags)}
    index2tag = {v: k for k, v in tag2index.iteritems()}
    word2index = {c: i for i, c in enumerate(vocab)}
    vocab_size = len(vocab)

    unique_chars = []
    for word in vocab:
        unique_chars.extend(list(word))
    unique_chars = set(unique_chars)
    num_of_chars = len(unique_chars)
    char2index = {c: i for i, c in enumerate(unique_chars)}

    model = dy.Model()
    trainer = dy.AdamTrainer(model)

    char_embedding = None
    char_rnn = None
    d_W = None
    d_b = None

    # here execute according to a, b, c or d
    if input_representation == 'a' or input_representation == 'c':
        rnnlayer1_input_dim = 128
        rnnlayer2_input_dim = 100
        hidden_dim = 50
        pH_dim1 = 32
        pH_dim2 = 100
    elif input_representation == 'b':
        char_embedding = model.add_lookup_parameters((num_of_chars, CHAR_EMBED_SIZE))
        char_rnn = dy.LSTMBuilder(layers=1, input_dim=CHAR_EMBED_SIZE, hidden_dim=64, model=model)
        pH_dim1 = 32
        pH_dim2 = 100
        rnnlayer1_input_dim = 64
        hidden_dim = 50
        rnnlayer2_input_dim = 100
        pH_dim1 = 32
        pH_dim2 = 100
    elif input_representation == 'd':
        rnnlayer1_input_dim = 200
        rnnlayer2_input_dim = 100
        hidden_dim = 50
        pH_dim1 = 32
        pH_dim2 = 100
        char_embedding = model.add_lookup_parameters((num_of_chars, CHAR_EMBED_SIZE))
        char_rnn = dy.LSTMBuilder(layers=1, input_dim=CHAR_EMBED_SIZE, hidden_dim=64, model=model)
        d_W = model.add_parameters((200, 192))
        d_b = model.add_parameters(200)

    word_embedding = model.add_lookup_parameters((vocab_size, WORD_EMBED_SIZE))


    pH = model.add_parameters((pH_dim1, pH_dim2))
    pO = model.add_parameters((len(tags), pH_dim1))

    # here we'll need to change the dim by the selected way of representation
    fwdRNN_layer1 = dy.LSTMBuilder(layers=1, input_dim=rnnlayer1_input_dim, hidden_dim=hidden_dim, model=model)
    bwdRNN_layer1 = dy.LSTMBuilder(layers=1, input_dim=rnnlayer1_input_dim, hidden_dim=hidden_dim, model=model)

    fwdRNN_layer2 = dy.LSTMBuilder(layers=1, input_dim=rnnlayer2_input_dim, hidden_dim=hidden_dim, model=model)
    bwdRNN_layer2 = dy.LSTMBuilder(layers=1, input_dim=rnnlayer2_input_dim, hidden_dim=hidden_dim, model=model)

    holder = ComponentHolder(input_representation, word2index, index2tag, tag2index, char2index,  fwdRNN_layer1, bwdRNN_layer1,
                             fwdRNN_layer2, bwdRNN_layer2, pH, pO, word_embedding, char_embedding, char_rnn, d_W, d_b)

    start_training_time = datetime.now()
    current_date = start_training_time.strftime("%d.%m.%Y")
    current_time = start_training_time.strftime("%H:%M:%S")

    # num_tagged = 0
    evaluation_results = []
    cum_loss = 0
    for iter in xrange(EPOCHS):
        pretrain_time = datetime.now()
        random.shuffle(train_batches)
        for i, sentence in enumerate(train_batches, 1):
            if i % EVALUATE_ITERATION == 0:
                eval = evaluate_set(dev_batches, holder)
                evaluation_results.append(eval)
                print 'epoch {}, batch {}, validation evaluation {}'.format(iter + 1, i, eval)
            words = [word for word, tag in sentence]
            tags = [tag for word, tag in sentence]
            loss_exp = calc_loss(words, tags, holder)
            cum_loss += loss_exp.scalar_value()
            # num_tagged += len(golds)
            loss_exp.backward()
            trainer.update()
        print 'epoch took: ' + str(datetime.now() - pretrain_time)

    total_training_time = datetime.now() - start_training_time
    print 'total training time was {}'.format(total_training_time)

    save_results_and_model(evaluation_results, current_date, current_time, tagging_type, input_representation, total_training_time, model)


if __name__ == '__main__':
    main()

c = 2