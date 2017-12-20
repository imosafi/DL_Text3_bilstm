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
CHAR_EMBED_SIZE = 30
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


def word_rep(w, word2index, word_embedding):
    w_index = word2index[w] if w in word2index else word2index[UNKNOWN_WORD]
    return word_embedding[w_index]


def build_graph(words, fwdRNN_layer1, bwdRNN_layer1, fwdRNN_layer2, bwdRNN_layer2, pH, pO, word2index, word_embedding):
    dy.renew_cg()
    # initialize the RNNs
    fl1_init = fwdRNN_layer1.initial_state()
    bl1_init = bwdRNN_layer1.initial_state()
    fl2_init = fwdRNN_layer2.initial_state()
    bl2_init = bwdRNN_layer2.initial_state()

    wembs = [word_rep(w, word2index, word_embedding) for w in words]
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


def calc_loss(words, tags, tag2index, fwdRNN_layer1, bwdRNN_layer1, fwdRNN_layer2, bwdRNN_layer2, pH, pO, word2index, word_embedding):
    vecs = build_graph(words, fwdRNN_layer1, bwdRNN_layer1, fwdRNN_layer2, bwdRNN_layer2, pH, pO, word2index, word_embedding)
    losses = []
    for v, t in zip(vecs, tags):
        tid = tag2index[t]
        loss = dy.pickneglogsoftmax(v, tid)
        losses.append(loss)
    return dy.esum(losses)


def predict_tags(words, index2tag, fwdRNN_layer1, bwdRNN_layer1, fwdRNN_layer2, bwdRNN_layer2, pH, pO, word2index, word_embedding):
    vecs = build_graph(words, fwdRNN_layer1, bwdRNN_layer1, fwdRNN_layer2, bwdRNN_layer2, pH, pO, word2index, word_embedding)
    vecs = [dy.softmax(v) for v in vecs]
    probs = [v.npvalue() for v in vecs]
    tags = []
    for prb in probs:
       tag = np.argmax(prb)
       tags.append(index2tag[tag])
    return zip(words, tags)


def evaluate_set(dev_batches, index2tag, fwdRNN_layer1, bwdRNN_layer1, fwdRNN_layer2, bwdRNN_layer2, pH, pO, word2index, word_embedding):
    good = 0
    bad = 0
    for sentence in dev_batches:
        words = [word for word, tag in sentence]
        real_tags = [tag for word, tag in sentence]
        tags = [tag for word, tag in predict_tags(words, index2tag, fwdRNN_layer1, bwdRNN_layer1, fwdRNN_layer2, bwdRNN_layer2, pH, pO, word2index, word_embedding)]
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

    vocab.append(UNKNOWN_WORD)
    vocab = set(vocab)
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

    model = dy.Model()
    trainer = dy.AdamTrainer(model)

    # here execute according to a, b, c or d

    word_embedding = model.add_lookup_parameters((vocab_size, WORD_EMBED_SIZE))
    # char_embedding = model.add_lookup_parameters((num_of_chars, CHAR_EMBED_SIZE))

    pH = model.add_parameters((32, 100))
    pO = model.add_parameters((len(tags), 32))

    # here we'll need to change the dim by the selected way of representation
    fwdRNN_layer1 = dy.LSTMBuilder(layers=1, input_dim=128, hidden_dim=50, model=model)
    bwdRNN_layer1 = dy.LSTMBuilder(layers=1, input_dim=128, hidden_dim=50, model=model)

    fwdRNN_layer2 = dy.LSTMBuilder(layers=1, input_dim=100, hidden_dim=50, model=model)
    bwdRNN_layer2 = dy.LSTMBuilder(layers=1, input_dim=100, hidden_dim=50, model=model)



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
                eval = evaluate_set(dev_batches, index2tag, fwdRNN_layer1, bwdRNN_layer1, fwdRNN_layer2, bwdRNN_layer2, pH, pO, word2index, word_embedding)
                evaluation_results.append(eval)
                print 'epoch {}, batch {}, validation evaluation {}'.format(iter + 1, i, eval)
            words = [word for word, tag in sentence]
            tags = [tag for word, tag in sentence]
            loss_exp = calc_loss(words, tags, tag2index, fwdRNN_layer1, bwdRNN_layer1, fwdRNN_layer2, bwdRNN_layer2, pH, pO, word2index, word_embedding)
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