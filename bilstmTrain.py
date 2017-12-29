import dynet as dy
from collections import Counter
import random
import sys
import numpy as np
from datetime import datetime
import os


POS_PATH = 'data/pos/'
NER_PATH = 'data/ner/'
UNKNOWN_WORD = 'unk'
UNKNOWN_PREFIX = 'unknown_pre'
UNKNOWN_SUFFIX = 'unknown_pre'
UNKNOW_CHAR = '~'

WORD_EMBED_SIZE = 150
CHAR_EMBED_SIZE = 30
EPOCHS = 5
EVALUATE_ITERATION = 2000
POS_VOCAB_SIZE = 35000
NER_VOCAB_SIZE = 20000


def read_dev_into_batches(file_name):
    batches = []
    current_batch = []
    for line in file(file_name):
        if line == '\n':
            batches.append(current_batch)
            current_batch = []
        else:
            text, label = line.strip().lower().split()
            current_batch.append((text, label))
    return batches


def read_train_data_into_batches(file_name):
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
            vocab.append(text)
            tags.append(label)
    return batches, vocab, tags


def get_subword_vocab(words):
    pre_voc = []
    suf_voc = []
    for word in words:
        if len(word) > 3:
            pre_voc.append(word[:3])
            suf_voc.append(word[-3:])
    return set(pre_voc), set(suf_voc)


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
        char_indexes = [holder.char2index[c] if c in holder.char2index else holder.char2index[UNKNOW_CHAR] for c in list(word)]
        char_embeddings = [holder.char_embedding[c_index] for c_index in char_indexes]
        return c_init.transduce(char_embeddings)[-1]
    elif holder.input_representation == 'c':
        if word not in holder.word2index:
            return holder.word_embedding[holder.word2index[UNKNOWN_WORD]]
        elif len(word) < 4:
            return holder.word_embedding[holder.word2index[word]] * 3
        else:
            return holder.word_embedding[holder.word2index[word]] + \
                   holder.prefix_embedding[holder.pre2index[word[:3]]] + \
                   holder.suffix_embedding[holder.suff2index[word[-3:]]]
    else:
        d_W = dy.parameter(holder.d_W)
        d_b = dy.parameter(holder.d_b)
        w_index = holder.word2index[word] if word in holder.word2index else holder.word2index[UNKNOWN_WORD]
        a = holder.word_embedding[w_index]
        c_init = holder.char_rnn.initial_state()
        char_indexes = [holder.char2index[c] if c in holder.char2index else holder.char2index[UNKNOW_CHAR] for c in list(word)]
        char_embeddings = [holder.char_embedding[c_index] for c_index in char_indexes]
        b = c_init.transduce(char_embeddings)[-1]
        ab = dy.concatenate([a, b])
        return d_W * ab + d_b


def build_graph(words, holder):
    dy.renew_cg()
    fl1_init = holder.fwdRNN_layer1.initial_state()
    bl1_init = holder.bwdRNN_layer1.initial_state()
    fl2_init = holder.fwdRNN_layer2.initial_state()
    bl2_init = holder.bwdRNN_layer2.initial_state()

    wembs = [get_word_rep(w, holder) for w in words]
    fws = fl1_init.transduce(wembs)
    bws = bl1_init.transduce(reversed(wembs))

    bi = [dy.concatenate([f, b]) for f, b in zip(fws, reversed(bws))]
    fws2 = fl2_init.transduce(bi)
    bws2 = bl2_init.transduce(reversed(bi))
    b_tag = [dy.concatenate([f2, b2]) for f2, b2 in zip(fws2, reversed(bws2))]
    W_ab = dy.parameter(holder.W_ab)
    b_ab = dy.parameter(holder.b_ab)

    return [(W_ab * x + b_ab) for x in b_tag]


def calc_loss(words, tags, holder):
    vecs = build_graph(words, holder)
    losses = []
    for v, t in zip(vecs, tags):
        tid = holder.tag2index[t]
        loss = dy.pickneglogsoftmax(v, tid)
        losses.append(loss)
    loss = dy.esum(losses)
    return loss / len(vecs)


def predict_tags(words, holder):
    vecs = build_graph(words, holder)
    vecs = [dy.softmax(v) for v in vecs]
    probs = [v.npvalue() for v in vecs]
    tags = []
    for prb in probs:
       tag = np.argmax(prb)
       tags.append(holder.index2tag[tag])
    return zip(words, tags)


def evaluate_set(dev_batches, holder):
    good = 0
    bad = 0
    for sentence in dev_batches:
        words = [word for word, tag in sentence]
        real_tags = [tag for word, tag in sentence]
        tags = [tag for word, tag in predict_tags(words, holder)]
        for t1, t2 in zip(real_tags, tags):
            if holder.tagging_type == 'ner' and t1 == 'o' and t2 == 'o':
                continue
            if t1 == t2:
                good += 1
            else:
                bad += 1
    return float(good) / (good + bad)


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def save_results_and_model(evaluation_results, current_date, current_time, tagging_type, input_representation, total_training_time, model, model_file):
    path = 'results/{}_{}'.format(current_date, current_time)
    ensure_directory_exists(path)
    with open(path + '/results.txt', 'w') as f:
        f.write('tagging type: {}\n'.format(tagging_type))
        f.write('representation type: {}\n'.format(input_representation))
        f.write('total training time: {}\n'.format(total_training_time))
        f.write('num of epochs: {}\n'.format(EPOCHS))
        f.write('evaluated every: {}\n'.format(EVALUATE_ITERATION))
    with open(path + '/' + tagging_type + '_' + input_representation + '_' + 'evaluation.txt', 'w') as f:
        f.write(''.join([str(x) + ' ' for x in evaluation_results]))
    model.save(path + '/' + model_file)


def get_train_words(file_name):
    vocab = []
    for line in file(file_name):
        if not line == '\n':
            text, _ = line.strip().lower().split()
            vocab.append(text)
    return vocab


def get_rare_words(words, vocab_size):
    counter = Counter(words)
    most_common_keys = [x[0] for x in counter.most_common(vocab_size)]
    keys_to_remove = [x for x in counter.keys() if x not in most_common_keys]
    return keys_to_remove


def get_rare_chars(unique_chars):
    counter = Counter(unique_chars)
    most_common_keys = [x[0] for x in counter.most_common(len(counter.keys()) - 2)]
    keys_to_remove = [x for x in counter.keys() if x not in most_common_keys]
    return keys_to_remove


class ComponentHolder:
    def __init__(self, input_representation, tagging_type, word2index, index2tag, tag2index, char2index,pre2index, suff2index, fwdRNN_layer1, bwdRNN_layer1, fwdRNN_layer2,
                 bwdRNN_layer2, W_ab, b_ab, word_embedding,prefix_embedding,suffix_embedding,  char_embedding, char_rnn, d_W, d_b):
        self.input_representation = input_representation
        self.tagging_type = tagging_type
        self.word2index = word2index
        self.index2tag = index2tag
        self.tag2index = tag2index
        self.char2index = char2index
        self.pre2index = pre2index
        self.suff2index = suff2index
        self.fwdRNN_layer1 = fwdRNN_layer1
        self.bwdRNN_layer1 = bwdRNN_layer1
        self.fwdRNN_layer2 = fwdRNN_layer2
        self.bwdRNN_layer2 = bwdRNN_layer2
        self.W_ab = W_ab
        self.b_ab = b_ab
        self.word_embedding = word_embedding
        self.prefix_embedding = prefix_embedding
        self.suffix_embedding = suffix_embedding
        self.char_embedding = char_embedding
        self.char_rnn = char_rnn
        self.d_W = d_W
        self.d_b = d_b


def main(in_rep, tag_type):
    input_representation = in_rep #sys.argv[1]
    train_file = sys.argv[2]
    model_file = sys.argv[3]
    tagging_type = tag_type #sys.argv[4]

    validate_args(input_representation, tagging_type)

    rare_words = get_rare_words(get_train_words((POS_PATH if tagging_type == 'pos' else NER_PATH) + 'train'), POS_VOCAB_SIZE if tagging_type == 'pos' else NER_VOCAB_SIZE)
    train_batches, vocab, tags = read_train_data_into_batches((POS_PATH if tagging_type == 'pos' else NER_PATH) + train_file)
    dev_batches = read_dev_into_batches((POS_PATH if tagging_type == 'pos' else NER_PATH) + 'dev')

    prefx_vocab = None
    suffix_vocab = None
    pref2index = None
    suff2index = None
    pre_embedding = None
    suff_embedding = None

    vocab = set(vocab)
    for word in rare_words:
        vocab.remove(word)
    vocab.add(UNKNOWN_WORD)
    vocab = list(vocab)
    if input_representation == 'c':
        prefx_vocab, suffix_vocab = get_subword_vocab(vocab)
        prefx_vocab.add(UNKNOWN_PREFIX)
        suffix_vocab.add(UNKNOWN_SUFFIX)
        pref2index = {c: i for i, c in enumerate(prefx_vocab)}
        suff2index = {c: i for i, c in enumerate(suffix_vocab)}
    tags = set(tags)


    tag2index = {c: i for i, c in enumerate(tags)}
    index2tag = {v: k for k, v in tag2index.iteritems()}
    word2index = {c: i for i, c in enumerate(vocab)}
    vocab_size = len(vocab)

    unique_chars = []
    for word in vocab:
        unique_chars.extend(list(word))
    rare_chars = get_rare_chars(unique_chars)
    unique_chars = set(unique_chars)
    for c in rare_chars:
        unique_chars.remove(c)
    unique_chars.add(UNKNOW_CHAR)
    num_of_chars = len(unique_chars)
    char2index = {c: i for i, c in enumerate(unique_chars)}

    model = dy.Model()
    trainer = dy.AdamTrainer(model)
    # trainer.learning_rate = 0.0005
    # trainer.learning_rate = 0.00025
    # trainer.learning_rate = 0.002

    char_embedding = None
    char_rnn = None
    d_W = None
    d_b = None
    l1_hidden_dim = 64
    l2_hidden_dim = 64

    if input_representation == 'a':
        rnnlayer1_input_dim = WORD_EMBED_SIZE
    elif input_representation == 'b':
        char_embedding = model.add_lookup_parameters((num_of_chars, CHAR_EMBED_SIZE))
        char_rnn = dy.LSTMBuilder(layers=1, input_dim=CHAR_EMBED_SIZE, hidden_dim=150, model=model)
        rnnlayer1_input_dim = 150
    elif input_representation == 'c':
        pre_embedding = model.add_lookup_parameters((len(prefx_vocab), WORD_EMBED_SIZE))
        suff_embedding = model.add_lookup_parameters((len(suffix_vocab), WORD_EMBED_SIZE))
    elif input_representation == 'd':
        rnnlayer1_input_dim = 150
        char_embedding = model.add_lookup_parameters((num_of_chars, CHAR_EMBED_SIZE))
        char_rnn = dy.LSTMBuilder(layers=1, input_dim=CHAR_EMBED_SIZE, hidden_dim=150, model=model)
        d_W = model.add_parameters((150, 300))
        d_b = model.add_parameters(150)

    rnnlayer2_input_dim = l1_hidden_dim * 2
    word_embedding = model.add_lookup_parameters((vocab_size, WORD_EMBED_SIZE))

    W_ab = model.add_parameters((len(tags), l2_hidden_dim * 2))
    b_ab = model.add_parameters(len(tags))

    fwdRNN_layer1 = dy.LSTMBuilder(layers=1, input_dim=rnnlayer1_input_dim, hidden_dim=l1_hidden_dim, model=model)
    bwdRNN_layer1 = dy.LSTMBuilder(layers=1, input_dim=rnnlayer1_input_dim, hidden_dim=l1_hidden_dim, model=model)

    fwdRNN_layer2 = dy.LSTMBuilder(layers=1, input_dim=rnnlayer2_input_dim, hidden_dim=l2_hidden_dim, model=model)
    bwdRNN_layer2 = dy.LSTMBuilder(layers=1, input_dim=rnnlayer2_input_dim, hidden_dim=l2_hidden_dim, model=model)

    holder = ComponentHolder(input_representation, tagging_type,  word2index, index2tag, tag2index, char2index,pref2index, suff2index,  fwdRNN_layer1, bwdRNN_layer1,
                             fwdRNN_layer2, bwdRNN_layer2, W_ab, b_ab, word_embedding, pre_embedding, suff_embedding, char_embedding, char_rnn, d_W, d_b)

    start_training_time = datetime.now()
    current_date = start_training_time.strftime("%d.%m.%Y")
    current_time = start_training_time.strftime("%H:%M:%S")

    evaluation_results = []
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
            loss_exp.backward()
            trainer.update()
        print 'epoch took: ' + str(datetime.now() - pretrain_time)

    total_training_time = datetime.now() - start_training_time
    print 'total training time was {}'.format(total_training_time)

    save_results_and_model(evaluation_results, current_date, current_time, tagging_type, input_representation, total_training_time, model, model_file)

if __name__ == '__main__':
    # main('a', 'pos')
    # main('b', 'pos')
    # main('c', 'pos')
    # main('d', 'pos')
    # main('a', 'ner')
    main('b', 'ner')
    # main('c', 'ner')
    # main('d', 'ner')
