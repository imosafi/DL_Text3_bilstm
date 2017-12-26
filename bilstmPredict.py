from bilstmTrain import *
import sys
import numpy as np
import dynet as dy

POS_PATH = 'data/pos/'
NER_PATH = 'data/ner/'


def read_test_batches(file_name):
    batches = []
    current_batch = []
    for line in file(file_name):
        if line == '\n':
            batches.append(current_batch)
            current_batch = []
        else:
            text = line.strip().lower()
            current_batch.append(text)
    return batches


def main():
    input_representation = sys.argv[1]
    model_file = sys.argv[2]
    test_file = sys.argv[3]
    tagging_type = sys.argv[4]
    validate_args(input_representation, tagging_type)

    rare_words = get_rare_words(get_train_words((POS_PATH if tagging_type == 'pos' else NER_PATH) + 'train'), POS_VOCAB_SIZE if tagging_type == 'pos' else NER_VOCAB_SIZE)
    train_batches, vocab, tags = read_train_data_into_batches((POS_PATH if tagging_type == 'pos' else NER_PATH) + 'train')
    dev_batches = read_dev_into_batches((POS_PATH if tagging_type == 'pos' else NER_PATH) + 'dev')
    test_batches = read_test_batches((POS_PATH if tagging_type == 'pos' else NER_PATH) + test_file)


    vocab = set(vocab)
    for word in rare_words:
        vocab.remove(word)
    vocab.add(UNKNOWN_WORD)
    vocab = list(vocab)
    if input_representation == 'c':
        vocab = list(get_subword_vocab(vocab))
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

    model2 = dy.Model()

    char_embedding = None
    char_rnn = None
    d_W = None
    d_b = None
    ab_tag_output = 100

    if input_representation == 'a' or input_representation == 'c':
        rnnlayer1_input_dim = 128
        rnnlayer2_input_dim = 100
        hidden_dim = 50
    elif input_representation == 'b':
        char_embedding = model2.add_lookup_parameters((num_of_chars, CHAR_EMBED_SIZE))
        char_rnn = dy.LSTMBuilder(layers=1, input_dim=CHAR_EMBED_SIZE, hidden_dim=64, model=model2)
        rnnlayer1_input_dim = 64
        hidden_dim = 50
        rnnlayer2_input_dim = 100
    elif input_representation == 'd':
        rnnlayer1_input_dim = 200
        rnnlayer2_input_dim = 100
        hidden_dim = 50
        char_embedding = model2.add_lookup_parameters((num_of_chars, CHAR_EMBED_SIZE))
        char_rnn = dy.LSTMBuilder(layers=1, input_dim=CHAR_EMBED_SIZE, hidden_dim=64, model=model2)
        d_W = model2.add_parameters((200, 192))
        d_b = model2.add_parameters(200)

    word_embedding = model2.add_lookup_parameters((vocab_size, WORD_EMBED_SIZE))

    W_ab = model2.add_parameters((len(tags), ab_tag_output))
    b_ab = model2.add_parameters(len(tags))

    fwdRNN_layer1 = dy.LSTMBuilder(layers=1, input_dim=rnnlayer1_input_dim, hidden_dim=hidden_dim, model=model2)
    bwdRNN_layer1 = dy.LSTMBuilder(layers=1, input_dim=rnnlayer1_input_dim, hidden_dim=hidden_dim, model=model2)

    fwdRNN_layer2 = dy.LSTMBuilder(layers=1, input_dim=rnnlayer2_input_dim, hidden_dim=hidden_dim, model=model2)
    bwdRNN_layer2 = dy.LSTMBuilder(layers=1, input_dim=rnnlayer2_input_dim, hidden_dim=hidden_dim, model=model2)

    holder = ComponentHolder(input_representation, tagging_type,  word2index, index2tag, tag2index, char2index,  fwdRNN_layer1, bwdRNN_layer1,
                             fwdRNN_layer2, bwdRNN_layer2, W_ab, b_ab, word_embedding, char_embedding, char_rnn, d_W, d_b)


    model2.populate(model_file)

    word_tag_tuples = []

    print evaluate_set(dev_batches, holder)

    for sentence in test_batches:
        words = [word for word in sentence]
        word_tag_tuples.extend(predict_tags(words, holder))

    with open('test4.{}'.format(tagging_type), 'w') as f:
        f.write('\n'.join('{} {}'.format(x[0], x[1]) for x in word_tag_tuples))


if __name__ == '__main__':
    main()


