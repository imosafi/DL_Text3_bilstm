from bilstmTrain import read_data_into_batches, validate_args, WORD_EMBED_SIZE, UNKNOWN_WORD, predict_tags, evaluate_set
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
    dev_batches = read_data_into_batches((POS_PATH if tagging_type == 'pos' else NER_PATH) + 'dev')
    test_batches = read_test_batches((POS_PATH if tagging_type == 'pos' else NER_PATH) + test_file)

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

    model2 = dy.Model()
    word_embedding = model2.add_lookup_parameters((vocab_size, WORD_EMBED_SIZE))
    # char_embedding = model2.add_lookup_parameters((num_of_chars, CHAR_EMBED_SIZE))

    pH = model2.add_parameters((32, 100))
    pO = model2.add_parameters((len(tags), 32))

    fwdRNN_layer1 = dy.LSTMBuilder(layers=1, input_dim=128, hidden_dim=50, model=model2)
    bwdRNN_layer1 = dy.LSTMBuilder(layers=1, input_dim=128, hidden_dim=50, model=model2)

    fwdRNN_layer2 = dy.LSTMBuilder(layers=1, input_dim=100, hidden_dim=50, model=model2)
    bwdRNN_layer2 = dy.LSTMBuilder(layers=1, input_dim=100, hidden_dim=50, model=model2)

    model2.populate(model_file)

    word_tag_tuples = []

    print evaluate_set(dev_batches, index2tag, fwdRNN_layer1, bwdRNN_layer1, fwdRNN_layer2, bwdRNN_layer2, pH, pO, word2index, word_embedding)

    for sentence in test_batches:
        words = [word for word in sentence]
        word_tag_tuples.extend(predict_tags(words, index2tag, fwdRNN_layer1, bwdRNN_layer1, fwdRNN_layer2, bwdRNN_layer2, pH, pO, word2index, word_embedding))

    with open('test4.{}'.format(tagging_type), 'w') as f:
        f.write('\n'.join('{} {}'.format(x[0], x[1]) for x in word_tag_tuples))


if __name__ == '__main__':
    main()


