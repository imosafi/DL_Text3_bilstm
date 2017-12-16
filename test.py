# import dynet as dy
# import numpy as np
# import random
# import datetime
#
# # acceptor LSTM
# class LstmAcceptor(object):
#     def __init__(self, in_dim, out_dim, mlp_hidden_dim, model):
#         self.builder = dy.VanillaLSTMBuilder(1, in_dim, out_dim, model)
#         self.W1 = model.add_parameters((mlp_hidden_dim, out_dim))
#         self.b1 = model.add_parameters(mlp_hidden_dim)
#         self.W2 = model.add_parameters((2, mlp_hidden_dim))
#         self.b2 = model.add_parameters(2)
#
#     def __call__(self, sequence):
#         lstm = self.builder.initial_state()
#         W1 = self.W1.expr()
#         W2 = self.W2.expr()
#         b1 = self.b1.expr()
#         b2 = self.b2.expr()
#         outputs = lstm.transduce(sequence)
#         # result = dy.softmax(W2 * (dy.tanh(W1 * outputs[-1]) + b1) + b2)
#         result = W2 * (dy.tanh(W1 * outputs[-1]) + b1) + b2
#         return result
#
#
# def read_data(fname):
#     data = []
#     for line in file(fname):
#         text, label = line.strip().split(' ', 1)
#         data.append((text, int(label)))
#     return data
#
#
# def devide_to_batches(data_set, batch_size):
#     num_of_batches = len(data_set) // batch_size  # // floors the answer
#     batches = []
#     for i in range(0, int(num_of_batches)):
#         batches.append((data_set[i * batch_size:i * batch_size + batch_size]))
#     return batches
#
#
# characters = list("abcd123456789")
# int2char = list(characters)
# char2int = {c:i for i,c in enumerate(characters)}
# VOCAB_SIZE = len(characters)
# EMBED_SIZE = 50
# EPOCHS = 10
# MLP_HIDDEN_DIM = 30
# BATCH_SIZE = 100
# LSTM_DIM = 2
# LSTM_OUT_DIM = 50
#
# m = dy.Model()
# trainer = dy.AdamTrainer(m)
# embeds = m.add_lookup_parameters((VOCAB_SIZE, EMBED_SIZE))
# acceptor = LstmAcceptor(in_dim=EMBED_SIZE, out_dim=LSTM_OUT_DIM, mlp_hidden_dim=MLP_HIDDEN_DIM, model=m)
#
# train = read_data('data/train.txt')
# test = read_data('data/test.txt')
#
# random.shuffle(train)
# train_batches = devide_to_batches(train, BATCH_SIZE)
# test_batches = devide_to_batches(test, BATCH_SIZE)
#
# pretrain_time = datetime.datetime.now()
# print 'current time: ' + str(pretrain_time) + ' start training:'
#
# # training code: batched.
# for epoch in xrange(EPOCHS):
#     random.shuffle(train_batches)
#     for i_batch, batch in enumerate(train_batches):
#         dy.renew_cg()     # we create a new computation graph for the epoch, not each item.
#         # we will treat all these 3 datapoints as a single batch
#         losses = []
#         for sequence, label in batch:
#             vecs = [embeds[char2int[i]] for i in sequence]
#             preds = acceptor(vecs)
#             loss = dy.pickneglogsoftmax(preds, label)
#             losses.append(loss)
#         # we accumulated the losses from all the batch.
#         # Now we sum them, and do forward-backward as usual.
#         # Things will run with efficient batch operations.
#         batch_loss = dy.esum(losses) / len(batch)
#         if i_batch % 100 == 0:
#             print 'epoch ' + str(epoch) + ' batch ' + str(i_batch) + ' loss ' + str(batch_loss.npvalue()) # this calls forward on the batch
#         batch_loss.backward()
#         trainer.update()
#
# training_duration = datetime.datetime.now() - pretrain_time
# print 'Done! training took: ' + str(training_duration) + '\n'
#
# print "\n\nPrediction time!\n"
# # prediction code:
# correct = 0
# for batch in test_batches:
#     dy.renew_cg() # new computation graph
#     batch_preds = []
#     for sequence, label in batch:
#         vecs = [embeds[char2int[i]] for i in sequence]
#         preds = dy.softmax(acceptor(vecs))
#         batch_preds.append(preds)
#
#     # now that we accumulated the prediction expressions,
#     # we run forward on all of them:
#     dy.forward(batch_preds)
#     # and now we can efficiently access the individual values:
#     for preds in batch_preds:
#         vals  = preds.npvalue()
#         if np.argmax(vals) == label:
#             correct += 1
#
# print 'accuracy: ' + str(float(correct) / len(test))

import xeger
import re
import os
from random import *
import math

LANGUAGE_REG_EXPRESSION = '[0-9]+[0-9]+'
UPPER_LIMIT = 40
LOWER_LIMIT = 10
SIZE_OF_TRAIN = 30000
SIZE_OF_TEST = 5000

# SIZE_OF_TRAIN = 20
# SIZE_OF_TEST = 2


def generate_train_test_sequneces():
    labeled_valid_sequences = [x + ' ' + '0' for x in generate_valid_examples((SIZE_OF_TRAIN + SIZE_OF_TEST) / 2)]
    labeled_invalid_sequences = [x + ' ' + '1' for x in generate_invalid_examples((SIZE_OF_TRAIN + SIZE_OF_TEST) / 2)]
    slicer = len(labeled_valid_sequences) - SIZE_OF_TEST / 2
    train = labeled_valid_sequences[:slicer] + labeled_invalid_sequences[:slicer]
    test = labeled_valid_sequences[slicer:] + labeled_invalid_sequences[slicer:]
    return train, test


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def generate_valid_sequence():
    while True:
        sequence = str(xeger.xeger(LANGUAGE_REG_EXPRESSION))
        if is_valid_sequence(sequence):
            break
    return sequence

def generate_invalid_sequence():
    while True:
        sequence = str(xeger.xeger(LANGUAGE_REG_EXPRESSION))
        if not is_valid_sequence(sequence):
            break
    return sequence


def is_valid_sequence(sequence):
    chars = list(sequence)
    return sum(int(x) for x in chars) % int(math.sqrt(len(chars))) == 0


def generate_valid_examples(size):
    sequences = []
    i = 0
    while i < size:
        xeger._limit = randint(LOWER_LIMIT, UPPER_LIMIT)
        example = generate_valid_sequence()
        if example not in sequences:
            sequences.append(example)
            i += 1
    return sequences


def generate_invalid_examples(size):
    sequences = []
    i = 0
    while i < size:
        xeger._limit = randint(LOWER_LIMIT, UPPER_LIMIT)
        example = generate_invalid_sequence()
        if example not in sequences:
            sequences.append(example)
            i += 1
    return sequences



if __name__ == '__main__':
    xeger = xeger.Xeger(UPPER_LIMIT)

    ensure_directory_exists('data_tries/')

    train, test = generate_train_test_sequneces()

    with open('data_tries/train.txt', 'w') as f:
        f.writelines([x + '\n' for x in train])

    with open('data_tries/test.txt', 'w') as f:
        f.writelines([x + '\n' for x in test])