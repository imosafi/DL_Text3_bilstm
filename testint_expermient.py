import dynet as dy
import numpy as np
import random
import datetime


class LstmAcceptor(object):
    def __init__(self, in_dim, out_dim, mlp_hidden_dim, model):
        self.builder = dy.VanillaLSTMBuilder(1, in_dim, out_dim, model)
        self.W1 = model.add_parameters((mlp_hidden_dim, out_dim))
        self.b1 = model.add_parameters(mlp_hidden_dim)
        self.W2 = model.add_parameters((2, mlp_hidden_dim))
        self.b2 = model.add_parameters(2)

    def __call__(self, sequence):
        lstm = self.builder.initial_state()
        W1 = self.W1.expr()
        W2 = self.W2.expr()
        b1 = self.b1.expr()
        b2 = self.b2.expr()
        outputs = lstm.transduce(sequence)
        # result = dy.softmax(W2 * (dy.tanh(W1 * outputs[-1]) + b1) + b2)
        result = W2 * (dy.tanh(W1 * outputs[-1]) + b1) + b2
        return result


def read_data(fname):
    data = []
    for line in file(fname):
        text, label = line.strip().split(' ', 1)
        data.append((text, int(label)))
    return data


def devide_to_batches(data_set, batch_size):
    num_of_batches = len(data_set) // batch_size  # // floors the answer
    batches = []
    for i in range(0, int(num_of_batches)):
        batches.append((data_set[i * batch_size:i * batch_size + batch_size]))
    return batches

# characters = list("0123456789")
characters = list("0123456789abcdefghijklmnopqrstuvwxyz")
int2char = list(characters)
char2int = {c:i for i,c in enumerate(characters)}
VOCAB_SIZE = len(characters)
EMBED_SIZE = 50
EPOCHS = 10
MLP_HIDDEN_DIM = 30
BATCH_SIZE = 100
LSTM_DIM = 2
LSTM_OUT_DIM = 50

m = dy.Model()
trainer = dy.AdamTrainer(m)
embeds = m.add_lookup_parameters((VOCAB_SIZE, EMBED_SIZE))
acceptor = LstmAcceptor(in_dim=EMBED_SIZE, out_dim=LSTM_OUT_DIM, mlp_hidden_dim=MLP_HIDDEN_DIM, model=m)

train = read_data('data_tries/train.txt')
test = read_data('data_tries/test.txt')

random.shuffle(train)
train_batches = devide_to_batches(train, BATCH_SIZE)
test_batches = devide_to_batches(test, BATCH_SIZE)

pretrain_time = datetime.datetime.now()
print 'current time: ' + str(pretrain_time) + ' start training:'

# training code: batched.
for epoch in xrange(EPOCHS):
    random.shuffle(train_batches)
    for i_batch, batch in enumerate(train_batches):
        dy.renew_cg()     # we create a new computation graph for the epoch, not each item.
        # we will treat all these 3 datapoints as a single batch
        losses = []
        for sequence, label in batch:
            vecs = [embeds[char2int[i]] for i in sequence]
            preds = acceptor(vecs)
            loss = dy.pickneglogsoftmax(preds, label)
            losses.append(loss)
        # we accumulated the losses from all the batch.
        # Now we sum them, and do forward-backward as usual.
        # Things will run with efficient batch operations.
        batch_loss = dy.esum(losses) / len(batch)
        if i_batch % 50 == 0:
            print 'epoch ' + str(epoch) + ' batch ' + str(i_batch) + ' loss ' + str(batch_loss.npvalue()) # this calls forward on the batch
        batch_loss.backward()
        trainer.update()

training_duration = datetime.datetime.now() - pretrain_time
print 'Done! training took: ' + str(training_duration) + '\n'

print "\n\nPrediction time!\n"
# prediction code:
correct = 0
for batch in test_batches:
    dy.renew_cg() # new computation graph
    batch_preds = []
    for sequence, label in batch:
        vecs = [embeds[char2int[i]] for i in sequence]
        preds = dy.softmax(acceptor(vecs))
        batch_preds.append(preds)

    # now that we accumulated the prediction expressions,
    # we run forward on all of them:
    dy.forward(batch_preds)
    # and now we can efficiently access the individual values:
    for preds in batch_preds:
        vals  = preds.npvalue()
        if np.argmax(vals) == label:
            correct += 1

print 'accuracy: ' + str(float(correct) / len(test))
