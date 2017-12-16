import xeger
import re
import os
from random import *

LANGUAGE_REG_EXPRESSION = '[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+'
UPPER_LIMIT = 40
NUM_VALID_EXAMPLES = 500
NUM_INVALID_EXAMPLES = 500
SIZE_OF_TRAIN = 30000
SIZE_OF_TEST = 5000

# SIZE_OF_TRAIN = 20
# SIZE_OF_TEST = 4


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


def get_regex_subitems():
    items = []
    items.append('[1-9]+')
    items.append('a+')
    items.append('[1-9]+')
    items.append('b+')
    items.append('[1-9]+')
    items.append('c+')
    items.append('[1-9]+')
    items.append('d+')
    items.append('[1-9]+')
    return items


def generate_invalid_sequence():
    while True:
        shuffle(regex_subitems)
        new_regex = ''.join(regex_subitems)
        sequence = str(xeger.xeger(new_regex))
        if not is_valid_sequence(sequence):
            break
    return sequence


def is_valid_sequence(sequence):
    return re.sub(LANGUAGE_REG_EXPRESSION, '', sequence, count=1) == ''


def generate_valid_examples(size):
    sequences = []
    i = 0
    while i < size:
        xeger._limit = randint(1, UPPER_LIMIT)
        example = str(xeger.xeger(LANGUAGE_REG_EXPRESSION))
        if example not in sequences:
            sequences.append(example)
            i += 1
    return sequences


def generate_invalid_examples(size):
    sequences = []
    i = 0
    while i < size:
        xeger._limit = randint(1, UPPER_LIMIT)
        example = generate_invalid_sequence()
        if example not in sequences:
            sequences.append(example)
            i += 1
    return sequences


def test_valid_examples(examples):
    assert (all(is_valid_sequence(x) for x in examples))
    print 'test_valid_examples passed'


def test_invalid_examples(examples):
    assert (all(not is_valid_sequence(x) for x in examples))
    print 'test_invalid_examples passed'


if __name__ == '__main__':
    xeger = xeger.Xeger(UPPER_LIMIT)
    regex_subitems = get_regex_subitems()
    valid_examples = generate_valid_examples(NUM_VALID_EXAMPLES)
    test_valid_examples(valid_examples)
    invalid_examples = generate_invalid_examples(NUM_INVALID_EXAMPLES)
    test_invalid_examples(invalid_examples)

    ensure_directory_exists('data/')

    with open('data/pos_examples.txt', 'w') as f:
        f.writelines([x + '\n' for x in valid_examples])

    with open('data/neg_examples.txt', 'w') as f:
        f.writelines([x + '\n' for x in invalid_examples])

    train, test = generate_train_test_sequneces()

    with open('data/train.txt', 'w') as f:
        f.writelines([x + '\n' for x in train])

    with open('data/test.txt', 'w') as f:
        f.writelines([x + '\n' for x in test])
