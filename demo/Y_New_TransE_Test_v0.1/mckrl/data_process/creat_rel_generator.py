# coding = utf-8
import numpy as np
import random


def next_batch(batch_size, data, shuffle=True, infinite=True):
    batch = (len(data) - 1) // batch_size + 1
    while True:
        if shuffle:
            random.shuffle(data)
        for i in range(batch):
            yield data[i::batch]
        if not infinite:
            break

def next_batch_(batch_size, data, shuffle=True, infinite=False):
    batch = (len(data) - 1) // batch_size + 1
    while True:
        if shuffle:
            random.shuffle(data)
        for i in range(batch):
            yield data[i::batch]
        if not infinite:
            break



# next_batch_(128, train_data, shuffle=True, infinite=False)
