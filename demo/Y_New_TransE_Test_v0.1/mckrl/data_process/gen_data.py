# print("w")
# def create_data_generator(shuffle=True, infinite=True):
#     print("1212")
#     while True:
#         if shuffle:
#             pass
#         for i in range(10):
#             print("this is inside", i)
#             yield i
#         if not infinite:
#             break
# create_data_generator()
import random
data = list(range(237))

def next_batch(batch_size, data, shuffle=True, infinite=True):
    batch = (len(data) - 1) // batch_size + 1
    print(batch)
    while True:
        if shuffle:
            random.shuffle(data)
        for i in range(batch):
            yield data[i::batch]
        if not infinite:
            break

import numpy as np

# random.shuffle(data)
# print(data)
g = next_batch(15,data)
for i in range(20):
    print(next(g))