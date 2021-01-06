import numpy as np


def next_test(batch_size, data):
    start = 0
    while start < len(data):
        end = min(start + batch_size, len(data))
        yield data[start:end]
        start = end


# 2019_12 = list(range(107))
# g = next_test(10,2019_12)
# for i in range(11):
#     print(next(g))

def next_batch(batch_size, data, infinite=False):
    batch = (len(data) - 1) // batch_size + 1
    print(batch)
    while True:
        for i in range(batch):
            yield data[i::batch]
        if not infinite:
            break


data = list(range(119))
g = next_batch(10, data)
print(next(g))
print(next(g))
# print(next(g))
# for i in range(12):
#     print(next(g))
    # random_id = np.random.permutation(len(2019_12))
    # start = 0
    # if start < len(2019_12):
    #     end = min(start + batch_size,len(2019_12))
    #     yield [2019_12[i] for i in ]
