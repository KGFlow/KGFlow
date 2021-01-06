import pickle
import numpy as np
rel_vec =[]

with open("/home/hyf/fanzhiguang/dateset/image-text/DB15K/image_text.txt","r",encoding="utf-8")as f:
    for line in f.readlines():
        line = line.strip().split()
        temp = []
        for i, value in enumerate(line):
            if i ==0:
                pass
            else:
                temp.append(float(value))
        rel_vec.append(temp)

file = "rel_text.pickle"


# with open(file, 'rb') as handle:
#     new_1hop = pickle.load(handle)
# rel_dic =gen_intra_context_edges(new_1hop)
with open(file, 'wb') as handle:
    pickle.dump(np.array(rel_vec), handle,
                protocol=pickle.HIGHEST_PROTOCOL)

