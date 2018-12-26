import numpy as np
import itertools

text_features = np.load('experiment/text_feat.npy')
cnnfeat = np.load('experiment/cnnfeat.npy')

n_cat = 20

def text2image(index_cat, index):
    vec = text_features[index_cat, index]
    x = np.argmin(np.array([np.linalg.norm(vec-cnnfeat[i,j]) for i,j in itertools.product(range(n_cat), range(n_cat))]))
    bool = False
    if x // n_cat == index_cat and x % n_cat == index:
        bool = True
    return x // n_cat, x % n_cat, bool

def image2text(index_cat, index):
    vec = cnnfeat[index_cat, index]
    x = np.argmin(np.array([np.linalg.norm(vec-text_features[i,j]) for i,j in itertools.product(range(n_cat), range(n_cat))]))
    bool = False
    if x // n_cat == index_cat and x % n_cat == index:
        bool = True
    return x // n_cat, x % n_cat, bool


# index_cat, index = 0 , 0
# print(text2image(index_cat, index))

count1 = 0
count2 = 0

for i in range(n_cat):
    for j in range(n_cat):
        count1 += text2image(i, j)[-1]
        count2 += image2text(i, j)[-1]

print(count1)
print(count2)
