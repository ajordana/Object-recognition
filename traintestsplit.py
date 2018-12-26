import os
import sys
import random
import numpy as np
from tqdm import tqdm
# rename original data folder: 'train_images_origin/'
# create a folder 'train_images_augmented/'  in bird_data_set
# run this code in bird_data_set

num_category = 20

N_train = 30
N_test =-12


if os.path.isdir('PascalSentenceDataset/validation_images/'):
    sys.exit("validation folder already existing")



os.chdir('PascalSentenceDataset/')
list_category = os.listdir('dataset/')
os.mkdir('validation_images/')
os.mkdir('validation_sentences/')

for category in tqdm(list_category):

    os.mkdir(os.path.join('validation_images/',os.path.join(category, '')))
    os.mkdir(os.path.join('validation_sentences/',os.path.join(category, '')))

    L = np.linspace(0,N_train + N_test -1 ,N_train + N_test).astype('int')
    random.shuffle(L)
    val_index = L[:N_test]
    images = os.listdir(os.path.join('dataset/',os.path.join(category, '')))
    sentences = os.listdir(os.path.join('sentence/',os.path.join(category, '')))
    val_im = [images[l] for l in val_index]
    val_sentence = [sentences[l] for l in val_index]
    for i in range(N_test):
        os.rename(os.path.join(os.path.join('dataset/',os.path.join(category, '')),val_im[i]), os.path.join(os.path.join('validation_images/',os.path.join(category, '')),val_im[i]))
        os.rename(os.path.join(os.path.join('sentence/',os.path.join(category, '')),val_sentence[i]), os.path.join(os.path.join('validation_sentences/',os.path.join(category, '')),val_sentence[i]))
