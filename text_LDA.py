import imp
from nltk.corpus import stopwords
from gensim import corpora, models
import gensim
import string
import os
import re
import numpy as np
from librairy import clean_text_simple


# stemmer = PorterStemmer()
stpwds = stopwords.words('english')
punct = string.punctuation.replace('-', '')


path_to_sentences = 'PascalSentenceDataset/sentence/'
path_to_val_sentences = 'PascalSentenceDataset/validation_sentences/'
categories_names = sorted(os.listdir(path_to_sentences))

list_text = []
for category in categories_names:
    path_to_descriptions = os.path.join(path_to_sentences, category)
    text_names = os.listdir(path_to_descriptions)
    for counter,filename in enumerate(text_names):
        # read file
        with open(os.path.join(path_to_descriptions, filename), 'r') as my_file:
            text = my_file.read().splitlines()
        text = ' '.join(text)
        # remove formatting
        text = re.sub('\s+', ' ', text)
        list_text.append(text)
        # print progress
        if counter  == len(text_names) - 1:
            print(counter + 1, 'files processed')


list_text_cleaned = []
for i in range(len(categories_names)):
    text_cat = list_text[i * 30 : (i+1)*30]
    for counter,filename in enumerate(text_cat):
        my_tokens = clean_text_simple(filename, my_stopwords=stpwds,punct=punct)
        list_text_cleaned.append(my_tokens)
        # print progress
        if counter  == len(text_names) - 1:
            print(counter + 1, 'files cleaned')

list_val_text = []
for category in categories_names:
    path_to_descriptions = os.path.join(path_to_val_sentences, category)
    text_names = os.listdir(path_to_descriptions)
    for counter,filename in enumerate(text_names):
        # read file
        with open(os.path.join(path_to_descriptions, filename), 'r') as my_file:
            text = my_file.read().splitlines()
        text = ' '.join(text)
        # remove formatting
        text = re.sub('\s+', ' ', text)
        list_val_text.append(text)
        # print progress
        if counter  == len(text_names) - 1:
            print(counter + 1, 'files processed')


list_val_text_cleaned = []
for i in range(len(categories_names)):
    text_cat = list_val_text[i * 20 : (i+1)*20]
    for counter,filename in enumerate(text_cat):
        my_tokens = clean_text_simple(filename, my_stopwords=stpwds,punct=punct)
        list_val_text_cleaned.append(my_tokens)
        # print progress
        if counter  == len(text_names) - 1:
            print(counter + 1, 'files cleaned')


# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(list_text_cleaned)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in list_text_cleaned]
val_corpus = [dictionary.doc2bow(text) for text in list_val_text_cleaned]

# generate LDA model
num_topics = 100
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=20, dtype=np.float64)

total_distribution = []
for x in corpus:
    proba_distrib = ldamodel.get_document_topics(x, minimum_probability=0.0)
    total_distribution.append([proba_distrib[i][1] for i in range(num_topics)])

np.save('experiment/lda_train', total_distribution)

total_val_distribution = []
for x in val_corpus:
    proba_distrib = ldamodel.get_document_topics(x, minimum_probability=0.0)
    total_val_distribution.append([proba_distrib[i][1] for i in range(num_topics)])

np.save('experiment/lda_val', total_val_distribution)
