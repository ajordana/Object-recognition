import re
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag

def clean_text_simple(text, my_stopwords, punct, remove_stopwords=True, pos_filtering=True, stemming=True):
    text = text.lower()
    text = ''.join(l for l in text if l not in punct) # remove punctuation (preserving intra-word dashes)
    text = re.sub(' +',' ',text) # strip extra white space
    text = text.strip() # strip leading and trailing white space

    # tokenize (split based on whitespace)
    ### fill the gap (store results as 'tokens') ###
    tokens = wordpunct_tokenize(text)

    if pos_filtering == True:
        # POS tag and retain only nouns and adjectives
        tagged_tokens = pos_tag(tokens)
        tokens_keep = []
        for item in tagged_tokens:
            if (
            item[1] == 'NN' or
            item[1] == 'NNS' or
            item[1] == 'NNP' or
            item[1] == 'NNPS' or
            item[1] == 'JJ' or
            item[1] == 'JJS' or
            item[1] == 'JJR'
            ):
                tokens_keep.append(item[0])
        tokens = tokens_keep
    if remove_stopwords:
        # remove stopwords from 'tokens'
        ### fill the gap ###
        filtered_list = [w for w in tokens if not w in my_stopwords]
        tokens = filtered_list

    if stemming:
        # apply Porter's stemmer
        stemmer = PorterStemmer()
        tokens_stemmed = list()
        for token in tokens:
            tokens_stemmed.append(stemmer.stem(token))
        tokens = tokens_stemmed
    return(tokens)
