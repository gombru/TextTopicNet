import os,sys,sys,re
import json

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

import numpy as np
import gensim
from gensim import utils, corpora, models
sys.path.append("../LDA")
from preprocess_text import preprocess

print '  '+sys.argv[0]
print '  builds a dictionary with images paths as keys and Word2Vec space probability distributions as values'
print '  the embedding of a document is the TF-IDF weighted mean of its words'
print '  these probability distributions are then used as labels'
print '  for training a CNN to predict the semantic context in which images appear'
print '  (...)'

NUM_TOPICS = 40
db_dir  = '../data/ImageCLEF_Wikipedia/'
train_dict_path = '../LDA/train_dict_ImageCLEF_Wikipedia.json'

if not os.path.isdir(db_dir):
    sys.exit('ERR: Dataset folder '+db_dir+' not found!')

if not os.path.isfile(train_dict_path):
    sys.exit('ERR: Train dictionary file '+train_dict_path+' not found!')

with open(train_dict_path) as f:
    train_dict = json.load(f)

# load id <-> term dictionary
if not os.path.isfile('./dictionary.dict'):
    sys.exit('ERR: ID <-> Term dictionary file ./dictionary.dict not found!')

print 'Loading id <-> term dictionary from ./dictionary.dict ...',
sys.stdout.flush()
dictionary = corpora.Dictionary.load('./dictionary.dict')
print ' Done!'
# ignore words that appear in less than 20 documents or more than 50% documents
dictionary.filter_extremes(no_below=20, no_above=0.5)

# load Word2Vec model
if not os.path.isfile('word2vecmodel'+str(NUM_TOPICS)+'.word2vec'):
    sys.exit('ERR: Word2Vec model file ../Word2Vec/word2vecmodel'+str(NUM_TOPICS)+'.word2vec not found!')

print 'Loading Word2Vec model from file ../Word2Vec/word2vecmodel'+str(NUM_TOPICS)+'.word2vec ...',
sys.stdout.flush()
word2vecmodel = models.Word2Vec.load('./Word2Vec/word2vecmodel'+str(NUM_TOPICS)+'.word2vec')
print ' Done!'

print 'Loading TFIDF model from file ./tfidfmodel.tfidf ...',
sys.stdout.flush()
tfidfmodel = models.TfidfModel.load('./tfidfmodel.tfidf')
tfidfdictionary = gensim.corpora.Dictionary.load('./tfidfmodel_docs.dict')
print ' Done!'

# transform ALL documents into Word2Vec space
target_labels = {}
for img_path in train_dict.keys():
    with open(db_dir+train_dict[img_path]) as fp: raw = fp.read()

    tokens = preprocess(raw)

    # ignore words that appear in less than 20 documents or more than 50% documents
    filtered_tokens = []
    for word in tokens:
        if word in dictionary.token2id: filtered_tokens.append(word)

    # Compute Word2Vec embedding for each word in the text and take its mean
    embedding = np.zeros(NUM_TOPICS)
    num_words = 0

    vec = tfidfdictionary.doc2bow(filtered_tokens)
    vec_tfidf = tfidfmodel[vec]
    for tok in vec_tfidf:
        try:
            word_embedding = word2vecmodel[tfidfdictionary[tok[0]]]
            embedding += word_embedding * tok[1]
            # num_words += 1
        except:
            print "Word not in model: " + word
            continue

    # if num_words > 1: embedding /= num_words

    # L2 normalize the embedding
    if min(embedding) < 0: embedding = embedding - min(embedding)
    if sum(embedding) > 0: embedding = embedding / np.linalg.norm(embedding)

    target_labels[img_path] = embedding.tolist()

    sys.stdout.write('\r%d/%d text documents processed...' % (len(target_labels),len(train_dict.keys())))
    sys.stdout.flush()
sys.stdout.write(' Done!\n')

# save key,labels pairs into json format file
with open('./training_labels'+str(NUM_TOPICS)+'.json','w') as fp:
  json.dump(target_labels, fp)