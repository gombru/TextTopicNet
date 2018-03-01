import os,sys,sys,re
import json

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

import numpy as np
import gensim
from gensim import utils, corpora, models
from gensim.corpora.wikicorpus import remove_markup
sys.path.append("../LDA")
from preprocess_text import preprocess

print '  '+sys.argv[0]
print '  builds a dictionary with images paths as keys and Doc2Vec space probability distributions as values'
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

# load document-term matrix
if not os.path.isfile('./bow.mm'):
    sys.exit('ERR: Document-term matrix file ./bow.mm not found!')

print 'Loading document-term matrix from ./bow.mm ...',
sys.stdout.flush()
corpus = gensim.corpora.MmCorpus('./bow.mm')
print ' Done!'

# load Doc2Vec model
if not os.path.isfile('doc2vecmodel'+str(NUM_TOPICS)+'.doc2vec'):
    sys.exit('ERR: Doc2Vec model file ./doc2vecmodel'+str(NUM_TOPICS)+'.doc2vec not found!')

print 'Loading Doc2Vec model from file ./doc2vecmodel'+str(NUM_TOPICS)+'.doc2vec ...',
sys.stdout.flush()
doc2vecmodel = models.Doc2Vec.load('doc2vecmodel'+str(NUM_TOPICS)+'.doc2vec')
print ' Done!'

# transform ALL documents into Doc2Vec space
target_labels = {}
for img_path in train_dict.keys():

    with open(db_dir+train_dict[img_path]) as fp: raw = fp.read()

    tokens = preprocess(raw)

    # ignore words that appear in less than 20 documents or more than 50% documents
    filtered_tokens = []
    for word in tokens:
        if word in dictionary.token2id: filtered_tokens.append(word)

    embedding = doc2vecmodel.infer_vector(filtered_tokens)

    # L2 normalize the embedding
    if min(embedding) < 0: embedding = embedding - min(embedding)
    if sum(embedding) > 0: embedding = embedding / np.linalg.norm(embedding)

    target_labels[img_path] = embedding

    sys.stdout.write('\r%d/%d text documents processed...' % (len(target_labels),len(train_dict.keys())))
    sys.stdout.flush()
sys.stdout.write(' Done!\n')

# save key,labels pairs into json format file
with open('./training_labels'+str(NUM_TOPICS)+'.json','w') as fp:
  json.dump(target_labels, fp)
