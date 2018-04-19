# Run with python3, pickle has a bug in python 2.7 saving and loadinh the model due to file size

import os,sys,sys,re
import json
from glove import Corpus, Glove
from gensim import utils, corpora, models
sys.path.append("../LDA")
from preprocess_text import preprocess

NUM_TOPICS = 40
db_dir  = '../data/ImageCLEF_Wikipedia/'
train_dict_path = '../LDA/train_dict_ImageCLEF_Wikipedia.json'

# print '  '+sys.argv[0]
print ('  Learns GloVe topic model with '+str(NUM_TOPICS)+' topics from corpora on '+train_dict_path)
print ('  (...)')

img_dir = db_dir+'images/'
xml_dir = db_dir+'metadata/'

if not os.path.isdir(db_dir):
    sys.exit('ERR: Dataset folder '+db_dir+' not found!')

if not os.path.isdir(img_dir):
    sys.exit('ERR: Dataset images folder '+img_dir+' not found!')

if not os.path.isdir(xml_dir):
    sys.exit('ERR: Dataset metadata folder '+xml_dir+' not found!')

if not os.path.isfile(train_dict_path):
    sys.exit('ERR: Train dictionary file '+train_dict_path+' not found!')

with open(train_dict_path) as f:
    train_dict = json.load(f)

if not os.path.isfile('./dictionary.dict') or not os.path.isfile('./bow.mm'):
    # list for tokenized documents in loop
    texts = []
    for text_path in train_dict.values():
        with open(db_dir+text_path) as f: raw = f.read()
        # add tokens to corpus list
        texts.append(preprocess(raw))
        sys.stdout.write('\rCreating a list of tokenized documents: %d/%d documents processed...' % (len(texts),len(train_dict.values())))
        sys.stdout.flush()
        #if len(texts) == 100 : break
    sys.stdout.write(' Done!\n')

# turn our tokenized documents into a id <-> term dictionary
if not os.path.isfile('./dictionary.dict'):
    print ('Turn our tokenized documents into a id <-> term dictionary ...',)
    sys.stdout.flush()
    dictionary = corpora.Dictionary(texts)
    dictionary.save('./dictionary.dict')
else:
    print ('Loading id <-> term dictionary from ./dictionary.dict ...',)
    sys.stdout.flush()
    dictionary = corpora.Dictionary.load('./dictionary.dict')
print (' Done!')

# ignore words that appear in less than 20 documents or more than 50% documents
print ("Filtering less and more frequent words ...")
dictionary.filter_extremes(no_below=20, no_above=0.5)
for i, text in enumerate(texts):
    filtered_text = []
    for w in text:
        if w in dictionary.token2id: filtered_text.append(w)
    texts[i] = filtered_text

del dictionary

print('Creating corpus ...')
corpus = Corpus()
corpus.fit(texts, window=8) #10
print('Dict size: %s' % len(corpus.dictionary))
print('Collocations: %s' % corpus.matrix.nnz)

# Learn the GloVe model
print ('Learning the GloVe model ...',)
sys.stdout.flush()
glove = Glove(no_components=NUM_TOPICS, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=25, no_threads=8, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('glovemodel'+str(NUM_TOPICS)+'.glove')
print (' Done!')

# Learn the GloVe model
print ('Learning the GloVe model ...',)
sys.stdout.flush()
glove = Glove(no_components=300, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=25, no_threads=8, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('glovemodel'+str(300)+'.glove')
print (' Done!')
