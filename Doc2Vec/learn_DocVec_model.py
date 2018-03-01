import os,sys,sys,re
import json

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

import gensim
from gensim import utils, corpora, models
from gensim.corpora.wikicorpus import remove_markup
sys.path.append("../LDA")
from preprocess_text import preprocess

NUM_TOPICS = 40
db_dir  = '../data/ImageCLEF_Wikipedia/'
train_dict_path = '../LDA/train_dict_ImageCLEF_Wikipedia.json'

print '  '+sys.argv[0]
print '  Learns Doc2Vec topic model with '+str(NUM_TOPICS)+' topics from corpora on '+train_dict_path
print '  (...)'

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
        # if len(texts) == 100 : break
    sys.stdout.write(' Done!\n')

# turn our tokenized documents into a id <-> term dictionary
if not os.path.isfile('./dictionary.dict'):
    print 'Turn our tokenized documents into a id <-> term dictionary ...',
    sys.stdout.flush()
    dictionary = corpora.Dictionary(texts)
    dictionary.save('./dictionary.dict')
else:
    print 'Loading id <-> term dictionary from ./dictionary.dict ...',
    sys.stdout.flush()
    dictionary = corpora.Dictionary.load('./dictionary.dict')
print ' Done!'

# ignore words that appear in less than 20 documents or more than 50% documents
print "Filtering less and more frequent words ..."
dictionary.filter_extremes(no_below=20, no_above=0.5)
for i, text in enumerate(texts):
    filtered_text = []
    for w in text:
        if w in dictionary.token2id: filtered_text.append(w)
    texts[i] = filtered_text

del dictionary

for i in range(0,len(texts)): texts[i] = gensim.models.doc2vec.TaggedDocument(texts[i], [i])

# Learn the Doc2Vec model
print 'Learning the Doc2Vec model ...',
sys.stdout.flush()
doc2vecmodel = models.doc2vec.Doc2Vec(size=NUM_TOPICS, iter=25, window=8, workers=8)
doc2vecmodel.build_vocab(texts)
doc2vecmodel.train(texts, total_examples=doc2vecmodel.corpus_count, epochs=doc2vecmodel.iter)
doc2vecmodel.save('doc2vecmodel'+str(NUM_TOPICS)+'.doc2vec')
print ' Done!'

# Learn the Doc2Vec model
print 'Learning the Doc2Vec model ...',
sys.stdout.flush()
doc2vecmodel = models.doc2vec.Doc2Vec(size=300, iter=25, window=8, workers=8)
doc2vecmodel.build_vocab(texts)
doc2vecmodel.train(texts, total_examples=doc2vecmodel.corpus_count, epochs=doc2vecmodel.iter)
doc2vecmodel.save('doc2vecmodel'+str(300)+'.doc2vec')
print ' Done!'
