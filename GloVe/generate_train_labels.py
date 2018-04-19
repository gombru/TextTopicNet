import os,sys,sys,re
import json
from glove import Glove
import numpy as np
from gensim import corpora
sys.path.append("../LDA")
from preprocess_text import preprocess

print ('  builds a dictionary with images paths as keys and GloVe space probability distributions as values')
print ('  these probability distributions are then used as labels')
print ('  for training a CNN to predict the semantic context in which images appear')
print ('  (...)')

NUM_TOPICS_arr = [40,300]

for NUM_TOPICS in NUM_TOPICS_arr:
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

    print ('Loading id <-> term dictionary from ./dictionary.dict ...',)
    sys.stdout.flush()
    dictionary = corpora.Dictionary.load('./dictionary.dict')
    print (' Done!')
    # ignore words that appear in less than 20 documents or more than 50% documents
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    # load GloVe model
    if not os.path.isfile('glovemodel'+str(NUM_TOPICS)+'.glove'):
        sys.exit('ERR: GloVe model file ./word2vecmodel'+str(NUM_TOPICS)+'.glove not found!')

    print ('Loading GloVe model from file ./glovemodel'+str(NUM_TOPICS)+'.glove ...',)
    sys.stdout.flush()
    glovemodel = Glove.load('glovemodel'+str(NUM_TOPICS)+'.glove')
    print (' Done!')

    # transform ALL documents into GloVe space
    target_labels = {}
    for img_path in train_dict.keys():
        with open(db_dir+train_dict[img_path]) as fp: raw = fp.read()

        tokens = preprocess(raw)

        # ignore words that appear in less than 20 documents or more than 50% documents
        filtered_tokens = []
        for word in tokens:
            if word in dictionary.token2id: filtered_tokens.append(word)

        # Compute GloVe embedding for each word in the text and take its mean
        embedding = np.zeros(NUM_TOPICS)
        num_words = 0
        for word in filtered_tokens:
            try:
                embedding += glovemodel.word_vectors[glovemodel.dictionary[word]]
                num_words += 1
            except:
                print ("Word not in model: " + word)
                continue
        if num_words > 1: embedding /= num_words

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
