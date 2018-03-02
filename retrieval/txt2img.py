# Retrieves nearest images given a text query and database representations and saves them in an given folder

import numpy as np
import operator
import os
from shutil import copyfile
from gensim import models

# Config
data = '../data/ImageCLEF_Wikipedia/regression_output/word2vec_dim40.txt'
model_name = '../Word2Vec/word2vecmodel40.word2vec'
embedding = 'word2vec' # 'word2vec' 'doc2vec'
num_topics = 40
num_results = 10
queries = ['history','tree','nature','mountain','river','man','animal','dog','king','computer']


# Load model
print("Loading " + embedding + " model ...")
if embedding == 'word2vec': model = models.Word2Vec.load(model_name)
elif embedding == 'doc2vec': model = models.Doc2Vec.load(model_name)

# Load dataset
print("Loading regressions from " + data + " ...")
database = {}
file = open(data, "r")
for line in file:
    d = line.split(',')
    regression_values = np.zeros(num_topics)
    for t in range(0, num_topics):
        regression_values[t] = d[t + 1]
    database[d[0]] = regression_values

# L2 normalize
for id in database:
    if min(database[id]) < 0: database[id] = database[id] - min(database[id])
    if sum(database[id]) > 0: database[id] = database[id] / np.linalg.norm(database[id])

for q in queries:
    print(q)
    results_path = "../data/ImageCLEF_Wikipedia/retrieval_results/" + data.split('/')[-1].split('.')[0] + "/" + q
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if embedding == 'word2vec': topics = model[q]
    elif embedding == 'doc2vec': topics = model.infer_vector(q)

    # L2 normalize
    if min(topics) < 0: topics = topics - min(topics)
    if sum(topics) > 0: topics = topics / np.linalg.norm(topics)

    # Create empty dict for distances
    distances = {}

    for id in database:
        distances[id] = np.dot(database[id],topics)

    #Sort dictionary
    distances = sorted(distances.items(), key=operator.itemgetter(1), reverse=True)

    # Get elements with min distances
    for idx,id in enumerate(distances):
        # Copy image results
        copyfile("../data/ImageCLEF_Wikipedia/images/" + id[0] + ".jpg", results_path + id[0].replace('/', '_') + ".jpg")
        if idx == num_results - 1: break