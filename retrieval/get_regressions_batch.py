import caffe
import numpy as np
from PIL import Image
import json

# Run in GPU
caffe.set_device(0)
caffe.set_mode_gpu()

with open('../LDA/train_dict_ImageCLEF_Wikipedia.json') as f:
    train_dict = json.load(f)
img_paths = train_dict.keys()

model = 'textTopicNet_train_Wikipedia_ImageCLEF'
output_file = open('../data/ImageCLEF_Wikipedia/regression_output/word2vec_dim40.txt', "w")

# load net
net = caffe.Net('../CNN/CaffeNet/deploy.prototxt', '../CNN/CaffeNet'+ model + '.caffemodel', caffe.TEST)

size = 227

# Reshape net
batch_size = 250
net.blobs['data'].reshape(batch_size, 3, size, size)

print 'Computing  ...'

count = 0
i = 0
while i < len(img_paths):
    indices = []
    if i % 100 == 0: print i

    # Fill batch
    for x in range(0, batch_size):
        if i > len(img_paths) - 1: break
        filename = '../data/ImageCLEF_Wikipedia/images/' + img_paths[i] + '.jpg'
        im = Image.open(filename)
        im_o = im
        im = im.resize((size, size), Image.ANTIALIAS)
        indices.append(img_paths[i])

        # Turn grayscale images to 3 channels
        if (im.size.__len__() == 2):
            im_gray = im
            im = Image.new("RGB", im_gray.size)
            im.paste(im_gray)

        #switch to BGR and substract mean
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= np.array((104, 117, 123))
        in_ = in_.transpose((2,0,1))

        net.blobs['data'].data[x,] = in_
        i += 1
    # run net and take scores
    net.forward()

    # Save results for each batch element
    for x in range(0,len(indices)):
        topic_probs = net.blobs['prob'].data[x]
        topic_probs_str = ''

        for t in topic_probs:
            topic_probs_str = topic_probs_str + ',' + str(t)

        output_file.write(indices[x].split(',')[0] + topic_probs_str + '\n')

output_file.close()

print "DONE"
print output_file_path


