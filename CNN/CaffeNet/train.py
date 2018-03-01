import caffe
import numpy as np
from pylab import zeros, arange, subplots, plt, savefig

caffe.set_device(0)
caffe.set_mode_gpu()

# Config
niter = 160000
disp_interval = 20
test_interval = 100
test_iters = 0 # number of validating images  is  test_iters * batchSize
save_interval = 10000

# Name for training plot and snapshots
dataset_root = '../../data/ImageCLEF_Wikipedia/'
training_id = 'textTopicNet_word2vec_dim40_train_Wikipedia_ImageCLEF'

# Set solver configuration
solver = caffe.get_solver('solver.prototxt')

# Set plots data
train_loss = zeros(niter/disp_interval)
val_loss = zeros(niter/test_interval)
it_axes = (arange(niter) * disp_interval) + disp_interval
it_val_axes = (arange(niter) * test_interval) + test_interval
_, ax1 = subplots()
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss (r)')#, val loss (g)')
loss = np.zeros(niter)

# RUN TRAINING
for it in range(niter):
    solver.step(1)
    loss[it] = solver.net.blobs['loss'].data.copy()

    # PLOT
    if it % disp_interval == 0 or it + 1 == niter:
        loss_disp = 'loss=' + str(loss[it])
        print '%3d) %s' % (it, loss_disp)
        train_loss[it/disp_interval] = loss[it]
        ax1.plot(it_axes[0:it/disp_interval], train_loss[0:it/disp_interval], 'r')
        # ax1.set_ylim([230,280])
        plt.title(training_id)
        plt.ion()
        plt.grid(True)
        plt.show()
        plt.pause(0.001)

    # VALIDATE
    if it % test_interval == 0 and it > 0 and test_iters > 0:
        loss_val = 0
        for i in range(test_iters):
            solver.test_nets[0].forward()
            loss_val += solver.test_nets[0].blobs['loss'].data
        loss_val /= test_iters
        print("Val loss: {:.3f}".format(loss_val))
        val_loss[it/test_interval - 1] = loss_val
        ax1.plot(it_val_axes[0:it/test_interval], val_loss[0:it/test_interval], 'g')
        # ax1.set_ylim([230,280])
        plt.title(training_id)
        plt.ion()
        plt.grid(True)
        plt.show()
        plt.pause(0.001)

    # SAVE
    if it % save_interval == 0:
        title = dataset_root + 'models/training/' + training_id + '_' + str(it) + '.png'  # Save graph to disk
        savefig(title, bbox_inches='tight')
        # filename = '../../data/ImageCLEF_Wikipedia/models/' + training_id + '.caffemodel'
        # solver.net.save(filename)

#Save the learned weights at the end of the training
filename = dataset_root + 'models/CNNRegression/' + training_id + '.caffemodel'
solver.net.save(filename)
print("DONE")



