import numpy as np
import os, time
import tensorflow as tf
from tensorflow._api.v1.layers import *
from Preprocessing import *
from cv2 import cv2 as cv
from modelJunk import *
import dataset
import tensorlayer as tl

task = "all"
modelName = "./checkpoint/unet_attention_{}.ckpt".format(all)

# FOLDERS Initialization
save_dir = "checkpoint"
tl.files.exists_or_mkdir(save_dir)
tl.files.exists_or_mkdir("test_samples/{}".format(task))
FOLDER_NAME = [
    "samples_dwi/{}".format(task),
    "samples_flair/{}".format(task),
    "samples_t1/{}".format(task),
    "samples_t2/{}".format(task),
    "samples_gt/{}".format(task),
    "samples_out/{}".format(task)
]
for folderName in FOLDER_NAME:
    tl.files.exists_or_mkdir(folderName)

# Prepare Dataset
XTest = dataset.X_test_input
yTest = dataset.X_test_target[:, :, :, np.newaxis]
yTest = (yTest > 0).astype(int)

XTest_original = np.copy(XTest)
yTest_original = np.copy(yTest)

# Normalize data
XTest, yTest = NormalizeImageSet(XTest, yTest)

# Utility functions
def v1(X, y_, y, path1, path2, path3):
    """ show one slice with target """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    if y_.ndim == 2:
        y_ = y_[:,:,np.newaxis]
    assert X.ndim == 3
    print("saved")
    print(np.array(X[:,:,0,np.newaxis].shape))
    mi = np.nanmin(np.asarray(X[:,:,0]))
    ma = np.nanmax(np.asarray(X[:,:,0]))
    if mi != ma:
    	tl.vis.save_image(np.asarray(X[:,:,0]), path1)
    	tl.vis.save_image(np.asarray(X[:,:,1]), path2)
    	tl.vis.save_image(np.asarray(X[:,:,2]), path3)


def v2(X, y_, y, path4, path5, path6):
    """ show one slice with target """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    if y_.ndim == 2:
        y_ = y_[:,:,np.newaxis]
    assert X.ndim == 3
    print("saved")
    #print((np.asarray(X[:,:,3])).type)
    print(np.array(X[:,:,0,np.newaxis].shape))
    mi = np.nanmin(np.asarray(X[:,:,3]))
    ma = np.nanmax(np.asarray(X[:,:,3]))
    if mi != ma:
       tl.vis.save_image(np.asarray(X[:,:,3]),path4)
       tl.vis.save_image(np.asarray(y_), path5)
       tl.vis.save_image(np.asarray(y), path6)

# hyperparams
batch_size = 10
lr = 0.0001
n_epoch = 120
evalEpoch = 100
dropout_value = 0.05

# Load model and test out
tf.reset_default_graph()
tf.reset_default_graph()
with tf.device("/cpu:0"):
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True))
    with tf.device("/gpu:0"):

        XInput = tf.placeholder(dtype = tf.float32, shape = [batch_size, 240, 240, 4], name = "input_image")
        yInput = tf.placeholder(dtype = tf.float32, shape = [batch_size, 240, 240, 1], name = "ground_truth")
        dropout_ph = tf.placeholder_with_default(tf.cast(dropout_value, dtype = tf.float32), shape = [], name = "dropout_probability")

        yOutput = UNetWithAttention(XInput, reuse = False, n_out = 1, dropout_ph = dropout_ph, batchNorm = True)

        yOutputTest = UNetWithAttention(XInput, reuse = True, n_out = 1, dropout_ph = dropout_ph, batchNorm = True)

        # defining loss
        loss = 1 - tl.cost.dice_coe(yOutput, yInput, axis = [0, 1, 2, 3])
        testLoss = 1 - tl.cost.dice_coe(yOutputTest, yInput, axis = [0, 1, 2, 3])


    trainVar = tl.layers.get_variables_with_name('u_net', True, True)
    with tf.device('/gpu:0'):
        with tf.variable_scope('learning_rate'):
            lrVar = tf.Variable(lr, trainable = False)

        optimizer = tf.train.AdamOptimizer(lrVar).minimize(loss, var_list = trainVar)

    sess.run(tf.global_variables_initializer())
    # load existing model if exist
    saver = tf.train.Saver(var_list = trainVar)

saver.restore(sess, tf.train.latest_checkpoint("./checkpoint"))
# EVALUATE
total_loss, n_batch = 0, 0
for batch, unnormalizedBatch in zip(tl.iterate.minibatches(inputs=XTest, targets=yTest, batch_size=batch_size, shuffle=False), tl.iterate.minibatches(inputs=XTest_original, targets=yTest_original, batch_size=batch_size, shuffle=False)):
    b_images, b_labels = batch
    u_b_images, u_b_label = unnormalizedBatch
    _loss, out = sess.run([testLoss, yOutputTest],{XInput: b_images, yInput: b_labels})
    total_loss += _loss
    n_batch += 1
    for i in range(batch_size):
            count+=1
            v1(u_b_images[i], b_labels[i], out[i], "samples_dwi/{}/test_{}.png".format(task,count), "samples_flair/{}/test_{}.png".format(task,count), "samples_t1/{}/test{}.png".format(task,count))
            v2(u_b_images[i], b_labels[i], out[i], "samples_t2/{}/test_{}.png".format(task,count), "samples_gt/{}/test_{}.png".format(task,count), "samples_out/{}/test{}.png".format(task,count))


print(" **"+" "*17+"test -- loss: %f" %(total_loss/n_batch))
print(" task: {}".format(task))
