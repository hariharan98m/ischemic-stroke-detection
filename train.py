import numpy as np
import os, time
import tensorflow as tf
from tensorflow._api.v1.layers import *
from Preprocessing import *
from cv2 import cv2 as cv
from modelJunk import *
import dataset

task = "all"
modelName = "./checkpoint/unet_attention_{}.ckpt".format(all)

# FOLDERS Initialization
save_dir = "checkpoint"
tl.files.exists_or_mkdir(save_dir)
tl.files.exists_or_mkdir("samples/{}".format(task))

# Prepare Dataset
XTrain = dataset.X_train_input
yTrain = dataset.X_train_target[:, :, :, np.newaxis]
XTest = dataset.X_dev_input
yTest = dataset.X_dev_target[:, :, :, np.newaxis]
yTrain = (yTrain > 0).astype(int)
yTest = (yTest > 0).astype(int)

# Normalize data
XTrain, yTrain = NormalizeImageSet(XTrain, yTrain)
XTest, yTest = NormalizeImageSet(XTest, yTest)

# hyperparams
batch_size = 10
lr = 0.0001
n_epoch = 120
evalEpoch = 100
dropout_value = 0.05

# Train Session Setups
tf.reset_default_graph()
with tf.device("/cpu:0"):
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True))
    with tf.device("/gpu:0"):

        XInput = tf.placeholder(dtype = tf.float32, shape = [batch_size, 240, 240, 4], name = "input_image")
        yInput = tf.placeholder(dtype = tf.float32, shape = [batch_size, 240, 240, 1], name = "ground_truth")
        dropout_ph = tf.placeholder_with_default(tf.cast(dropout_value, dtype = tf.float32), shape = [], name = "dropout_probability")

        yOutput = UNetWithAttention(XInput, reuse = False, n_out = 1, dropout_ph = dropout_ph)

        yOutputTest = UNetWithAttention(XInput, reuse = True, n_out = 1, dropout_ph = dropout_ph)

        # defining loss
        loss = 1 - tl.cost.dice_coe(yOutput, yInput, axis = [0, 1, 2, 3])
        testLoss = 1 - tl.cost.dice_coe(yOutputTest, yInput, axis = [0, 1, 2, 3])


    trainVar = tl.layers.get_variables_with_name('u_net_attention', True, True)
    with tf.device('/gpu:0'):
        with tf.variable_scope('learning_rate'):
            lrVar = tf.Variable(lr, trainable = False)

        optimizer = tf.train.AdamOptimizer(lrVar).minimize(loss, var_list = trainVar)

    sess.run(tf.global_variables_initializer())
    # sess.run(tf.random.set_random_seed(1))
    # load existing model if exist
    saver = tf.train.Saver(var_list = trainVar)

# TRAINING
# Train
try:
    if os.path.exists("./checkpoint/u_net_attention_{}.ckpt".format(task)):
        saver.restore(sess, "./checkpoint/u_net_attention_{}.ckpt".format(task))
        print("[+] Presaved model restored")
    else:
        print("[+] Preloaded model not available")
    for epoch in range(0, n_epoch + 1):
        epochTime = time.time()
        total_loss, n_batch = 0, 0
        for batch in tl.iterate.minibatches(inputs=XTrain, targets=yTrain, batch_size=batch_size, shuffle=True):
            images, labels = batch
            step_time = time.time()
            ## data augumentation for a batch of Flair, T1, T1c, T2 images
            # and label maps synchronously.
            data = tl.prepro.threading_data([_ for _ in zip(images[:,:,:,0, np.newaxis],
                    images[:,:,:,1, np.newaxis], images[:,:,:,2, np.newaxis],
                    images[:,:,:,3, np.newaxis], labels)],
                    fn = DistortImages) # (10, 5, 240, 240, 1)
            b_images = data[:,0:4,:,:,:]  # (10, 4, 240, 240, 1)
            b_labels = data[:,4,:,:,:]
            b_images = b_images.transpose((0,2,3,1,4))
            b_images.shape = (batch_size, 240, 240, 4)

            ## update network
            _, _loss, out = sess.run([optimizer, loss, yOutput], feed_dict = {XInput:b_images, yInput:b_labels})
            total_loss += _loss
            n_batch += 1
            print("Epoch %d step %d 1 - dice: %f time taken: %fsecs          " %(epoch, n_batch, _loss, time.time() - step_time), end = "\r")

            if np.isnan(_loss):
                exit(" ** NaN loss found during training, stop training")

            if np.isnan(out).any():
                exit(" ** NaN found in output images during training, stop training")

        print("\n ** Epoch [%d/%d] trained ==> loss: %f -- time taken: %fsec" %(epoch, n_epoch, total_loss/n_batch, time.time() - epochTime))

        ## save a predition of training set
        for i in range(batch_size):
            if np.max(b_images[i]) > 0:
                VisualizeImageWithPrediction(b_images[i], b_labels[i], out[i], "samples/{}/train_{}.png".format(task, epoch))
                break
            elif i == batch_size-1:
                VisualizeImageWithPrediction(b_images[i], b_labels[i], out[i], "samples/{}/train_{}.png".format(task, epoch))

        # EVALUATE
        total_loss, n_batch = 0, 0
        for batch in tl.iterate.minibatches(inputs=XTest, targets=yTest, batch_size=batch_size, shuffle=True):
            b_images, b_labels = batch
            _loss, out = sess.run([testLoss, yOutputTest],{XInput: b_images, yInput: b_labels})
            total_loss += _loss
            n_batch += 1

        print(" **"+" "*17+"test -- loss: %f" %(total_loss/n_batch))
        print(" task: {}".format(task))
        ## save a predition of test set
        for i in range(batch_size):
            if np.max(b_images[i]) > 0:
                VisualizeImageWithPrediction(b_images[i], b_labels[i], out[i], "samples/{}/test_{}.png".format(task, epoch))
                break
            elif i == batch_size-1:
                VisualizeImageWithPrediction(b_images[i], b_labels[i], out[i], "samples/{}/test_{}.png".format(task, epoch))

        savePath = saver.save(sess, "./checkpoint/u_net_attention_{}.ckpt".format(task))
        print("[+] Checkpoint saved in ", savePath)

except KeyboardInterrupt:
    savePath = saver.save(sess, "./checkpoint/u_net_attention_{}.ckpt".format(task))
    print("[+] KeyBoard Interrupted. Model Saving")

sess.close()
