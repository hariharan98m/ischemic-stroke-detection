import numpy as np
import tensorflow as tf
from tensorflow._api.v1.layers import *
import tensorlayer as tl

def AttentionBlock(x, shortcut, num_filters, blockName, reuse = False):
    
    g1 = conv2d(shortcut, num_filters, kernel_size = (1, 1), padding = "same", name = blockName + "_g1", kernel_initializer = "he_normal")
    x1 = conv2d(x, num_filters, kernel_size = (1, 1), padding = "same", name = blockName + "_x1", kernel_initializer = "he_normal")

    g1_x1 = tf.add(g1, x1, name = blockName + "_g1x1")
    psi = tf.nn.relu(g1_x1, name = "_psi_relu")
    psi = conv2d(psi, 1, kernel_size = (1, 1), padding = "same", name = blockName + "_psi_conv", kernel_initializer = "he_normal")
    psi = tf.nn.sigmoid(psi, name = blockName + "_psi_sigmoid")
    psi = tf.broadcast_to(psi, x.shape, name = blockName + "_psi_final")
    x = tf.multiply(x, psi, name = blockName + "_attn_out")
    return x

def CustomConvLayer(x, num_filters, kernel_size = (3, 3), padding = "same", level = "l1", kernel_initializer = "he_normal", batchNorm = False):
    '''
    Tensorflow block to setup a levelwise conv layer for the UNet
    
    Arguments:
        x {input tensor} -- The input tensor
        num_filters {int} -- the total amount of filters to use
    
    Keyword Arguments:
        kernel_size {tuple} -- The size of the kernel to use (default: {(3, 3)})
        padding {str} -- the padding option, "Valid" or "same" (default: {"same"})
        level {str} -- the level of the block (default: {"l1"})
        kernel_initializer {str} -- the initializer for the kernel (default: {"he_normal"})
        batchNorm {bool} -- the option to choose batch normalization (default: {False})
    '''
    conv1 = conv2d(x, num_filters , kernel_size, activation= "relu", padding = padding, kernel_initializer = kernel_initializer, name = "conv1_" + level + "_1")
    if batchNorm:
        conv1 = batch_normalization(conv1)
    conv1 = conv2d(x, num_filters , kernel_size, activation= "relu", padding = padding, kernel_initializer = kernel_initializer, name = "conv" + level + "_2")
    if batchNorm:
        conv1 = batch_normalization(conv1)
    return conv1
    
def UNetWithAttention(x, reuse = False, n_out = 1, dropout_ph = 0.0, batchNorm = True):
    '''
    The UNET model
    '''
    nnx = int(x.shape[1])
    nny = int(x.shape[2])
    nnz = int(x.shape[3])
    print(" * Input: size of image: %d %d %d" % (nnx, nny, nnz))
    with tf.variable_scope("u_net_attention", reuse = reuse):
        # Encoder
        # level 1
        conv1 = CustomConvLayer(x, 64, (3, 3), padding = "same", level = "l1", kernel_initializer = "he_normal", batchNorm = batchNorm)
        pool1 = max_pooling2d(conv1, (2, 2), (2, 2), name = "pool1")
        pool1 = dropout(pool1, rate = dropout_ph * 0.5, name = "drop1")
        # level 2
        conv2 = CustomConvLayer(pool1, 128, (3, 3), padding = "same", level = "l2", kernel_initializer = "he_normal", batchNorm = batchNorm)
        pool2 = max_pooling2d(conv2, (2, 2), (2, 2), name = "pool2")
        pool2 = dropout(pool2, rate = dropout_ph, name = "drop2")
        # level 3
        conv3 = CustomConvLayer(pool2, 256, (3, 3), padding = "same", level = "l3", kernel_initializer = "he_normal", batchNorm = batchNorm)
        pool3 = max_pooling2d(conv3, (2, 2), (2, 2), name = "pool3")
        pool3 = dropout(pool3, rate = dropout_ph, name = "drop3")
        # level 4
        conv4 = CustomConvLayer(pool3, 512, (3, 3), padding = "same", level = "l4", kernel_initializer = "he_normal", batchNorm = batchNorm)
        pool4 = max_pooling2d(conv4, (2, 2), (2, 2), name = "pool4")
        pool4 = dropout(pool4, rate = dropout_ph, name = "drop4")
        # level 5 - feature level
        conv5 = CustomConvLayer(pool4, 1024, (3, 3), padding = "same", level = "l5", kernel_initializer = "he_normal", batchNorm = batchNorm)
        
        # Decoder
        # level 4
        up1 = conv2d_transpose(conv5, 512, (3, 3), (2, 2), padding = "same", name = "deconv_1", kernel_initializer = "he_normal")
        attn1 = AttentionBlock(up1, conv4, 512, blockName = "attn1", reuse = reuse)
        up1 = tf.concat([attn1, up1], axis = -1, name = "concat_1")
        up1 = dropout(up1, rate = dropout_ph, name = "drop5")
        uconv1 = CustomConvLayer(up1, 512, (3, 3), padding = "same", level = "ul1", kernel_initializer = "he_normal", batchNorm = batchNorm)
        # level 3
        up2 = conv2d_transpose(uconv1, 256, (3, 3), (2, 2), padding = "same", name = "deconv_2", kernel_initializer = "he_normal")
        attn2= AttentionBlock(up2, conv3, 256, blockName = "attn2", reuse = reuse)
        up2 = tf.concat([attn2, up2], axis = -1, name = "concat_2")
        up2 = dropout(up2, rate = dropout_ph, name = "drop6")
        uconv2 = CustomConvLayer(up2, 256, (3, 3), padding = "same", level = "ul2", kernel_initializer = "he_normal", batchNorm = batchNorm)
        # level 2
        up3 = conv2d_transpose(uconv2, 128, (3, 3), (2, 2), padding = "same", name = "deconv_3", kernel_initializer = "he_normal")
        attn3 = AttentionBlock(up3, conv2, 128, blockName = "attn3", reuse = reuse)
        up3 = tf.concat([attn3, up3], axis = -1, name = "concat_3")
        up3 = dropout(up3, rate = dropout_ph, name = "drop7")
        uconv3 = CustomConvLayer(up3, 128, (3, 3), padding = "same", level = "ul3", kernel_initializer = "he_normal", batchNorm = batchNorm)
        # level 1
        up4 = conv2d_transpose(uconv3, 64, (3, 3), (2, 2), padding = "same", name = "deconv_4", kernel_initializer = "he_normal")
        attn4 = AttentionBlock(up4, conv1, 64, blockName = "attn4", reuse = reuse)
        up4 = tf.concat([attn4, up4], axis = -1, name = "concat_4")
        up4 = dropout(up4, rate = dropout_ph, name = "drop8")
        uconv4 = CustomConvLayer(up4, 64, (3, 3), padding = "same", level = "ul4", kernel_initializer = "he_normal", batchNorm = batchNorm)
        # output
        outputs = conv2d(uconv4, n_out, (1, 1), activation = "sigmoid", padding = "same", name = "conv_output", kernel_initializer = "he_normal")
    
    return outputs

def UNet(x, reuse = False, n_out = 1, dropout_ph = 1.0):
    '''
    The UNET model
    '''
    
    _, nx, ny, nz = x.get_shape().as_list()
    nnx = int(x._shape[1])
    nny = int(x._shape[2])
    nnz = int(x._shape[3])
    print(" * Input: size of image: %d %d %d" % (nnx, nny, nnz))
    
    with tf.variable_scope("u_net", reuse = reuse):
        # Encoder
        # level 1
        conv1 = conv2d(x, 64, (3, 3), activation = "relu", padding = "same", name = "conv1_1", kernel_initializer = "he_normal")
        conv1 = conv2d(conv1, 64, (3, 3), activation = "relu", padding = "same", name = "conv1_2", kernel_initializer = "he_normal")
        pool1 = max_pooling2d(conv1, (2, 2), (2, 2), name = "pool1")
        pool1 = dropout(pool1, rate = dropout_ph * 0.5, name = "drop1")
        # level 2
        conv2 = conv2d(pool1, 128, (3, 3), activation = "relu", padding = "same", name = "conv2_1", kernel_initializer = "he_normal")
        conv2 = conv2d(conv2, 128, (3, 3), activation = "relu", padding = "same", name = "conv2_2", kernel_initializer = "he_normal")
        pool2 = max_pooling2d(conv2, (2, 2), (2, 2), name = "pool2")
        pool2 = dropout(pool2, rate = dropout_ph, name = "drop2")
        # level 3
        conv3 = conv2d(pool2, 256, (3, 3), activation = "relu", padding = "same", name = "conv3_1", kernel_initializer = "he_normal")
        conv3 = conv2d(conv3, 256, (3 ,3), activation = "relu", padding = "same", name = "conv3_2", kernel_initializer = "he_normal")
        pool3 = max_pooling2d(conv3, (2, 2), (2, 2), name = "pool3")
        pool3 = dropout(pool3, rate = dropout_ph, name = "drop3")
        # level 4
        conv4 = conv2d(pool3, 512, (3, 3), activation = "relu", padding = "same", name = "conv4_1", kernel_initializer = "he_normal")
        conv4 = conv2d(conv4, 512, (3, 3), activation = "relu", padding = "same", name = "conv4_2", kernel_initializer = "he_normal")
        pool4 = max_pooling2d(conv4, (2, 2), (2, 2), name = "pool4")
        pool4 = dropout(pool4, rate = dropout_ph, name = "drop4")
        # level 5 - feature level
        conv5 = conv2d(pool4, 1024, (3, 3), activation = "relu", padding = "same", name = "conv5_1", kernel_initializer = "he_normal")
        conv5 = conv2d(conv5, 1024, (3, 3), activation = "relu", padding = "same", name = "conv5_2", kernel_initializer = "he_normal")
        
        # Decoder
        # level 4
        up1 = conv2d_transpose(conv5, 512, (3, 3), (2, 2), padding = "same", name = "deconv_1", kernel_initializer = "he_normal")
        up1 = tf.concat([up1, conv4], axis = -1, name = "concat_1")
        up1 = dropout(up1, rate = dropout_ph, name = "drop5")
        uconv1 = conv2d(up1, 512, (3, 3), activation = "relu", padding = "same", name = "uconv1_1", kernel_initializer = "he_normal")
        uconv1 = conv2d(uconv1, 512, (3, 3), activation = "relu", padding = "same", name = "uconv1_2", kernel_initializer = "he_normal")
        # level 3
        up2 = conv2d_transpose(uconv1, 256, (3, 3), (2, 2), padding = "same", name = "deconv_2", kernel_initializer = "he_normal")
        up2 = tf.concat([up2, conv3], axis = -1, name = "concat_2")
        up2 = dropout(up2, rate = dropout_ph, name = "drop6")
        uconv2 = conv2d(up2, 256, (3, 3), activation = "relu", padding = "same", name = "uconv2_1", kernel_initializer = "he_normal")
        uconv2 = conv2d(uconv2, 256, (3, 3), activation = "relu", padding = "same", name = "uconv2_2", kernel_initializer = "he_normal")
        # level 2
        up3 = conv2d_transpose(uconv2, 128, (3, 3), (2, 2), padding = "same", name = "deconv_3", kernel_initializer = "he_normal")
        up3 = tf.concat([up3, conv2], axis = -1, name = "concat_3")
        up3 = dropout(up3, rate = dropout_ph, name = "drop7")
        uconv3 = conv2d(up3, 128, (3, 3), activation = "relu", padding = "same", name = "uconv3_1", kernel_initializer = "he_normal")
        uconv3 = conv2d(uconv3, 128, (3, 3), activation = "relu", padding = "same", name = "uconv3_2", kernel_initializer = "he_normal")
        # level 1
        up4 = conv2d_transpose(uconv3, 64, (3, 3), (2, 2), padding = "same", name = "deconv_4", kernel_initializer = "he_normal")
        up4 = tf.concat([up4, conv1], axis = -1, name = "concat_4")
        up4 = dropout(up4, rate = dropout_ph, name = "drop8")
        uconv4 = conv2d(up4, 64, (3, 3), activation = "relu", padding = "same", name = "uconv4_1", kernel_initializer = "he_normal")
        uconv4 = conv2d(uconv4, 64, (3, 3), activation = "relu", padding = "same", name = "uconv4_2", kernel_initializer = "he_normal")
        
        # output
        outputs = conv2d(uconv4, n_out, (1, 1), activation = "sigmoid", padding = "same", name = "conv_output", kernel_initializer = "he_normal")
    
    return outputs
