import numpy as np
import tensorflow as tf

import os
import urllib.request
import tarfile
import pickle


# Loads Cifar From The Internet To Your Disk
def download_cifar(data_path, data_url):
    filename = data_url.split('/')[-1]
    file_path = os.path.join(data_path, filename)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    print("Downloading CIFAR10")
    file_path, _ = urllib.request.urlretrieve(url=data_url,filename=file_path)

    print("Download finished. Extracting files.")
    tarfile.open(name=file_path, mode="r:gz").extractall(data_path)
    print("Done.")



# Loads Cifar Data From Your Disk Into Memeory
def load_cifar(data_path):
    train_samples = []
    train_labels = []
    val_samples = []
    val_labels = []

    for i in range(5):
        #  äwith open(data_path + 'cifar-10-batches-py/data_batch_' + str(i+1), 'rb') as fo:
        with open('/media/zimmerw/Daten/Daten/walter.zimmer/Documents/private/master/2_semester_31_ECTS/Masterpraktikum_10ECTS/Praktikum_Perception_and_Learning_in_Robotics_and_Augmented_Reality_10_ECTS/exercises/plarr17/exercise3/data/CIFAR-10/cifar-10-batches-py/data_batch_' + str(i+1), 'rb') as fo:

            dict = pickle.load(fo, encoding='bytes')
            train_samples.append(dict[b'data'])
            train_labels.append(dict[b'labels'])

    #  with open(data_path + 'cifar-10-batches-py/test_batch', 'rb') as fo:
    with open('/media/zimmerw/Daten/Daten/walter.zimmer/Documents/private/master/2_semester_31_ECTS/Masterpraktikum_10ECTS/Praktikum_Perception_and_Learning_in_Robotics_and_Augmented_Reality_10_ECTS/exercises/plarr17/exercise3/data/CIFAR-10/cifar-10-batches-py/test_batch', 'rb') as fo:

        dict = pickle.load(fo, encoding='bytes')
        val_samples.append(dict[b'data'])
        val_labels.append(dict[b'labels'])        
            

    train_samples = np.array(train_samples).reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1]) / 255.
    val_samples = np.array(val_samples).reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1]) / 255.

    train_labels = np.array(train_labels).reshape(-1)
    val_labels = np.array(val_labels).reshape(-1)

    return train_samples, train_labels, val_samples, val_labels



# Builds The Network
def buildNetwork(inputs, batch_size, NUM_CLASSES=10,keep_prob=1.0):
    def conv_layer(x, num_channels_out, spatial_stride=2):
        """ Layer for 3x3 convolutions.

        Args:
          x: A 4-D float32 Tensor with shape [num_images, height, width, num_channels].
          num_channels_out: An integer. The number of output channels we'll compute
            (with one convolutional filter per output channel).
          spatial_stride: A positive integer. If this is 1, we obtain a typical
            convolution; if 2, we'll have one output pixel for every *two* input
            pixels; and so on.

        Returns:
          A 4-D float32 Tensor with shape [num_images, new_height, new_width, num_channels_out].
        """
        num_channels_in = x.get_shape().as_list()[-1]
        conv_strides = [1, spatial_stride, spatial_stride, 1]
        W_shape = [5, 5, num_channels_in, num_channels_out]
        W = tf.Variable(tf.truncated_normal(
            W_shape,
            mean=0.0,
            stddev=5e-2,
            dtype=tf.float32,
            seed=None,
            name=None
        ))
        b = tf.Variable(tf.zeros([num_channels_out]))
        conv = tf.nn.conv2d(x, W, conv_strides, 'SAME')
        conv_with_bias = conv + b
        return conv_with_bias

    def linear_layer(x, num_outputs):
        """ A simple linear layer.

        Args:
          x: A 2-D float32 Tensor with shape [num_images, num_inputs]. (Each
            image is represented by a vector with dimensionality num_inputs.)
          num_outputs: An integer.

        Returns:
          A 2-D float32 Tensor with shape [num_images, num_outputs].
        """
        num_inputs = x.get_shape().as_list()[-1]
        W_shape = [num_inputs, num_outputs]
        W = tf.Variable(tf.truncated_normal(
            W_shape,
            mean=0.0,
            stddev=5e-2,
            dtype=tf.float32,
            seed=None,
            name=None
        ))
        b = tf.Variable(tf.zeros([num_outputs]))
        ret = tf.nn.xw_plus_b(x, W, b)
        return ret
        
    x = conv_layer(inputs, num_channels_out=64)
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                     padding='SAME', name='pool1')
    x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                name='norm1')
#    x= tf.nn.dropout(x,keep_prob)

    x = conv_layer(x, num_channels_out=32, spatial_stride=1)
    x = tf.nn.relu(x)
    x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
    x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    x= tf.nn.dropout(x,keep_prob)
    x = conv_layer(x, num_channels_out=64, spatial_stride=1)
    x = tf.nn.relu(x)
 #   x= tf.nn.dropout(x,keep_prob)
    x = tf.reshape(x, [batch_size, -1])

    x = linear_layer(x, num_outputs=384)
    x = tf.nn.relu(x)
    x= tf.nn.dropout(x,keep_prob)
    x = linear_layer(x, num_outputs=192)
    x = tf.nn.relu(x)
  #  x= tf.nn.dropout(x,keep_prob)

    logits = linear_layer(x, num_outputs=NUM_CLASSES)

    return logits



