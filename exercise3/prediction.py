import numpy as np
import tensorflow as tf
import operator
from utils import buildNetwork, load_cifar


# Let's start a Session
sess = tf.Session()


batch_size = 1
HEIGHT = 32
WIDTH = 32
CHANNELS = 3
NUM_CLASSES = 10


data_path = "data/CIFAR-10/"

train_samples, train_labels, val_samples, val_labels = load_cifar(data_path)


# TODO add placeholder for inputs
inputs = tf.placeholder(tf.float32,name='input',shape=[batch_size,WIDTH,HEIGHT,CHANNELS])

logits = buildNetwork(inputs, batch_size)


# TODO restore the saved checkpoints ./checkpoints/model.ckpt
saver = tf.train.Saver()
saver.restore(sess, "./checkpoints/model.ckpt")




# TODO add an appropriate op to convert the logits into probabilities
probs = tf.nn.softmax(logits)


label_to_name = ['airplane', 'automobile', 'bird', 'cat', 'deer' , 'dog', 'frog', 'horse', 'ship', 'truck']

# TODO plot the first five *validation* images 
for index in range(5):
#    index=2
    feed_dict = {inputs: [val_samples[index]]}
    classification = sess.run(probs, feed_dict)
    print(classification)
    idx = max(enumerate(classification[0]), key=operator.itemgetter(1))[0]
    #plt.figure(figsize=(1,1))
    print('actual: '+label_to_name[val_labels[index]]+' predicted: '+ label_to_name[idx])
    #plt.imshow(val_samples[index])

# TODO for each image, print the predicted class and the probability vector for all classes

