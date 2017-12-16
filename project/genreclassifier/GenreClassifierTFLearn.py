import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tensorflow as tf
#from tensorflow.contrib.data import Dataset, Iterator
import numpy as np
import matplotlib.pyplot as plt
import glob
import random

# Bailed on this, have to use HDF5 format to handle big data that doesn't fit in RAM
# https://github.com/tflearn/tflearn/issues/8


def input_parser(img_path, label_value):
    label = tf.one_hot(label_value, NUM_CLASSES)
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=1)
    return img_decoded, label


def load_and_shuffle_data():
    data = []
    for file in glob.glob("data/*.png"):
        # extract label
        filename = file.split("/")[len(file.split("/")) - 1]
        genre = filename.split("_")[0]

        # if we can't extract the label from the image we should not train on it
        if genre not in label_dict:
            continue

        label_val = int(label_dict.get(genre))
        data.append((file, label_val))

    random.shuffle(data)
    image_paths = [x[0] for x in data]
    labels = [x[1] for x in data]
    return tf.constant(image_paths), tf.constant(labels)


# Validate our data, plot first element
def validate_data(tf_data):
    # Create TensorFlow Iterator object
    iterator = tf.data.Iterator.from_structure(tf_data.output_types, tf_data.output_shapes)
    next_element = iterator.get_next()
    # Create two initialization ops to switch between the datasets
    training_init_op = iterator.make_initializer(tf_data)
    with tf.Session() as sess:
        # initialize the iterator on the data
        sess.run(training_init_op)
        elem = sess.run(next_element)
        print("label:" + str(elem[1]))
        two_d_image = elem[0].reshape(128, 128)
        plt.imshow(two_d_image, cmap='Greys')
        plt.show()


def build_cnn():
    # Taken from https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_mnist.py
    network = input_data(shape=[None, 128, 128, 1], name='input')
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')
    return network


if __name__ == "__main__":

    # Our music genre labels
    label_dict = {
        'Classical': 0,
        'Electronic': 1,
        'Pop': 2,
        'HipHop': 3,
        'Metal': 4,
        'Rock': 5
    }
    NUM_CLASSES = len(label_dict)

    images, labels = load_and_shuffle_data()

    # Create TensorFlow Dataset objects
    tf_data = tf.data.Dataset.from_tensor_slices((images, labels))
    tf_data = tf_data.map(input_parser)

    validate_data(tf_data)

    cnn = build_cnn()

    # Training
    model = tflearn.DNN(cnn, tensorboard_verbose=0)
    model.fit({'input': X}, {'target': Y}, n_epoch=20,
              validation_set=({'input': testX}, {'target': testY}),
              snapshot_step=100, show_metric=True, run_id='cnn_genreclassification')