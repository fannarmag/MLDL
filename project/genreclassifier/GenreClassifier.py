import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import numpy as np
import matplotlib.pyplot as plt
import glob
import random


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

    # create TensorFlow Dataset objects
    tf_data = tf.data.Dataset.from_tensor_slices((images, labels))
    tf_data = tf_data.map(input_parser)

    # validate our data, plot first element
    # create TensorFlow Iterator object
    iterator = Iterator.from_structure(tf_data.output_types,
                                       tf_data.output_shapes)
    next_element = iterator.get_next()

    # create two initialization ops to switch between the datasets
    training_init_op = iterator.make_initializer(tf_data)

    with tf.Session() as sess:
        # initialize the iterator on the data
        sess.run(training_init_op)
        elem = sess.run(next_element)
        print("label:" + str(elem[1]))
        two_d_image = elem[0].reshape(128, 128)
        plt.imshow(two_d_image, cmap='Greys')
        plt.show()


