import tensorflow as tf
#from tensorflow.contrib.data import Dataset, Iterator
import matplotlib.pyplot as plt
import glob
import random
import sys

# See:
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_DataManagement/tensorflow_dataset_api.py


def input_parser(img_path, label_value):
    label = tf.one_hot(label_value, n_classes)
    img_file = tf.read_file(img_path)
    img_decoded = tf.cast(tf.image.decode_image(img_file, channels=1), tf.float32)
    return img_decoded, label


def load_and_shuffle_data(data_folder):
    data = []
    for file in glob.glob("data/" + data_folder + "/*/**.png"):
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
    return image_paths, labels


def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 128, 128, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out


# Validate our data, plot first element
def validate_data(data_set):
    # Create TensorFlow Iterator object
    iterator = tf.data.Iterator.from_structure(data_set.output_types, data_set.output_shapes)
    next_element = iterator.get_next()
    training_init_op = iterator.make_initializer(data_set)
    with tf.Session() as sess:
        # Initialize the iterator on the data
        sess.run(training_init_op)
        elem = sess.run(next_element)
        print("label:" + str(elem[1]))
        two_d_image = elem[0].reshape(128, 128)
        plt.imshow(two_d_image, cmap='Greys')
        plt.show()


if __name__ == "__main__":

    sess = tf.Session()

    # Our music genre labels
    label_dict = {
        'Classical': 0,
        'Techno': 1,
        'Pop': 2,
        'HipHop': 3,
        'Metal': 4,
        'Rock': 5
    }

    # Parameters
    n_classes = len(label_dict)
    learning_rate = 0.001
    batch_size = 100
    num_steps = 100000 / batch_size    # Size of data set
    display_step = 10
    dropout = 0.75  # Dropout, probability to keep units

    # Load train data
    images_train, labels_train = load_and_shuffle_data("train")
    tf_data_set_train = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
    tf_data_set_train = tf_data_set_train.map(input_parser)

    # Load test data
    #images_test, labels_test = load_and_shuffle_data("test")
    #tf_data_set_test = tf.data.Dataset.from_tensor_slices((images_test, labels_test))
    #tf_data_set_test = tf_data_set_test.map(input_parser)

    # Uncomment to check out the data we are working on
    # validate_data(tf_data_set)

    # Set up batching for data sets
    batch_train_data_set = tf_data_set_train.batch(batch_size)
    #batch_test_data_set = tf_data_set_test.batch(batch_size)

    # Iterators (reinitializable)
    train_iterator = tf.data.Iterator.from_structure(batch_train_data_set.output_types, batch_train_data_set.output_shapes)
    training_iterator_init_op = train_iterator.make_initializer(batch_train_data_set)
    #test_iterator = tf.data.Iterator.from_structure(tf_data_set_test.output_types, tf_data_set_test.output_shapes)
    #test_iterator_init_op = test_iterator.make_initializer(tf_data_set_test)

    # Initialize iterators
    sess.run(training_iterator_init_op)
    #sess.run(test_iterator_init_op)

    # images, labels
    X_train, Y_train = train_iterator.get_next()
    #X_test, Y_test = test_iterator.get_next()

    # Verify batching
    # with tf.Session() as sess:
    #print (sess.run(Y_test))

    # Create a graph for training
    logits_train = conv_net(X_train, n_classes, dropout, reuse=False, is_training=True)
    # Create another graph for testing that reuse the same weights, but has
    # different behavior for 'dropout' (not applied).
    logits_test = conv_net(X_train, n_classes, dropout, reuse=True, is_training=False)

    # Define loss and optimizer (with train logits, for dropout to take effect)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_train, labels=Y_train))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(Y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Run the initializer
    sess.run(init)

    # Training cycle
    for step in range(1, num_steps + 1):

        try:
            sess.run(train_op)
        except tf.errors.OutOfRangeError:
            print("Reached end of data set in train op - reinitializing iterator")
            #sess.run(training_iterator_init_op)
            sys.exit(0)

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            # (note that this consume a new batch of data)
            try:
                loss, acc = sess.run([loss_op, accuracy])
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
            except tf.errors.OutOfRangeError:
                print("Reached end of data set on loss step - reinitializing iterator")
                #sess.run(training_iterator_init_op)
                #sess.run(test_iterator_init_op)
                sys.exit(0)

    print("Optimization Finished!")





