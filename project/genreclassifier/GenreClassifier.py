import tensorflow as tf
#from tensorflow.contrib.data import Dataset, Iterator
import matplotlib.pyplot as plt
import glob
import random
import sys


def input_parser(img_path, label_value):
    label = tf.one_hot(label_value, n_classes)
    img_file = tf.read_file(img_path)
    img_decoded = tf.cast(tf.image.decode_image(img_file, channels=1), tf.float32)
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
        'Electronic': 1,
        'Pop': 2,
        'HipHop': 3,
        'Metal': 4,
        'Rock': 5
    }

    n_classes = len(label_dict)
    learning_rate = 0.001
    num_steps = 4 # Size of data set
    batch_size = 128
    display_step = 100
    # Network Parameters
    #n_input = 784  # MNIST data input (img shape: 28*28)
    #n_classes = 10  # MNIST total classes (0-9 digits)
    dropout = 0.75  # Dropout, probability to keep units

    images, labels = load_and_shuffle_data()

    # create TensorFlow Dataset objects
    tf_data_set = tf.data.Dataset.from_tensor_slices((images, labels))
    tf_data_set = tf_data_set.map(input_parser)

    # validate_data(tf_data_set)

    # Set up batching
    batch_data_set = tf_data_set.batch(1)
    #iterator = batch_data_set.make_one_shot_iterator()
    #iterator = batch_data_set.make_initializable_iterator()
    # Reinitializable iterator
    iterator = tf.data.Iterator.from_structure(batch_data_set.output_types, batch_data_set.output_shapes)
    training_iterator_init_op = iterator.make_initializer(batch_data_set)

    #_data = tf.placeholder(tf.float32, [None, 128, 128, 1])
    #_labels = tf.placeholder(tf.float32, [None, n_classes])
    # Initialize the iterator
    #sess.run(iterator.initializer, feed_dict={_data: images, _labels: labels})

    # images, labels
    X, Y = iterator.get_next()

    # Verify batching
    # with tf.Session() as sess:
    #print (sess.run(Y))

    # Create a graph for training
    logits_train = conv_net(X, n_classes, dropout, reuse=False, is_training=True)
    # Create another graph for testing that reuse the same weights, but has
    # different behavior for 'dropout' (not applied).
    logits_test = conv_net(X, n_classes, dropout, reuse=True, is_training=False)

    # Define loss and optimizer (with train logits, for dropout to take effect)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_train, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Run the initializer
    sess.run(init)

    # Training cycle
    sess.run(training_iterator_init_op)
    for step in range(1, num_steps + 1):

        # # NOTE: For initializable iterator
        # try:
        #     # Run optimization
        #     sess.run(train_op)
        # except tf.errors.OutOfRangeError:
        #     # Reload the iterator when it reaches the end of the dataset
        #     sess.run(iterator.initializer, feed_dict={_data: images, _labels: labels})
        #     sess.run(train_op)

        try:
            sess.run(train_op)
            print("Running train op")
        except tf.errors.OutOfRangeError:
            print("Reached end of data set in train op")
            sess.run(training_iterator_init_op)

        if True: #step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            # (note that this consume a new batch of data)
            try:
                loss, acc = sess.run([loss_op, accuracy])
                print("Running loss op")
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
            except tf.errors.OutOfRangeError:
                print("Reached end of data set on loss step - done")
                sess.run(training_iterator_init_op)

    print("Optimization Finished!")





