import tensorflow as tf
#from tensorflow.contrib.data import Dataset, Iterator
import matplotlib.pyplot as plt
import glob
import random
import sys
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils

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


# Adapted from the medium guy
def conv_net2(x, n_classes, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = tf.reshape(x, shape=[-1, 128, 128, 1])
        conv = tf.layers.conv2d(x, 64, 2, activation=tf.nn.relu)
        conv = tf.layers.max_pooling2d(conv, 2, 2)
        conv = tf.layers.conv2d(conv, 128, 2, activation=tf.nn.relu)
        conv = tf.layers.max_pooling2d(conv, 2, 2)
        conv = tf.layers.conv2d(conv, 256, 2, activation=tf.nn.relu)
        conv = tf.layers.max_pooling2d(conv, 2, 2)
        conv = tf.layers.conv2d(conv, 512, 2, activation=tf.nn.relu)
        conv = tf.layers.max_pooling2d(conv, 2, 2)
        conv = tf.contrib.layers.flatten(conv)
        conv = tf.layers.dense(conv, 1024)
        conv = tf.layers.dropout(conv, rate=dropout, training=is_training)
        out = tf.layers.dense(conv, n_classes)
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
    images_train, labels_train = load_and_shuffle_data("training")
    tf_data_set_train = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
    tf_data_set_train = tf_data_set_train.map(input_parser)

    # Load validation data
    images_validation, labels_validation = load_and_shuffle_data("validation")
    tf_data_set_validation = tf.data.Dataset.from_tensor_slices((images_validation, labels_validation))
    tf_data_set_validation = tf_data_set_validation.map(input_parser)

    # Uncomment to check out the data we are working on
    #validate_data(tf_data_set_validation)

    # Set up batching for data sets
    batch_train_data_set = tf_data_set_train.batch(batch_size)
    batch_validation_data_set = tf_data_set_validation.batch(batch_size)

    # Iterator (feedable and reinitializable)
    # The train and validation data sets have the same shape, can use either one here
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, batch_train_data_set.output_types, batch_train_data_set.output_shapes)
    X, Y = iterator.get_next()  # images, labels
    # Iterators for the different data sets
    training_iterator = batch_train_data_set.make_initializable_iterator()
    validation_iterator = batch_validation_data_set.make_initializable_iterator()
    # Handles for the iterators
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    # Verify batching
    # with tf.Session() as sess:
    #   print (sess.run(Y))

    # Create a graph for training
    logits_train = conv_net2(X, n_classes, dropout, reuse=False, is_training=True)
    # Create another graph for testing that reuse the same weights, but has
    # different behavior for 'dropout' (not applied).
    logits_validation = conv_net2(X, n_classes, dropout, reuse=True, is_training=False)

    # Stuff needed to save the model
    values, indices = tf.nn.top_k(logits_validation, n_classes)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        tf.constant([str(i) for i in xrange(n_classes)]))
    prediction_classes = table.lookup(tf.to_int64(indices))

    # Define loss and optimizer (with train logits, for dropout to take effect)
    with tf.name_scope('loss_op'):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_train, labels=Y))
    tf.summary.scalar('loss_op', loss_op)

    # Optimizer and train operation

    # https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.005
    # decay every 100 steps with a base of 0.96
    decay_steps = 100
    decay_rate = 0.96
    # If the argument staircase is True, then global_step / decay_steps is an integer division
    # and the decayed learning rate follows a staircase function.
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate,
                                               staircase=True)
    # Note: Passing global_step to minimize() will increment it at each step.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=global_step)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(logits_validation, 1), tf.argmax(Y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Summaries
    merged_summaries = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter('/Users/tts/Development/school/MLDL/project/genreclassifier/summaries', sess.graph)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Run the initializer
    sess.run(init)

    # Training loop
    sess.run(training_iterator.initializer)
    sess.run(validation_iterator.initializer)
    num_epochs = 20
    for epoch in range(num_epochs):
        print("Starting epoch {}".format(str(epoch)))
        for step in range(1, num_steps + 1):

            # Training - use training iterator
            try:
                sess.run(train_op, feed_dict={handle: training_handle})
            except tf.errors.OutOfRangeError:
                print("Reached end of data set in train op - reinitializing iterator and starting next epoch")
                sess.run(training_iterator.initializer)
                break

            # Validation - use validation iterator
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy (on the validation set)
                # (note that this consumes a new batch of data - from the validation set)
                # (note that the loss op doesn't actually train the model)
                try:
                    summary, loss, acc = sess.run([merged_summaries, loss_op, accuracy], feed_dict={handle: validation_handle})
                    test_writer.add_summary(summary, step)
                    print("Epoch " + str(epoch) + " - Step " + str(step)
                          + ", Minibatch Loss= " + "{:.4f}".format(loss)
                          + ", Training Accuracy= " + "{:.3f}".format(acc))
                except tf.errors.OutOfRangeError:
                    print("Reached end of data set on loss step - reinitializing validation iterator")
                    sess.run(validation_iterator.initializer)

    test_writer.close()
    print("Optimization Finished!")

    # Saving model
    # https://github.com/llSourcell/How-to-Deploy-a-Tensorflow-Model-in-Production/blob/master/custom_model.py
    # https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py

    export_path = "/Users/tts/Development/school/MLDL/project/genreclassifier/export"
    print 'Exporting trained model to', export_path
    builder = saved_model_builder.SavedModelBuilder(export_path)

    classification_inputs = utils.build_tensor_info(X)
    classification_outputs_classes = utils.build_tensor_info(prediction_classes)
    classification_outputs_scores = utils.build_tensor_info(values)

    classification_signature = signature_def_utils.build_signature_def(
        inputs={signature_constants.CLASSIFY_INPUTS: classification_inputs},
        outputs={
            signature_constants.CLASSIFY_OUTPUT_CLASSES:
                classification_outputs_classes,
            signature_constants.CLASSIFY_OUTPUT_SCORES:
                classification_outputs_scores
        },
        method_name=signature_constants.CLASSIFY_METHOD_NAME)

    tensor_info_x = utils.build_tensor_info(X)
    tensor_info_y = utils.build_tensor_info(Y)

    prediction_signature = signature_def_utils.build_signature_def(
        inputs={'images': tensor_info_x},
        outputs={'scores': tensor_info_y},
        method_name=signature_constants.PREDICT_METHOD_NAME)

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                classification_signature,
        },
        legacy_init_op=legacy_init_op)

    builder.save()






