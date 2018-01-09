from __future__ import division
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from glob import glob
import shutil
import sys
from collections import defaultdict
from operator import add


# Script that takes an mp3 file and predicts genre for it
# Adapted from: https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def image_to_array(img_path):
    img_file = Image.open(img_path)
    img_decoded = np.array(img_file)
    return img_decoded.reshape(-1, 128, 128, 1)


def get_image_file_paths(folder_path):
    paths = glob(folder_path + '/*.png')
    return paths


def get_genre_png(file_path):
    file_name = os.path.basename(file_path)
    # png file names are of the form genre_....png
    genre = os.path.splitext(file_name)[0].split("_")[0]
    return genre


# https://stackoverflow.com/a/3787983
def all_same(items):
    return all(x == items[0] for x in items)


def print_prediction_dictionary(name, dictionary, label_genre_map):
    print(name)
    for key, value in dictionary.items():
        print(str(label_genre_map.get(key)) + ": " + str(value))


def get_final_verdict(prediction_dict, label_genre_map):
    verdict = ""
    max_value = 0
    for key, value in prediction_dict.items():
        if value >= max_value:
            max_value = value
            verdict = key
    return str(label_genre_map.get(verdict))


if __name__ == '__main__':

    frozen_model_path = "tm_11000_learning_rate=0.0001.dropout=0.5/frozen_model.pb"
    # frozen_model_path = "tm_11000_learning_rate=0.0005.dropout=0.25/frozen_model.pb"

    # mp3_path = "mp3s/01-dageneral-second_thought_(original_mix).mp3"
    # mp3_path = "mp3s/03 The Four Seasons, Op. 8, _Spring__ Allegro.mp3"
    mp3_path = "mp3s/04 Adagio for Strings.mp3"
    # mp3_path = "mp3s/Death - [The Sound Of Perseverance 1998] - 01 - Scavanger Of Human Sorrow.mp3"
    # mp3_path = "mp3s/Death - [Scream Bloody Gore 1987] - 02 - Zombie Ritual.mp3"
    # mp3_path = "mp3s/Queens Of The Stone Age - [Songs For The Deaf 2002] - 02 - No One Knows.mp3"
    # mp3_path = "mp3s/Common - [Be 2005] - 02 - The Corner.mp3"
    # mp3_path = "mp3s/Nick Cave And The Bad Seeds - [The Boatmans Call 1997] - 01 - Into My Arms.mp3"

    spectrogram_output_folder = "mp3s/generated_spectrograms"
    spectrogram_splits_output_folder = "mp3s/generated_split_spectrograms"

    # Generate spectrograms
    command = "python ../dataprep/ProcessSongs.py '../genreclassifier/{}' '../genreclassifier/{}'"\
        .format(mp3_path, spectrogram_output_folder)
    os.system(command)

    # Split spectrograms
    spectrogram_paths = get_image_file_paths(spectrogram_output_folder)
    if len(spectrogram_paths) == 0:
        print("No spectrograms found, exiting")
        sys.exit(1)

    command = "python ../dataprep/SpectrogramSplitter.py '../genreclassifier/{}' '../genreclassifier/{}'"\
        .format(spectrogram_paths[0], spectrogram_splits_output_folder)
    os.system(command)

    split_spectrogram_paths = glob(spectrogram_splits_output_folder + '/**/*.png')  # Splits get put under Genre/*.png

    # Load graph

    graph = load_graph(frozen_model_path)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)

    # We access the input and output nodes
    # Note: We have two different graphs types, one from the TFOS model and one from the TF Estimator API

    # Input
    # x = graph.get_tensor_by_name('prefix/Reshape:0')    # TFOS model
    x = graph.get_tensor_by_name('prefix/ConvNet_1/Reshape:0')  # Estimator model

    # Output
    # y = graph.get_tensor_by_name('prefix/prediction:0')    # TFOS model
    # pkeep = graph.get_tensor_by_name('prefix/pkeep:0') # TFOS model
    y = graph.get_tensor_by_name('prefix/ArgMax:0') # Estimator model


    # NOTE: Model expects batches of 100. Can't do smaller batches, but can do a single image.
    # Will also return 100 identical prediction values for each image.
    # Just hacking around this by predicting each and every image.

    label_dict = {
        'Classical': 0,
        'Techno': 1,
        'Pop': 2,
        'HipHop': 3,
        'Metal': 4,
        'Rock': 5
    }

    label_to_genre_mapping = {
        0: 'Classical',
        1: 'Techno',
        2: 'Pop',
        3: 'HipHop',
        4: 'Metal',
        5: 'Rock'
    }

    # We launch a Session
    with tf.Session(graph=graph) as sess:

        print("Processing {} spectrograms".format(len(split_spectrogram_paths)))

        predictions = defaultdict(int)

        total_predictions = 0
        total_correct = 0
        count = 1
        for image_path in split_spectrogram_paths:

            if count % 10 == 0 or count == 1:
                print("Processing image {} of {}".format(count, len(split_spectrogram_paths)))

            genre = get_genre_png(image_path)
            label = label_dict.get(genre)

            y_out = sess.run(y, feed_dict={
                x: image_to_array(image_path),
                # pkeep: 1   # TFOS model
            })

            # Since the TFOS model expects batches of 100 we get 100 identical values as the prediction (with pkeep=1)
            # (The prediction tensor is of shape 100x6)
            if not all_same(y_out):
                print("WARNING - not all the predicted values are the same for {}".format(image_path))

            predicted_label = y_out[0]
            match = label == predicted_label
            if match:
                total_correct = total_correct + 1
            total_predictions = total_predictions + 1
            # print("{} - {} - Predicted: {} - Match: {}".format(label, genre, predicted_label, match))
            predictions[predicted_label] = add(predictions[predicted_label], 1)
            count = count + 1

        print_prediction_dictionary("\n{} - Predictions:".format(mp3_path), predictions, label_to_genre_mapping)
        # print("\nTotal: {} - Correct: {} - Accuracy: {}"
        #      .format(str(total_predictions), str(total_correct), str(total_correct / total_predictions)))
        print("\nResult: {}".format(get_final_verdict(predictions, label_to_genre_mapping)))

        # Cleanup

        print("\nCleaning up...")

        # Remove spectrogram splits
        for path in os.listdir(spectrogram_splits_output_folder):
            joined_path = os.path.join(spectrogram_splits_output_folder, path)
            if os.path.isdir(joined_path):
                shutil.rmtree(joined_path)

        # Remove generated spectrogram
        for spectrogram_path in spectrogram_paths:
            os.remove(spectrogram_path)



