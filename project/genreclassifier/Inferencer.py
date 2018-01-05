import tensorflow as tf
import argparse
from PIL import Image
import numpy as np

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


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="trained_model_3/frozen_model.pb",
                        type=str, help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions

    # We access the input and output nodes
    # x = graph.get_tensor_by_name('prefix/Placeholder/inputs_placeholder:0')
    x = graph.get_tensor_by_name('prefix/Reshape:0')
    # y = graph.get_tensor_by_name('prefix/Accuracy/predictions:0')
    y = graph.get_tensor_by_name('prefix/prediction:0')
    pkeep = graph.get_tensor_by_name('prefix/pkeep:0')


    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants
        y_out = sess.run(y, feed_dict={
            x: image_to_array("data/testing/Pop/Pop_1_4.png"),
            pkeep: 1
        })
        # I taught a neural net to recognise when a sum of numbers is bigger than 45
        # it should return False in this case
        print(y_out)  # [[ False ]] Yay, it works!