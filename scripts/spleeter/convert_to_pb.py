#!/usr/bin/env python3

# Code in this file is modified from
# https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
#
# Please see ./run.sh for usages
import argparse
import os

import tensorflow as tf


def freeze_graph(model_dir, output_node_names, output_filename):
    """Extract the sub graph defined by the output nodes and convert all its
    variables into constant

    Args:
      model_dir:
        the root folder containing the checkpoint state file
      output_node_names:
        a string, containing all the output node's names, comma separated
      output_filename:
        Filename to save the graph.
    """
    if not tf.compat.v1.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir
        )

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split("/")[:-1])
    output_graph = output_filename

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.compat.v1.train.import_meta_graph(
            input_checkpoint + ".meta", clear_devices=clear_devices
        )

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.compat.v1.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(
                ","
            ),  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.compat.v1.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=str, default="", help="Model folder to export"
    )
    parser.add_argument(
        "--output-node-names",
        type=str,
        default="vocals_spectrogram/mul,accompaniment_spectrogram/mul",
        help="The name of the output nodes, comma separated.",
    )

    parser.add_argument(
        "--output-filename",
        type=str,
    )
    args = parser.parse_args()

    freeze_graph(args.model_dir, args.output_node_names, args.output_filename)
