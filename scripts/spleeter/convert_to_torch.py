#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)

# Please see ./run.sh for usage

import argparse

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

from unet import UNet


def load_graph(frozen_graph_filename):
    # This function is modified from
    # https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc

    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.compat.v1.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        #  tf.import_graph_def(graph_def, name="prefix")
        tf.import_graph_def(graph_def, name="")
    return graph


def generate_waveform():
    np.random.seed(20230821)
    waveform = np.random.rand(60 * 44100).astype(np.float32)

    # (num_samples, num_channels)
    waveform = waveform.reshape(-1, 2)
    return waveform


def get_param(graph, name):
    with tf.compat.v1.Session(graph=graph) as sess:
        constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
        for constant_op in constant_ops:
            if constant_op.name != name:
                continue

            value = sess.run(constant_op.outputs[0])
            return torch.from_numpy(value)


@torch.no_grad()
def main(name):
    graph = load_graph(f"./2stems/frozen_{name}_model.pb")
    #  for op in graph.get_operations():
    #      print(op.name)
    x = graph.get_tensor_by_name("waveform:0")
    #  y = graph.get_tensor_by_name("Reshape:0")
    y0 = graph.get_tensor_by_name("strided_slice_3:0")
    #  y1 = graph.get_tensor_by_name("leaky_re_lu_5/LeakyRelu:0")
    #  y1 = graph.get_tensor_by_name("conv2d_5/BiasAdd:0")
    #  y1 = graph.get_tensor_by_name("conv2d_transpose/BiasAdd:0")
    #  y1 = graph.get_tensor_by_name("re_lu/Relu:0")
    #  y1 = graph.get_tensor_by_name("batch_normalization_6/cond/FusedBatchNorm_1:0")
    #  y1 = graph.get_tensor_by_name("concatenate/concat:0")
    #  y1 = graph.get_tensor_by_name("concatenate_1/concat:0")
    #  y1 = graph.get_tensor_by_name("concatenate_4/concat:0")
    #  y1 = graph.get_tensor_by_name("batch_normalization_11/cond/FusedBatchNorm_1:0")
    #  y1 = graph.get_tensor_by_name("conv2d_6/Sigmoid:0")
    y1 = graph.get_tensor_by_name(f"{name}_spectrogram/mul:0")

    unet = UNet()
    unet.eval()

    # For the conv2d in tensorflow, weight shape is (kernel_h, kernel_w, in_channel, out_channel)
    # default input shape is NHWC

    # For the conv2d in torch, weight shape is (out_channel, in_channel, kernel_h, kernel_w)
    # default input shape is NCHW
    state_dict = unet.state_dict()
    #  print(list(state_dict.keys()))

    if name == "vocals":
        state_dict["conv.weight"] = get_param(graph, "conv2d/kernel").permute(
            3, 2, 0, 1
        )
        state_dict["conv.bias"] = get_param(graph, "conv2d/bias")

        state_dict["bn.weight"] = get_param(graph, "batch_normalization/gamma")
        state_dict["bn.bias"] = get_param(graph, "batch_normalization/beta")
        state_dict["bn.running_mean"] = get_param(
            graph, "batch_normalization/moving_mean"
        )
        state_dict["bn.running_var"] = get_param(
            graph, "batch_normalization/moving_variance"
        )

        conv_offset = 0
        bn_offset = 0
    else:
        state_dict["conv.weight"] = get_param(graph, "conv2d_7/kernel").permute(
            3, 2, 0, 1
        )
        state_dict["conv.bias"] = get_param(graph, "conv2d_7/bias")

        state_dict["bn.weight"] = get_param(graph, "batch_normalization_12/gamma")
        state_dict["bn.bias"] = get_param(graph, "batch_normalization_12/beta")
        state_dict["bn.running_mean"] = get_param(
            graph, "batch_normalization_12/moving_mean"
        )
        state_dict["bn.running_var"] = get_param(
            graph, "batch_normalization_12/moving_variance"
        )
        conv_offset = 7
        bn_offset = 12

    for i in range(1, 6):
        state_dict[f"conv{i}.weight"] = get_param(
            graph, f"conv2d_{i+conv_offset}/kernel"
        ).permute(3, 2, 0, 1)
        state_dict[f"conv{i}.bias"] = get_param(graph, f"conv2d_{i+conv_offset}/bias")
        if i >= 5:
            continue
        state_dict[f"bn{i}.weight"] = get_param(
            graph, f"batch_normalization_{i+bn_offset}/gamma"
        )
        state_dict[f"bn{i}.bias"] = get_param(
            graph, f"batch_normalization_{i+bn_offset}/beta"
        )
        state_dict[f"bn{i}.running_mean"] = get_param(
            graph, f"batch_normalization_{i+bn_offset}/moving_mean"
        )
        state_dict[f"bn{i}.running_var"] = get_param(
            graph, f"batch_normalization_{i+bn_offset}/moving_variance"
        )

    if name == "vocals":
        state_dict["up1.weight"] = get_param(graph, "conv2d_transpose/kernel").permute(
            3, 2, 0, 1
        )
        state_dict["up1.bias"] = get_param(graph, "conv2d_transpose/bias")

        state_dict["bn5.weight"] = get_param(graph, "batch_normalization_6/gamma")
        state_dict["bn5.bias"] = get_param(graph, "batch_normalization_6/beta")
        state_dict["bn5.running_mean"] = get_param(
            graph, "batch_normalization_6/moving_mean"
        )
        state_dict["bn5.running_var"] = get_param(
            graph, "batch_normalization_6/moving_variance"
        )
        conv_offset = 0
        bn_offset = 0
    else:
        state_dict["up1.weight"] = get_param(
            graph, "conv2d_transpose_6/kernel"
        ).permute(3, 2, 0, 1)
        state_dict["up1.bias"] = get_param(graph, "conv2d_transpose_6/bias")

        state_dict["bn5.weight"] = get_param(graph, "batch_normalization_18/gamma")
        state_dict["bn5.bias"] = get_param(graph, "batch_normalization_18/beta")
        state_dict["bn5.running_mean"] = get_param(
            graph, "batch_normalization_18/moving_mean"
        )
        state_dict["bn5.running_var"] = get_param(
            graph, "batch_normalization_18/moving_variance"
        )
        conv_offset = 6
        bn_offset = 12

    for i in range(1, 6):
        state_dict[f"up{i+1}.weight"] = get_param(
            graph, f"conv2d_transpose_{i+conv_offset}/kernel"
        ).permute(3, 2, 0, 1)

        state_dict[f"up{i+1}.bias"] = get_param(
            graph, f"conv2d_transpose_{i+conv_offset}/bias"
        )

        state_dict[f"bn{5+i}.weight"] = get_param(
            graph, f"batch_normalization_{6+i+bn_offset}/gamma"
        )
        state_dict[f"bn{5+i}.bias"] = get_param(
            graph, f"batch_normalization_{6+i+bn_offset}/beta"
        )
        state_dict[f"bn{5+i}.running_mean"] = get_param(
            graph, f"batch_normalization_{6+i+bn_offset}/moving_mean"
        )
        state_dict[f"bn{5+i}.running_var"] = get_param(
            graph, f"batch_normalization_{6+i+bn_offset}/moving_variance"
        )

    if name == "vocals":
        state_dict["up7.weight"] = get_param(graph, "conv2d_6/kernel").permute(
            3, 2, 0, 1
        )
        state_dict["up7.bias"] = get_param(graph, "conv2d_6/bias")
    else:
        state_dict["up7.weight"] = get_param(graph, "conv2d_13/kernel").permute(
            3, 2, 0, 1
        )
        state_dict["up7.bias"] = get_param(graph, "conv2d_13/bias")

    unet.load_state_dict(state_dict)

    with tf.compat.v1.Session(graph=graph) as sess:
        y0_out, y1_out = sess.run([y0, y1], feed_dict={x: generate_waveform()})
        #  y0_out = sess.run(y0, feed_dict={x: generate_waveform()})
        #  y1_out = sess.run(y1, feed_dict={x: generate_waveform()})
        #  print(y0_out.shape)
        #  print(y1_out.shape)

    # for the batchnormalization in tensorflow,
    # default input shape is NHWC

    # for the batchnormalization in torch,
    # default input shape is NCHW

    # NHWC to NCHW
    torch_y1_out = unet(torch.from_numpy(y0_out).permute(0, 3, 1, 2))

    #  print(torch_y1_out.shape, torch.from_numpy(y1_out).permute(0, 3, 1, 2).shape)
    assert torch.allclose(
        torch_y1_out, torch.from_numpy(y1_out).permute(0, 3, 1, 2), atol=1e-1
    ), ((torch_y1_out - torch.from_numpy(y1_out).permute(0, 3, 1, 2)).abs().max())
    torch.save(unet.state_dict(), f"2stems/{name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        choices=["vocals", "accompaniment"],
    )
    args = parser.parse_args()
    print(vars(args))
    main(args.name)
