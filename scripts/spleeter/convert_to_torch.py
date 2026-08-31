#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)

# Please see ./run.sh for usage

import argparse

import numpy as np
import tensorflow as tf
import torch

from unet import UNet

# Spleeter's instrument_list, in the order configs/<model>/base_config.json
# declares it. That order is what fixes each stem's block of TF op names.
STEMS = {
    "2stems": ["vocals", "accompaniment"],
    "4stems": ["vocals", "drums", "bass", "other"],
}

# The U-Net activations, also from base_config.json. 2stems ships "params": {},
# i.e. the unet.unet defaults; 4stems asks for ELU on both. Architecture and
# weight shapes are identical either way, so getting this wrong loads cleanly
# and returns garbage rather than raising.
ACTIVATIONS = {
    "2stems": ("LeakyReLU", "ReLU"),
    "4stems": ("ELU", "ELU"),
}


def op(family, index):
    """TF names the first op of a family with no suffix: conv2d, conv2d_1, ..."""
    return family if index == 0 else f"{family}_{index}"


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
def main(name, model):
    graph = load_graph(f"./{model}/frozen_{name}_model.pb")
    x = graph.get_tensor_by_name("waveform:0")
    y0 = graph.get_tensor_by_name("strided_slice_3:0")
    y1 = graph.get_tensor_by_name(f"{name}_spectrogram/mul:0")

    # Each stem's U-Net owns a contiguous block of TF op names, in
    # instrument_list order. Per stem: 7 conv2d (conv..conv5 plus the final
    # up7), 6 conv2d_transpose, and 12 batch_normalization indices -- of which
    # only 11 exist, because conv5 has no BatchNorm, but the 12th index is
    # allocated regardless. Hence the 6 rather than 5 below.
    index = STEMS[model].index(name)
    conv_offset = 7 * index
    tconv_offset = 6 * index
    bn_offset = 12 * index

    unet = UNet(*ACTIVATIONS[model])
    unet.eval()

    # For the conv2d in tensorflow, weight shape is (kernel_h, kernel_w, in_channel, out_channel)
    # default input shape is NHWC
    #
    # For the conv2d in torch, weight shape is (out_channel, in_channel, kernel_h, kernel_w)
    # default input shape is NCHW
    state_dict = unet.state_dict()

    state_dict["conv.weight"] = get_param(
        graph, f"{op('conv2d', conv_offset)}/kernel"
    ).permute(3, 2, 0, 1)
    state_dict["conv.bias"] = get_param(graph, f"{op('conv2d', conv_offset)}/bias")

    for i in range(1, 6):
        state_dict[f"conv{i}.weight"] = get_param(
            graph, f"{op('conv2d', i + conv_offset)}/kernel"
        ).permute(3, 2, 0, 1)
        state_dict[f"conv{i}.bias"] = get_param(
            graph, f"{op('conv2d', i + conv_offset)}/bias"
        )

    state_dict["up7.weight"] = get_param(
        graph, f"{op('conv2d', 6 + conv_offset)}/kernel"
    ).permute(3, 2, 0, 1)
    state_dict["up7.bias"] = get_param(graph, f"{op('conv2d', 6 + conv_offset)}/bias")

    for i in range(6):
        state_dict[f"up{i + 1}.weight"] = get_param(
            graph, f"{op('conv2d_transpose', i + tconv_offset)}/kernel"
        ).permute(3, 2, 0, 1)
        state_dict[f"up{i + 1}.bias"] = get_param(
            graph, f"{op('conv2d_transpose', i + tconv_offset)}/bias"
        )

    # for the batchnormalization in tensorflow, default input shape is NHWC
    # for the batchnormalization in torch, default input shape is NCHW
    #
    # bn..bn4 sit after conv..conv4; bn5..bn10 after up1..up6. Index 5 of the
    # block is the one conv5 would have used and is skipped.
    bn_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    bn_keys = ["bn"] + [f"bn{i}" for i in range(1, 11)]
    for key, i in zip(bn_keys, bn_indices):
        scope = op("batch_normalization", i + bn_offset)
        state_dict[f"{key}.weight"] = get_param(graph, f"{scope}/gamma")
        state_dict[f"{key}.bias"] = get_param(graph, f"{scope}/beta")
        state_dict[f"{key}.running_mean"] = get_param(graph, f"{scope}/moving_mean")
        state_dict[f"{key}.running_var"] = get_param(graph, f"{scope}/moving_variance")

    unet.load_state_dict(state_dict)

    with tf.compat.v1.Session(graph=graph) as sess:
        y0_out, y1_out = sess.run([y0, y1], feed_dict={x: generate_waveform()})

    torch_y1_out = unet(torch.from_numpy(y0_out).permute(3, 0, 1, 2))
    torch_y1_out = torch_y1_out.permute(1, 0, 2, 3)

    assert torch.allclose(
        torch_y1_out, torch.from_numpy(y1_out).permute(0, 3, 1, 2), atol=1e-1
    ), ((torch_y1_out - torch.from_numpy(y1_out).permute(0, 3, 1, 2)).abs().max())
    torch.save(unet.state_dict(), f"{model}/{name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="2stems", choices=list(STEMS))
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()
    if args.name not in STEMS[args.model]:
        parser.error(f"{args.model} has no stem {args.name!r}; "
                     f"choose from {STEMS[args.model]}")
    print(vars(args))
    main(args.name, args.model)
