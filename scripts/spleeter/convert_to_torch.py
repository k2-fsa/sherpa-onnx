#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn


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


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 16, kernel_size=5, stride=(2, 2), padding=0)
        self.bn = torch.nn.BatchNorm2d(
            16, track_running_stats=True, eps=1e-3, momentum=0.01
        )
        #
        self.conv1 = torch.nn.Conv2d(16, 32, kernel_size=5, stride=(2, 2), padding=0)
        self.bn1 = torch.nn.BatchNorm2d(
            32, track_running_stats=True, eps=1e-3, momentum=0.01
        )

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, stride=(2, 2), padding=0)
        self.bn2 = torch.nn.BatchNorm2d(
            64, track_running_stats=True, eps=1e-3, momentum=0.01
        )

        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=5, stride=(2, 2), padding=0)
        self.bn3 = torch.nn.BatchNorm2d(
            128, track_running_stats=True, eps=1e-3, momentum=0.01
        )

        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=5, stride=(2, 2), padding=0)
        self.bn4 = torch.nn.BatchNorm2d(
            256, track_running_stats=True, eps=1e-3, momentum=0.01
        )

        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=5, stride=(2, 2), padding=0)

        self.up1 = torch.nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2)
        self.bn5 = torch.nn.BatchNorm2d(
            256, track_running_stats=True, eps=1e-3, momentum=0.01
        )

        self.up2 = torch.nn.ConvTranspose2d(512, 128, kernel_size=5, stride=2)
        self.bn6 = torch.nn.BatchNorm2d(
            128, track_running_stats=True, eps=1e-3, momentum=0.01
        )

    def forward(self, x):
        x = torch.nn.functional.pad(x, (1, 2, 1, 2), "constant", 0)
        conv1 = self.conv(x)
        batch1 = self.bn(conv1)
        rel1 = torch.nn.functional.leaky_relu(batch1, negative_slope=0.2)

        x = torch.nn.functional.pad(rel1, (1, 2, 1, 2), "constant", 0)
        conv2 = self.conv1(x)  # (3, 32, 128, 256)
        batch2 = self.bn1(conv2)
        rel2 = torch.nn.functional.leaky_relu(
            batch2, negative_slope=0.2
        )  # (3, 32, 128, 256)

        x = torch.nn.functional.pad(rel2, (1, 2, 1, 2), "constant", 0)
        conv3 = self.conv2(x)  # (3, 64, 64, 128)
        batch3 = self.bn2(conv3)
        rel3 = torch.nn.functional.leaky_relu(
            batch3, negative_slope=0.2
        )  # (3, 64, 64, 128)

        x = torch.nn.functional.pad(rel3, (1, 2, 1, 2), "constant", 0)
        conv4 = self.conv3(x)  # (3, 128, 32, 64)
        batch4 = self.bn3(conv4)
        rel4 = torch.nn.functional.leaky_relu(
            batch4, negative_slope=0.2
        )  # (3, 128, 32, 64)

        x = torch.nn.functional.pad(rel4, (1, 2, 1, 2), "constant", 0)
        conv5 = self.conv4(x)  # (3, 256, 16, 32)
        batch5 = self.bn4(conv5)
        rel6 = torch.nn.functional.leaky_relu(
            batch5, negative_slope=0.2
        )  # (3, 256, 16, 32)

        x = torch.nn.functional.pad(rel6, (1, 2, 1, 2), "constant", 0)
        conv6 = self.conv5(x)  # (3, 512, 8, 16)

        up1 = self.up1(conv6)
        up1 = up1[:, :, 1:-2, 1:-2]  # (3, 256, 16, 32)
        up1 = torch.nn.functional.relu(up1)
        batch7 = self.bn5(up1)
        merge1 = torch.cat([conv5, batch7], axis=1)  # (3, 512, 16, 32)

        up2 = self.up2(merge1)
        up2 = up2[:, :, 1:-2, 1:-2]
        up2 = torch.nn.functional.relu(up2)
        batch8 = self.bn6(up2)

        merge2 = torch.cat([conv4, batch8], axis=1)

        return merge2


def get_param(graph, name):
    with tf.compat.v1.Session(graph=graph) as sess:
        constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
        for constant_op in constant_ops:
            if constant_op.name != name:
                continue

            value = sess.run(constant_op.outputs[0])
            return torch.from_numpy(value)


@torch.no_grad()
def main():
    graph = load_graph("./2stems/frozen_model.pb")
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
    y1 = graph.get_tensor_by_name("concatenate_1/concat:0")

    unet = UNet()
    unet.eval()

    # For the conv2d in tensorflow, weight shape is (kernel_h, kernel_w, in_channel, out_channel)
    # default input shape is NHWC

    # For the conv2d in torch, weight shape is (out_channel, in_channel, kernel_h, kernel_w)
    # default input shape is NCHW
    state_dict = unet.state_dict()
    print(list(state_dict.keys()))

    state_dict["conv.weight"] = get_param(graph, "conv2d/kernel").permute(3, 2, 0, 1)
    state_dict["conv.bias"] = get_param(graph, "conv2d/bias")

    state_dict["bn.weight"] = get_param(graph, "batch_normalization/gamma")
    state_dict["bn.bias"] = get_param(graph, "batch_normalization/beta")
    state_dict["bn.running_mean"] = get_param(graph, "batch_normalization/moving_mean")
    state_dict["bn.running_var"] = get_param(
        graph, "batch_normalization/moving_variance"
    )

    for i in range(1, 6):
        state_dict[f"conv{i}.weight"] = get_param(graph, f"conv2d_{i}/kernel").permute(
            3, 2, 0, 1
        )
        state_dict[f"conv{i}.bias"] = get_param(graph, f"conv2d_{i}/bias")
        if i >= 5:
            continue
        state_dict[f"bn{i}.weight"] = get_param(graph, f"batch_normalization_{i}/gamma")
        state_dict[f"bn{i}.bias"] = get_param(graph, f"batch_normalization_{i}/beta")
        state_dict[f"bn{i}.running_mean"] = get_param(
            graph, f"batch_normalization_{i}/moving_mean"
        )
        state_dict[f"bn{i}.running_var"] = get_param(
            graph, f"batch_normalization_{i}/moving_variance"
        )

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

    for i in range(1, 2):
        state_dict[f"up{i+1}.weight"] = get_param(
            graph, f"conv2d_transpose_{i}/kernel"
        ).permute(3, 2, 0, 1)

        state_dict[f"up{i+1}.bias"] = get_param(graph, f"conv2d_transpose_{i}/bias")

        state_dict[f"bn{5+i}.weight"] = get_param(
            graph, f"batch_normalization_{6+i}/gamma"
        )
        state_dict[f"bn{5+i}.bias"] = get_param(
            graph, f"batch_normalization_{6+i}/beta"
        )
        state_dict[f"bn{5+i}.running_mean"] = get_param(
            graph, f"batch_normalization_{6+i}/moving_mean"
        )
        state_dict[f"bn{5+i}.running_var"] = get_param(
            graph, f"batch_normalization_{6+i}/moving_variance"
        )

    unet.load_state_dict(state_dict)

    with tf.compat.v1.Session(graph=graph) as sess:
        y0_out, y1_out = sess.run([y0, y1], feed_dict={x: generate_waveform()})
        #  y0_out = sess.run(y0, feed_dict={x: generate_waveform()})
        #  y1_out = sess.run(y1, feed_dict={x: generate_waveform()})
        print(y0_out.shape)
        print(y1_out.shape)

    # for the batchnormalization in tensorflow,
    # default input shape is NHWC

    # for the batchnormalization in torch,
    # default input shape is NCHW

    # NHWC to NCHW
    torch_y1_out = unet(torch.from_numpy(y0_out).permute(0, 3, 1, 2))

    print(torch_y1_out.shape, torch.from_numpy(y1_out).permute(0, 3, 1, 2).shape)
    assert torch.allclose(
        torch_y1_out, torch.from_numpy(y1_out).permute(0, 3, 1, 2), atol=1e-2
    ), ((torch_y1_out - torch.from_numpy(y1_out).permute(0, 3, 1, 2)).abs().max())


if __name__ == "__main__":
    main()
