# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)

import torch


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

        self.up3 = torch.nn.ConvTranspose2d(256, 64, kernel_size=5, stride=2)
        self.bn7 = torch.nn.BatchNorm2d(
            64, track_running_stats=True, eps=1e-3, momentum=0.01
        )

        self.up4 = torch.nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2)
        self.bn8 = torch.nn.BatchNorm2d(
            32, track_running_stats=True, eps=1e-3, momentum=0.01
        )

        self.up5 = torch.nn.ConvTranspose2d(64, 16, kernel_size=5, stride=2)
        self.bn9 = torch.nn.BatchNorm2d(
            16, track_running_stats=True, eps=1e-3, momentum=0.01
        )

        self.up6 = torch.nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2)
        self.bn10 = torch.nn.BatchNorm2d(
            1, track_running_stats=True, eps=1e-3, momentum=0.01
        )

        # output logit is False, so we need self.up7
        self.up7 = torch.nn.Conv2d(1, 2, kernel_size=4, dilation=2, padding=3)

    def forward(self, x):
        """
        Args:
          x: (num_audio_channels, num_splits, 512, 1024)
        Returns:
          y: (num_audio_channels, num_splits, 512, 1024)
        """
        x = x.permute(1, 0, 2, 3)

        in_x = x
        # in_x is (3, 2, 512, 1024) = (T, 2, 512, 1024)
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

        merge2 = torch.cat([conv4, batch8], axis=1)  # (3, 256, 32, 64)

        up3 = self.up3(merge2)
        up3 = up3[:, :, 1:-2, 1:-2]
        up3 = torch.nn.functional.relu(up3)
        batch9 = self.bn7(up3)

        merge3 = torch.cat([conv3, batch9], axis=1)  # (3, 128, 64, 128)

        up4 = self.up4(merge3)
        up4 = up4[:, :, 1:-2, 1:-2]
        up4 = torch.nn.functional.relu(up4)
        batch10 = self.bn8(up4)

        merge4 = torch.cat([conv2, batch10], axis=1)  # (3, 64, 128, 256)

        up5 = self.up5(merge4)
        up5 = up5[:, :, 1:-2, 1:-2]
        up5 = torch.nn.functional.relu(up5)
        batch11 = self.bn9(up5)

        merge5 = torch.cat([conv1, batch11], axis=1)  # (3, 32, 256, 512)

        up6 = self.up6(merge5)
        up6 = up6[:, :, 1:-2, 1:-2]
        up6 = torch.nn.functional.relu(up6)
        batch12 = self.bn10(up6)  # (3, 1, 512, 1024)  = (T, 1, 512, 1024)

        up7 = self.up7(batch12)
        up7 = torch.sigmoid(up7)  # (3, 2, 512, 1024)

        ans = up7 * in_x
        return ans.permute(1, 0, 2, 3)
