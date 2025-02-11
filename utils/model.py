import torch
import torch.nn as nn

architecture_config = [
    (7, 64, 2, 3),
    'm',
    (3, 192, 1, 1),
    'm',
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    'm',
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    'm',
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leakyrelu(self.batchnorm(self.conv(x)))
        return x


class YOLOV1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(YOLOV1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.dark_net = self._conv_net(self.architecture)
        self.fcs = self._fcs_net(**kwargs)

    def _conv_net(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == tuple:
                layers += [CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2))]
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_layers = x[2]

                for _ in range(num_layers):
                    layers += [CNNBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3])]
                    # in_channels = conv1[1]

                    layers += [CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3])]
                    in_channels = conv2[1]
        
        return nn.Sequential(*layers)

    def _fcs_net(self, grid, num_boxes, num_classes):
        S, B, C = grid, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S *(C + B*5))
        )

    def forward(self, x):
        x = self.dark_net(x)
        return self.fcs(torch.flatten(x, start_dim=1))


