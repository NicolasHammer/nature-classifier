import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d


class BasicMultiClassCNN(nn.Module):
    def __init__(self, in_channels: int, height: int, width: int, classes: int):
        super(BasicMultiClassCNN, self).__init__()

        self.convolutional_block = nn.Sequential(
            # Input layer
            nn.Conv2d(in_channels=in_channels,
                      out_channels=64, kernel_size=(3, 3)),  # (64, height - 2, width - 2))
            ReLU(),
            MaxPool2d((2, 2)),  # (64, height//2 - 1, width//2 - 1)

            # First hidden layer
            # (128, height//2 - 4, width//2 - 4)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4)),
            ReLU(),
            MaxPool2d((2, 2)),  # (128, height//4 - 2, width//4 - 2)

            # Second hidden layer
            # (64,  height//4 - 4,  width//4 - 2)
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3)),
            ReLU(),
            MaxPool2d((2, 2))  # (64,  height//8 - 2, width//8 - 2)
        )

        self.MLP = nn.Sequential(
            # Flatten layer
            nn.Flatten(),

            # Dense layer
            nn.Linear(in_features=64*(height//8 - 2) * \
                      (width//8 - 2), out_features=32),
            ReLU(),

            # Output layer
            nn.Linear(in_features=32, out_features=classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, data):
        conv_out = self.convolutional_block(data)
        mlp_out = self.MLP(conv_out)
        return mlp_out
