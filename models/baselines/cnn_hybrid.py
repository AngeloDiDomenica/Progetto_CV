import torch.nn as nn
import torch

class HybridSN(nn.Module):

    def __init__(self,num_classes=16):

        super().__init__()

        self.conv3d_1 = nn.Conv3d(
            1,
            8,
            kernel_size=(7,3,3)
        )

        self.conv3d_2 = nn.Conv3d(
            8,
            16,
            kernel_size=(5,3,3)
        )

        self.conv3d_3 = nn.Conv3d(
            16,
            32,
            kernel_size=(3,3,3)
        )

        self.conv2d = nn.LazyConv2d(64, kernel_size=3)

        self.fc1 = nn.LazyLinear(256)

        self.fc2 = nn.Linear(
            256,
            128
        )

        self.fc3 = nn.Linear(
            128,
            num_classes
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.4)

    def forward(self,x):

        x = self.relu(self.conv3d_1(x))
        x = self.relu(self.conv3d_2(x))
        x = self.relu(self.conv3d_3(x))

        b,c,d,h,w = x.shape

        x = x.reshape(
            b,
            c*d,
            h,
            w
        )

        x = self.relu(self.conv2d(x))

        x = x.view(x.size(0),-1)

        x = self.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.relu(self.fc2(x))

        x = self.dropout(x)

        x = self.fc3(x)

        return x