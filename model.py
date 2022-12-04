# @Time    : 12/4/22 10:50 PM
# @Author  : Zhou-Lin-yong
# @File    : model.py
# @SoftWare: PyCharm
import torch.nn as nn
import torch.nn.functional as F

class Model_Minst(nn.Module):
    def __init__(self):
        super(Model_Minst, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size()[0], -1)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Model_CIFAR10(nn.Module):
    def __init__(self):
        super(Model_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 5, padding=2)
        self.conv2 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn_conv1 = nn.BatchNorm2d(128)
        self.bn_conv2 = nn.BatchNorm2d(128)
        self.bn_conv3 = nn.BatchNorm2d(256)
        self.bn_conv4 = nn.BatchNorm2d(256)
        self.bn_dense1 = nn.BatchNorm1d(1024)
        self.bn_dense2 = nn.BatchNorm1d(512)
        self.dropout_conv = nn.Dropout2d(p=0.25)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def conv_layers(self, x):
        out = F.relu(self.bn_conv1(self.conv1(x)))
        out = F.relu(self.bn_conv2(self.conv2(out)))
        out = self.pool(out)
        out = self.dropout_conv(out)
        out = F.relu(self.bn_conv3(self.conv3(out)))
        out = F.relu(self.bn_conv4(self.conv4(out)))
        out = self.pool(out)
        out = self.dropout_conv(out)
        return out

    def dense_layers(self, x):
        out = F.relu(self.bn_dense1(self.fc1(x)))
        out = self.dropout(out)
        out = F.relu(self.bn_dense2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(-1, 256 * 8 * 8)
        out = self.dense_layers(out)
        return out

class Model_HTRU(nn.Module):
    def __init__(self):
        super(Model_HTRU, self).__init__()
        self.input_dim = 1
        self.filter1 = 64
        self.filter2 = 128
        self.output_dim = 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, self.filter1, 2, 2),
            nn.BatchNorm2d(self.filter1),
            nn.ReLU(),

            nn.Conv2d(self.filter1, self.filter2, 5, 1, 1),
            nn.BatchNorm2d(self.filter2),
            nn.ReLU(),

            nn.Conv2d(self.filter2, self.filter2, 5, 1, 1),
            nn.BatchNorm2d(self.filter2),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(self.filter2, self.filter2, 5, 1, 1),
            nn.BatchNorm2d(self.filter2),
            nn.ReLU(),

            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.filter2 * 6 * 6, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, self.output_dim)
        )

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, self.filter2 * 6 * 6)
        x = self.fc1(x)
        return x