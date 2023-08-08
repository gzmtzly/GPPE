import torch.nn as nn
import torch.nn.functional as F
import torch
class Model_Mnist(nn.Module):
    def __init__(self, num_class=2):
        super(Model_Mnist, self).__init__()
        self.num_class = num_class
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_class)

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


class Model_Cifar10(nn.Module):
  def __init__(self, num_class=2):
    super(Model_Cifar10, self).__init__()
    self.num_class = num_class
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
    self.fc3 = nn.Linear(512, self.num_class)

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


class Model_Cifar100(nn.Module):
  def __init__(self, num_class=2):
    super(Model_Cifar100, self).__init__()
    self.num_class = num_class
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
    self.fc3 = nn.Linear(512, self.num_class)

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


class Model_Caltech101(nn.Module):
    def __init__(self, num_class=101, init_weights=False):
        super(Model_Caltech101, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6 * 6 * 128, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_class),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  
                nn.init.constant_(m.bias, 0)


class Model_Ttru(nn.Module):
    def __init__(self):
        super(Model_Ttru, self).__init__()
        self.input_dim = 1
        self.filter1 = 64
        self.filter2 = 128
        self.output_dim = 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, self.filter1, 2, 2),   # 64 ---> 32
            nn.BatchNorm2d(self.filter1),
            nn.ReLU(),

            nn.Conv2d(self.filter1, self.filter2, 5, 1, 1),  # 32 ---> 30
            nn.BatchNorm2d(self.filter2),
            nn.ReLU(),

            nn.Conv2d(self.filter2, self.filter2, 5, 1, 1),  # 30 ---> 28
            nn.BatchNorm2d(self.filter2),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(self.filter2, self.filter2, 5, 1, 1),  # 14 ---> 12
            nn.BatchNorm2d(self.filter2),
            nn.ReLU(),

            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.filter2 * 6 * 6, 1024),  # 128*6*6 ---> 1024
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),  # 1024 ---> 512
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, self.output_dim)  # 512 ---> 2
        )

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, self.filter2 * 6 * 6)
        x = self.fc1(x)
        return x

