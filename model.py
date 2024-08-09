
import torch as t


class ResBlock(t.nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()

        self.kernel_size = 3
        self.intermediate_channels = out_channels

        self.conv2d_1 = t.nn.Conv2d(in_channels, self.intermediate_channels, self.kernel_size, stride=stride, padding=1)
        self.bn_1 = t.nn.BatchNorm2d(self.intermediate_channels)
        self.relu_1 = t.nn.ReLU()
        self.conv2d_2 = t.nn.Conv2d(self.intermediate_channels, out_channels, self.kernel_size, stride=1, padding=1)
        self.conv1x1 = t.nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        self.bn_res = t.nn.BatchNorm2d(out_channels)
        self.bn_2 = t.nn.BatchNorm2d(out_channels)
        self.relu_2 = t.nn.ReLU()

    def forward(self, x):

        # skip path
        skip_x = self.conv1x1(x)
        skip_x = self.bn_res(skip_x)

        # go through residual path
        res_x = self.conv2d_1(x)
        res_x = self.bn_1(res_x)
        res_x = self.relu_1(res_x)
        res_x = self.conv2d_2(res_x)
        res_x = self.bn_2(res_x)

        # sum skip and residual connection
        x = skip_x + res_x

        # activation after skip + residual
        x = self.relu_2(x)

        return x


class ResNet(t.nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

        self.conv2d_1 = t.nn.Conv2d(3,64,7,stride=2)
        self.bn_1 = t.nn.BatchNorm2d(64)
        self.relu_1 = t.nn.ReLU()
        self.max_pool_1 = t.nn.MaxPool2d(3,stride=2)
        self.res_block_1 = ResBlock(64,64,1)
        self.res_block_2 = ResBlock(64,128,2)
        self.res_block_3 = ResBlock(128,256,2)
        self.res_block_4 = ResBlock(256,512,2)
        self.global_avg = t.nn.AdaptiveAvgPool2d((1,1))
        self.flatten = t.nn.Flatten()
        self.fc = t.nn.Linear(512,2)
        self.sigmoid = t.nn.Sigmoid()

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.max_pool_1(x)
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.res_block_4(x)
        x = self.global_avg(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
