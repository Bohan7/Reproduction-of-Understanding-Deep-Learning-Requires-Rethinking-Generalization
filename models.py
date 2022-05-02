import torch
import torch.nn as nn
from torchsummary import summary
import math

class conv_module(nn.Module):
    """ conv module for Inception """
    def __init__(self, in_channel, C=96, K=3, S=1, padding='same', use_bn=True, **kwargs):
        super(conv_module, self).__init__()
        
        self.use_bn = use_bn
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=C, kernel_size=(K, K), stride=(S, S),
                              padding=padding, bias=True)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(C, eps=0.001, momentum=0.1, affine=False, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)

        return x


class Inception_module(nn.Module):
    """ Inception module for Inception """
    def __init__(self, in_channel, Ch1, Ch2, **kwargs):
        super(Inception_module, self).__init__()

        self.conv_module_1 = conv_module(in_channel=in_channel, C=Ch1, K=1, S=1, **kwargs)  # 0
        self.conv_module_2 = conv_module(in_channel=in_channel, C=Ch2, K=3, S=1, **kwargs)  # 1

    def forward(self, x):
        x1 = self.conv_module_1(x)
        x2 = self.conv_module_2(x)
        x = torch.cat([x1, x2], 1)

        return x


class downsample_module(nn.Module):
    """ Downsample module for Inception """
    def __init__(self, in_channel, Ch3, **kwargs):
        super(downsample_module, self).__init__()

        self.conv_module = conv_module(in_channel=in_channel, C=Ch3, K=3, S=2, padding=1, **kwargs)
        self.max_pool = nn.MaxPool2d((3, 3), stride=2, ceil_mode=True)

    def forward(self, x):
        x1 = self.conv_module(x)
        x2 = self.max_pool(x)

        #print('conv size: {}, pool size: {}'.format(x1.size(), x2.size()))
        x = torch.cat([x1, x2], 1)

        return x


class Inception_small(nn.Module):
    
    """ Simplified Inception, adapted to fit CIFAR10 images """
    def __init__(self, num_classes=10, dropout_prob=0.0, init_scale=0.5, **kwargs):
        super(Inception_small, self).__init__()
        self.dropout_prob = dropout_prob
        
        self.conv_module = conv_module(in_channel=3, C=96, K=3, S=1, padding=1, **kwargs)
        self.Inception_module_1 = Inception_module(in_channel=96, Ch1=32, Ch2=32, **kwargs)
        self.Inception_module_2 = Inception_module(in_channel=64, Ch1=32, Ch2=48, **kwargs)
        self.downsample_module_1 = downsample_module(in_channel=80, Ch3=80, **kwargs)

        self.Inception_module_3 = Inception_module(in_channel=160, Ch1=112, Ch2=48, **kwargs)
        self.Inception_module_4 = Inception_module(in_channel=160, Ch1=96, Ch2=64, **kwargs)
        self.Inception_module_5 = Inception_module(in_channel=160, Ch1=80, Ch2=80, **kwargs)
        self.Inception_module_6 = Inception_module(in_channel=160, Ch1=48, Ch2=96, **kwargs)
        self.downsample_module_2 = downsample_module(in_channel=144, Ch3=96, **kwargs)

        self.Inception_module_7 = Inception_module(in_channel=240, Ch1=176, Ch2=160, **kwargs)
        self.Inception_module_8 = Inception_module(in_channel=336, Ch1=176, Ch2=160, **kwargs)
        self.mean_pooling = nn.AvgPool2d((7, 7))
        self.fc = nn.Linear(in_features=336, out_features=num_classes, bias=True)
        
        if self.dropout_prob > 0:
            self.dropout = nn.Dropout(p=self.dropout_prob)
            print(self.dropout)
        
        for m in self.modules():
            # print('Initialization')      
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, init_scale * math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

                size = m.weight.size()
                fan_out = size[0] # number of rows
                fan_in = size[1] # number of columns
                variance = math.sqrt(2.0/(fan_in + fan_out))
                m.weight.data.normal_(0.0, init_scale * variance)        

    def forward(self, x):
        x = self.conv_module(x)

        x = self.Inception_module_1(x)
        x = self.Inception_module_2(x)
        x = self.downsample_module_1(x)

        x = self.Inception_module_3(x)
        x = self.Inception_module_4(x)
        x = self.Inception_module_5(x)
        x = self.Inception_module_6(x)
        x = self.downsample_module_2(x)

        x = self.Inception_module_7(x)
        x = self.Inception_module_8(x)
        x = self.mean_pooling(x)
        
        if self.dropout_prob > 0:
            x = self.dropout(x)
            
        x = x.view(-1, 336)
        x = self.fc(x)

        return x

class Alexnet_module(nn.Module):
    
    """ Alexnet module for AlexNet """
    def __init__(self, in_channel, out_channel, init_scale=1.0):
        super(Alexnet_module, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(5, 5), padding='same')
        self.bn = nn.BatchNorm2d(out_channel, eps=0.001, momentum=0.1, affine=False,
                                 track_running_stats=True)  # added BN to alexnet so that it converges
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3)
        self.lrn = nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75)  # same setting as original Alexnet

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, init_scale * math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

                size = m.weight.size()
                fan_out = size[0] # number of rows
                fan_in = size[1] # number of columns
                variance = math.sqrt(2.0/(fan_in + fan_out))
                m.weight.data.normal_(0.0, init_scale * variance) 


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.lrn(x)

        return x


class Alexnet_small(nn.Module):
    
    """ Simplified AlexNet model to fit CIFAR10 images """
    def __init__(self):
        super(Alexnet_small, self).__init__()

        self.Alexnet_module_1 = Alexnet_module(in_channel=3, out_channel=64)
        self.Alexnet_module_2 = Alexnet_module(in_channel=64, out_channel=256)
        self.fc = nn.Sequential(nn.Linear(in_features=256 * 3 * 3, out_features=384, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=384, out_features=192, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=192, out_features=10, bias=True))

    def forward(self, x):
        x = self.Alexnet_module_1(x)
        x = self.Alexnet_module_2(x)

        x = x.view(-1, 256 * 3 * 3)
        x = self.fc(x)

        return x

    
class MLP(nn.Module):
    
    """ MLP with 1 or 3 layers, each hidden layer contains 512 hidden units """
    def __init__(self, layer=1):
        super(MLP, self).__init__()

        if layer == 1:
            self.fc = nn.Sequential(nn.Linear(in_features=3 * 28 * 28, out_features=512, bias=True),
                                    nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=False, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(in_features=512, out_features=10))

        if layer == 3:
            self.fc = nn.Sequential(nn.Linear(in_features=3 * 28 * 28, out_features=512, bias=True),
                                    nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=False, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(in_features=512, out_features=512, bias=True),
                                    nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=False, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(in_features=512, out_features=512, bias=True),
                                    nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=False, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(in_features=512, out_features=10, bias=True))

    def forward(self, x):

        x = x.view(-1, 3 * 28 * 28)
        x = self.fc(x)

        return x