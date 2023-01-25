#Deep Residual Learning for Image Recognition

#TODO *2 num channels works, now residual connection for that layer
import torch
import torch.nn as nn
import torchvision
import numpy as np

class Block(nn.Module):
    def __init__(self, input_channels, downsample=False):
        super().__init__()
        self.input_channels = input_channels
        self.downsample = downsample
        if self.downsample == False:
            self.conv_x = nn.Sequential(
                nn.Conv2d(self.input_channels, self.input_channels , 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.input_channels),
                nn.ReLU(),
                nn.Conv2d(self.input_channels,self.input_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.input_channels),
        )
        else:
            self.conv_x_2 = nn.Sequential(
                nn.Conv2d(self.input_channels,self.input_channels * 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(self.input_channels * 2),
                nn.ReLU(),
                nn.Conv2d(self.input_channels * 2,self.input_channels * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.input_channels * 2),
            )
            self.downsampling = nn.Sequential(
                nn.Conv2d(self.input_channels,self.input_channels * 2, 1, 2, 0, bias=False),
                nn.BatchNorm2d(self.input_channels * 2)

            )

    def forward(self, x):

        if self.downsample == True:

            out = self.conv_x_2(x)

            out = out + self.downsampling(x)

            return out
        else:

            out = self.conv_x(x)

            out = out + x

            return out


class Blocks(nn.Module):
    def __init__(self, in_channels, num_layer_of_blocks, num_blocks):
        super().__init__()
        self.num_layer_of_blocks = num_layer_of_blocks
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        
        if self.num_layer_of_blocks == 0:
            self.layer = nn.ModuleList([Block(self.in_channels) for _ in range(self.num_blocks)])

        else:
            self.layer = nn.ModuleList([Block(self.in_channels * 2) for _ in range(self.num_blocks - 1)])
            #adding downsample block at index 0 to self.layer
            self.layer.insert(0, Block(self.in_channels, True))
    def forward(self, x):
        
        out = x

        for f in self.layer:
            out = f(out)

        return out

class Resnet(nn.Module):
    def __init__(self,num_blocks = []):
        super().__init__()

        self.num_blocks = num_blocks

        self.conv1 = nn.Conv2d(3, 64, 7, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.relu = nn.ReLU()

        self.con2_x = Blocks(64, 0, num_blocks[0])
        
        self.con3_x = Blocks(64, 1, num_blocks[1])

        self.con4_x = Blocks(128, 2, num_blocks[2])

        self.con5_x = Blocks(256, 3, num_blocks[3])

        self.pool2 = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512, 1000, bias=True)
        """
        self.con4_x = Block(256, 256)

        self.con5_x = Block(512, 512)
        """
        
#32 512 512 1024 -> 32 1024
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.ReLU(self.pool1(out))
        out = self.con2_x(out)
        out = self.con3_x(out)
        out = self.con4_x(out)
        out = self.con5_x(out)
        out = self.pool2(out)
        out = out.squeeze(2).squeeze(2)
        out = self.linear(out)
        return out

model = Resnet([3,4,6,3])
print(model)
model2 = torchvision.models.resnet34()
# print(model)
print(model2)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

model_parameters = filter(lambda p: p.requires_grad, model2.parameters())
params2 = sum([np.prod(p.size()) for p in model_parameters])
print(f'number of parameters in pytorch implementation :{params}, \nnumber in parameters in this implemenation: {params2}')

