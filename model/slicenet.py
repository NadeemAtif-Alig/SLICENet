import torch
import torch.nn as nn
import torch.nn.functional as F


class InitialModule(nn.Module):
    def __init__(self, nIn, nOut):  # nIn = 3, nOut = 19
        super().__init__()
        n_int = int(nOut - nIn)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv = CBR(nIn, n_int, 3, stride=2)
        self.conv1 = CBR(n_int, n_int, 3, stride=1)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.conv1(self.conv(input))
        output = torch.cat([output, self.pool(input)], dim = 1)
        output = self.act(output)
        return output

class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        #self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        #self.conv1 = nn.Conv2d(nOut, nOut, (1, kSize), stride=1, padding=(0, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        #output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output

class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output

class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output

class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, groups = 1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, groups = groups)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class DownSampler(nn.Module):
    def __init__(self, nIn, nOut):  # lets say. nIn = 64, nOut = 128
        super().__init__() 
        n = int(nOut/5)  # 25
        self.conv_1x1 = nn.Conv2d(nIn, n, 1)                      # 19 --> 12
        self.conv = C(n, n, 3, 2, groups = n)                        # 12 --> 12
        self.conv_d2 = CDilated(n, n, 3, 1, 2)             # 12 --> 16
        self.conv_d4 = CDilated(n, n, 3, 1, 4)     # 12 --> 12
        self.conv_d8 = CDilated(n, n, 3, 1, 8)     # 12 --> 12
        self.conv_d16 =CDilated(n, n, 3, 1, 16)   # 12 --
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.conv(self.conv_1x1(input))
        d2 = self.conv_d2(output1)
        add1 = d2 + output1
        d4 = self.conv_d4(add1)
        d4 = d4 + d2               # This is the 2nd layer of add block

        d8 = self.conv_d8(output1)
        add2 = d8 + output1
        d16 = self.conv_d16(add2)
        d16 = d16 + d8
        
        combine = torch.cat([output1, d2, d4, d8, d16],1)
        output = self.bn(combine)
        output = self.act(output)
        return output

class BasicBlock(nn.Module):
    
    def __init__(self, nIn, nOut, add=True):
        
        super().__init__()
        n = int(nOut/5)                                           # n_squeezed = 12
        #n1 = nOut - 4*n_squeezed                                           # n1 = 16
        self.conv_1x1 = nn.Conv2d(nIn, n, 1)                      # 19 --> 12
        self.conv = C(n, n, 3, 1, groups = n)                        # 12 --> 12
        self.conv_d2 = CDilated(n, n, 3, 1, 2)             # 12 --> 16
        self.conv_d4 = CDilated(n, n, 3, 1, 4)     # 12 --> 12
        self.conv_d8 = CDilated(n, n, 3, 1, 8)     # 12 --> 12
        self.conv_d16= CDilated(n, n, 3, 1, 16)   # 12 --> 12

        self.bn = BR(nOut)
        self.add = add  # bool

    def forward(self, input):
        output1 = self.conv(self.conv_1x1(input))
        d2 = self.conv_d2(output1)
        add1 = d2 + output1
        d4 = self.conv_d4(add1)
        d4 = d4 + d2               # This is the 2nd layer of add block

        d8 = self.conv_d8(output1)
        add2 = d8 + output1
        d16 = self.conv_d16(add2)
        d16 = d16 + d8

        combine = torch.cat([output1, d2, d4, d8, d16], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output


class Network(nn.Module):
    
    def __init__(self, num_classes=19):
        
        super().__init__()
        self.stageE1 = InitialModule(3, 35)  # nIn, nOut, kSize, stride
        self.bn_relu1 = BR(35)    
        
        self.stageE2 = nn.ModuleList()
        self.stageE2.append(DownSampler(35, 65))
        self.stageE2.append(BasicBlock(65 , 65))
        self.stageE2.append(BasicBlock(65 , 65))
        self.stageE2.append(BasicBlock(65 , 65))
        self.stageE2.append(BasicBlock(65 , 65))
        self.bn_relu2 = BR(65)  
        
        self.stageE3 = nn.ModuleList()
        self.stageE3.append(DownSampler(65, 130))
        self.stageE3.append(BasicBlock(130 , 130))
        self.stageE3.append(BasicBlock(130 , 130))
        self.stageE3.append(BasicBlock(130 , 130))
        self.stageE3.append(BasicBlock(130 , 130))
        self.stageE3.append(BasicBlock(130 , 130))
        self.stageE3.append(BasicBlock(130 , 130))
        self.stageE3.append(BasicBlock(130 , 130))
        self.stageE3.append(BasicBlock(130 , 130))
        self.bn_relu3 = BR(130)

        self.projection = C(130, num_classes, 1, 1)

    def forward(self, input):
    
        output = self.bn_relu1(self.stageE1(input))  
 
        for layer in self.stageE2:
            output = layer(output) 
        output = self.bn_relu2(output)

        for layer in self.stageE3:
            output = layer(output)  
        output = self.bn_relu3(output)
        output = self.projection(output)
        output = F.interpolate(output, scale_factor=8, mode='bilinear', align_corners = True)
        return output      


