import torch
import torch.nn as nn
import torch.nn.functional as F

from Unet_Tool import *


# class ResBlock(nn.Module):
#     def __init__(self, inchannel, outchannel, dia=0):
#         super(ResBlock, self).__init__()
#         self.left = nn.Sequential(
#             nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(outchannel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(outchannel)
#         )
#
#         self.left1 =nn.Sequential(
#             nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(outchannel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=3, dilation=3),
#             nn.BatchNorm2d(outchannel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=5, dilation=5),
#             nn.BatchNorm2d(outchannel),
#         )
#
#         self.left2 = nn.Sequential(
#             nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(outchannel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(outchannel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=4, dilation=4),
#             nn.BatchNorm2d(outchannel),
#         )
#         self.shortcut = nn.Sequential()
#         # self.p1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)
#         # self.p2 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=3, dilation=3)
#         # self.p3 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=5, dilation=5)
#         # self.shortcut = nn.Sequential()
#         if inchannel != outchannel:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1),
#                 nn.BatchNorm2d(outchannel),
#             )
#         self.ReLU= nn.ReLU()
#         # self.De = nn.Conv2d(3*outchannel, outchannel, kernel_size=1, stride=1)
#     def forward(self, x, Pre_Feature, num):
#
#         # p1 = self.p1(x)
#         # p2 = self.p2(x)
#         # p3 = self.p3(x)
#         x1 = self.left(x)
#         x2 = self.left1(x)
#         x3 = self.left2(x)
#         # f = self.De(torch.cat([p1, p2, p3], dim=1))
#         if num==0:
#             out = self.ReLU(self.shortcut(x) + self.left(x) + 0.2*self.left1(x) + 0.2*self.left2(x))
#         # else:
#         #     out = self.ReLU(self.shortcut(x) + x1)
#
#         return out
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 64, num_block[1], 1)
        self.conv4_x = self._make_layer(block, 64, num_block[2], 1)
        self.conv5_x = self._make_layer(block, 64, num_block[3], 1)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        # output = self.avg_pool(output)
        # output = output.view(output.size(0), -1)
        # output = self.fc(output)

        return output


def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [6, 6, 6, 6])


class Get_RG(nn.Module):
    # the residual generation (RG) module
    def __init__(self, input_channel=256, beta=4, gamma=4):
        super(Get_RG, self).__init__()
        # self.D_r = Descriptor(input_channel, gamma)
        block = []
        for i in range(beta):
            block.append(nn.Conv2d(input_channel, 3, 2 * i + 1, 1, padding=i))
        self.conv_module = nn.ModuleList(block)
        self.activation = nn.Tanh()

    def forward(self, f_r):
        for i, module in enumerate(self.conv_module):
            if i == 0:
                r = module(f_r)
            else:
                r = r + module(f_r)
        r = self.activation(r)
        return r


class DP(nn.Module):
    # dilation pyramid
    def __init__(self, input_channel=512, beta=4, gamma=4):
        super(DP, self).__init__()
        # self.D_r = Descriptor(input_channel, gamma)
        block = []
        for i in range(beta):
            block.append(nn.Sequential(
                nn.Conv2d(input_channel, 256, 2 * i + 1, 1, padding=i),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 2 * i + 1, 1, padding=i),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 2 * i + 1, 1, padding=i),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 2 * i + 1, 1, padding=i),
            )
            )
        self.conv_module = nn.ModuleList(block)
        self.Encode = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        )
        self.activation = nn.ReLU()

    def forward(self, f_r):
        for i, module in enumerate(self.conv_module):
            if i == 0:
                r = module(f_r)
            else:
                r = torch.cat([module(f_r), r], dim=1)
        re = self.Encode(r)
        return re


class DP_Fea(nn.Module):
    # dilation pyramid
    def __init__(self, in_channel=64, out_channel=3, depth=32, gamma=3):
        super(DP_Fea, self).__init__()
        # self.Encode = nn.Conv2d(in_channel, 128, kernel_size=3, padding=1, stride=1)
        self.gamma = gamma
        self.ReLU = nn.ReLU()
        block = []
        block1 = []
        for i in range(gamma + 1):
            block.append(nn.Conv2d(in_channel, depth, 3, 1, padding=2 ** i, dilation=2 ** i))
        self.block = nn.ModuleList(block)

    def forward(self, feature):
        # feature = self.ReLU(self.Encode(fea))
        for i, block in enumerate(self.block):
            if i == 0:
                output = self.ReLU(block(feature))
            else:
                output = torch.cat([output, block(feature)], dim=1)
        return output


class Pyramid_maxout(nn.Module):
    def __init__(self, in_channel=64, depth=3, beta=4):
        super(Pyramid_maxout, self).__init__()
        block = []
        for i in range(beta):
            block.append(nn.Conv2d(in_channel, depth, 2 * i + 1, 1, padding=i))
        self.activation = nn.ReLU()
        self.conv_module = nn.ModuleList(block)

    def forward(self, f):
        for i, module in enumerate(self.conv_module):
            if i == 0:
                conv_result = module(f).unsqueeze(0)
            else:
                temp = module(f).unsqueeze(0)
                conv_result = torch.cat([conv_result, temp], dim=0)
        result, _ = torch.max(conv_result, dim=0)
        return self.activation(result)


class A_Get(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(A_Get, self).__init__()
        self.Sequential = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
        )

    def forward(self, x):
        return self.Sequential(x)


class Feature_Get(nn.Module):
    def __init__(self, in_channels):
        super(Feature_Get, self).__init__()
        self.ResNet = ResNet(BasicBlock, [8, 8, 8, 8])
        # self.block = []
        # # self.start = nn.Conv2d(3, 16, kernel_size=7, padding=3, stride=1)
        # self.block1 = ResBlock(in_channels,64)
        # self.block.append(self.block1)
        # self.block2 = ResBlock(64,64)
        # self.block.append(self.block2)
        # self.block3 = ResBlock(64, 64)
        # self.block.append(self.block3)
        # self.block4 = ResBlock(64, 64)
        # self.block.append(self.block4)
        # self.block5 = ResBlock(64, 64)
        # self.block.append(self.block5)
        # self.block6 = ResBlock(64, 64)
        # self.block.append(self.block6)
        # self.block7 = ResBlock(64, 64)
        # self.block.append(self.block7)
        # self.block8 = ResBlock(64, 64)
        # self.block.append(self.block8)
        # self.block9 = ResBlock(64, 64)
        # self.block.append(self.block9)
        # self.block10 = ResBlock(64, 64)
        # self.block.append(self.block10)
        # self.block11 = ResBlock(64, 64)
        # self.block.append(self.block11)
        # self.block12 = ResBlock(64, 64)
        # self.block.append(self.block12)
        # self.ReLU = nn.ReLU()
        # self.Dp = DP_Fea()

    def forward(self, x, Pre_Feature, num):
        x = self.ResNet(x)
        return x


# class RNN_Denow(nn.Module):
#     def __init__(self):
#         super(RNN_Denow, self).__init__()
#         self.Seg = Seg_UNet(n_channels=3, bilinear=True)
#
#
#     def forward(self, x):
#         # if num == 0:
#         # x_comb = torch.cat([x,x],dim=1)
#         # Fea_x = self.fea_get(x, [], num)
#         # Temp_Fea = self.fea_DP(Fea_x)
#         # Light_Mask = self.mask_out(Temp_Fea)
#
#         # Fea_con_x = self.fea_get_sec(con_x, [], num)
#         # Light_Mask = self.mask_out(Fea_con_x)
#         # Heavy_Predict = self.seg_heavy(Fea)
#         # else:
#         #     x_comb = torch.cat([x, Pre_x], dim=1)
#         #     Fea, f1 = self.fea_get(x_comb, Pre_Feature, num)
#         #     Light_Mask = self.mask_out(Fea) + Pre_Light_Mask
#
#         # Light_Mask[Light_Mask <= 0] = 0
#         # Light_Mask[Light_Mask >= 1] = 1
#         # fir = (-self.MaxPool(-Light_Mask))*0.85+self.AvgPool(Light_Mask)*0.15
#         # sec = (-self.MaxPool(-fir))*0.85+self.AvgPool(fir)*0.15
#         # thr = (-self.MaxPool(-sec))*0.85+self.AvgPool(sec)*0.15
#         # fou = (-self.MaxPool(-thr))*0.85+self.AvgPool(thr)*0.15
#
#         # Fea_fou, A_fou = self.UN_fou(Max_fou,fou,4)
#         # Fea_thr, A_thr = self.UN_thr(Max_thr, thr, A_fou, Fea_fou, 3)
#         # Fea_sec, A_sec = self.UN_sec(Max_sec, sec, A_thr, Fea_thr, 2)
#         # Fea_fir, A_fir, = self.UN_fir(Max_fir, fir, A_sec, Fea_sec, 1)
#         # Fea, A , = self.UN(con_x, Light_Mask, A_fir, Fea_fir, 0)
#         M = self.Seg(x, True)
#         # Mask = torch.argmax(M, dim=1)
#
#         return M
#         # , fir, sec, thr, fou
#
#
# class RNN_Desnow_Net(nn.Module):
#     def __init__(self):
#         super(RNN_Desnow_Net, self).__init__()
#         self.De_fir = RNN_Denow()
#         # self.De_sec = RNN_Denow()
#
#     def forward(self, x):
#         # Light_Mask, Fea, Fea_sec, out, End_Out, A= self.De_fir(x, [], [], [], [], 0)
#         # Light_Mask_sec, Fea_sec, Fea_sec_sec, out_sec, End_Out_sec, A_sec= self.De_sec(x, End_Out, Light_Mask, Fea, Fea_sec, 1)
#
#         # return Light_Mask, out, End_Out, Light_Mask_sec, out_sec, End_Out_sec, A, A_sec
#
#         # Light_Mask, A_Mask, A_fir, A_sec, A_thr, A_fou, out = self.De_fir(x, con_x,snow_fir_img, snow_sec_img,snow_thr_img,snow_fou_img,0)
#
#         A = self.De_fir(x)
#
#         return A

