import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, dia=0):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=2**dia, dilation=2**dia),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=2**dia, dilation=2**dia),
        )

        self.left1 =nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.ReLU(),
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=5, dilation=5),
        )

        self.left2 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=4, dilation=4),
        )

        self.p1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.p2 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=3, dilation=3)
        self.p3 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=5, dilation=5)
        self.shortcut = nn.Sequential()
        if inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1),
            )
        self.ReLU= nn.ReLU()
        self.De = nn.Conv2d(3*outchannel, outchannel, kernel_size=1, stride=1)
    def forward(self, x):
        # if connection==True :
        #     out = self.left(x)
        #     out = out + self.shortcut(x)
        # else :
        #     out = self.left(x)
        # if num == 0:
        #     out = self.ReLU(out)
        # else:
        #     out = self.ReLU(out) + Pre_Feature
        # return out
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        x1 = self.left(x)
        x2 = self.left1(x)
        x3 = self.left2(x)
        # print (x1.shape,x2.shape,x3.shape)
        f = self.De(torch.cat([p1, p2, p3], dim=1))
        out = self.ReLU(self.shortcut(x) + x1 + 0.1*x2 + 0.1*x3 + 0.1*f)

        return out



class DP(nn.Module):
    # dilation pyramid
    def __init__(self, in_channel=384, out_channel=3, depth=64, gamma=3):
        super(DP, self).__init__()
        # self.Encode = nn.Conv2d(in_channel, 128, kernel_size=3, padding=1, stride=1)
        self.gamma = gamma
        self.ReLU = nn.ReLU()
        block = []
        block1 = []
        for i in range(gamma + 1):
            block.append(nn.Conv2d(128, depth, 3, 1, padding=2 ** i, dilation=2 ** i))
            # block1.append(nn.Conv2d(64, depth, 3, 1, padding=2 ** i, dilation=2 ** i))
        self.block = nn.ModuleList(block)
        self.Decode = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, padding=2 ** 1, dilation=2 ** 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=2 ** 2, dilation=2 ** 2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=7, padding=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=1),
            # nn.ReLU(),
            # nn.Conv2d(64, 3, kernel_size=3, padding=1, stride=1),
        )

    def forward(self, feature):
        # feature = self.ReLU(self.Encode(fea))
        for i, block in enumerate(self.block):
            if i == 0:
                output = self.ReLU(block(feature))
            else:
                output = torch.cat([output, block(feature)], dim=1)
        return self.Decode(output)

class Pyramid_maxout(nn.Module):
    def __init__(self, in_channel=128, depth=3, beta=4):
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
    def __init__(self,in_channel,out_channel):
        super(A_Get, self).__init__()
        self.Sequential=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
        )
    def forward(self, x):
        return self.Sequential(x)


class Feature_Get(nn.Module):
    def __init__(self,inchannel=3,outchannel=128):
        super(Feature_Get, self).__init__()

        self.block = []
        # self.start = nn.Conv2d(3, 16, kernel_size=7, padding=3, stride=1)
        self.block1 = ResBlock(3,16)
        self.block.append(self.block1)
        self.block2 = ResBlock(16,64)
        self.block.append(self.block2)
        self.block3 = ResBlock(64, 64)
        self.block.append(self.block3)
        self.block4 = ResBlock(64, 64)
        self.block.append(self.block4)
        self.block5 = ResBlock(64, 64)
        self.block.append(self.block5)
        self.block6 = ResBlock(64, 64)
        self.block.append(self.block6)
        self.block7 = ResBlock(64, 64)
        self.block.append(self.block7)
        self.block8 = ResBlock(64, 64)
        self.block.append(self.block8)
        self.block9 = ResBlock(64, 64)
        self.block.append(self.block9)
        self.block10 = ResBlock(64, 64)
        self.block.append(self.block10)
        self.block11 = ResBlock(64, 64)
        self.block.append(self.block11)
        self.block12 = ResBlock(64, 64)
        self.block.append(self.block12)
        self.ReLU = nn.ReLU()
    def forward(self, x):
        # x = self.ReLU(self.start(x))
        for i in range(12):
            if i <=9:
                c = True
            else:
                c = False
            x = self.block[i].forward(x)
            # f.append(x)
        # x = self.ReLU(self.block11(x))
        # # f.append(x)
        # x = self.ReLU(self.block12(x))
        # f.append(x)
        # else:
        #     for i in range(11):
        #         x = self.block[i].forward(x,Pre_Feature[i],1)
        #         f.append(x)
        #     x = self.ReLU(self.block12(x)) + Pre_Feature[11]
        #     f.append(x)
        #     x = self.ReLU(self.block13(x)) + Pre_Feature[12]
        #     f.append(x)

        return x

class RNN_Denow(nn.Module):
    def __init__(self):
        super(RNN_Denow, self).__init__()
        self.fea_get = Feature_Get(inchannel=3,outchannel=64)
        # self.Fea_get = Feature_Get(inchannel=6,outchannel=3)
        self.mask_out = Pyramid_maxout(in_channel=64,depth=3)
        self.a_out = A_Get(in_channel=64,out_channel=3)
        # self.get_light_re = DP(in_channel=9,out_channel=3)
        self.get_heavy_re = DP(in_channel=6, out_channel=3)
        self.seg_heavy = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2,kernel_size=3, stride=1, padding=1),
        )
        self.get_re = nn.Sequential(
            nn.Conv2d(70, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        )
        self.get_re_next = nn.Sequential(
            nn.Conv2d(70, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        )
        self.Decode1 =  nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        )
        self.Decode2 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
                                     )
        self.All_Decode = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
                                    # nn.ReLU(),
                                    # nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
                                    )
    def forward(self, x):

        # # if num == 0:
        # Fea = self.fea_get(x)
        # # else:
        # #     Fea= self.fea_get(x, Pre_Feature, num)
        #
        # Light_Mask = self.mask_out(Fea)
        # Heavy_Predict = self.seg_heavy(Fea)
        # # else:
        # #     Light_Mask = self.mask_out(Fea)
        # #     Heavy_Predict = self.seg_heavy(Fea)
        #
        # A = self.a_out(Fea)
        # A_Mask = self.get_re(torch.cat([A,Light_Mask,Fea], dim=1))
        # Heavy_Mask = torch.argmax(Heavy_Predict, dim=1, keepdim=True)
        # H_M = torch.cat([Heavy_Mask,Heavy_Mask,Heavy_Mask],dim=1)
        # H_Re = Heavy_Mask * H_M
        # Img_one_stage = x - A_Mask
        # Img_one_stage[Img_one_stage<=0] = 0
        # # Img_one_detach = Img_one_stage.detach()
        # # Re_Light = self.get_light_re(torch.cat([Miss,Light_Mask,Light_Mask*Miss],dim=1))
        # # Img_one_stage = Miss + Re_Light
        # # Re_Heavy = Light_Mask * self.All_Decode(self.get_heavy_re(torch.cat([Light_Mask, Img_one_stage], dim=1)))
        # if iteration>=9:
        #     Img_temp = Img_one_stage * (1-H_M) + H_Re
        #     A_next_mask = self.get_re_next(torch.cat([A,Light_Mask*H_M,Fea], dim=1))
        #     Img_two_stage = Img_temp - A_next_mask
        # else :
        #     Img_two_stage = Img_one_stage

        Fea = self.fea_get(x)
        Light_Mask = self.mask_out(Fea)
        Heavy_Predict = self.seg_heavy(Fea)
        # return Img_one_stage, Img_two_stage, Heavy_Predict, Light_Mask, Heavy_Mask, A
        return Light_Mask, Heavy_Predict
class RNN_Desnow_Net(nn.Module):
    def __init__(self):
        super(RNN_Desnow_Net, self).__init__()
        self.De_fir = RNN_Denow()
        # self.De_sec = RNN_Denow()

    def forward(self, x):
        #Image_1, Image_2, Heavy_Predict_fir, Light_mask_fir, Heavy_Mask, A_fir = self.De_fir(x, iteration)
        # Image_3, Image_4, Heavy_Predict_sen, Light_mask_sen, Res, Heavy_Mask, A_sec, Re_Heavy= self.De_sec(Image_2, Light_mask_fir, Res, Heavy_Predict_fir, 1)

        # return Image_1, Image_2, Image_3, Image_4, Heavy_Predict_fir, Heavy_Predict_sen, Light_mask_fir, Light_mask_sen,Heavy_Mask, A_fir,A_sec, Re_Heavy
        #return Image_1, Image_2, Heavy_Predict_fir, Light_mask_fir, A_fir, Heavy_Mask
        Light_Mask, Heavy_Predict = self.De_fir(x)

        return Light_Mask,Heavy_Predict

class RNN_Denow_End(nn.Module):
    def __init__(self):
        super(RNN_Denow_End, self).__init__()
        self.fea_get = Feature_Get(inchannel=3,outchannel=64)
        # self.Fea_get = Feature_Get(inchannel=6,outchannel=3)
        self.mask_out = Pyramid_maxout(in_channel=64,depth=3)
        self.a_out = A_Get(in_channel=64,out_channel=3)
        # self.get_light_re = DP(in_channel=9,out_channel=3)
        self.get_heavy_re = DP(in_channel=6, out_channel=3)
        self.seg_heavy = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2,kernel_size=3, stride=1, padding=1),
        )
        self.get_re = nn.Sequential(
            nn.Conv2d(70, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        )
        self.get_re_next = nn.Sequential(
            nn.Conv2d(70, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        )
        self.A_Decode =  nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        )

        self.M_Decode = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                      )
        self.All_Encode = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
                                    )
        # self.Decode2 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        #                              nn.ReLU(),
        #                              nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #                              nn.ReLU(),
        #                              nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #                              nn.ReLU(),
        #                              nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        #                              )
        # self.All_Decode = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        #                             nn.ReLU(),
        #                             nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
        #                             nn.ReLU(),
        #                             nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
        #                             # nn.ReLU(),
        #                             # nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        #                             )
    def forward(self, x):

        # # if num == 0:
        Fea = self.fea_get(x)
        # # else:
        # #     Fea= self.fea_get(x, Pre_Feature, num)
        #
        # Light_Mask = self.mask_out(Fea)
        Heavy_Predict = self.seg_heavy(Fea)
        # # else:
        # #     Light_Mask = self.mask_out(Fea)
        # #     Heavy_Predict = self.seg_heavy(Fea)
        #
        Light_Mask = self.mask_out(Fea)
        A = self.a_out(Fea)
        A_Mask = self.get_re(torch.cat([A,Light_Mask,Fea], dim=1))
        Heavy_Mask = torch.argmax(Heavy_Predict, dim=1, keepdim=True)
        # H_M = torch.cat([Heavy_Mask,Heavy_Mask,Heavy_Mask],dim=1)
        # H_Re = Heavy_Mask * H_M
        Img = x - A_Mask
        # Img_one_stage[Img_one_stage<=0] = 0
        # # Img_one_detach = Img_one_stage.detach()
        # # Re_Light = self.get_light_re(torch.cat([Miss,Light_Mask,Light_Mask*Miss],dim=1))
        # # Img_one_stage = Miss + Re_Light
        # # Re_Heavy = Light_Mask * self.All_Decode(self.get_heavy_re(torch.cat([Light_Mask, Img_one_stage], dim=1)))
        # if iteration>=9:
        #     Img_temp = Img_one_stage * (1-H_M) + H_Re
        #     A_next_mask = self.get_re_next(torch.cat([A,Light_Mask*H_M,Fea], dim=1))
        #     Img_two_stage = Img_temp - A_next_mask
        # else :
        #     Img_two_stage = Img_one_stage

        # Fea = self.fea_get(x)
        # Light_Mask = self.mask_out(Fea)
        # Heavy_Predict = self.seg_heavy(Fea)
        # # return Img_one_stage, Img_two_stage, Heavy_Predict, Light_Mask, Heavy_Mask, A
        # return Light_Mask, Heavy_Predict
        return Img, A, Light_Mask, Heavy_Predict
class RNN_Desnow_Net_End(nn.Module):
    def __init__(self):
        super(RNN_Desnow_Net_End, self).__init__()
        self.De_sen = RNN_Denow_End()
        # self.De_sec = RNN_Denow()

    def forward(self, x):
        #Image_1, Image_2, Heavy_Predict_fir, Light_mask_fir, Heavy_Mask, A_fir = self.De_fir(x, iteration)
        # Image_3, Image_4, Heavy_Predict_sen, Light_mask_sen, Res, Heavy_Mask, A_sec, Re_Heavy= self.De_sec(Image_2, Light_mask_fir, Res, Heavy_Predict_fir, 1)

        # return Image_1, Image_2, Image_3, Image_4, Heavy_Predict_fir, Heavy_Predict_sen, Light_mask_fir, Light_mask_sen,Heavy_Mask, A_fir,A_sec, Re_Heavy
        #return Image_1, Image_2, Heavy_Predict_fir, Light_mask_fir, A_fir, Heavy_Mask
        Img, A, Light_Mask, H_Pre= self.De_sen(x)

        return Img, A, Light_Mask, H_Pre

