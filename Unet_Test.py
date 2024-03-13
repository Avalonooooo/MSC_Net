import os
import torch
import argparse
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader

# import click
import numpy as np
from DataSet import *
from Unet_Tool import *

class conv(nn.Module):
    def __init__(self):
        super(conv, self).__init__()
        # self.dir =  [(0,1),(0,-1),(1,0),(-1,0)]
        self.snow_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        # print (self.snow_conv.weight.shape)
        self.x = torch.Tensor([[0,1,0],[1,0,1],[0,1,0]])
        self.x1 = torch.stack((self.x, self.x, self.x),dim=0)
        self.weight = torch.unsqueeze(self.x1,dim=1)
        self.weight = self.weight.to(device='cuda:0')
        self.snow_weight = nn.Parameter(data=self.weight, requires_grad=False)
        self.snow_conv.weight = self.snow_weight
        # torch.nn.init.constant_(self.snow_conv.weight, self.weight)
        # self.snow_conv.weight =
        for param in self.snow_conv.parameters():
            param.requires_grad = False




        self.mask_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        self.mask_conv.weight = self.snow_weight
        for param in self.mask_conv.parameters():
            param.requires_grad = False
        # self.Pad = nn.ConstantPad2d(padding=(1, 1, 1, 1), value=1)
    def forward(self, snow, mask, i):

        no_update_holes = mask==0
        # print (torch.count_nonzero(no_update_holes).item(),'@@@@@@@@@@@')

        output = self.snow_conv(snow*mask)

        output_mask = self.mask_conv(mask)
        # save_image(output, "Padding_Test/re_%d.png" % i, nrow=1, normalize=False)
        # print (self.snow_conv.weight)
        update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(update_holes, 1.0)
        output = output/mask_sum

        # output = output.masked_fill_(update_holes, 0.0)
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(update_holes, 0.0)
        sum = torch.count_nonzero(update_holes).item()
        return output, new_mask, no_update_holes, sum
        # l = []
        # mask_img = copy.deepcopy(mask)
        # snow_img = copy.deepcopy(snow)
        # for e in li:
        #     i, j, k, m = e[0], e[1], e[2], e[3]
        #     num = 0
        #     avg = 0
        #     # print(i, j, k, m)
        #     for n in self.dir:
        #         if k + n[0] < h and k + n[0] >= 0 and m + n[1] < w and m + n[1] >= 0 and mask_img[i, j, k + n[0], m + n[1]] < 0.5:
        #             # print ("!!!!!")
        #             num += 1
        #             avg += snow_img[i, j, k + n[0], m + n[1]]
        #             # avg = max(avg,snow_img[i,j,k+n[0],m+n[1]])
        #     if num > 0:
        #         snow[i, j, k, m] = avg * 1.0 / (num * 1.0)
        #         mask[i, j, k, m] = 0
        #     else:
        #         l.append((i, j, k, m))
        #
        # return snow, mask, l

class partial_avg(nn.Module):
    def __init__(self):
        super(partial_avg, self).__init__()
        self.conv = conv()

    def forward(self, snow, mask):
        # b, c, h, w = mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3]
        # num = b*c*h*w
        # li = []
        # li = torch.nonzero(mask >= 0.5)
        # print(len(li))
        i = 0
        while 1:
            i+=1
            # mask = self.Pad(mask)
            output_snow, new_mask, pre_mask, sum= self.conv(snow, mask, i)
            snow = snow*(~pre_mask) + pre_mask*output_snow
            mask = new_mask
            # save_image(snow, "Padding_Test/re_%d.png" % i, nrow=1, normalize=False)
            # print (sum)
            if sum == 0 :
                break
            # print(len(l), "!!!!!!!!!!!!!!!!!!!!!")
            # if len(l) == 0:
            #     break
        return snow
if __name__ == '__main__':



    argparser = argparse.ArgumentParser(description='Train the model')

    argparser.add_argument(
        '--device',
        type=str,
        default='cuda:0'
    )

    argparser.add_argument(
        '-r',
        '--root',
        default='C:/DataSet/t/all/all',
        type=str,
        help='root directory of trainset'
    )

    argparser.add_argument(
        '-dir',
        type=str,
        default='weight/',
        help='path to store the model checkpoints'
    )

    argparser.add_argument(
        '-iter',
        '--iterations',
        type=int,
        default=2000
    )

    argparser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        default=1e-3
    )

    argparser.add_argument(
        '--batch_size',
        type=int,
        default=1
    )

    argparser.add_argument(
        '-beta',
        type=int,
        default=4,
        help='the scale of the pyramid maxout'
    )

    argparser.add_argument(
        '-gamma',
        type=int,
        default=4,
        help='the levels of the dilation pyramid'
    )

    argparser.add_argument(
        '--weight_decay',
        type=float,
        default=5e-4
    )

    argparser.add_argument(
        '--weight_mask',
        type=float,
        default=3,
        help='the weighting to leverage the importance of snow mask'
    )

    argparser.add_argument(
        '--save_schedule',
        type=int,
        nargs='+',
        default=[],
        help='the schedule to save the model'
    )

    argparser.add_argument(
        '--mode',
        type=str,
        default='original',
        help='the architectural mode of DesnowNet'
    )

    # argparser.add_argument('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')

    args = argparser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # net, starting_epoch = build_network(snapshot, backend)
    # data_path = os.path.abspath(os.path.expanduser(data_path))
    # models_path = os.path.abspath(os.path.expanduser(models_path))
    # os.makedirs(models_path, exist_ok=True)
    gt_root = os.path.join(args.root, 'gt')
    mask_root = os.path.join(args.root, 'mask')
    synthetic_root = os.path.join(args.root, 'synthetic')
    dataset = snow_dataset(gt_root, mask_root, synthetic_root, mode='train')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=1,
                                              pin_memory=True)
    '''
        To follow this training routine you need a DataLoader that yields the tuples of the following format:
        (Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y_cls) where
        x - batch of input images,
        y - batch of groung truth seg maps,
        y_cls - batch of 1D tensors of dimensionality N: N total number of classes, 
        y_cls[i, T] = 1 if class T is present in image i, 0 otherwise
    '''
    net = AttU_Net().to(device=args.device)
    # net_end = RNN_Desnow_Net_End().to(device=args.device)
    # optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True)
    # seg_criterion = nn.CrossEntropyLoss()
    # crit = nn.L1Loss()
    # crit_sum = nn.L1Loss(reduction='sum')
    # P_Avg = partial_avg().to(device=args.device)
    # criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device=args.device)
    # se_loss = LossFunc_One().to(device=args.device)

    checkpoint = torch.load(os.path.join(args.dir, 'checkpoints_Unet_seg_Zero_point_two_five_itea90.pth'))
    net.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # check_point = torch.load(os.path.join(args.dir, 'checkpoints_End_itea21.pth'))
    # net_end.load_state_dict(check_point['model_state_dict'])
    # optimizer.load_state_dict(check_point['optimizer_state_dict'])

    Max = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
    Avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
    iteration = 0
    number = 0
    sum = 0
    net.eval()
    # net_end.eval()
    # while iteration < args.iterations:
    #     iteration += 1
    number = 0
    #     sum = 0
    with torch.no_grad():
        for data in data_loader:
            number += 1
            if number > 10:
                break;



            snow_img, gt_img, mask_h, mask_l = data

            gt_img = gt_img.to(device=args.device)
            snow_img = snow_img.to(device=args.device)

            mask_hh = torch.LongTensor(mask_h.numpy())
            mask_hh = mask_hh.to(device=args.device)

            mask_h = mask_h.to(device=args.device)
            mask_l = mask_l.to(device=args.device)

            print (mask_l.shape)
            out = net(snow_img)
            out = torch.argmax(out,dim=1,keepdim=True)
            print (out.shape)
            out = torch.cat([out,out,out],dim=1)

            print (mask_h.shape)
            mask_h = torch.unsqueeze(mask_h, dim=1)
            mask_h = torch.cat([mask_h,mask_h,mask_h],dim=1)

            snow_fir_img = 0.85 * Max(snow_img) + 0.15 * Avg(snow_img)
            snow_sec_img = 0.85 * Max(snow_img) + 0.15 * Avg(snow_fir_img)
            snow_thr_img = 0.85 * Max(snow_img) + 0.15 * Avg(snow_sec_img)
            snow_fou_img = 0.85 * Max(snow_img) + 0.15 * Avg(snow_thr_img)
            # Light_Mask,Heavy_Predict = net(snow_img)
            # Heavy_Mask = torch.argmax(Heavy_Predict, dim=1, keepdim=True)
            # with torch.no_grad():
            #     Light_Mask, Heavy_Predict = net(snow_img)
            # Heavy_Mask = torch.argmax(Heavy_Predict, dim=1, keepdim=True)
            # H_M = 1-torch.cat([Heavy_Mask,Heavy_Mask,Heavy_Mask],dim=1)
            # with torch.no_grad():
            #     snow_img = torch.tensor(snow_img,dtype=torch.float32)
            #     mask_l = torch.tensor(mask_l,dtype=torch.float32)
            #     H_M_one = torch.tensor(H_M,dtype=torch.float32)
            #     H_M_sec = torch.tensor(H_M, dtype=torch.float32)
            #     avg_snow_img = P_Avg(snow_img, H_M_one)
            #     avg_snow_mask = P_Avg(mask_l,H_M_sec)
            #     Avg_Light_Mask, P_Avg_Predict = net(avg_snow_img)
            # con_snow = 1-snow_img
            # print (hea_mask.shape, lig_mask.shape)
            # print (final.shape, mask_h.shape, mask_l.shape)
            #res = torch.cat([mask_l,gt_img, snow_img, Image_1, Image_2, Light_mask_fir, torch.cat([Heavy_Mask,Heavy_Mask,Heavy_Mask],dim=1)], dim=3)
            #L_M = torch.cat([Light_Mask,Light_Mask,Light_Mask],dim=1)
            # mask_h = torch.unsqueeze(mask_h,dim=1)
            # mask_l = torch.cat([mask_l,mask_l,mask_l],dim=1)
            # res = torch.cat([snow_img,gt_img,snow_img-gt_img,1-gt_img,Avg_fir_img,Avg_sec_img,Avg_thr_img,Avg_fou_img,fir_img,sec_img,thr_img,fou_img],dim=3)
            # res_next = torch.cat([mask_h, mask_l], dim = 2)
            save_image(torch.cat([snow_img,out,mask_l,mask_h],dim=3), "result/end_%d.png" % number, nrow=1, normalize=False)



