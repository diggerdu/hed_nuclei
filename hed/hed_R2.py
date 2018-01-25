# -*- coding: UTF-8 -*-
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from utils.dataset import ISBI
from models.Hed_R2 import Hed_R2
from models.Hed_R import Hed_R
import torch_deform_conv.utils as utils
import torch.nn.init as init


parser = argparse.ArgumentParser(description='train hed model')
parser.add_argument('--batchSize', type=int, default=1, help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=15, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--outf', default='checkpoints/hed_R2/', help='folder to output images and model checkpoints')
parser.add_argument('--dataPath', default='facades/train/', help='path to training images')
parser.add_argument('--loadSize', type=int, default=512, help='scale image to this size')
parser.add_argument('--fineSize', type=int, default=512, help='random crop image to this size')
parser.add_argument('--input_nc', type=int, default=3, help='channel number of input image')
parser.add_argument('--output_nc', type=int, default=1, help='channel number of output image')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

lr_decay_epoch = {3, 5, 8 , 10}
lr_decay = 1./10
# pretrain_model = Hed_R(opt.input_nc, opt.output_nc, opt.ngf)
# pretrain_model.load_state_dict(torch.load('/home/wangbin/wb/pytorch/hed/checkpoints/hed_R_8.pth'))
###########   DATASET   ###########
isbi = ISBI()
train_loader = torch.utils.data.DataLoader(dataset=isbi,
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=2)

torch.cuda.set_device(1)

###########   MODEL   ###########
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        print (classname)
        init.xavier_uniform(m.weight.data)
        init.constant(m.bias.data, 0.1)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

ngf = opt.ngf
nc = 3

net = Hed_R2(opt.input_nc, opt.output_nc, opt.ngf)

net.cuda()

net.apply(weights_init)
# utils.transfer_weights(pretrain_model,net)
# net.load_state_dict(torch.load('/home/wangbin/wb/pytorch/hed/checkpoints/hed_R2_10.pth'))

print(net)

###########   LOSS & OPTIMIZER   ##########
criterion = nn.BCELoss()
lr = opt.lr
optimizer = torch.optim.Adam(net.parameters(),lr=lr, betas=(opt.beta1, 0.999))
# optimizer = torch.optim.SGD(net.parameters(),lr=lr, momentum=0.9, weight_decay=0.0002)

###########   GLOBAL VARIABLES   ###########



########### Training   ###########
net.train()
for epoch in range(1,opt.niter+1):
    for i, image in enumerate(train_loader):
        ########### fDx ###########
        net.zero_grad()

        imgA = image[0]
        imgB = image[1]
        img = Variable(imgA).cuda()
        gt = Variable(imgB).cuda()

        output = net(img)

        side_output1 = output[0]
        side_output2 = output[1]
        side_output3 = output[2]
        side_output4 = output[3]
        side_output5 = output[4]
        final_output = output[5]
        side_output1_R = output[6]
        side_output2_R = output[7]
        side_output3_R = output[8]
        side_output4_R = output[9]
        side_output5_R = output[10]
        final_output_R = output[11]
        side_output1_R2 = output[12]
        side_output2_R2 = output[13]
        side_output3_R2 = output[14]
        side_output4_R2 = output[15]
        side_output5_R2 = output[16]
        final_output_R2 = output[17]
        final_output_F = output[18]

        loss_side1 = criterion(side_output1, gt)
        loss_side2 = criterion(side_output2, gt)
        loss_side3 = criterion(side_output3, gt)
        loss_side4 = criterion(side_output4, gt)
        loss_side5 = criterion(side_output5, gt)
        final_loss = criterion(final_output, gt)
        loss_side1_R = criterion(side_output1_R, gt)
        loss_side2_R = criterion(side_output2_R, gt)
        loss_side3_R = criterion(side_output3_R, gt)
        loss_side4_R = criterion(side_output4_R, gt)
        loss_side5_R = criterion(side_output5_R, gt)
        final_loss_R = criterion(final_output_R, gt)
        loss_side1_R2 = criterion(side_output1_R2, gt)
        loss_side2_R2 = criterion(side_output2_R2, gt)
        loss_side3_R2 = criterion(side_output3_R2, gt)
        loss_side4_R2 = criterion(side_output4_R2, gt)
        loss_side5_R2 = criterion(side_output5_R2, gt)
        final_loss_R2 = criterion(final_output_R2, gt)
        final_loss_F = criterion(final_output_F, gt)


        loss = (loss_side1 + loss_side2 + loss_side3 + loss_side4 + loss_side5 + final_loss + loss_side1_R + loss_side2_R + loss_side3_R + loss_side4_R + loss_side5_R + final_loss_R \
            + loss_side1_R2 + loss_side2_R2 + loss_side3_R2 + loss_side4_R2 + loss_side5_R2 + final_loss_R2 + final_loss_F) 
        loss.backward()
        optimizer.step()


        ########### Logging ##########
        if(i % 10 == 0):
            print('[%d/%d][%d/%d] Loss_1_R2: %.4f Loss_2_R2: %.4f Loss_3_R2: %.4f Loss_4_R2: %.4f Loss_5_R2: %.4f Loss_all_R2: %.4f Loss_all_F: %.4f lr= %.8f'
                      % (epoch, opt.niter, i, len(train_loader),
                         loss_side1_R2.data[0], loss_side2_R2.data[0], loss_side3_R2.data[0], loss_side4_R2.data[0], loss_side5_R2.data[0], final_loss_R2.data[0], final_loss_F.data[0], lr))
        if(i % 1000 == 0):
            vutils.save_image(final_output.data,
                       'tmp/samples_i_%d_%03d.png' % (epoch, i),
                       normalize=True)

    if epoch in lr_decay_epoch:
            lr *= lr_decay
            optimizer = torch.optim.Adam(net.parameters(),lr=lr, betas=(opt.beta1, 0.999))


    # ########## Visualize #########
    # if(epoch % 5 == 0):
    #     vutils.save_image(fake_G.data,
    #                 'results/fake_samples_epoch_%03d.png' % (epoch),
    #                 normalize=True)

    torch.save(net.state_dict(), '%s/hed_R2_%d.pth' % (opt.outf, epoch))
