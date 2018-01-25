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
from models.Hed import Hed


parser = argparse.ArgumentParser(description='train hed model')
parser.add_argument('--batchSize', type=int, default=1, help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=15, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--outf', default='checkpoints/', help='folder to output images and model checkpoints')
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

lr_decay_epoch = {5, 8 , 10}
lr_decay = 1./10
###########   DATASET   ###########
isbi = ISBI()
train_loader = torch.utils.data.DataLoader(dataset=isbi,
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=2)

torch.cuda.set_device(0)

###########   MODEL   ###########
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        print (classname)
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

ngf = opt.ngf
nc = 3

net = Hed(opt.input_nc, opt.output_nc, opt.ngf)

net.cuda()


# net.load_state_dict(torch.load('/home/wangbin/wb/pytorch/unet_gan/checkpoints/netD_rre_4.pth'))
net.apply(weights_init)
print(net)

###########   LOSS & OPTIMIZER   ##########
criterion = nn.BCELoss()
lr = opt.lr
optimizer = torch.optim.Adam(net.parameters(),lr=lr, betas=(opt.beta1, 0.999))

###########   GLOBAL VARIABLES   ###########
input_nc = opt.input_nc
output_nc = opt.output_nc
fineSize = opt.fineSize

img = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
gt = torch.FloatTensor(opt.batchSize, output_nc, fineSize, fineSize)

img = Variable(img)
gt = Variable(gt)

img= img.cuda()
gt= gt.cuda()


########### Training   ###########
net.train()
for epoch in range(1,opt.niter+1):
    for i, image in enumerate(train_loader):
        ########### fDx ###########
        net.zero_grad()

        imgA = image[0]
        imgB = image[1]
        img.data.resize_(imgA.size()).copy_(imgA)
        gt.data.resize_(imgB.size()).copy_(imgB)

        output = net(img)

        side_output1 = output[0]
        side_output2 = output[1]
        side_output3 = output[2]
        side_output4 = output[3]
        side_output5 = output[4]
        final_output = output[5]

        loss_side1 = criterion(side_output1, gt)
        loss_side2 = criterion(side_output2, gt)
        loss_side3 = criterion(side_output3, gt)
        loss_side4 = criterion(side_output4, gt)
        loss_side5 = criterion(side_output5, gt)
        final_loss = criterion(final_output, gt)

        loss = (loss_side1 + loss_side2 + loss_side3 + loss_side4 + loss_side5 + final_loss) 
        loss.backward()
        optimizer.step()


        ########### Logging ##########
        if(i % 10 == 0):
            print('[%d/%d][%d/%d] Loss_1: %.4f Loss_2: %.4f Loss_3: %.4f Loss_4: %.4f Loss_5: %.4f Loss_all: %.4f lr= %.4f'
                      % (epoch, opt.niter, i, len(train_loader),
                         loss_side1.data[0], loss_side2.data[0], loss_side3.data[0], loss_side4.data[0], loss_side5.data[0], final_loss.data[0], lr))
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

    torch.save(net.state_dict(), '%s/hed_%d.pth' % (opt.outf, epoch))
