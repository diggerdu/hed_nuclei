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
from utils.dataset import ISBI_TEST
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
isbi = ISBI_TEST()
test_loader = torch.utils.data.DataLoader(dataset=isbi,
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


net.load_state_dict(torch.load('/home/wangbin/wb/pytorch/hed/checkpoints/hed_10.pth'))
# net.apply(weights_init)
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

img = Variable(img)

img= img.cuda()



########### testing   ###########
topilimage = transforms.ToPILImage()

for i, image in enumerate(test_loader):
    ########### fDx ###########

    imgA = image[0]
    name = image[1]
    img.data.resize_(imgA.size()).copy_(imgA)

    # train with fake
    output = net(img)

    side_output1 = output[0].cpu()
    side_output2 = output[1].cpu()
    side_output3 = output[2].cpu()
    side_output4 = output[3].cpu()
    side_output5 = output[4].cpu()
    final_output = output[5].cpu()

    print(i) 
    fn = name[0]

    save_dir = '/home/wangbin/wb/pytorch/hed/test/hed/'
    save_side1 = save_dir + '1/'
    save_side2 = save_dir + '2/'
    save_side3 = save_dir + '3/'
    save_side4 = save_dir + '4/'
    save_side5 = save_dir + '5/'
    save_fuse = save_dir + 'fuse/'

    if not os.path.exists(save_side1):
        os.makedirs(save_side1)
    if not os.path.exists(save_side2):
        os.makedirs(save_side2)
    if not os.path.exists(save_side3):
        os.makedirs(save_side3)
    if not os.path.exists(save_side4):
        os.makedirs(save_side4)
    if not os.path.exists(save_side5):
        os.makedirs(save_side5)
    if not os.path.exists(save_fuse):
        os.makedirs(save_fuse)


    to_pil_image = transforms.ToPILImage()
    side1 = to_pil_image(side_output1.data[0])
    side2 = to_pil_image(side_output2.data[0]) 
    side3 = to_pil_image(side_output3.data[0])
    side4 = to_pil_image(side_output4.data[0])
    side5 = to_pil_image(side_output5.data[0])
    fuse = to_pil_image(final_output.data[0])   
    
    side1.save(save_side1 + fn)
    side2.save(save_side2 + fn)
    side3.save(save_side3 + fn)
    side4.save(save_side4 + fn)
    side5.save(save_side5 + fn)
    fuse.save(save_fuse + fn)
#    vutils.save_image(fake_G.data,
#                   'test/%s' % (name))
