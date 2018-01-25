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
from models.Hed_R2_rec import Hed_R2


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

###########   DATASET   ###########
isbi = ISBI_TEST()
test_loader = torch.utils.data.DataLoader(dataset=isbi,
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=2)

torch.cuda.set_device(1)

###########   MODEL   ###########

ngf = opt.ngf
nc = 3

net = Hed_R2(opt.input_nc, opt.output_nc, opt.ngf)

net.cuda()


net.load_state_dict(torch.load('checkpoints/rec/hed_R2_14.pth'))
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

    final_output = output[5].cpu()
    final_output_R = output[11].cpu()
    final_output_R2 = output[17].cpu()
    final_output_F = output[18].cpu()

    print(i) 
    fn = name[0]

    save_dir = 'test/rec/'
    save_fuse = save_dir + 'fuse/'
    save_fuse_R = save_dir + 'R/'
    save_fuse_R2 = save_dir + 'R2/'
    save_fuse_F = save_dir + 'F/'

    if not os.path.exists(save_fuse):
        os.makedirs(save_fuse)
    if not os.path.exists(save_fuse_R):
        os.makedirs(save_fuse_R)
    if not os.path.exists(save_fuse_R2):
        os.makedirs(save_fuse_R2)
    if not os.path.exists(save_fuse_F):
        os.makedirs(save_fuse_F)


    to_pil_image = transforms.ToPILImage()
    fuse = to_pil_image(final_output.data[0]) 
    R = to_pil_image(final_output_R.data[0]) 
    R2 = to_pil_image(final_output_R2.data[0]) 
    F = to_pil_image(final_output_F.data[0])   
    
    fuse.save(save_fuse + fn)
    R.save(save_fuse_R + fn)
    R2.save(save_fuse_R2 + fn)
    F.save(save_fuse_F + fn)
#    vutils.save_image(fake_G.data,
#                   'test/%s' % (name))
