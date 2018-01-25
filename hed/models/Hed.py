import torch.nn as nn
import torch
import sys
sys.path.insert(0,'../')
from torch_deform_conv.layers import ConvOffset2D as DEF

class Hed(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Hed,self).__init__()

        self.conv1_1 = nn.Conv2d(input_nc, ngf, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(ngf, ngf, 3, 1, 1)

        self.conv2_1 = nn.Conv2d(ngf, ngf * 2, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1)

        self.conv3_1 = nn.Conv2d(ngf * 2, ngf * 4, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(ngf * 4, ngf * 4, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(ngf * 4, ngf * 4, 3, 1, 1)

        self.conv4_1 = nn.Conv2d(ngf * 4, ngf * 8, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)

        self.conv5_1 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)

        self.convD_23 = DEF(ngf * 2)
        self.convD_34 = DEF(ngf * 4)
        self.convD_45 = DEF(ngf * 8)

        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.conv1_s = nn.Conv2d(ngf, output_nc, 1, 1, 0)
        self.conv2_s = nn.Conv2d(ngf * 2, output_nc, 1, 1, 0)
        self.conv3_s = nn.Conv2d(ngf * 4, output_nc, 1, 1, 0)
        self.conv4_s = nn.Conv2d(ngf * 8, output_nc, 1, 1, 0)
        self.conv5_s = nn.Conv2d(ngf * 8, output_nc, 1, 1, 0)
        self.conv6 = nn.Conv2d(5, output_nc, 1, 1, 0)

        self.upsample_2 = nn.ConvTranspose2d(output_nc, output_nc, 4, 2, 1)
        self.upsample_4 = nn.ConvTranspose2d(output_nc, output_nc, 8, 4, 2)
        self.upsample_8 = nn.ConvTranspose2d(output_nc, output_nc, 16, 8, 4)
        self.upsample_16 = nn.ConvTranspose2d(output_nc, output_nc, 32, 16, 8)

        self.batch_norm = nn.BatchNorm2d(ngf)
        self.batch_norm2 = nn.BatchNorm2d(ngf * 2)
        self.batch_norm4 = nn.BatchNorm2d(ngf * 4)
        self.batch_norm8 = nn.BatchNorm2d(ngf * 8)

        # self.upsample_2 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.upsample_4 = nn.UpsamplingBilinear2d(scale_factor=4)
        # self.upsample_8 = nn.UpsamplingBilinear2d(scale_factor=8)
        # self.upsample_16 = nn.UpsamplingBilinear2d(scale_factor=16)

        self.relu = nn.ReLU(True)

        self.Sigmoid = nn.Sigmoid()

    def forward(self, input):

        e1_1 = self.batch_norm(self.relu(self.conv1_1(input)))
        e1_2 = self.batch_norm(self.relu(self.conv1_2(e1_1)))
        p1 = self.maxpool(e1_2) 

        e2_1 = self.batch_norm2(self.relu(self.conv2_1(p1)))
        e2_2 = self.batch_norm2(self.relu(self.conv2_2(e2_1)))
        p2 = self.maxpool(e2_2) 

        def23 = self.convD_23(p2)
        e3_1 = self.batch_norm4(self.relu(self.conv3_1(def23)))
        e3_2 = self.batch_norm4(self.relu(self.conv3_2(e3_1)))
        e3_3 = self.batch_norm4(self.relu(self.conv3_3(e3_2))) 
        p3 = self.maxpool(e3_3) 

        def34 = self.convD_34(p3)
        e4_1 = self.batch_norm8(self.relu(self.conv4_1(def34)))
        e4_2 = self.batch_norm8(self.relu(self.conv4_2(e4_1)))
        e4_3 = self.batch_norm8(self.relu(self.conv4_3(e4_2))) 
        p4 = self.maxpool(e4_3) 

        def45 = self.convD_45(p4)
        e5_1 = self.batch_norm8(self.relu(self.conv5_1(def45)))
        e5_2 = self.batch_norm8(self.relu(self.conv5_2(e5_1))) 
        e5_3 = self.batch_norm8(self.relu(self.conv5_2(e5_2))) 

        s1_1 = self.conv1_s(e1_2)
        side_output1 = self.Sigmoid(s1_1)

        s2_1 = self.conv2_s(e2_2)
        s2_2 = self.upsample_2(s2_1)
        side_output2 = self.Sigmoid(s2_2)

        s3_1 = self.conv3_s(e3_3)
        s3_2 = self.upsample_4(s3_1)
        side_output3 = self.Sigmoid(s3_2)

        s4_1 = self.conv4_s(e4_3)
        s4_2 = self.upsample_8(s4_1)
        side_output4 = self.Sigmoid(s4_2)

        s5_1 = self.conv5_s(e5_3)
        s5_2 = self.upsample_16(s5_1)
        side_output5 = self.Sigmoid(s5_2)

        c1 = torch.cat((s1_1, s2_2, s3_2, s4_2, s5_2), 1)

        c2 = self.conv6(c1)

        output = self.Sigmoid(c2)
        return side_output1, side_output2, side_output3, side_output4, side_output5, output
