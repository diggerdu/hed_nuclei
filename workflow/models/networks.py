import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules import Module
import numpy as np
from collections import OrderedDict
from . import densenet_efficient as dens
from . import time_frequence as tf

###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or classname.find(
            'InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    else:
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer
    # return None

def define_G(opt):
    use_gpu = len(opt.gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    #netG = AuFCNWrapper(n_fft, hop, gpu_ids)
    if opt.which_model_netG == 'cnn':
        netG = CNN(opt.nClasses, opt.gpu_ids)
    elif opt.which_model_netG == 'drcnn':
        netG = DrCNN(opt.nClasses, opt.gpu_ids)
    elif opt.which_model_netG == 'lstm':
        assert len(opt.gpu_ids) == 1
        netG = LSTM(64, 128, opt.nClasses, opt.batchSize)
    elif opt.which_model_netG == 'bilstm':
        assert len(opt.gpu_ids) == 1
        netG = BiLSTM(64, 128, opt.nClasses, opt.batchSize)
    elif opt.which_model_netG == 'cnnFeatureExtractor':
        netG = cnnFeatureExtractor(opt.nClasses, opt.gpu_ids)
    elif opt.which_model_netG == 'LSTMFeatureExtractor':
        netG = LSTMFeatureExtractor(64, 128, opt.nClasses, opt.batchSize)
    elif opt.which_model_netG == 'fusenet':
        netG = FuseNet(opt.nClasses, 256, opt.gpu_ids)
    #netG = LSTM()
    if len(opt.gpu_ids) > 0:
        netG.cuda(device_id=opt.gpu_ids[0])
    netG.apply(weights_init)
    return netG




def define_D(input_nc,
             ndf,
             which_model_netD,
             n_layers_D=3,
             norm='batch',
             use_sigmoid=False,
             gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(
            input_nc,
            ndf,
            n_layers=3,
            norm_layer=norm_layer,
            use_sigmoid=use_sigmoid,
            gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(
            input_nc,
            ndf,
            n_layers_D,
            norm_layer=norm_layer,
            use_sigmoid=use_sigmoid,
            gpu_ids=gpu_ids)
    else:
        print('Discriminator model name [%s] is not recognized' %
              which_model_netD)
    if use_gpu:
        netD.cuda(device_id=gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self,
                 use_lsgan=True,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None)
                            or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(
                    real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None)
                            or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(
                    fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class cnnFeatureExtractor(nn.Module):
    def __init__(self, numClass, gpu_ids):
        super(cnnFeatureExtractor, self).__init__()
        self.gpu_ids = gpu_ids
        modelList = list()
        width = 32
        modelList.append(nn.ReflectionPad2d((1, 0, 1, 0)))
        modelList.append(nn.Conv2d(1, width, kernel_size=2))

        for i in range(4):
            modelList.append(nn.BatchNorm2d(width))
            modelList.append(nn.LeakyReLU(inplace=True))
            modelList.append(nn.ReflectionPad2d((1, 0, 1, 0)))
            modelList.append(nn.Conv2d(width, width * 2, kernel_size=3, stride=2))
            width *= 2

        modelList.append(nn.BatchNorm2d(width))
        modelList.append(nn.LeakyReLU(inplace=True))
        modelList.append(nn.ReflectionPad2d((1, 0, 1, 0)))
        modelList.append(nn.Conv2d(width, width, kernel_size=3, stride=2))


        self.model = nn.ModuleList([*modelList,
                                    flatten(),
                                    nn.Linear(2048, 512),
                                    nn.SELU(inplace=True),
                                    nn.Linear(512, 128),
                                    nn.SELU(inplace=True),
                                    nn.Linear(128, numClass),
                                    nn.LogSoftmax()
                                    ])

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data,torch.cuda.FloatTensor):
            output = input
            for index, model in enumerate(self.model):
                output = model(output)
                if index == (len(self.model)-3):
                    return {'features': output, 'logits': None}
        else:
            return self.model(input)

class CNN(nn.Module):
    def __init__(self, numClass, gpu_ids):
        super(CNN, self).__init__()
        self.gpu_ids = gpu_ids
        modelList = list()
        width = 32
        modelList.append(nn.ReflectionPad2d((1, 0, 1, 0)))
        modelList.append(nn.Conv2d(1, width, kernel_size=2))

        for i in range(4):
            modelList.append(nn.BatchNorm2d(width))
            modelList.append(nn.LeakyReLU(inplace=True))
            modelList.append(nn.ReflectionPad2d((1, 0, 1, 0)))
            modelList.append(nn.Conv2d(width, width * 2, kernel_size=3, stride=2))
            width *= 2

        modelList.append(nn.BatchNorm2d(width))
        modelList.append(nn.LeakyReLU(inplace=True))
        modelList.append(nn.ReflectionPad2d((1, 0, 1, 0)))
        modelList.append(nn.Conv2d(width, width, kernel_size=3, stride=2))


        self.model = nn.Sequential(*modelList,
                                    flatten(),
                                    nn.Linear(2048, 512),
                                    nn.SELU(inplace=True),
                                    nn.Linear(512, 128),
                                    nn.SELU(inplace=True),
                                    nn.Linear(128, numClass),
                                                      nn.LogSoftmax()
                )

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data,torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            return {'logits': output}
        else:
            return self.model(input)

class DrCNN(nn.Module):
    def __init__(self, numClass, gpu_ids):
        super(DrCNN, self).__init__()
        self.gpu_ids = gpu_ids
        modelList = list()
        width = 32
        modelList.append(nn.ReflectionPad2d((1, 0, 1, 0)))
        modelList.append(nn.Conv2d(1, width, kernel_size=2))

        for i in range(4):
            modelList.append(nn.BatchNorm2d(width))
            modelList.append(nn.LeakyReLU(inplace=True))
            modelList.append(nn.ReflectionPad2d((1, 0, 1, 0)))
            modelList.append(nn.Conv2d(width, width * 2, kernel_size=3, stride=2))
            width *= 2

        modelList.append(nn.BatchNorm2d(width))
        modelList.append(nn.LeakyReLU(inplace=True))
        modelList.append(nn.ReflectionPad2d((1, 0, 1, 0)))
        modelList.append(nn.Conv2d(width, width, kernel_size=3, stride=2))


        self.model = nn.Sequential(*modelList,
                                    flatten(),
                                    nn.Linear(2048, 512),
                                    nn.SELU(inplace=True),
                                    nn.AlphaDropout(p=0.2),
                                    nn.Linear(512, 128),
                                    nn.SELU(inplace=True),
                                    nn.Linear(128, numClass),
                                    nn.LogSoftmax()
                )

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data,torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            return {'logits': output}
        else:
            return self.model(input)

class pCNN(nn.Module):
    def __init__(self, numClass, gpu_ids):
        super(pCNN, self).__init__()
        self.gpu_ids = gpu_ids
        modelList = list()
        width = 32
        modelList.append(nn.Conv2d(1, width, kernel_size=1))
        modelList.append(nn.ReflectionPad2d((1, 0, 1, 0)))

        for i in range(4):
            modelList.append(nn.BatchNorm2d(width))
            modelList.append(nn.LeakyReLU(inplace=True))
            modelList.append(nn.ReflectionPad2d((1, 1, 1, 1)))
            modelList.append(nn.Conv2d(width, width * 2, kernel_size=3, stride=1))
            modelList.append(nn.ReflectionPad2d((1, 0, 1, 0)))
            modelList.append(nn.MaxPool2d(3, stride2=2))
            width *= 2

        modelList.append(nn.BatchNorm2d(width))
        modelList.append(nn.LeakyReLU(inplace=True))
        modelList.append(nn.ReflectionPad2d((1, 1, 1, 1)))
        modelList.append(nn.Conv2d(width, width, kernel_size=3, stride=1))
        modelList.append(nn.ReflectionPad2d((1, 0, 1, 0)))
        modelList.append(nn.MaxPool2d(3, stride2=2))


        self.model = nn.Sequential(*modelList,
                                    flatten(),
                                    nn.Linear(2048, 512),
                                    nn.SELU(inplace=True),
                                    nn.AlphaDropout(p=0.2),
                                    nn.Linear(512, 128),
                                    nn.SELU(inplace=True),
                                    nn.Linear(128, numClass),
                                    nn.LogSoftmax()
                )

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data,torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            return {'logits': output}
        else:
            return self.model(input)

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim, batchSize):
        super(LSTMFeatureExtractor, self).__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.batchSize = batchSize
        self.lstm = nn.LSTM(inputDim, hiddenDim, 4)
        self.hidden2logits = nn.Linear(hiddenDim, outputDim)
        self.hidden = self.initHidden(self.batchSize)
        self.logSoftmax = nn.LogSoftmax()

    def initHidden(self, batchSize):
        h0 = Variable(torch.zeros(4, batchSize, self.hiddenDim).cuda())
        c0 = Variable(torch.zeros(4, batchSize, self.hiddenDim).cuda())
        return (h0, c0)

    def forward(self, x):
        x = torch.squeeze(x)
        x = x.transpose(0, 1)
        try:
            assert x.size()[1] == self.batchSize and x.size()[2] == self.inputDim
            output, features = self.lstm(x, self.hidden)
        except:
            print('shape not match')
            output, features = self.lstm(x, self.initHidden(x.size()[1]))

        # features = self.hidden.view(features.shape[1], -1)
        logits = self.hidden2logits(output[-1])
        logits = self.logSoftmax(logits)
        return {'logits':logits,
                'features':features}

class LSTM(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim, batchSize):
        super(LSTM, self).__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.batchSize = batchSize
        self.lstm = nn.LSTM(inputDim, hiddenDim, 4)
        self.hidden2logits = nn.Linear(hiddenDim, outputDim)
        self.hidden = self.initHidden(self.batchSize)
        self.logSoftmax = nn.LogSoftmax()

    def initHidden(self, batchSize):
        h0 = Variable(torch.zeros(4, batchSize, self.hiddenDim).cuda())
        c0 = Variable(torch.zeros(4, batchSize, self.hiddenDim).cuda())
        return (h0, c0)

    def forward(self, x):
        x = torch.squeeze(x)
        x = x.transpose(0, 1)
        try:
            assert x.size()[1] == self.batchSize and x.size()[2] == self.inputDim
            # output, _ = self.lstm(x, self.hidden)
            output, _ = self.lstm(x)
        except:
            print('shape not match')
            # output, _ = self.lstm(x, self.initHidden(x.size()[1]))
            output, _ = self.lstm(x)

        logits = self.hidden2logits(output[-1])
        logits = self.logSoftmax(logits)
        return {'logits':logits}

class BiLSTM(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim, batchSize):
        super(BiLSTM, self).__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.batchSize = batchSize
        self.lstm = nn.LSTM(inputDim, hiddenDim, num_layers=4, bidirectional=True)
        self.hidden2logits = nn.Sequential(
                                            nn.Linear(hiddenDim * 2, hiddenDim),
                                            nn.SELU(inplace=True),
                                            # nn.AlphaDropout(p=0.05),
                                            nn.Linear(hiddenDim, outputDim)
                                        )
        self.hidden = self.initHidden(self.batchSize)
        self.logSoftmax = nn.LogSoftmax()

    def initHidden(self, batchSize):
        h0 = Variable(torch.zeros(4 * 2, batchSize, self.hiddenDim).cuda())
        c0 = Variable(torch.zeros(4 * 2, batchSize, self.hiddenDim).cuda())
        return (h0, c0)

    def forward(self, x):
        x = torch.squeeze(x)
        x = x.transpose(0, 1)
        try:
            assert x.size()[1] == self.batchSize and x.size()[2] == self.inputDim
            output, _ = self.lstm(x, self.hidden)
        except:
            print('shape not match')
            output, _ = self.lstm(x, self.initHidden(x.size()[1]))
            # output, _ = self.lstm(x)

        logits = self.hidden2logits(output[-1])
        logits = self.logSoftmax(logits)
        return {'logits':logits}


class FuseNet(nn.Module):
    def __init__(self, numClass, inputDim, gpu_ids):
        super(FuseNet, self).__init__()
        self.gpu_ids = gpu_ids
        self.model  = list()
        self.model.append(nn.Linear(inputDim, 512))
        self.model.append(nn.SELU(inplace=True))
        self.model.append(nn.Linear(512, 1024))
        self.model.append(nn.SELU(inplace=True))
        self.model.append(nn.AlphaDropout(p=0.2))
        self.model.append(nn.Linear(1024, 512))
        self.model.append(nn.SELU(inplace=True))
        self.model.append(nn.AlphaDropout(p=0.2))
        self.model.append(nn.Linear(512, numClass))
        self.model.append(nn.LogSoftmax())
        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data,torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            return {'logits': output}
        else:
            return self.model(input)
