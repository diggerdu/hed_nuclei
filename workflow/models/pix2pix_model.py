import numpy as np
import pandas as pd
import csv
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from torch.optim import lr_scheduler
import util.util as util
from util.image_pool import ImagePool
import pickle
from .base_model import BaseModel
from . import networks
from . import time_frequence as tf


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.gan_loss = opt.gan_loss
        self.isTrain = opt.isTrain

        #set table and csv file to write
        self.table = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
                # define tensors self.Tensor has been reloaded
        self.inputAudio = self.Tensor(opt.batchSize, opt.len).cuda(device=self.gpu_ids[0])
        self.inputLabel = self.Tensor(opt.batchSize, opt.nClasses).cuda(device=self.gpu_ids[0])
        # load/define networks
        self.netG = networks.define_G(opt)


        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
            #                              opt.which_model_netD,
            #                              opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            from util.getPatch import getLabelDict
            # self.relabelDict = getLabelDict('./labeling')
            self.relabelDict = dict()

            self.load_network(self.netG, 'G', opt.which_epoch)
            # if self.isTrain:
            #    self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            # self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions

             # self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            # TODO
            CEWeights = torch.ones((self.opt.nClasses))
            #CEWeights[-1] = 1e-3
            #CEWeights[-2] = 0.9
            #######
            self.criterion = torch.nn.NLLLoss(weight=CEWeights.cuda())

            # initialize optimizers
            self.TrainableParam = list()
            param = self.netG.named_parameters()
            IgnoredParam = [id(P) for name, P in param if 'stft' in name]

            if self.opt.optimizer == 'Adam':
                self.optimizer_G = torch.optim.Adam(
                        self.netG.parameters(),
                        lr=opt.lr,
                        betas=(opt.beta1, 0.999),
                        weight_decay = self.opt.weightDecay)

            if self.opt.optimizer == 'sgd':
                self.optimizer_G = torch.optim.SGD(
                    filter(lambda P: id(P) not in IgnoredParam,
                       self.netG.parameters()),
                        lr=opt.lr)

            self.lrSche = lr_scheduler.ReduceLROnPlateau(self.optimizer_G, 'min', patience=6, verbose=True)

            print('---------- Networks initialized ---------------')
            networks.print_network(self.netG)
            # networks.print_network(self.netD)
            print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        inputAudio = input['Audio']
        inputLabel = input['Label']

        if self.opt.isTrain and self.opt.mixup is True:
            inputAudio, inputLabel = self.mixup(inputAudio, inputLabel)

        self.inputFname = input['Fname']
        self.inputAudio.resize_(inputAudio.size()).copy_(inputAudio)
        self.inputLabel.resize_(inputLabel.size()).copy_(inputLabel)
        self.image_paths = 'NOTIMPLEMENT'

    def mixup(self, inputs, targets):
        batchSize = inputs.shape[0]
        indexB = torch.randperm(batchSize)
        inputsB = inputs[indexB]
        targetsB = targets[indexB]

        alpha = self.opt.mixupAlpha
        mixRatio = np.random.beta(alpha, alpha, [batchSize, ])
        mixRatioInputs = np.broadcast_to(mixRatio[..., None, None, None], inputs.shape)
        mixRatioInputs = torch.from_numpy(mixRatioInputs).float()
        mixRatioTargets = np.broadcast_to(mixRatio[..., None], targets.shape)
        mixRatioTargets = torch.from_numpy(mixRatioTargets).float()

        mixInputs = mixRatioInputs * inputs + (1-mixRatioInputs) * inputsB
        mixTargets = mixRatioTargets * targets + (1-mixRatioTargets) * targetsB
        try:
            assert np.abs(np.sum(mixTargets.numpy()) - batchSize) < 1e-2
        except:
            import ipdb as pdb; pdb.set_trace()

        return mixInputs, mixTargets

    def forward(self):
        self.input = Variable(self.inputAudio)
        output = self.netG.forward(self.input)

        #import pdb; pdb.set_trace()
        self.predLogits = output['logits']
        return output


    # no backprop gradients
    def test(self):
        self.sub_name = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'submission_{}.csv'.format(self.opt.which_epoch))
        if not os.path.isfile(self.sub_name):
            with open(self.sub_name, 'w') as f:
                header = ['fname', 'label']
                writer = csv.writer(f)
                writer.writerow(header)


        labeledList = list(self.relabelDict.keys())
        self.netG.eval()
        self.forward()

        logitsArray = self.predLogits.cpu().data.numpy()
        prediction = np.argmax(logitsArray, axis=1).astype(int)

        LabelCodeArray = np.argmax(self.inputLabel.cpu().numpy(), axis=1)

        print(np.sum(LabelCodeArray == prediction) / max(prediction.shape))
        predictLabel = [self.table[i] for i in prediction]
        relabelCount = 0

        # confidence
        confPath = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'confidence.csv')
        confData = pd.DataFrame(np.exp(logitsArray), index=self.inputFname)
        try:
            os.remove(confPath)
        except:
            pass

        with open(confPath, 'a') as f:
            confData.to_csv(f, header=False)

        with open(self.sub_name, 'a') as f:
            message = [m for m in zip(self.inputFname, predictLabel)]
            writer = csv.writer(f)
            for row in message:
                '''
                if row[0] in labeledList and self.relabelDict[row[0]] != row[1]:
                    row = (row[0],self.relabelDict[row[0]])
                    relabelCount += 1
                assert row[1] in self.table
                '''
                writer.writerow(row)

        print('relabeling ', relabelCount)
        f.close()
        self.netG.train()

        return prediction

    def extractFeatures(self):
        assert self.opt.which_model_netG in ['cnnFeatureExtractor', 'LSTMFeatureExtractor']
        self.netG.eval()
        output = self.forward()
        if self.opt.which_model_netG == 'LSTMFeatureExtractor':
            return {'features':output['features'][0][-1].cpu().data.numpy()}
        if self.opt.which_model_netG == 'cnnFeatureExtractor':
            return {'features':output['features'].cpu().data.numpy()}
        # self.input = Variable(self.inputAudio)
        '''
        # import pdb; pdb.set_trace()
        featPath = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'featrues.csv')
        labelPath = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'labels.csv')
        features = self.netG.forward(self.input)['features'].cpu().data.numpy()
        self.netG.train()
        assert features.shape[1] == 128
        with open(featPath, 'a') as f:
            pd.DataFrame(features, index=self.inputFname).to_csv(f, header=False)
        with open(labelPath, 'a') as f:
            for fname, label in zip(self.inputFname,
                    np.argmax(self.inputLabel.cpu().numpy(), axis=1).tolist()):
                print('{},{}'.format(fname, self.table[label]), file=f)
        '''




    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):
        Label = Variable(self.inputLabel, requires_grad=False)

        self.loss_G = torch.mean(-self.predLogits * Label)
        #self.loss_G = self.criterion(self.predLogits, Label)

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        if self.gan_loss:
            return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                                ('G_L1', self.loss_G_L1.data[0]),
                                ('D_real', self.loss_D_real.data[0]),
                                ('D_fake', self.loss_D_fake.data[0]),
                                ('Logits', self.predLogits)
                                ])
        else:
            # print("#############clean sample mean#########")
            # sample_data = self.input_B.cpu().numpy()

            # print("max value", np.max(sample_data))
            # print("mean value", np.mean(np.abs(sample_data)))
            return OrderedDict([('Logits', self.predLogits.data.cpu().numpy()),
                                ('G_LOSS', self.loss_G.data.cpu().numpy())])
            # return self.loss_G.data[0]

    def get_current_visuals(self):
        real_A = self.real_A.data.cpu().numpy()
        fakeB = self.fakeB.data.cpu().numpy()
        realB = self.realB.data.cpu().numpy()
        clean = self.clean.cpu().numpy()
        noise = self.noise.cpu().numpy()
        return OrderedDict([
            ('est_ratio', fakeB),
            ('clean', clean),
            ('ratio', realB),
            ('noise', noise),
        ])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        if self.gan_loss:
            self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        # lrd = self.opt.lr / self.opt.niter_decay
        # lr = self.old_lr - lrd
        lr = self.old_lr * 0.6
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
