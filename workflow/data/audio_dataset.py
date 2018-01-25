import os.path
from data.base_dataset import BaseDataset
from data.audio_folder import make_dataset, getMel
import librosa
import soundfile as sf
import numpy as np
import random

class freqFilter(object):
    def __init__(self, stopFreq=49):
        self.stopFreq = 49
    def __call__(self, dataDict):
        dataDict['Audio'][0, self.stopFreq:,:] = 0.
        return dataDict

class AudioDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.Dir = opt.Path
        self.Data, self.Labels, self.Fnames = make_dataset(opt)
        if self.opt.isTrain and self.opt.which_model_netG in ['cnn', 'lstm', 'drcnn', 'bilstm']:
            self.pinkNoise = getMel('/home/diggerdu/dataset/tfsrc/train/_background_noise_/pink_noise.wav', opt)

        self.oriLen = len(self.Data)
        self.SR = opt.SR
        self.hop = opt.hop
        self.nfft = self.opt.nfft
        self.table = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
        if opt.isTrain:
            self.augFuncList = [lambda x:x, freqFilter()]
        else:
            self.augFuncList = [lambda x:x]


    def __getitem__(self, index):
        Data = np.array(self.Data[index % self.oriLen])
        if self.opt.isTrain and self.opt.which_model_netG in ['cnn', 'lstm', 'drcnn', 'bilstm']:
            Data = Data + self.getRatio() * self.pinkNoise
        Label = self.Labels[index % self.oriLen]
        Fname = self.Fnames[index % self.oriLen]
        if self.opt.which_model_netG in ['fusenet']:
            return {'Audio':Data, 'Label':np.array(Label), 'Fname':Fname}

        assert len(Data.shape) == 2
        assert type(Label) is not np.ndarray
        Audio = np.expand_dims(Data, axis=0)

#        if index > self.oriLen:
#            assert np.sum(Data[49:,:]) == 0.

        # Audio = self.load_audio(Data)

        assert Audio.dtype==np.float32

        LabelCode = self._label2Code(Label)

        dataDict = {'Audio': Audio,
                    'Label': self._one_hot(LabelCode),
                    'Fname': Fname}

        return self.augFuncList[index // self.oriLen](dataDict)

    def getRatio(self):
        import scipy.stats as stats
        lower, upper = -0.1, 0.5
        mu, sigma = 0.15, 0.1
        X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        ratio = X.rvs()
        assert ratio > lower - 1e-3 and ratio < upper + 1e-3
        if ratio < 0.:
            ratio = 0.
        return ratio

    def _one_hot(self, index):
        arr = np.zeros((self.opt.nClasses), dtype=np.float32)
        arr[index] = 1.
        return arr
    def _label2Code(self, labelStr):
        try:
            LabelCode = self.table.index(labelStr)
        except:
            LabelCode = self.table.index('unknown')
        return LabelCode
    def _mixup(self, dataDict):
        labelCode = np.argmax(dataDict['Label'])
        if labelCode == self.table.index('silence'):
            return dataDict

        index = random.choice([i for i in range(self.oriLen) if self.Labels[i] != labelCode])
        assert type(index) == int
        # TODO
        alpha = 1.
        mixRatio = np.random.beta(alpha, alpha)
        bData = np.array(self.Data[index])
        bLabel = self._one_hot(self._label2Code(self.Labels[index]))
        bName = self.Fnames[index]
        if self.Labels[index] == self.table.index('silence'):
            mixLabel = dataDict['Label']
            mixRatio = max(0.7+1e-5, mixRatio)
            assert mixRatio > 0.7
        else:
            mixLabel = mixRatio * dataDict['Label'] + (1-mixRatio) * bLabel

        mixData = mixRatio * dataDict['Audio'] + (1-mixRatio) * bData
        mixName = '{}*{}+{}*{}'.format(mixRatio, dataDict['Fname'], 1-mixRatio, bName)
        return {'Audio' : mixData,
                'Label' : mixLabel,
                'Fname' : mixName
                }

    def __len__(self):
        # return len(self.FilesClean)
        return self.oriLen * len(self.augFuncList)
        # return max(len(self.Clean), len(self.Noise))

    def name(self):
        return "AudioDataset"

    def load_audio(self, data):
        target_len = self.opt.len
        if data.shape[0] >= target_len:
            head = random.randint(0, data.shape[0] - target_len)
            data = data[head:head + target_len]
        if data.shape[0] < target_len:
            ExtraLen = target_len - data.shape[0]
            PrevExtraLen = np.random.randint(ExtraLen)
            PostExtraLen = ExtraLen - PrevExtraLen
            PrevExtra = np.zeros((PrevExtraLen, ), dtype=np.float32)
            PostExtra = np.zeros((PostExtraLen, ), dtype=np.float32)
            data = np.concatenate((PrevExtra, data, PostExtra))

        data = data - np.mean(data)
        assert data.dtype == np.float32

        assert data.shape[0] == self.opt.len
        return data
