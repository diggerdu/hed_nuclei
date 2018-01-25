import os
import os.path
import time
import librosa
import soundfile as sf
import librosa as lb
import numpy as np
import shutil
import random
import gc
import multiprocessing
import sys
sys.path.append('/home/diggerdu/fuck/labeling/util/')
from getPatch import getLabelDict

AUDIO_EXTENSIONS = [
    '.wav',
    '.WAV',
]

def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

def getMel(path, opt):
    try:
        wav, sr = sf.read(path, dtype='float32')
    except:
        time.sleep(6)
        wav, sr = sf.read(path, dtype='float32')

    try:
        assert sr == opt.SR
    except AssertionError:
        wav = librosa.resample(wav, sr, opt.SR)
    print('{}:{}'.format(path, wav.shape[0]))
    target_len = opt.len - 512

    if wav.shape[0] > target_len:
        offsetSum = 0
        maxSum = 0
        maxHead = 0
        for head in range(wav.shape[0] - target_len):
            if offsetSum > maxSum:
                offsetSum = maxSum
                maxHead = head
            offsetSum -= np.abs(wav[head])
            offsetSum += np.abs(wav[head + target_len])
        print('maxHead: ', maxHead)

        wav = wav[maxHead:maxHead + target_len]

    if wav.shape[0] < target_len:
        ExtraLen = target_len - wav.shape[0]
        PrevExtraLen = np.random.randint(ExtraLen)
        PostExtraLen = ExtraLen - PrevExtraLen
        PrevExtra = np.zeros((PrevExtraLen, ), dtype=np.float32)
        PostExtra = np.zeros((PostExtraLen, ), dtype=np.float32)
        wav = np.concatenate((PrevExtra, wav, PostExtra))

    assert wav.shape[0] == target_len

    wav = wav - np.mean(wav)
    if np.max(np.abs(wav)) > 0:
        wav = wav / np.max(np.abs(wav))
    melsp = librosa.feature.melspectrogram(
                    y=wav,
                    sr=sr,
                    S=None,
                    n_fft=opt.nfft,
                    hop_length=opt.hop,
                    power=2.0,
                    n_mels=64,
                    fmax=sr // 2)
    # TODO
    eps = 1e-3
    melsp = np.log(melsp + eps)
    return melsp.astype(np.float32)


def procFile(paraList):
    path, prevData, opt = paraList
    audios, labels, fnames = [list() for i in range(3)]
    if True:
        label, fname = path.split('/')[-2:]
        print(label, fname)

        if prevData is not None and fname in prevData['fnames']:
            melsp = prevData['audios'][prevData['fnames'].index(fname)]
        else:
            melsp = getMel(path, opt)

        audios.append(melsp)
        labels.append(label)
        fnames.append(fname)


        # TODO
        if opt.isTrain:
            tmpFname = fname.split('.')[0] + '_tempo1.9.wav'
            if prevData is not None and tmpFname in prevData['fnames']:
                melsp = prevData['audios'][prevData['fnames'].index(tmpFname)]
            else:
                if not os.path.isfile('/tmp/{}'.format(tmpFname)):
                    os.system("sox {0} /tmp/{1} tempo -s 1.9".format(path, tmpFname))
                melsp = getMel('/tmp/{}'.format(tmpFname), opt)
                # os.remove('/tmp/{}'.format(tmpFname))
            audios.append(melsp)
            labels.append(label)
            fnames.append(tmpFname)

            '''
            tmpFname = fname.split('.')[0] + '_tempo0.75.wav'
            if prevData is not None and tmpFname in prevData['fnames']:
                melsp = prevData['audios'][prevData['fnames'].index(tmpFname)]
            else:
                os.system("sox {0} /tmp/{1} tempo -s 0.75".format(path, tmpFname))
                melsp = getMel('/tmp/{}'.format(tmpFname), opt)
                os.remove('/tmp/{}'.format(tmpFname))
            audios.append(melsp)
            labels.append(label)
            fnames.append(tmpFname)


            tmpFname = fname.split('.')[0] + '_pitch500.wav'
            if prevData is not None and tmpFname in prevData['fnames']:
                melsp = prevData['audios'][prevData['fnames'].index(tmpFname)]
            else:
                os.system("sox {0} /tmp/{1} pitch 500".format(path, tmpFname))
                melsp = getMel('/tmp/{}'.format(tmpFname), opt)
                os.remove('/tmp/{}'.format(tmpFname))
            audios.append(melsp)
            labels.append(label)
            fnames.append(tmpFname)

            tmpFname = fname.split('.')[0] + '_pitch-500.wav'
            if prevData is not None and tmpFname in prevData['fnames']:
                melsp = prevData['audios'][prevData['fnames'].index(tmpFname)]
            else:
                os.system("sox {0} /tmp/{1} pitch -500".format(path, tmpFname))
                melsp = getMel('/tmp/{}'.format(tmpFname), opt)
                os.remove('/tmp/{}'.format(tmpFname))
            audios.append(melsp)
            labels.append(label)
            fnames.append(tmpFname)
            '''

    return {'audios':audios, 'labels':labels, 'fnames':fnames}

def loadData(Dir, opt, prevData=None):
    assert os.path.isdir(Dir), '%s is not a valid directory' % dir
    assert prevData is None
    audios = list()
    labels = list()
    fnames = list()
    processPool = multiprocessing.Pool(32)

    paraList = list()
    for root, _, fns in sorted(os.walk(Dir)):
        for fname in fns:
            if is_audio_file(fname):
                paraList.append([os.path.join(root, fname), prevData, opt])

    # import dill
    # processFunc = lambda path:procFile(path, prevData, opt.isTrain)
    output = processPool.map(procFile, paraList)
    audios = [a for item in output for a in item['audios']]
    labels = [l for item in output for l in item['labels']]
    fnames = [f for item in output for f in item['fnames']]


    return {'audios':audios, 'labels':labels, 'fnames':fnames}



def make_dataset(opt):
    audios = []
    labels = []
    fnames = []
    prevData = None
#    try:
    if True:
        print("######CAUTION:starting load generated datasets###########")
        audios = np.load(opt.dumpPath + "/audios.npy")
        labels = np.load(opt.dumpPath + "/labels.npy").tolist()
        fnames = np.load(opt.dumpPath + "/fnames.npy").tolist()
        # prevData={'audios':audios, 'labels':labels, 'fnames':fnames}

        print("######CAUTION:previous generated dataset loaded###########")
        if opt.patchPath is not None:
            print("starting patch")
            labelDict = getLabelDict(opt.patchPath)
            indexs = list()
            for i in range(len(fnames)):
                if not 'clip' in fnames[i]:
                    if opt.name == 'kfold':
                        continue
                    if not opt.isTrain:
                        continue
                    #print(fnames[i])
                    indexs.append(i)
                else:
                    # clip_4c6766436_tempo
                    if 'tempo' in fnames[i]:
                        if opt.name == 'kfold':
                            continue
                            print('continue')
                        if not opt.isTrain:
                            continue
                    oriFname = fnames[i][:len('clip_4c6766436')] + '.wav'
                    if oriFname in labelDict.keys():
                        if labels[i] != labelDict[oriFname]:
                            print(oriFname, labelDict[oriFname], labels[i])
                        indexs.append(i)
                        labels[i] = labelDict[oriFname]

            np.save("/home/diggerdu/audios.npy", audios[indexs])
            np.save("/home/diggerdu/labels.npy", np.array(np.array(labels))[indexs])
            np.save("/home/diggerdu/fnames.npy", np.array(np.array(fnames))[indexs])
            import sys; sys.exit(0)

            return audios[indexs], np.array(labels)[indexs].tolist(), np.array(fnames)[indexs].tolist()
        return audios, labels, fnames

    '''
    except:
        #import pdb; pdb.set_trace()
        if os.path.isfile(opt.dumpPath + 'audios.npy'):
            print('##########CAUTION#########')
            print('##########CAUTION#########')
            print('##########CAUTION#########')
            print('##########CAUTION#########')
            print('##########CAUTION#########')
            print('##########CAUTION#########')
            print('##########CAUTION#########')
            print('##########CAUTION#########')
            print('##########FILE LOCK#########')
            sys.exit(0)
            nothings = input()
        pass
    '''
    try:
        #import pdb; pdb.set_trace()
        #shutil.rmtree(opt.dumpPath)
        pass
    except:
        pass
    os.mkdir(opt.dumpPath)

    data = loadData(opt.Path, opt, prevData)
    audios = data['audios']
    labels = data['labels']
    fnames = data['fnames']
    if opt.additionPath is not None:
        additionData = loadData(opt.additionPath, opt, prevData)
        audios.extend(additionData['audios'])
        labels.extend(additionData['labels'])
        fnames.extend(additionData['fnames'])
    np.save(opt.dumpPath + "/audios.npy", np.array(audios))
    np.save(opt.dumpPath + "/labels.npy", np.array(labels))
    np.save(opt.dumpPath + "/fnames.npy", np.array(fnames))
    return audios, labels, fnames
