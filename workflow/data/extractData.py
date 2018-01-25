import numpy as np
import os
oriPath = 'trainDump'
outPath = 'trainDump_tempo1.9'
oriAudio = np.load(os.path.join(oriPath, 'audios.npy'))
oriFname = np.load(os.path.join(oriPath, 'fnames.npy'))
oriLabel = np.load(os.path.join(oriPath, 'labels.npy'))



Audio = list()
Fname = list()
Label = list()
for i in range(len(oriFname.tolist())):
    if oriFname[i].split('.')[-1] in ['wav', 'wav_tempo1.9']:
        Audio.append(oriAudio[i])
        Fname.append(oriFname[i])
        Label.append(oriLabel[i])

del oriAudio
del oriFname
del oriLabel

np.save(os.path.join(outPath, 'audios.npy'), np.array(Audio))
np.save(os.path.join(outPath, 'fnames.npy'), np.array(Fname))
np.save(os.path.join(outPath, 'labels.npy'), np.array(Label))


