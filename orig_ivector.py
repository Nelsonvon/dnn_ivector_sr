import bob.learn.em
import bob.ap
import numpy as np
import scipy.io.wavfile
import os
import datetime

"""
train a GMM as UBM
"""
print("Program starts, time:{}".format(str(datetime.datetime.now())))
##################################
###########--MFCC--###############
wave_path = "/home/nelson/Data/Seminar/voxceleb/vox1_test_wav/wav/id10270/5r0dWxy17C8/00001.wav"
rate, signal = scipy.io.wavfile.read(str(wave_path))

win_length_ms = 25 # The window length of the cepstral analysis in milliseconds
win_shift_ms = 10 # The window shift of the cepstral analysis in milliseconds
n_filters = 24 # The number of filter bands
n_ceps = 19 # The number of cepstral coefficients
f_min = 0. # The minimal frequency of the filter bank
f_max = 4000. # The maximal frequency of the filter bank
delta_win = 2 # The integer delta value used for computing the first and second order derivatives
pre_emphasis_coef = 1.0 # The coefficient used for the pre-emphasis
dct_norm = True # A factor by which the cepstral coefficients are multiplied
mel_scale = True # Tell whether cepstral features are extracted on a linear (LFCC) or Mel (MFCC) scale

c = bob.ap.Ceps(rate, win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
c.with_energy = True
c.with_delta = True



dataset_path = "/home/nelson/Data/Seminar/voxceleb/vox1_test_wav/wav/id10270"
dirs = os.listdir(dataset_path)
fea_set = []
for dir in dirs:
    files = os.listdir(dataset_path + '/' + dir)
    for file in files:
        _, signal = scipy.io.wavfile.read(dataset_path + '/' + dir + '/' + file)
        signal = np.cast['float'](signal)  # vector should be in **float**
        mfcc = c(signal)
        #print(type(mfcc))
        if len(fea_set)==0:
            fea_set = mfcc
        else:
            fea_set = np.append(fea_set, mfcc,axis=0)

print(fea_set)
print("Finish read audio, time:{}".format(str(datetime.datetime.now())))
##########################################
###########--Training UBM--###############
dim = 40
data = np.array(fea_set,dtype = 'float64')
print(np.size(data))
kmeans =    bob.learn.em.KMeansMachine(2048, dim)
kmeansTrainer = bob.learn.em.KMeansTrainer()
bob.learn.em.train(kmeansTrainer, kmeans, data, max_iterations = 200, convergence_threshold = 1e-5)
print("Kmeans Training over, time:{}".format(str(datetime.datetime.now())))
ubm_machine = bob.learn.em.GMMMachine(2048, dim)
ubm_machine.means = kmeans.means
ubm_trainer = bob.learn.em.ML_GMMTrainer(True,True,True)  # update means/variances/weights at each iteration
bob.learn.em.train(ubm_trainer, ubm_machine, data, max_iterations = 200, convergence_threshold = 1e-5)
print("UBM Training over, time:{}".format(str(datetime.datetime.now())))

#tv_machine = bob.learn.em.IVectorMachine(ubm_machine,dim)
#tv_machine.variance_threshold = 1e-5

#tv_trainer = bob.learn.em.IVectorTrainer(update_sigma = True)
#TRAINING_STATS_flatten = [gs11, gs12, gs21, gs22]

