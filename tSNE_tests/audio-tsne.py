#! /usr/bin/env python3

# from: https://github.com/ml4a/ml4a-guides/blob/master/notebooks/audio-tsne.ipynb


from matplotlib import pyplot as plt
import matplotlib.cm as cm
import fnmatch
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
from sklearn.manifold import TSNE
import json

#source_audio = '/home/marcello/Music/AI_Tomomi/Sample/selection/1PutifMayToJuneOneHour_selection.wav' 
#source_audio = '/home/marcello/Music/Listen/Chin, Unsuk - Rocana, Violin Concero (2009, Hagner, Nagano) [FLAC]/01. Rocana.flac' 

source_audio = '/home/marcello/Music/Samples/Huggermugger/cello_01.wav' 

def get_features(y, sr):
    #    y = y[0:sr]  # analyze just first second
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    #print("data: %s \n" % mfcc  )
    delta_mfcc = librosa.feature.delta(mfcc, mode='nearest')
    delta2_mfcc = librosa.feature.delta(mfcc, order=2, mode='nearest')
    feature_vector = np.concatenate((np.mean(mfcc,1), np.mean(delta_mfcc,1), np.mean(delta2_mfcc,1)))
    print("feature_vector1: %s" % feature_vector)
    feature_vector = (feature_vector-np.mean(feature_vector))/np.std(feature_vector)
    print("feature_vector2: %s" % feature_vector)
    return feature_vector




hop_length = 512
y, sr = librosa.load(source_audio)
onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)

print("audio time series: %s" % y )
print("onsets: %s" % onsets )
times = [hop_length * onset / sr for onset in onsets]

# where to save our new clips to
path_save_intervals = '/home/marcello/Music/AI_Tomomi/python/tSNE_tests/audio_segs/'  

# make new directory to save them 
if not os.path.isdir(path_save_intervals):
    os.mkdir(path_save_intervals)

# grab each interval, extract a feature vector, and save the new clip to our above path
feature_vectors = []
errors = 0
for i in range(len(onsets)-1):
    try:
        idx_y1 = onsets[i  ] * hop_length  # first sample of the interval
        idx_y2 = onsets[i+1] * hop_length  # last sample of the interval
        y_interval = y[idx_y1:idx_y2]
        features = get_features(y_interval, sr)   # get feature vector for the audio clip between y1 and y2
        file_path = '%s/onset_%d.wav' % (path_save_intervals, i)   # where to save our new audio clip
        feature_vectors.append({"file":file_path, "features":features})   # append to a feature vector
        librosa.output.write_wav(file_path, y_interval, sr)   # save to disk
    except:
        print("==== ERROR ====")
        print("y_interval of %d: %s \n length: %d" % (i,y_interval, len(y_interval) ) )
        errors = errors + 1
        if i % 50 == 0:
            print( "analyzed %d/%d = %s"%(i+1, len(onsets)-1, file_path) )
        print( "analyzed %d/%d = %s"%(i+1, len(onsets)-1, file_path) )
print("Number of errors/analyzed: %d/%d" % (errors, len(onsets)-1 ) ) 

# save results to this json file
tsne_path = "/home/marcello/Music/AI_Tomomi/python/tSNE_tests/json-data/example-audio-tSNE-onsets.json"

# feature_vectors has both the features and file paths in it. let's pull out just the feature vectors
features_matrix = [f["features"] for f in feature_vectors]

# calculate a t-SNE and normalize it
model = TSNE(n_components=2, learning_rate=150, perplexity=30, verbose=2, angle=0.1).fit_transform(features_matrix)
x_axis, y_axis = model[:,0], model[:,1] # normalize t-SNE
x_norm = (x_axis - np.min(x_axis)) / (np.max(x_axis) - np.min(x_axis))
y_norm = (y_axis - np.min(y_axis)) / (np.max(y_axis) - np.min(y_axis))

data = [{"path":os.path.abspath(f['file']), "point":[x, y]} for f, x, y in zip(feature_vectors, x_norm, y_norm)]
with open(tsne_path, 'w') as outfile:
    json.dump(data, outfile)

print("saved %s to disk!" % tsne_path)




colors = cm.rainbow(np.linspace(0, 1, len(x_axis)))
plt.figure(figsize = (8,6))
plt.scatter(x_axis, y_axis, color=colors)
plt.show()


