import logging
import glob
import json
import resampy
from load_edf import load_annotations, load_data
from generate_features import split_data, split_annotations
from feature_extraction import feature_extraction

dataFolder = '/home/jonathan/Documents/Byteflies/mxspir'

j = 0
for patient in glob.iglob(dataFolder + '/Seize*.EDF'):
    # Load data
    logging.info('Loading patient: {}'.format(patient))
    bipolar_data, fs, channel_labels = load_data(patient)
    resampled_data = list()
    for channel in bipolar_data:
        resampled_data.append(resampy.resample(channel, fs, 128))
    fs = 128
    seizures = load_annotations(patient)

    data = split_data(resampled_data, fs, duration=2, overlap=50)
    annotations = split_annotations(seizures, len(resampled_data[0]), fs,
                                    percentage=75, duration=2, overlap=50)
    
    feature_set = list()
    for i, label in enumerate(annotations[:-2]):
        feature = [label]
        for channel in data:
            feature += feature_extraction(channel[i],channel[i+1])
        feature_set.append(feature)
    with open(str(j) + '.json', 'w') as f:
        json.dump(feature_set, f)
    j += 1

