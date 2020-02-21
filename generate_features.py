"""Generate feature set from continuous EEG data.
"""
import numpy as np


def split_data(data, fs, duration=2, overlap=0):
    """Split data into epochs.

    Args:
        data: Data contained in an array (row = channels, column = samples)
        fs: Sampling frequency in Hz
        duration: Length of each epoch in seconds (default=2)
        overlap: Percentage of overlap [0-100] (default=0)
    Returns:
        list: List of list of epochs. Top level list is one list per channel.
    """
    epochs = list()
    for channel in data:
        epochs.append([])
        i = 0
        while (i+duration)*fs <= len(channel):
            epochs[-1].append(channel[int(i*fs):int((i+duration)*fs)])
            i += duration - overlap*duration/100
    return epochs


def split_annotations(annotations, dataLen, fs, percentage=75, duration=2,
                      overlap=0):
    """list of annotations into epoch binary labels.

    Args:
        annotations: list of events times in seconds. Each row contains two
            columns: [start time, end time]
        dataLen: Number of samples in the full data vector.
        fs: Sampling frequency in Hz
        percentage: Percentage of samples that should be True to consider the
            event as True. [0-100] (default=75)
        duration: Length of each epoch in seconds (default=2)
        overlap: Percentage of overlap [0-100] (default=0)
    Returns:
        list: List of binary labels.
    """
    annotation_mask = _eventList2Mask(annotations, dataLen, fs)
    epochs = list()
    i = 0
    while (i+duration)*fs <= len(annotation_mask):
        poz_samples = np.sum(annotation_mask[int(i*fs):int((i+duration)*fs)])
        if poz_samples >= duration*fs*percentage/100:
            epochs.append(True)
        else:
            epochs.append(False)
        i += duration - overlap*duration/100
    return epochs


def _eventList2Mask(events, totalLen, fs):
    """Convert list of events to a binary mask.

    Returns a logical array of length totalLen.
    All event epochs are set to True

    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        totalLen: length of array to return in samples
        fs: sampling frequency of the data in Hertz
    Return:
        mask: logical array set to True during event epochs and False the rest
              if the time.
    """
    mask = np.zeros((totalLen,), dtype=bool)
    for event in events:
        for i in range(int(event[0]*fs), int(event[1]*fs)):
            mask[i] = True
    return mask
