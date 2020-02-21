""" Load Gasthuisberg EDF files containing

Files should contain seizure annotations and the two behind the ear channels. 
"""
import logging
import numpy as np
import pyedflib
import re


def load_annotations(filename, interictal=False):
    """Parse annotations in an edf file to extract seizure events.

    Args:
        filename: string containing path to EDF files
        interictal: include markings for interictal eeg (default: False)
    Returns:
        list: list of events times in seconds. Each row contains two
            columns: [start time, end time]
    """
    with pyedflib.EdfReader(filename) as edf:
        annotations = edf.readAnnotations()
        edf._close()
    events = list()
    for i, text in enumerate(annotations[2]):
        if ((re.match('^begin_.*abs.*', text, re.IGNORECASE) is not None) or
            (re.match('^absence.*', text, re.IGNORECASE) is not None) or
            (re.match('^begin_aanval.*', text, re.IGNORECASE) is not None) or
            (re.match('^Note : absence start.*', text, re.IGNORECASE) is not None) or
            (interictal and re.match('^begin_inter.*', text, re.IGNORECASE) is not None) or
            (interictal and re.match('^inter-ictaal.*', text, re.IGNORECASE) is not None) or
            (interictal and re.match('^\(?inter.*', text, re.IGNORECASE) is not None)):
            if len(events) > 0 and len(events[-1]) != 2:
                events.pop()
            events.append([annotations[0][i]])
        elif ((re.match('^einde_.*abs.*', text, re.IGNORECASE) is not None) or
            (re.match('^einde_aanval.*', text, re.IGNORECASE) is not None) or
            (re.match('^Note : absence stop.*', text, re.IGNORECASE) is not None) or
            (interictal and re.match('^einde_inter.*', text, re.IGNORECASE) is not None)):
            events[-1].append(annotations[0][i])
        else:
            if re.match('^\+.*', text, re.IGNORECASE) is not None:
                pass
            else:
                pass

    events_length = list()
    for event in events:
        events_length.append((event[1]-event[0]))

    logging.info('A total of {} events with a median duration of {:.2f}s and a total duration of {:.2f}s.'.format(
    len(events_length), np.median(events_length), np.sum(events_length)))

    return events


def bipolar_rereference(data, channel_labels):
    """ Rereference data to bipolar montage

    Args:
        data: data contained in an array (row = channels, column = samples)
        channel_labels: label of each channel (in a list)
    Return:
        np.array: data contained in an array (row = channels, column = samples)
        list: list of channel labels
    """
    bipolar_pairs = (('LiOorTop', 'LiOorAchter'),
                     ('ReOorTop', 'ReOorAchter'),
                     ('LiOorAchter', 'ReOorAchter'))
    bipolar_data = list()
    bipolar_labels = list()
    for i, bipolar_pair in enumerate(bipolar_pairs):
        try:
            pos_i = channel_labels.index(bipolar_pair[0])
            neg_i = channel_labels.index(bipolar_pair[1])

            bipolar_data.append(data[pos_i, :] - data[neg_i, :])
            bipolar_labels.append(bipolar_pair[0] + ' - ' + bipolar_pair[1])
        except ValueError:
            raise KeyError('{} not found'.format(bipolar_pair))

    bipolar_data = np.array(bipolar_data)
    return bipolar_data, bipolar_labels


def load_data(filename):
    """Load edf file.

    Load EDF file and return an array with the two behind the ear channels.

    Args:
        filename: string containing path to EDF files
    returns:
        np.array: data of the two behind the ear channels [left, right]
        int: Sampling frequency in Hz
        list: List of channel names
    """
    channels = ['LiOorTop',
                'LiOorAchter',
                'ReOorTop',
                'ReOorAchter']

    with pyedflib.EdfReader(filename) as edf:
        fs = edf.getSampleFrequency(0)

        # Get channel labels
        channel_labels = edf.getSignalLabels()
        channel_indices = list()

        for channel in channels:
            found = False
            for i, name in enumerate(channel_labels):
                if channel.lower() in name.lower():
                    found = True
                    channel_indices.append(i)
                    fs = edf.getSampleFrequency(i)
                    break
            if not found:
                raise KeyError('{} not found'.format(channel))

        # Load data
        data = []
        for i in channel_indices:
            data.append(edf.readSignal(i))
        data = np.array(data)
        edf._close()

    bipolar_data, channel_labels = bipolar_rereference(data, channels)

    return bipolar_data, fs, channel_labels
