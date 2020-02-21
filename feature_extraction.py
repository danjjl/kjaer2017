"""Absence feature extraction module.

This module implements the feature extraction described in Kjaer, T. W.,
Sorensen, H. B. D., Groenborg, S., Pedersen, C. R., & Duun-Henriksen, J. (2017)
Detection of Paroxysms in Long-Term, Single-Channel EEG-Monitoring of Patients
with Typical Absence Seizures. IEEE Journal of Translational Engineering in
Health and Medicine, 5. https://doi.org/10.1109/JTEHM.2017.2649491
"""

import numpy as np
import pywt
from scipy import signal
from scipy.spatial import distance


def feature_extraction(data, data1):
    """ Extract features.

    Extract the ten features described in the Kjaer et al. paper.

        Args:
            data: Data vector at epoch n. Should be sampled at 128Hz and is
                expected to be 2 seconds long.
            data1: Data vector at epoch n + 1. Should be sampled at 128Hz and
                is expected to be 2 seconds long.
        Return:
            features: The ten features are ordered as follows (4 x wavelet,
                2 x power, 2 x correlation, 1 x phase, 1 x distance)
    """

    def extract_log_sum_wavelets(data):
        """ Wavelet based features.

        Extracts 5 levels of the multiscale db4 wavelet transform. Feature is
        the log-transformed absolute sum of a given level. Scale 4 is not
        retained.

        Args:
            data: Data vector. Should be sampled at 128Hz and is expected to be
                2 seconds long.
        Return:
            list: The four features corresponding to scales 5, 3, 2, 1.
        """
        coefficients = pywt.wavedec(data, 'db4', 5)
        wavelet_feat = [np.log(np.sum(np.abs(x))) for x in coefficients[1:]]
        del wavelet_feat[1]  # drop scale 4

        return wavelet_feat

    def extract_power(data_bp1_30, data_bp3_12):
        """ Power based features.

        First feature is the power of bandpassed [1-30Hz] signal. Transformed
        with an exponential of 1/10. Second feature is the ratio of power in
        the [3-12Hz] band and the [1-30Hz] band.

        Args:
            data_bp1_30: Bandpassed [1-30Hz] data vector
            data_bp3_12: Bandpassed [3-12Hz] data vector
        Return:
            list: The two power features (power [1-30], [3-12]/[1-30])
        """
        power1_30 = np.sum(np.square(data_bp1_30))
        power3_12 = np.sum(np.square(data_bp3_12))

        return [power1_30**0.1, power3_12/power1_30]

    def extract_correlate(data, data1, data_bp1_30, data_bp3_12):
        """ Cross-correlation based features.

        First feature is the cross-correlation between an epoch and the next
        one. The second feature is the cross-correlation between the bandpassed
        signal with a filter of [1-30Hz] and [3-12Hz]

        Args:
            data: Data vector at epoch n
            data1: Data vector at epoch n+1
            data_bp1_30: Bandpassed [1-30Hz] data vector
            data_bp3_12: Bandpassed [3-12Hz] data vector
        Return:
            list: The two correlation features
        """
        return [np.correlate(data, data1)[0]**0.5,
                np.correlate(data_bp1_30, data_bp3_12)[0]**2]

    def extract_phase(data):
        """ Mean phase variance feature.

        Means phase variance based on the imaginary part of an Hilbert
        transform.

        Args:
            data: Data vector.
        Return:
            float: mean phase variance feature.
        """
        return np.var(signal.hilbert(data - np.mean(data)))

    def extract_mahalanobis(data_bp1_30, data_bp3_12):
        """ Mahalanobis variance feature.

        Calculated as the distance between the distribution of points in the
        bandpassed [1-30Hz] and [3-12Hz] signals.

        Args:
            data_bp1_30: Bandpassed [1-30Hz] data vector
            data_bp3_12: Bandpassed [3-12Hz] data vector
        Return:
            float: Mahalanobis variance feature
        """
        cov_mat = np.outer(data_bp1_30, data_bp3_12)
        try:
            inv_cov = np.linalg.inv(cov_mat)
        except np.linalg. LinAlgError:
            inv_cov = np.eye(len(data_bp1_30))

        return np.var(distance.mahalanobis(data_bp1_30, data_bp3_12,
                                           inv_cov))**0.25

    # Feature extraction #
    data_bp1_30 = np.convolve(data, signal.firwin(467, [1, 30], fs=128))
    data_bp3_12 = np.convolve(data, signal.firwin(467, [3, 12], fs=128))

    features = [extract_log_sum_wavelets(data) +
                extract_power(data_bp1_30, data_bp3_12) +
                extract_correlate(data, data1, data_bp1_30, data_bp3_12) +
                [extract_phase(data)] +
                [extract_mahalanobis(data_bp1_30, data_bp3_12)]]
    return features
