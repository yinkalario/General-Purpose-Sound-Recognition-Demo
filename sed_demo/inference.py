#!/usr/bin python
# -*- coding:utf-8 -*-


"""
This module contains functionality to convert audio waveforms to logmel
spectrograms and extract the categories using a pretrained PyTorch model.
"""


import numpy as np
import librosa
import torch


# ##############################################################################
# # AUDIO MODEL INFERENCE
# ##############################################################################
class AudioModelInference:
    """
    This class performs the following steps that depend heavily on the
    pretrained model:

    1. Convert audio waveform to log mel spectrograms using a mel filterbank
      with given number of mels and frequency range
    2. Normalize spectrogram with given means and standard deviations (see
      class attributes)
    3. Feed normalized spectrograms to PyTorch model and return results.
    """

    LOGMEL_MEANS = np.float32([
        -14.050895, -13.107869, -13.1390915, -13.255364, -13.917199,
        -14.087848, -14.855916, -15.266642,  -15.884036, -16.491768,
        -17.067415, -17.717588, -18.075916,  -18.84405,  -19.233824,
        -19.954256, -20.180824, -20.695705,  -21.031914, -21.33451,
        -21.758745, -21.917028, -22.283598,  -22.737364, -22.920172,
        -23.23437,  -23.66509,  -23.965239,  -24.580393, -24.67597,
        -25.194445, -25.55243,  -25.825129,  -26.309643, -26.703104,
        -27.28697,  -27.839067, -28.228388,  -28.746237, -29.236507,
        -29.937782, -30.755503, -31.674414,  -32.853516, -33.959763,
        -34.88149,  -35.81145,  -36.72929,   -37.746593, -39.000496,
        -40.069244, -40.947514, -41.79767,   -42.81981,  -43.8541,
        -44.895683, -46.086784, -47.255924,  -48.520145, -50.726765,
        -52.932228, -54.713795, -56.69902,   -59.078354])
    LOGMEL_STDDEVS = np.float32([
        22.680508, 22.13264,  21.857653, 21.656355, 21.565693, 21.525793,
        21.450764, 21.377304, 21.338581, 21.3247,   21.289171, 21.221565,
        21.175856, 21.049534, 20.954664, 20.891844, 20.849905, 20.809206,
        20.71186,  20.726717, 20.72358,  20.655743, 20.650305, 20.579372,
        20.583157, 20.604849, 20.5452,   20.561695, 20.448244, 20.46753,
        20.433657, 20.412025, 20.47265,  20.456116, 20.487215, 20.387547,
        20.331848, 20.310328, 20.292257, 20.292326, 20.241796, 20.19396,
        20.23783,  20.564362, 21.075726, 21.332186, 21.508852, 21.644777,
        21.727905, 22.251642, 22.65972,  22.800117, 22.783764, 22.78581,
        22.86413,  22.948992, 23.12939,  23.180748, 23.03542,  23.131435,
        23.454556, 23.39839,  23.254364, 23.198978])

    def __init__(self, model, winsize=1024, stft_hopsize=512, samplerate=32000,
                 stft_window="hahn", n_mels=64, mel_fmin=50, mel_fmax=14000):
        """
        :param model: A pretrained, ready-to-use PyTorch model that admits a
          batch of shape ``(b, 64, w)`` and returns a batch of predictions
          with shape ``(b, num_classes)``.
        :param winsize: This is the window size for the STFT and mel
          operations. Should match training settings.
        :param samplerate: Audio samplerate. Ideally it should match the one
          used during model training.
        :param n_mels: Number of mel bins. Should match training settings,
          in this case 64.
        :param mel_fmin: Lowest mel bin. Should match training settings.
        :param mel_fmax: Highest mel bin. Should match training settings.
        """
        self.model = model
        self.model.eval()
        #
        self.winsize = winsize
        self.stft_hopsize = stft_hopsize
        self.stft_window = stft_window
        #
        self.mel_filt = librosa.filters.mel(sr=samplerate,
                                            n_fft=winsize,
                                            n_mels=n_mels,
                                            fmin=mel_fmin,
                                            fmax=mel_fmax)

    def wav_to_logmel(self, wav_arr):
        """
        :param wav_arr: 1D audio array (float)
        :returns: normalized log-mel spectrogram of shape ``(t, n_mels)``
        """
        stft_spec = np.abs(librosa.stft(y=wav_arr,
                                        n_fft=self.winsize,
                                        hop_length=self.stft_hopsize,
                                        center=True,
                                        window=self.stft_window,
                                        pad_mode="reflect")) ** 2
        mel_spec = np.dot(self.mel_filt, stft_spec).T
        logmel_spec = librosa.power_to_db(mel_spec, ref=1.0, amin=1e-10,
                                          top_db=None)
        logmel_spec -= self.LOGMEL_MEANS
        logmel_spec /= self.LOGMEL_STDDEVS
        #
        return logmel_spec  # (t, nbins)

    def __call__(self, wav_arr):
        """
        Performs model inference on the given audio waveform.
        :param wav_arr: 1D audio array (float)
        :returns: Predictions with shape ``(num_output_classes,)``.
        """
        logmel_spec = self.wav_to_logmel(wav_arr)  # (t, nbins)
        with torch.no_grad():
            logmel_spec = torch.from_numpy(
                logmel_spec.astype(np.float32)).unsqueeze(0)  # (1, t, nbins)
            preds = self.model(logmel_spec).to("cpu").numpy().squeeze(axis=0)
        return preds  # shape: (num_classes,)


# ##############################################################################
# # PREDICTION TRACKER
# ##############################################################################
class PredictionTracker:
    """
    Post-processing module to filter out undesired classes from the output and
    retrieve top-k class names and confidences, sorted in descending order by
    their confidence.
    """

    def __init__(self, all_labels, allow_list=None, deny_list=None):
        """
        :param all_labels: List with all categories as returned by the model.
        :param allow_list: If not ``None``, contains the allowed categories.
        :param deny_list: If not ``None``, contains the categories ignored.
        """
        self.all_labels = all_labels
        self.all_lbls_to_idxs = {l: i for i, l in enumerate(all_labels)}
        if allow_list is None:
            allow_list = all_labels
        if deny_list is None:
            deny_list = []
        self.labels = [l for l in all_labels
                       if l in allow_list and l not in deny_list]
        self.lbls_to_idxs = {l: self.all_lbls_to_idxs[l] for l in self.labels}
        self.idxs = sorted(self.lbls_to_idxs.values())

    def __call__(self, model_probs, top_k=6, sorted_by_p=True):
        """
        :param model_probs: Result as provided by ``AudioModelInference``.
        :param top_k: How many top-confidence classes will be retrieved.
        :param sorted_by_p: If true, result is provided sorted by confidence in
          descending order.
        """
        assert top_k >= 1, "Only integer >= 1 allowed for top_k!"
        top_k += 1
        #
        tracked_probs = model_probs[self.idxs]
        top_idxs = np.argpartition(tracked_probs, -top_k)[-top_k:]
        top_probs = tracked_probs[top_idxs]
        top_labels = [self.labels[idx] for idx in top_idxs]
        result = list(zip(top_labels, top_probs))
        if sorted_by_p:
            result = sorted(result, key=lambda elt: elt[1], reverse=True)
        #
        return result
