#!/usr/bin python
# -*- coding:utf-8 -*-


"""
This module contains various utilities.

A substantial part has been borrowed from:
https://github.com/qiuqiangkong/audioset_tagging_cnn
"""


import torch
import csv


# ##############################################################################
# # I/O
# ##############################################################################
def load_csv_labels(labels_csv_path):
    """
    Given the path to a 3-column CSV file ``(index, class_ID, class_name)``
    with comma-separated entries, this function ignores the first row and
    returns the triple ``(number_of_classes, IDs, names)``, in order of
    appearance.
    """
    with open(labels_csv_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        lines = list(reader)
    idxs, ids, labels = zip(*lines[1:])
    num_classes = len(labels)
    return num_classes, ids, labels


# ##############################################################################
# # PYTORCH
# ##############################################################################
def move_data_to_device(x, device):
    """
    """
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def interpolate(x, ratio):
    """Interpolate the prediction to compensate the downsampling operation in a
    CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to upsample
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


def do_mixup(x, mixup_lambda):
    """
    """
    out = x[0::2].transpose(0, -1) * mixup_lambda[0::2] + \
        x[1::2].transpose(0, -1) * mixup_lambda[1::2]
    return out.transpose(0, -1)
