#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This package implements a real-time sound event detection demo. This is
achieved with the following structure:

* A GUI module, written in ``tkinter``, containing most of the frontend
* An audio module, using ``pyaudio``, to record audio from the system
* An inference module, using ``PyTorch``, to detect events from the audio
* A main module that connects all elements into an application, optionally
  collects custom options from the user, and runs the app.
* Other auxiliary modules

Check the respective docstrings for more details.
"""


import os
from . import __path__ as ROOT_PATH


AI4S_BANNER_PATH = os.path.join(ROOT_PATH[0], "assets", "ai4s_banner.png")
SURREY_LOGO_PATH = os.path.join(ROOT_PATH[0], "assets", "surrey_logo.png")
CVSSP_LOGO_PATH = os.path.join(ROOT_PATH[0], "assets", "CVSSP_logo.png")
EPSRC_LOGO_PATH = os.path.join(ROOT_PATH[0], "assets", "EPSRC_logo.png")
#
AUDIOSET_LABELS_PATH = os.path.join(
    ROOT_PATH[0], "assets", "audioset_labels.csv")
