# General-Purpose-Sound-Recognition-Demo


This is the origininal version of our general-purpose sound recognition demo, developed in 2019. Check [here](https://github.com/yinkalario/General-Purpose-Sound-Recognition-Demo) for newer versions.

## Example
We apply the audio tagging system to build a sound event detection (SED) system. The SED prediction is obtained by applying the audio tagging system on consecutive 2-second segments. The video of demo can be viewed at: <br>
https://www.youtube.com/watch?v=7TEtDMzdLeY

## Download models
This is a demo based on our AudioSet work. Please download trained models from https://zenodo.org/record/3576599, and save models in **models** folder. This demo uses `CNN9`(~150MB). You can download it using `wget` as follows:

```
wget https://zenodo.org/record/3576599/files/Cnn9_GMP_64x64_300000_iterations_mAP%3D0.37.pth?download=1
```

Please also check our AudioSet work on https://github.com/qiuqiangkong/audioset_tagging_cnn, where the model of this demo is trained.

## Install & Run

We recommend creating a Python environment using Anaconda and installing the dependencies as follows:

```shell
conda create -c anaconda -n sed_demo python=3.7 pyaudio
conda activate sed_demo
pip install -r requirements.txt
```

Alternatively, you can find a full list of working dependencies (tested on Ubuntu 20.04) [here](all_dependencies.txt).

Then, the following command starts the app:

```shell
python MSoS_demo_recognition.py
```

## Citation
If you use our codes in any format, please consider citing the following paper:

[1] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." arXiv preprint arXiv:1912.10211 (2019).

## Authors
Yin Cao, Qiuqiang Kong, Christian Kroos, Turab Iqbal, Wenwu Wang, Mark Plumbley
