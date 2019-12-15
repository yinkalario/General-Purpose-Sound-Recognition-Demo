import h5py
import librosa
import numpy as np
import pandas as pd
import pyaudio
import torch

import models_code as models

# parameters
CHANNELS = 1            # channels of recordings
RATE = 32000            # sample rate
REC_BUFFER_SIZE = 1024  # recorder buffer size, every records has channel's number of samples, if channel = 2, each record has 2 samples
                        # this parameter also determines the delay between recording and playback, LEN_REC_BUFFER / RATE (s) is the actual delay
chunk_len = 62          # 62 * REC_BUFFER_SIZE / RATE, approximately 2 (s) data for inference        (*algorithm cost*)
chunk_hs =  10          # 7 * REC_BUFFER_SIZE / RATE, approximately 1280 (ms), chunk level hop size   (*algorithm cost*)

nfft = 1024             # nfft for stft
hsfft = 500             # hsfft for fft hop size
chunk_stft_len = 128    # neural network time axis
mel_bins = 64           # neural network frequency axis
window = 'hann'
fmin = 50
fmax = 14000

cuda = False

model_dir = './models/'
model_path = model_dir + 'Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth'
scalar_fn = model_dir + 'scalar.h5'
csv_fname = model_dir + 'validate_meta.csv'


class DataRecorded():

    def __init__(self):
        

        self.chunk = np.zeros((chunk_len, REC_BUFFER_SIZE))
        self.chunk_idx = 0
        self.chunk_hs_count = 0
        self.Flag_Chunk_Full = False

        self.chunk_ready = np.zeros((chunk_len, REC_BUFFER_SIZE))

        self.melW = librosa.filters.mel(sr=RATE,
                                        n_fft=nfft,
                                        n_mels=mel_bins,
                                        fmin=fmin,
                                        fmax=fmax)

    def logmel_extract(self):

        S = np.abs(librosa.stft(y=self.chunk_ready,
                                n_fft=nfft,
                                hop_length=hsfft,
                                center=True,
                                window=window,
                                pad_mode='reflect'))**2

        mel_S = np.dot(self.melW, S).T
        log_mel_S = librosa.power_to_db(mel_S, ref=1.0, amin=1e-10, top_db=None)

        return log_mel_S   


def load_class_label_indices(class_labels_indices_path):    
    df = pd.read_csv(class_labels_indices_path, sep=',')
    labels = df['display_name'].tolist()
    lb_to_ix = {lb: i for i, lb in enumerate(labels)}
    ix_to_lb = {i: lb for i, lb in enumerate(labels)}    
    return labels, lb_to_ix, ix_to_lb


class AudioDetection():
    
    def __init__(self, num_results):

        # load nn model
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        # import pdb;pdb.set_trace()
        self.model = models.Cnn9_GMP_64x64(527)
        self.model.load_state_dict(checkpoint['model'])
        if cuda:
            self.model.cuda()
        
        self.num_results = num_results
        # load scalar
        with h5py.File(scalar_fn, 'r') as hf:
            self.mean = hf['mean'][:]
            self.std = hf['std'][:]

        # load label names
        _, _, self.ix_to_lb = load_class_label_indices(csv_fname)

        # initialize data format
        self.data = DataRecorded()

        # set record stream
        self.p = pyaudio.PyAudio()

        # start recording stream
        self.stream = self.p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=RATE,
                input=True, # record
                output=False, # playback
                frames_per_buffer=REC_BUFFER_SIZE,
                stream_callback=self.callback,
                start=False)
        
        self.running = True
        

    def callback(self, in_data, frame_count, time_info, status):
        '''

        Callback method is to unblockingly record and inference.
        Realtime signal processing could be added into this block.
        Segmentation of the time domain signal is done here.
        '''
        self.data.chunk[self.data.chunk_idx, :] = np.fromstring(in_data, dtype=np.float32)
        self.data.chunk_idx += 1
        if self.data.chunk_idx == chunk_len:
            self.data.chunk_idx = 0

        self.data.chunk_hs_count += 1
        if self.data.chunk_hs_count == chunk_hs:
            self.data.chunk_hs_count = 0
            self.data.Flag_Chunk_Full = True
            
        return (in_data, pyaudio.paContinue)
    

    def inference(self, x):
        '''

        Inference output for single instance from neural network
        '''
        x = torch.Tensor(x).view(1, x.shape[0], x.shape[1])
        if cuda:
            x = x.cuda()
        
        with torch.no_grad():
            self.model.eval()
            y = self.model(x)

        prob = y.data.cpu().numpy().squeeze(axis=0)
        predict_idxs = prob.argsort()[-self.num_results:][::-1]
        predict_probs = prob[predict_idxs]
        return predict_idxs, predict_probs
    

    def start_detection(self, resultLabel, posthocProb):

        self.stream.start_stream()
        
        while self.running and self.stream.is_active():

            if self.data.Flag_Chunk_Full:

                self.data.chunk_ready = \
                    np.vstack((self.data.chunk[self.data.chunk_idx:,:], self.data.chunk[0:self.data.chunk_idx,:])).reshape(-1,)
                x = self.data.logmel_extract()
                x = (x - self.mean) / self.std

                # obtain labels and probabilities
                predict_idxs, predict_probs = self.inference(x)
                predict_labels = []
                for idx in predict_idxs:
                    predict_labels.append(self.ix_to_lb[idx])
                
                # set GUI display
                for n in range(self.num_results):
                    posthocProb[n].set(predict_probs[n])
                    tempStr='{0:<15.15}'.format(predict_labels[n])
                    resultLabel[n].set(tempStr)
                    
    
    def stop_detection(self):

        self.stream.stop_stream()
        
    
    def terminate_detection(self):

        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
