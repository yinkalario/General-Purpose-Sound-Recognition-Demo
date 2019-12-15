import h5py
import librosa
import numpy as np
import sounddevice as sd
import torch
import models_generalisation as models

model_fn = './models_generalisation/model.pth'
scalar_fn = './models_generalisation/scalar.h5'

labels = ['Effects', 'Human', 'Music', 'Nature', 'Urban'] # Don't change the order, this order is used in model training procedure
lb_to_ix = {lb: ix for ix, lb in enumerate(labels)}
ix_to_lb = {ix: lb for ix, lb in enumerate(labels)}


class AudioAnalysis:

    def __init__(self):

        self.duration = 5 # seconds
        self.fs = 32000
        sd.default.samplerate = self.fs
        self.data = None

        self.melW = librosa.filters.mel(sr=self.fs,
                                        n_fft=1024,
                                        n_mels=64,
                                        fmin=50)

        # load nn model
        checkpoint = torch.load(model_fn, map_location=lambda storage, loc: storage)
        self.model = models.Cnn13(len(labels))
        self.model.load_state_dict(checkpoint['state_dict'])

        # load scalar
        with h5py.File(scalar_fn, 'r') as hf:
            self.mean = hf['mean'][:]
            self.std = hf['std'][:]
            

    def load(self, soundFilename):

        self.data, _ = librosa.load(soundFilename, sr=self.fs, mono=True)

    
    
    def record(self):

        self.data = sd.rec(int(self.duration * self.fs), channels=1)
        sd.wait()

    
    
    def play(self):

        if not(self.data is None):
            sd.play(self.data)

    

    def logmel_extract(self):

        S = np.abs(librosa.stft(y=self.data,
                                n_fft=1024,
                                hop_length=500,
                                center=True,
                                window='hann',
                                pad_mode='reflect'))**2

        mel_S = np.dot(self.melW, S).T
        log_mel_S = librosa.power_to_db(mel_S, ref=1.0, amin=1e-10, top_db=None)

        return log_mel_S   



    def analysis(self):
        
        if not(self.data is None):
            
            self.data = np.squeeze(self.data)
            x = self.logmel_extract()
            x = (x - self.mean) / self.std
    
            x = torch.Tensor(x).view(1, x.shape[0], x.shape[1])
    
            with torch.no_grad():
                self.model.eval()
                y = self.model(x)
    
            prob = torch.exp(y).data.cpu().numpy().squeeze(axis=0)
            prob = np.array([prob[3], prob[2], prob[1], prob[0], prob[4]]) # change the order to ['Nature', 'Music', 'Human', 'Effects', 'Urban']
            predict_idxs = prob.argsort()[-5:][::-1]
    
            return prob, predict_idxs


if __name__ == '__main__':
    
    print('Import without zombies') 