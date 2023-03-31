import os
import glob
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt    
import numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainAudioDataset(Dataset):
    def __init__(self):
        # class_names = sorted(os.listdir('./data/train_audio/'))
        # class_names_dic = {class_names[i]: i for i in range(0, len(class_names))}
        paths = glob.glob("./data/train_audio/**/*.ogg")

        df = pd.DataFrame(data={'x': paths, 'y': [x.split('\\')[-2] for x in paths], 'name': [x.split('\\')[-1] for x in paths]})
        df['y'] = df['y'].astype('category')
        # df['y'] = df['y'].map(class_names_dic)

        self.dataFrame = df

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, idx):
        row = self.dataFrame.iloc[idx]
        label = row['y'] 
        waveform, sample_rate = torchaudio.load(row['x'])
        return (waveform.cuda()[0], sample_rate, row['name']), label

dataSet = TrainAudioDataset()
batch_size = 1
n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128
audio_spectogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=32000,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    pad_mode="reflect", 
    power=2.0, 
    norm='slaney',
    mel_scale="htk",
    center=True,
).cuda()


class DataProcessing: 
    @staticmethod
    def record_to_frames(waveform, sample_rate, frame_size=5):
        p1d = (1, sample_rate * frame_size)
        out = torch.nn.functional.pad(waveform, p1d, "constant", 0)
        return out.unfold(0, sample_rate * frame_size, sample_rate * frame_size)

    @staticmethod
    def my_collate(batch):
        frames = []
        labels = []
        for (data,label) in batch:
                waveform, sample_rate, name = data
                l_frames = DataProcessing.record_to_frames(waveform, sample_rate)
            
                for index in range(l_frames.size()[0]):
                    frame = l_frames[index]
                    # audio_spectogram = spectogram(frame)
                    # audio_spectogram = audio_spectogram.repeat(3, 1, 1)
                    frames.append((frame, sample_rate, f'{name}_{index}'))
                    labels.append(label) 
        return [frames, labels]
    
    @staticmethod
    def melgram_v2(audio, to_file):
    
        plt.figure(figsize=(4.9,2))
        plt.axis('off')  # no axis
        plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
        melspectrogram = audio_spectogram(audio)
       
        # height = S.shape[0]  
        # image_cropped = S[int(height*0.2  ):int(height*0.8 ),:]
        plt.imshow(melspectrogram.repeat(3, 1, 1).log2()[0,:,:].cpu().numpy() )
        plt.savefig(to_file, bbox_inches=None, pad_inches=0)
        plt.close()




dataLoader = DataLoader(dataSet, batch_size=batch_size, collate_fn=DataProcessing.my_collate)


for batch, (X, y) in tqdm(enumerate(dataLoader), total=len(dataLoader), leave=False):
    for x in range(0, len(X)):
        label = y[x] 
        (frame, sample_rate, name) = X[x]
        Path(f'./data/train_melspectrogram/{label}/').mkdir(parents=True, exist_ok=True)
        DataProcessing.melgram_v2(frame,  f'./data/train_melspectrogram/{label}/{name}.png')
        


# item = next(iter(dataLoader))
# (frame, sample_rate, name) = item[0][0]
# # print(frame.size(), sample_rate)

# DataProcessing.melgram_v2(frame, './qqqqqq.png')

# class SpectogrammProvider:
#     def __init__(self, dataProvider: DataProvider):
#         self.dataProvider = dataProvider
    
