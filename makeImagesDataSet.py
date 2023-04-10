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
        paths = glob.glob("./data/test_soundscapes/**/*.ogg")

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
    
    # @staticmethod
    # def melgram_v1(audio_file_path, to_file):
    #     sig, fs = librosa.load(audio_file_path)
        
    #     plt.figure(figsize=(3,2))
    #     plt.axis('off')  # no axis
    #     plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
    #     S = librosa.feature.melspectrogram(y=sig, sr=32000)
       
    #     height = S.shape[0]  
    #     image_cropped = S[int(height*0.25  ):int(height*0.75 ),:]
    #     librosa.display.specshow(librosa.power_to_db(image_cropped, ref=np.max))
    #     plt.savefig(to_file, bbox_inches=None, pad_inches=0)
    #     plt.close()
    
    @staticmethod
    def melgram_v2(audio, sample_rate,  to_file):
         
        plt.figure(figsize=(3,2))
        plt.axis('off')  # no axis
        plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
        melspectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
       
        # height = S.shape[0]  
        # image_cropped = S[int(height*0.2  ):int(height*0.8 ),:]
        librosa.display.specshow(librosa.power_to_db(melspectrogram, ref=np.max))
        plt.savefig(to_file, bbox_inches=None, pad_inches=0)
        plt.close()




dataLoader = DataLoader(dataSet, batch_size=batch_size, collate_fn=DataProcessing.my_collate)


for batch, (X, y) in tqdm(enumerate(dataLoader), total=len(dataLoader), leave=False):
    for x in range(0, len(X)):
        label = y[x] 
        (frame, sample_rate, name) = X[x]
        Path(f'./data/test_melspectrogram/{label}/').mkdir(parents=True, exist_ok=True)
        DataProcessing.melgram_v2(frame.cpu().numpy(), sample_rate, f'./data/test_melspectrogram/{label}/{name}.png')
        


# item = next(iter(dataLoader))
# (frame, sample_rate) = item[0][0]
# print(frame.size(), sample_rate)

# DataProcessing.melgram_v2(frame.numpy(), sample_rate, './qqqqqq.png')

# class SpectogrammProvider:
#     def __init__(self, dataProvider: DataProvider):
#         self.dataProvider = dataProvider
    
