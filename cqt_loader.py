import os,sys
from torchvision import transforms
import torch, torch.utils
import numpy as np
from torch.utils.data import Dataset
import random
import PIL
import more_itertools as mit

from torch.utils.data.sampler import BatchSampler


class BalancedBatchSampler(BatchSampler):

    #[set_id, in_path , x]
    def __init__(self, dataset, max_batch_size):
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.frame_dataset = {}
        for idx, item in enumerate(self.dataset):
            l = len(item[2])
            if l not in self.frame_dataset:
                self.frame_dataset[l] = []
            self.frame_dataset[l].append(idx)
        self.batch_dataset = []
        for key in self.frame_dataset.keys(): 
            self.batch_dataset.extend([self.frame_dataset[key][i:i + self.max_batch_size] for i in range(0, len(self.frame_dataset[key]),self.max_batch_size)])
        print('Number Batch ' , len(self.batch_dataset))
    def __iter__(self):
        for batch in self.batch_dataset:
            yield batch
        
    def __len__(self):
        return len(self.batch_dataset)


np.random.seed(42)
def cut_data(data, out_length):
    if out_length is not None:
        if data.shape[0] > out_length:
            max_offset = data.shape[0] - out_length
            offset = np.random.randint(max_offset)
            data = data[offset:(out_length+offset),:]
        else:
            offset = out_length - data.shape[0]
            data = np.pad(data, ((0,offset),(0,0)), "constant")
    if data.shape[0] < 200:
        offset = 200 - data.shape[0]
        data = np.pad(data, ((0,offset),(0,0)), "constant")
    return data
def cut_data_front(data, out_length):
    if out_length is not None:
        if data.shape[0] > out_length:
            max_offset = data.shape[0] - out_length
            offset = 0
            data = data[offset:(out_length+offset),:]
        else:
            offset = out_length - data.shape[0]
            data = np.pad(data, ((0,offset),(0,0)), "constant")
    if data.shape[0] < 200:
        offset = 200 - data.shape[0]
        data = np.pad(data, ((0,offset),(0,0)), "constant")
    return data
def shorter(feature, mean_size=2):
    length, height  = feature.shape
    new_f = np.zeros((int(length/mean_size),height),dtype=np.float64)
    for i in range(int(length/mean_size)):
        new_f[i,:] = feature[i*mean_size:(i+1)*mean_size,:].mean(axis=0)
    return new_f

class CQT(Dataset):
    def __init__(self, filepath , out_length=None):
        self.indir = filepath
        self.file_list = list(os.listdir(filepath))
        self.out_length = out_length
    def __getitem__(self, index):
        transform_train = transforms.Compose([
            lambda x : x.T,
            lambda x : change_speed(x, 0.7, 1.3),
            # lambda x : x-np.mean(x),
            lambda x : x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6),
            lambda x : cut_data(x, self.out_length),
            lambda x : torch.Tensor(x),
            lambda x : x.permute(1,0).unsqueeze(0),
        ])
        
        filename = self.file_list[index].strip()
        set_id, version_id = filename.split('.')[0].split('-')
        in_path = os.path.join(self.indir, filename)
        data = np.load(in_path) # from 12xN to Nx12

        data = transform_train(data)
        return data, int(set_id)
    def __len__(self):
        return len(self.file_list)


class CQTSiamese(Dataset):
    def __init__(self, filepath , out_length=None, song_factor=2):
        self.transform = transforms.Compose([
            lambda x : x.T,
            lambda x : change_speed(x, 0.9, 1.1),
            # lambda x : x-np.mean(x),
            lambda x : x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6),
            lambda x : cut_data(x, self.out_length),
            lambda x : torch.Tensor(x),
            lambda x : x.permute(1,0).unsqueeze(0),
        ])
        self.indir = filepath
        self.file_list = list(os.listdir(filepath))
        self.out_length = out_length
        self.hums = []
        self.songs = {}
        self.labels = []

        for i in range(len(self.file_list)):
            fileName = self.file_list[i]
            id, t = fileName.split('-')[0].split('_')
            self.labels.append(id)
            if t == 'hum':
              self.hums.append([id, i, fileName])
            elif t == 'song':
              if id not in self.songs:
                self.songs[id] = []
              self.songs[id].append([i, fileName])

        self.labels_set = set(self.labels)

        self.cqtFeature = []
        
        for i in range(len(self.file_list)):
            in_path = os.path.join(self.indir, self.file_list[i])
            data = np.load(in_path)
            data = self.transform(data)
            self.cqtFeature.append(data)

        self.posPair = []
        self.negPair = []
        for hum in self.hums:
            id, humIdx, _ = hum
            for song in self.songs[id]:
                self.posPair.append([humIdx, song[0]])

            negSongId =  np.random.choice(list(self.labels_set - set([id])))
            for song in self.songs[negSongId]:
                self.negPair.append([humIdx, song[0]])
        
        self.pairData = []
        for p in self.posPair:
            self.pairData.append([p, 1])
        for p in self.negPair:
            self.pairData.append([p, 0])

        random.seed(42)
        random.shuffle(self.pairData)

        
    def __getitem__(self, index):
        pair, target = self.pairData[index]
        humIdx, songIdx = pair
        data1 = self.cqtFeature[humIdx]
        data2 = self.cqtFeature[songIdx]
        

        return (data1, data2), target

    def __len__(self):
        return len(self.pairData)


class CQTVal(Dataset):
    def __init__(self, filepath , out_length=None):
        self.indir = filepath
        self.file_list = list(os.listdir(filepath))
        self.out_length = out_length
    def __getitem__(self, index):
        transform_test = transforms.Compose([
            lambda x : x.T,
            # lambda x : x-np.mean(x),
            lambda x : x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6),
            lambda x : cut_data_front(x, self.out_length),
            lambda x : torch.Tensor(x),
            lambda x : x.permute(1,0).unsqueeze(0),
        ])
        
        filename = self.file_list[index].strip()
        set_id, version_id = filename.split('.')[0].split('-')
        in_path = os.path.join(self.indir, filename)
        data = np.load(in_path) # from 12xN to Nx12

        data = transform_test(data)
        return data, [set_id, version_id]
    def __len__(self):
        return len(self.file_list)

# hum_len = len(hum_feat)
#     hum_pad = 0.1*hum_len
#     for track_id in vocals_features.keys():
#         vocal_feat = vocals_features[track_id][0]
#         for search_len in [hum_len - hum_pad, hum_len, hum_len + hum_pad ]:
#           windows = list(mit.windowed(vocal_feat, n=search_len, step=hum_pad))
#           windows = [list(filter(None, w)) for w in windows]


class CQTVocal(Dataset):
    def __init__(self, filepath , hum_length, file_list):
        self.indir = filepath
        self.file_list = file_list
        self.hum_length = hum_length
        self.hum_pad = int(0.05 * hum_length)
        self.stride = int(0.1 * hum_length)

        self.dataset = []
        for filename in self.file_list:
            set_id, _ = filename.split('.')[0].split('-')
            in_path = os.path.join(self.indir, filename)
            vocal_feat = np.load(in_path) 
            vocal_indxs = list(range(vocal_feat.shape[1]))
            frame_idx = []
            for search_len in [self.hum_length + self.hum_pad*x for x in list(range(-3, 4))]:
                windows = list(mit.windowed(vocal_indxs, n=search_len, step=self.stride))
                windows = [ [x for x in list(w) if x is not None] for w in windows]
                frame_idx.extend(windows)
            self.dataset.extend([[set_id, in_path , x] for x in frame_idx])
            
    def __getitem__(self, index):
        transform_test = transforms.Compose([
            lambda x : x.T,
            # lambda x : x-np.mean(x),
            lambda x : x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6),
            lambda x : cut_data_front(x, None),
            lambda x : torch.Tensor(x),
            lambda x : x.permute(1,0).unsqueeze(0),
        ])
        
        set_id, in_path, frame_idx = self.dataset[index]

        data = np.load(in_path) # from 12xN to Nx12
        data = data.T[frame_idx].T

        data = transform_test(data)

        return data, [set_id, 0]
    def __len__(self):
        return len(self.dataset) 

class CQTHum(Dataset):
    def __init__(self, filepath , out_length=None):
        self.indir = filepath
        self.file_list = list(os.listdir(filepath))
        self.out_length = out_length
    def __getitem__(self, index):
        transform_test = transforms.Compose([
            lambda x : x.T,
            # lambda x : x-np.mean(x),
            lambda x : x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6),
            lambda x : cut_data_front(x, self.out_length),
            lambda x : torch.Tensor(x),
            lambda x : x.permute(1,0).unsqueeze(0),
        ])
        
        filename = self.file_list[index].strip()
        hum_id = filename.split('.')[0]
        in_path = os.path.join(self.indir, filename)
        data = np.load(in_path) # from 12xN to Nx12

        data = transform_test(data)
        return data, hum_id
    def __len__(self):
        return len(self.file_list)

def change_speed(data, l=0.7, r=1.5): # change data.shape[0]
    new_len = int(data.shape[0]*np.random.uniform(l,r))
    maxx = np.max(data)+1
    data0 = PIL.Image.fromarray((data*255.0/maxx).astype(np.uint8))
    transform = transforms.Compose([
        transforms.Resize(size=(new_len,data.shape[1])), 
    ])
    new_data = transform(data0)
    return np.array(new_data)/255.0*maxx

if __name__=='__main__':
    train_dataset = HPCP('train', 394)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=12, shuffle=True)
