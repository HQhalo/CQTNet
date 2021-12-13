import os
import argparse
from cqt_loader import *
from torch.utils.data import DataLoader
import models
from tqdm import tqdm
import numpy as np
import torch
from utility import *
from numpy import dot
from numpy.linalg import norm
import csv 

DEVICE= "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser(description = "infer");

parser.add_argument('--model',         type=str,   default='CQTNetAngular',   help='Demucs model');

parser.add_argument('--load_model_path',     type=str,   default='/content/saved_models/last.pth',    help='model path');


parser.add_argument('--vocal_path',     type=str,   default='/content/database/vocals_npy',    help='vocal path');
parser.add_argument('--hum_path',     type=str,   default='/content/database/hum_npy',    help='hum vocal path');

parser.add_argument('--hum_length',     type=int,   default=None,    help='hum length');

parser.add_argument('--result_filename',     type=str,   default='/content/submit.csv',    help='result file name');

args = parser.parse_args();

def main():
  model = getattr(models, args.model)() 
  if DEVICE == "cuda": 
    model = torch.nn.DataParallel(model)
    model.module.load(args.load_model_path)
  else:
    model.load(args.load_model_path, DEVICE)

  model.to(DEVICE)

  hum_data = CQTHum(args.hum_path, out_length=None)
  hum_dataloader = DataLoader(hum_data, 1, shuffle=False,num_workers=1)

  model.eval()

  hum_features = {}  
  hum_lengths = []
  for ii, (data, label) in tqdm(enumerate(hum_dataloader)):
    input = data.to(DEVICE)
    score, feature = model(input)
    feature = feature.data.cpu().numpy().reshape(-1)
    hum_features[label[0]] = [ data.shape[3] ,feature]
    hum_lengths.append(data.shape[3])
  
  hum_length = args.hum_length if args.hum_length is not None else int(np.mean(hum_lengths))
  print('Hum length: ', hum_length)
  
  vocals_data = CQTVocal(args.vocal_path, hum_length)
  vocal_dataloader = DataLoader(vocals_data, batch_sampler=BalancedBatchSampler(vocals_data.dataset, 32), shuffle=False,num_workers=1, pin_memory=True)
  vocals_features = {}

  for ii, (data, label) in tqdm(enumerate(vocal_dataloader)):
    input = data.to(DEVICE)
    score, feature = model(input)
    for idx, song_feat in enumerate(feature.data.cpu().numpy()):
      song_id = label[0][idx]
      if song_id not in vocals_features:
        vocals_features[song_id] = []    
      vocals_features[song_id].append(song_feat.reshape(-1))
    break
  # print(vocals_features)


  result = []
  hum_ids = list(hum_features.keys())
  hum_ids.sort()
  for hum_id in tqdm(hum_ids):
    hum_len, hum_feat = hum_features[hum_id]
    
    topVocal = topTen(hum_feat, vocals_features)
    
    result.append([f"{hum_id}.mp3"] + list(map(lambda x: x[0] , topVocal)))

  with open(args.result_filename, 'w') as f: 
      write = csv.writer(f) 
      write.writerows(result) 

def topTen(hum_feat, vocals_features):
    scores = []
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    for track_id in vocals_features.keys():
        vocal_feats = vocals_features[track_id]
        in1 = torch.Tensor(vocal_feats).cuda()
        in2 = torch.Tensor(hum_feat).cuda()
        maxScore = torch.max(cos(in1, in2)).cpu().item()
        scores.append([track_id, maxScore])
    scores.sort(reverse=True, key = lambda x: x[1])
    return scores[:10]


if __name__=='__main__':
  main()