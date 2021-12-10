import os
import argparse
from cqt_loader import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import models
from config import opt
from tqdm import tqdm
import numpy as np
import torch
from utility import *
from numpy import dot
from numpy.linalg import norm

parser = argparse.ArgumentParser(description = "infer");

parser.add_argument('--model',         type=str,   default='mdx_extra',   help='Demucs model');

## Data loader
parser.add_argument('--load_model_path',     type=str,   default='/content/saved_models/last.pth',    help='model path');
parser.add_argument('--parallel',     type=bool,   default=True,    help='gpu');
parser.add_argument('--vocal_path',     type=str,   default='/content/database/vocals_npy',    help='vocal path');
parser.add_argument('--hum_path',     type=str,   default='/content/database/hum_npy',    help='hum vocal path');

args = parser.parse_args();

def main():
  model = getattr(models, args.model)() 
  if args.parallel is True: 
      model = torch.nn.DataParallel(model)
  if args.parallel is True:
    model.module.load(opt.load_model_path)

  vocals_data = CQTVal(args.vocal_path, out_length=None)
  hum_data = CQTVal(args.vocal_path, out_length=None)

  vocal_dataloader = DataLoader(vocals_data, 1, shuffle=False,num_workers=1)
  hum_dataloader = DataLoader(hum_data, 1, shuffle=False,num_workers=1)


  model.eval()

  for ii, (data, label) in enumerate(vocal_dataloader): 
    print(label)