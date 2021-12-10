import glob
import os
from tqdm import tqdm
import subprocess
import argparse
from nnAudio import features
from scipy.io import wavfile
import torch
import numpy as np

DEVICE= "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
cqt_extractor = features.CQT(sr=SAMPLE_RATE).to(DEVICE)

parser = argparse.ArgumentParser(description = "gencqt");

parser.add_argument('--vocal_path',     type=str,   default='/content/vocals',    help='vocal path');
parser.add_argument('--out_vocals_path',     type=str,   default='/content/database/vocals',    help='out vocal path');

parser.add_argument('--hum_path',     type=str,   default='/content/public_test/hum',    help='hum path');
parser.add_argument('--out_hum_path',     type=str,   default='/content/database/hum',    help='out hum path');

args = parser.parse_args();



def extractCqt(filename):
    sr, song = wavfile.read(filename) # Loading your audio
    assert sr == SAMPLE_RATE
    x = torch.tensor(song, device=DEVICE).float() # casting the array into a PyTorch Tensor
    spec = cqt_extractor(x)
    spec = spec[0].cpu().detach().numpy() 

    mean_size = 4

    height, length = spec.shape
    new_cqt = np.zeros((height,int(length/mean_size)),dtype=np.float64)
    for i in range(int(length/mean_size)):
        new_cqt[:,i] = spec[:,i*mean_size:(i+1)*mean_size].mean(axis=1)
    return new_cqt


def convert2wav(inputdir, outputdir, ext='.wav'):
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    files = glob.glob(f'{inputdir}/*{ext}')
    files.sort()

    for fname in tqdm(files):
        name = fname.split('/')[-1].split('.')[0]
        outfile = f"{outputdir}/{name}.wav"            
        out = subprocess.call('ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null' %(fname,outfile), shell=True)
        if out != 0:
            raise ValueError('Conversion failed %s.'%fname)

def getCqt(inputDir):
    outdir = inputDir+"_npy"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for filename in tqdm(list(glob.glob(f"{inputDir}/*.wav"))):
        new_cqt =  extractCqt(filename)
        n = filename.split('/')[-1].split('.')[0]
        with open(f'{outdir}/{n}.npy', 'wb') as f:
            np.save(f, new_cqt)
            
if __name__=='__main__':
    convert2wav(args.vocal_path, args.out_vocals_path)
    convert2wav(args.hum_path, args.out_hum_path, ext='.mp3')

    getCqt(args.out_vocals_path)
    getCqt(args.out_hum_path)
        

