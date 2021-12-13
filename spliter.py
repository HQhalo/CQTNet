import io
import os
import argparse
from pathlib import Path
import select
import subprocess as sp
import sys
from typing import Dict, Tuple, Optional, IO
import glob
from tqdm import tqdm
import pickle 

parser = argparse.ArgumentParser(description = "demucs");

parser.add_argument('--model',         type=str,   default='mdx_extra',   help='Demucs model');

## Data loader
parser.add_argument('--in_path',     type=str,   default='/content/public_test/full_song',    help='Input path');
parser.add_argument('--out_separated_path',     type=str,   default='/content/train_separated',    help='Output path');
parser.add_argument('--out_vocal_path',     type=str,   default='/content/vocals',    help='Output vocal path');
parser.add_argument('--sub_file_path',     type=str,   default='/content/drive/MyDrive/colabdrive/humming/fake_hum/sub_infiles_1.pickle',    help='Output vocal path');


args = parser.parse_args();

extensions = ["mp3"]
SHIFTS = 1

def find_files(in_path):
    out = []
    for file in Path(in_path).iterdir():
        if file.suffix.lower().lstrip(".") in extensions:
            out.append(file)
    return out

def copy_process_streams(process: sp.Popen):
    def raw(stream: Optional[IO[bytes]]) -> IO[bytes]:
        assert stream is not None
        if isinstance(stream, io.BufferedIOBase):
            stream = stream.raw
        return stream

    p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)
    stream_by_fd: Dict[int, Tuple[IO[bytes], io.StringIO, IO[str]]] = {
        p_stdout.fileno(): (p_stdout, sys.stdout),
        p_stderr.fileno(): (p_stderr, sys.stderr),
    }
    fds = list(stream_by_fd.keys())

    while fds:
        # `select` syscall will wait until one of the file descriptors has content.
        ready, _, _ = select.select(fds, [], [])
        for fd in ready:
            p_stream, std = stream_by_fd[fd]
            raw_buf = p_stream.read(2 ** 16)
            if not raw_buf:
                fds.remove(fd)
                continue
            buf = raw_buf.decode()
            std.write(buf)
            std.flush()

def separate(files, outp, model):
    cmd = ["python3", "-m", "demucs.separate", "-o", str(outp), "-n", model, "--shifts", str(SHIFTS)]
    
    
    print("Going to separate the files:")
    print('\n'.join(files))
    print("With command: ", " ".join(cmd))
    p = sp.Popen(cmd + files, stdout=sp.PIPE, stderr=sp.PIPE)
    copy_process_streams(p)
    p.wait()
    if p.returncode != 0:
        print("Command failed, something went wrong.")
        return False
    return True

if __name__=='__main__':
    if not os.path.exists(args.out_separated_path):
        print('Create out_separated_path folder')
        os.makedirs(args.out_separated_path)

    if not os.path.exists(args.out_vocal_path):
        print('Create out_vocal_path folder')
        os.makedirs(args.out_vocal_path)
    
    with open(args.sub_file_path, 'rb') as handle:
        sub_infiles = pickle.load(handle)
    
    for sub_infile in sub_infiles:
        retry = 3
        while retry > 0:
            res = separate(sub_infile, args.out_separated_path, args.model)
            if res == False:
                retry -= 1
            else:
                retry = 0

    files = glob.glob(f'{args.out_separated_path}/mdx_extra/*/vocals.wav')
    for fname in tqdm(files):
        outfile = f"{args.out_vocal_path}/{fname.split('/')[-2]}-song_vocals.wav"
        os.system(f"cp {fname} {outfile}")