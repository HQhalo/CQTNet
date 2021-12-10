FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
WORKDIR /model

RUN pip3 install torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY ./ ./


