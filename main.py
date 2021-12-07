import torch
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


torch.manual_seed(42)
torch.cuda.manual_seed(42)
# multi_size train
def multi_train(**kwargs):
    parallel = True 
    opt.model = 'CQTNetAngular'
    opt.notes='CQTNetAngular'
    # opt.batch_size=32
    #opt.load_latest=True
    #opt.load_model_path = ''
    opt._parse(kwargs)
    # step1: configure model
    
    model = getattr(models, opt.model)() 
    if parallel is True: 
        model = torch.nn.DataParallel(model)
    if parallel is True:
        if opt.load_latest is True:
            model.module.load_latest(opt.notes)
        elif opt.load_model_path:
            model.module.load(opt.load_model_path)
    else:
        if opt.load_latest is True:
            model.load_latest(opt.notes)
        elif opt.load_model_path:
            model.load(opt.load_model_path)
    model.to(opt.device)
    print(model)
    # step2: data
   
    train_data = CQT('/content/train_npy' ,out_length=None)
    val_data = CQTVal('/content/val_npy', out_length=None)
   
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, 1, shuffle=False,num_workers=1)
  
    #step3: criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    if parallel is True:
        optimizer = torch.optim.Adam(model.module.parameters(), lr=lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode='min',factor=opt.lr_decay,patience=2, verbose=True,min_lr=5e-6)
    #train
    songMrr = val_MRR(model, val_dataloader, -1)
    print(f"Epoch {-1}, running loss: {-1}, val MRR: {songMrr}")

    for epoch in range(opt.max_epoch):
        running_loss = 0
        num = 0
        for (data, label) in tqdm(train_dataloader):
            # train model
            input = data.requires_grad_()
            input = input.to(opt.device)
            target = label.to(opt.device)

            optimizer.zero_grad()
            loss = model(input, target, embed=False)
            # loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num += target.shape[0]
        running_loss /= num 
        
        # update learning rate
        scheduler.step(running_loss) 
        songMrr = val_MRR(model, val_dataloader, epoch)
        print(f"Epoch {epoch}, running loss: {running_loss}, val MRR: {songMrr}")

        if parallel is True:
            model.module.save(epoch, songMrr, opt.dir_save)
        else:
            model.save(epoch, songMrr, opt.dir_save)
        model.train()

   
 
@torch.no_grad()
def val_MRR(model, dataloader, epoch):
    model.eval()
    labels = []
    features = None
    
    for ii, (data, label) in enumerate(dataloader):
        input = data.to(opt.device)
        #print(input.shape)
        score, feature = model(input)
        feature = feature.data.cpu().numpy()
        labels.append([label[0][0], label[1][0]])
        if features is not None:
            features = np.concatenate((features, feature), axis=0)
        else:
            features = feature
   
    hums = []
    vocals  = {}
    songs = {}
    for i in range(len(labels)):
        track_id, version = labels[i]
    
        if version.split('-')[0] == 'hum':
            hums.append([track_id, features[i]])
        # if version.split('-')[1] == 'vocals':
        #     vocals[track_id] = features[i]
        else:
            songs[track_id] = features[i]

    # # calc song 
    top10s = []
    for hum in hums:
        track_id_hum, feat_hum = hum
        top10s.append(topTen(feat_hum, track_id_hum, songs))
    songMrr = mean_reciprocal_rank(top10s)
    # # calc vocal

    # top10v = []
    # for hum in hums:
    #     track_id_hum, feat_hum = hum
    #     top10v.append(topTen(feat_hum, track_id_hum, vocals))
    # vocalMrr = mean_reciprocal_rank(top10v)
    # print(f"Vocal MRR at epoch {epoch}: ", vocalMrr)

    model.train()
    return songMrr

def topTen(hum_feat, hum_label, songs):
    scores = []
    for track_id in songs.keys():
        score = dot(hum_feat, songs[track_id])/(norm(hum_feat)*norm(songs[track_id]))
        scores.append([track_id, score])
    scores.sort(reverse=True, key = lambda x: x[1])
    return list(map(lambda x: x[0] == hum_label , scores[0:10]))

def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])
    


if __name__=='__main__':
    import fire
    fire.Fire()
