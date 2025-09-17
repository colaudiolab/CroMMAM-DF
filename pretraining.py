"""
Anonymous release of VFD
Part of the framework is borrowed from
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
Many thanks to these authors!
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import os
import math
import cv2
from PIL import Image
from util import util
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import torch
import random

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import warnings
# warnings.filterwarnings("ignore")
import gc

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def auc(real, fake):
    label_all = []
    target_all = []
    for ind in real:
        target_all.append(1)
        label_all.append(ind)

    for ind in fake:
        target_all.append(0)
        label_all.append(ind)

    from sklearn.metrics import roc_auc_score
    return roc_auc_score(target_all, label_all)
    
if __name__ == '__main__':
    opt = TrainOptions().parse()
    # opt.name = 'onlyMAE_no_modality'   # checkpoint dir
    opt.setting = 'train.csv'
    dataset = create_dataset(opt)
    dataset_size = len(dataset)

    # opt.mode = 'val'
    # opt.setting = 'val.csv'
    # opt.serial_batches = False
    # dataset_val = create_dataset(opt)
    # dataset_val_size = len(dataset_val)
    print('The number of training images dir = %d' % dataset_size)
    # print('The number of val images dir = %d' % dataset_val_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations
    loss_G_AV_all = 0

    loss_epo = 0
    show_loss_iters=2000
    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        time_start = epoch_start_time
        epoch_iter = 0
        iter_start_time = time.time()

        
        # test(dataset, model, device)
        # break
        for i, data in tqdm(enumerate(dataset)):
            # -----------------------------------
            # imgs: [batch, channel, frames, H, W]
            # audio: [batch, 1, T, F]
            # -----------------------------------
            if opt.dataset_mode == 'DFDC':
                imgs, mask, audio, _ = data
                show_loss_iters = 200
            else:
                imgs, mask, audio = data
            imgs = imgs.to(device)
            mask = mask.to(device).bool()
            audio = audio.to(device)[:,:,:-1,:]
            # audio = audio.to(device)
            loss = model.optimize_parameters(audio, imgs, mask)
            loss_G_AV_all += loss.to('cpu').detach().item()
            # loss_epo += loss_G_AV
            total_iters += 1

            if total_iters % show_loss_iters == 0:
                print('epoch %d, total_iters %d: loss_G_AV: %.3f(%.3f), time cost: %.2f s' %
                      (epoch, total_iters, loss, loss_G_AV_all / show_loss_iters, time.time() - time_start))
                loss_G_AV_all = 0
                time_start = time.time()
                # gc.collect()


        iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            # model.save_networks(opt.dataset_mode)
            model.save_networks(epoch, opt.dataset_mode)
        #     model.eval()
        #     with torch.no_grad():
        #         loss = []
        #         for i, data in tqdm(enumerate(dataset_val)):
        #             imgs, mask, audio = data
        #             imgs = imgs.to(device)
        #             mask = mask.to(device).bool()
        #             audio = audio.to(device)[:,:,:-1,:]
        #             loss.append(model.forward(audio, imgs, mask).to('cpu').detach().item())

        #     model.train()
        # print(f'Val loss: {np.mean(loss)}')
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

        if epoch >=50 : break

# python pretraining.py --dataroot /mnt/200ssddata2t/yejianbin/Voxceleb2 --dataset_mode VoxCeleb2 --model MAV --batch_size 40 --gpu_ids 5,6 --name MAE_CL_1-1_nomask