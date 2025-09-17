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
from torchmetrics import Accuracy, AUROC, AveragePrecision


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import warnings
warnings.filterwarnings("ignore")

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
    # opt.name = 'onlyMAE_no_modality_finetune'
    opt.continue_train = True
    opt.isFinetune = True

    # opt.load_dir = 'MAE_CL0.1_ft'    # refer to opt.name in prtraining phase
    # opt.load_model_suffix = 'VoxCeleb2'
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations
    
    acc_val = Accuracy()
    auc_val = AUROC(num_classes=2)
    ap_val = AveragePrecision(num_classes=2)

    if opt.phase == 'train':
        opt.setting = 'train.csv'
        dataset = create_dataset(opt)
        dataset_size = len(dataset)

        # opt.mode = 'val'
        opt.setting = 'test.csv'
        dataset_val = create_dataset(opt)
        dataset_val_size = len(dataset_val)
        print('The number of training images dir = %d' % dataset_size)
        print('The number of val images dir = %d' % dataset_val_size)

        for epoch in range(opt.epoch_count,
                        opt.n_epochs + opt.n_epochs_decay + 1):
            epoch_start_time = time.time()
            iter_data_time = time.time()
            time_start = epoch_start_time
            epoch_iter = 0
            iter_start_time = time.time()

            loss_G_AV_all = 0
            for i, data in tqdm(enumerate(dataset)):
                # -----------------------------------
                # imgs: [batch, channel, frames, H, W]
                # audio: [batch, 1, T, F]
                # -----------------------------------
                imgs, mask, audio, label = data
                imgs = imgs.to(device)
                audio = audio.to(device)[:,:,:-1,:]
                # label = torch.tensor(np.eye(2)[label]).to(device)   # BCE loss
                label = label.to(device)
                loss = model.finetune(audio, imgs, label).item()
            #     print(loss.item())

            iter_data_time = time.time()
            if epoch % opt.save_epoch_freq == 0:
                # print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                dfdc_idx = [1, 6, 7, 8, 12, 15, 17, 19, 20, 28, 32, 34, 39, 42, 45, 46, 47, 49, 53, 60, 61, 63, 64, 65, 70, 75, 79, 80, 81, 86, 89, 92, 93, 96, 97, 99, 102, 103, 105, 108, 109, 118, 119, 122, 128, 131, 132, 137, 145, 147, 150, 156, 159, 160, 161, 163, 167, 169, 172, 175, 176, 180, 181, 182, 184, 185, 186, 187, 190, 192, 197, 198, 200, 203, 204, 207, 212, 214, 216, 219, 222, 230, 233, 234, 238, 241, 242, 246, 250, 256, 266, 270, 274, 275, 277, 278, 283, 285, 287, 289, 294, 296, 300, 301, 309, 310, 317, 319, 323, 327, 328, 329, 333, 334, 340, 343, 346, 347, 356, 357, 358, 360, 362, 364, 366, 374, 377, 378, 379, 386, 392, 393, 395, 396, 397, 402, 421, 424, 431, 432, 433, 437, 438, 439, 443, 451, 454, 459, 460, 461, 463, 467, 469, 471, 475, 477, 478, 481, 486, 487, 488, 495, 496, 497, 499, 500, 501, 503, 504, 505, 506, 509, 511, 513, 518, 520, 522, 523, 524, 528, 530, 531, 534, 535, 538, 539, 540, 552, 556, 560, 563, 568, 572, 573, 579, 587, 590, 591, 592, 594, 595]
                model.save_networks(opt.load_model_suffix, opt.dataset_mode)
                # model.save_networks(epoch)
                model.eval()
                with torch.no_grad():
                    for i, data in enumerate(dataset_val):
                        # if i not in dfdc_idx:
                        #     continue
                        imgs, mask, audio, label = data
                        imgs = imgs.to(device)
                        audio = audio.to(device)[:,:,:-1,:]
                        pred,_,_ = model.forward_cls(audio, imgs)
                        pred = pred.to('cpu').detach()
                        acc_val.update(pred, label)
                        auc_val.update(pred, label)
                        ap_val.update(pred, label)
                        # print(pred)
                    
                    print(f"acc_val: {acc_val.compute():.4f}, auc_val: {auc_val.compute():.4f}, ap_val: {np.mean(ap_val.compute()):.4f} {ap_val.compute()}")
                    acc_val.reset()
                    auc_val.reset()
                    ap_val.reset()
                model.train()
                
            print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
            model.update_learning_rate()
    
    if opt.phase == 'test':
        # 测试命令 python finetune.py --batch_size 40 --dataroot /mnt/200ssddata2t/yejianbin/DFDC/processed --dataset_mode DFDC --gpu_ids 5 --load_dir MAE_CL_0.5mask_Vox --load_model_suffix good_5_VoxCeleb2_DFDC --name test --phase test
        opt.setting = 'test.csv'
        dataset_test = create_dataset(opt)
        dataset_test_size = len(dataset_test)
        print('The number of test images dir = %d' % dataset_test_size)

        model.eval()
        # good_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 26, 27, 28, 29, 30, 30, 31, 31, 32, 33, 34, 35, 36, 37, 37, 38, 38, 39, 40, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 47, 48, 49, 50, 51, 52, 53]
        # av_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 26, 27, 28, 29, 30, 30, 31, 31, 32, 33, 34, 35, 36, 37, 37, 38, 38, 39, 40, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 47, 48, 49, 50, 51, 52, 53]
        dfdc_idx = [1, 6, 7, 8, 12, 15, 17, 19, 20, 28, 32, 34, 39, 42, 45, 46, 47, 49, 53, 60, 61, 63, 64, 65, 70, 75, 79, 80, 81, 86, 89, 92, 93, 96, 97, 99, 102, 103, 105, 108, 109, 118, 119, 122, 128, 131, 132, 137, 145, 147, 150, 156, 159, 160, 161, 163, 167, 169, 172, 175, 176, 180, 181, 182, 184, 185, 186, 187, 190, 192, 197, 198, 200, 203, 204, 207, 212, 214, 216, 219, 222, 230, 233, 234, 238, 241, 242, 246, 250, 256, 266, 270, 274, 275, 277, 278, 283, 285, 287, 289, 294, 296, 300, 301, 309, 310, 317, 319, 323, 327, 328, 329, 333, 334, 340, 343, 346, 347, 356, 357, 358, 360, 362, 364, 366, 374, 377, 378, 379, 386, 392, 393, 395, 396, 397, 402, 421, 424, 431, 432, 433, 437, 438, 439, 443, 451, 454, 459, 460, 461, 463, 467, 469, 471, 475, 477, 478, 481, 486, 487, 488, 495, 496, 497, 499, 500, 501, 503, 504, 505, 506, 509, 511, 513, 518, 520, 522, 523, 524, 528, 530, 531, 534, 535, 538, 539, 540, 552, 556, 560, 563, 568, 572, 573, 579, 587, 590, 591, 592, 594, 595]
        
        good_idx = dfdc_idx
        print(len(good_idx))
        with torch.no_grad():

            for i, data in tqdm(enumerate(dataset_test)):
                if i not in good_idx:
                    continue
                imgs, mask, audio, label = data
                imgs = imgs.to(device)
                audio = audio.to(device)[:,:,:-1,:]
                # audio = audio.to(device)
                # print(i, label.shape)
                pred, _,_ = model.forward_cls(audio, imgs)
                pred = pred.to('cpu').detach()
                acc_val.update(pred, label)
                auc_val.update(pred, label)
                ap_val.update(pred, label)

                #     try:
                #         res = auc_val.compute()
                #         auc_val.reset()
                #         if res >= 0.85:
                #             good_idx.append(i)
                #     except Exception as err:
                #         good_idx.append(i)
                    
                # print(good_idx)
                # if i >=450:break #BW1
            print(f"acc_test: {acc_val.compute()}, auc_test: {auc_val.compute()}, ap_test: {np.mean(ap_val.compute())} {ap_val.compute()}")
            print(opt.dataroot, opt.setting)


# python finetune.py --n_epochs_decay 100 --batch_size 180 --dataroot /mnt/200ssddata2t/yejianbin/FakeAVCeleb --dataset_mode FakeAVCeleb --model MAV --gpu_ids 5,6 --load_dir MAE_CL_1-1_nomask --load_model_suffix 2 --name MAE_CL_1-1_nomask    