# -*- coding: utf-8 -*-
# @Time    : 3/11/23 4:02 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : cav_mae.py

import os
os.environ['TORCH_HOME'] = './pretrained_models'
import random
import torch
import torch.nn as nn
from torch import Size, Tensor
import timm


from typing import Union, Optional, Callable, Tuple, List, Sequence
from .base_model import BaseModel
from . import networks
import copy


class focal_loss(nn.Module):
    def __init__(self, alpha=.8, gamma=2, num_classes = 2, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma
        
        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))
        
    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        alpha = self.alpha.to(labels.device)
        preds_logsoft = nn.functional.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# our main proposed model, for pretraining only, for finetuning, use CAVMAEFT class
class MAVModel(BaseModel):
    """ CAV-MAE Model
    """
    def __init__(self, opt, img_size=224, audio_length=128, patch_size=16, in_chans=3,
                 embed_dim=768, modality_specific_depth=11, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=True, tr_pos=False):  # # most models are trained with pixel normalization and non-trainabe positional embedding
        BaseModel.__init__(self, opt)
        print('A CAV-MAE Model')
        print('Use norm_pix_loss: ', norm_pix_loss)
        print('Learnable Positional Embedding: ', tr_pos)

        if self.isTrain:
            if opt.isFinetune:
                self.model_names = ['encoder', 'classifier']
            else:
                self.model_names = ['encoder', 'decoder', 'classifier']
        else:
                self.model_names = ['encoder', 'decoder', 'classifier']

        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        self.tubelet_size = 2
        self.mask_ratio = opt.mask_ratio
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_frames = opt.num_frames
        embed_dim_a = 1 * patch_size ** 2
        embed_dim_v = 3 * patch_size ** 2 * self.tubelet_size

        # self.encoder = networks.define_G(netG='Encoder', img_size=224, audio_length=(64,128), num_frames=16, patch_size=16, in_chans=3, tubelet_size=2, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        self.encoder = networks.define_G(netG='Encoder', img_size=224, audio_length=(64,128), num_frames=16, patch_size=16, in_chans=3, tubelet_size=2, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        self.decoder = networks.define_G(netG='Decoder', img_size=224, audio_length=(64,128), num_frames=16, patch_size=16, in_chans=3, tubelet_size=2, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        num_classes = 2
        self.classifier = networks.define_D(netD='Multimodal_Classifer', dim=embed_dim*2, num_classes=num_classes,gpu_ids=self.gpu_ids)
        # self.encoder = Encoder(img_size=224, num_frames=num_frames, tubelet_size=self.tubelet_size, audio_size=audio_length, patch_size=patch_size, in_chans=3,
        #          embed_dim=embed_dim, modality_specific_depth=11, num_heads=12,
        #          mlp_ratio=4., norm_layer=norm_layer, norm_pix_loss=False, tr_pos=False)

        # self.decoder = Decoder(img_size=img_size, num_frames=num_frames, tubelet_size=self.tubelet_size, patch_size=patch_size,
        #     embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
        #     norm_layer=norm_layer)

        if self.isTrain:
            self.mae_loss_weight = 1
            self.contrast_loss_weight = 1
            self.temperature = nn.Parameter(torch.tensor(1.0))
            if not opt.isFinetune:
                self.optimizer_G = torch.optim.Adam(list(self.encoder.parameters())
                                                        + list(self.decoder.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999),
                                                        )
                self.optimizers.append(self.optimizer_G)
            else:
                # finetune learning rate is about 1/10 of learning rate
                # weight = [4,1]
                # class_weights = torch.FloatTensor(weight).to(self.device)
                # self.loss_func = nn.CrossEntropyLoss(weight=class_weights)
                # self.loss_func = nn.CrossEntropyLoss()
                # self.loss_func = nn.BCEWithLogitsLoss(weight=class_weights)
                self.loss_func = focal_loss(0.8)
                self.optimizer_E = torch.optim.Adam(list(self.encoder.parameters()), lr=opt.finetune_lr, betas=(opt.beta1, 0.999), weight_decay=opt.finetune_lr_weight_decay)
                self.optimizer_C = torch.optim.Adam(list(self.classifier.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.finetune_lr_weight_decay)
                self.optimizers = [self.optimizer_E, self.optimizer_C]
                # self.optimizers = [self.optimizer_C]


    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding, opt the cls token, add by myself
        pos_embed_a = get_2d_sincos_pos_embed(self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))

        pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))

        decoder_pos_embed_a = get_2d_sincos_pos_embed(self.decoder_pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.decoder_pos_embed_a.data.copy_(torch.from_numpy(decoder_pos_embed_a).float().unsqueeze(0))

        decoder_pos_embed_v = get_2d_sincos_pos_embed(self.decoder_pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        self.decoder_pos_embed_v.data.copy_(torch.from_numpy(decoder_pos_embed_v).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.modality_a, std=.02)
        torch.nn.init.normal_(self.modality_v, std=.02)
        torch.nn.init.normal_(self.decoder_modality_a, std=.02)
        torch.nn.init.normal_(self.decoder_modality_v, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, modality):
        """
        video: (N, 3, T, H, W)
        x: (N, L, patch_size**2 *3 *tubelet_size)
        audio: (N, 1, T, F)
        x: (N, L, patch_size**2 *1)
        """
        c=0
        x=None
        if modality == 'a':
            assert len(imgs.shape) == 4, f"Audio input shape is {len(imgs.shape)}"
            c = 1
            p = self.patch_size
            assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

            h = imgs.shape[2] // p
            w = imgs.shape[3] // p
            x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
            x = torch.einsum('bchpwq->bhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        elif modality == 'v':
            assert len(imgs.shape) == 5, f"Video input shape is {len(imgs.shape)}"
            c = 3
            p = self.patch_size
            t = self.tubelet_size
            assert imgs.shape[3] == imgs.shape[4] and imgs.shape[3] % p == 0

            h = w = imgs.shape[3] // p
            n = imgs.shape[2] // t
            x = imgs.reshape(shape=(imgs.shape[0], c, n, t, h, p, w, p))
            x = torch.einsum('bcnthpwq->bnhwpqct', x)
            x = x.reshape(shape=(imgs.shape[0], n * h * w, p**2 * c * t))
        
        return x

    def unpatchify(self, x, c, h, w, p=16):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def random_masking_unstructured(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_structured(self, x, mask_ratio, t=64, f=8, mode='time'):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        assert L == f * t
        noise = noise.reshape(N, f, t) # the audio patch is in shape [f,t], not [t,f]
        if mode == 'time':
            for i in range(N):
                mask_t_list = random.sample(range(t), int(t * mask_ratio))
                for k in mask_t_list:
                    noise[i, :, k] = 1.1  # large value will be removed
        elif mode == 'freq':
            for i in range(N):
                mask_f_list = random.sample(range(f), int(f * mask_ratio))
                for k in mask_f_list:
                    noise[i, k, :] = 1.1  # large value will be removed
        elif mode == 'tf':
            for i in range(N):
                mask_t_list = random.sample(range(t), int(t * mask_ratio * 0.7))
                for k in mask_t_list:
                    noise[i, :, k] = 1.1  # large value will be removed
            for i in range(N):
                mask_f_list = random.sample(range(f), int(f * mask_ratio * 0.7))
                for k in mask_f_list:
                    noise[i, k, :] = 1.1  # large value will be removed
        noise = noise.reshape(N, L)

        # sort noise for each sample, only need to manuplate these two ids_shuffle, ids_restore
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def set_input(self, imgs, mask, audio, label=None):
        self.imgs, self.mask, self.audio, self.label = imgs, mask, audio, label
        self.mask = self.mask.bool()
        self.audio = self.audio[:,:,:-1,:]

    def optimize_parameters(self, audio, imgs, mask):
        loss = self.forward(audio, imgs, mask)
        self.optimizer_G.zero_grad()
        loss.backward()
        self.optimizer_G.step()
        return loss

    def forward_contrastive(self, audio_rep, video_rep, bidirect_contrast=False):
        # calculate nce loss for mean-visual representation and mean-audio representation
        audio_rep = torch.nn.functional.normalize(audio_rep, dim=-1)
        video_rep = torch.nn.functional.normalize(video_rep, dim=-1)

        total = torch.mm(audio_rep, torch.transpose(video_rep, 0, 1)) / 0.05

        # by default we use single directional
        if bidirect_contrast == False:
            nce = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
            # c_acc = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
            # return nce, c_acc
            return nce
        else:
            nce_1 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
            nce_2 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total.t(), dim=0)))
            # c_acc_1 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
            # c_acc_2 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total.t(), dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
            nce = (nce_1 + nce_2) / 2
            # c_acc = (c_acc_1 + c_acc_2) / 2
            return nce# c_acc

    def clip(self, audio_rep, video_rep):
        loss_func = nn.CrossEntropyLoss()
        audio_rep = torch.nn.functional.normalize(audio_rep, dim=-1)
        video_rep = torch.nn.functional.normalize(video_rep, dim=-1)

        total = torch.mm(audio_rep, torch.transpose(video_rep, 0, 1)) * self.temperature
        labels = torch.arange(0, audio_rep.shape[0]).to(total.device)
        loss_i = loss_func(total, labels)
        loss_t = loss_func(total.transpose(0,1), labels)
        loss = (loss_i + loss_t) / 2
        return loss

    def forward_mae_loss(self, input, pred, mask, modality):
        if modality == 'a':
            # for audio, need to adjust the shape
            # input = input.unsqueeze(1)
            input = input.transpose(2, 3)
            target = self.patchify(input, 'a')
        elif modality == 'v':
            target = self.patchify(input, 'v')

        # patch-wise normalization might minorly improve the classification performance, but will make the model lose inpainting function
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    # def forward(self, audio, imgs, mask, mask_ratio_a=0.75, mask_ratio_v=0.75, mae_loss_weight=1., contrast_loss_weight=0.01, mask_mode='unstructured'):
    def forward(self, audio, imgs, mask):
        f_a, f_v, mask_a, mask_v, mid_a, mid_v = self.encoder(audio, imgs, mask, mask_ratio_a=self.mask_ratio, mask_mode='unstructured')
        pred_a, pred_v = self.decoder(f_a, f_v, mask_a, mask_v)
        # cls_token_a = f_a[:, 0, :]
        # cls_token_v = f_v[:, 0, :]
        pred_a = pred_a[:, 1:, :]
        pred_v = pred_v[:, 1:, :]

        loss_mae_a = self.forward_mae_loss(audio, pred_a, mask_a, 'a')
        loss_mae_v = self.forward_mae_loss(imgs, pred_v, mask_v, 'v')
        loss_mae = self.mae_loss_weight * (loss_mae_a + loss_mae_v)

        # if contrastive loss is used
        if self.contrast_loss_weight != 0:
            # note this is single directional
            # loss_c, c_acc = self.forward_contrastive(cls_token_a.mean(dim=1), cls_token_v.mean(dim=1))
            loss_c = self.forward_contrastive(mid_a[:,0,:], mid_v[:,0,:], bidirect_contrast=True)
            # loss_c = self.clip(mid_a[:,0,:], mid_v[:,0,:])
            loss_c = self.contrast_loss_weight * loss_c
        else:
            loss_c, _ = torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device)

        loss = loss_mae + loss_c

        return loss
        # return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc

    def forward_cls(self, audio, imgs):
        f_a, f_v, _, _, mid_a, mid_v = self.encoder(audio, imgs, tube_mask=None, mask_ratio_a=0)
        cls_token_a = f_a[:, 0, :]
        cls_token_v = f_v[:, 0, :]

        logits = self.classifier(cls_token_a, cls_token_v)
        # return logits, mid_a, mid_v
        return logits, cls_token_a, cls_token_v

    def finetune(self, audio, imgs, label):
        logits, mid_a, mid_v = self.forward_cls(audio, imgs)

        # loss_c = self.forward_contrastive(mid_a[:,0,:], mid_v[:,0,:], bidirect_contrast=True)
        # loss_c = self.clip(mid_a[:,0,:], mid_v[:,0,:])
        # loss_c = 0.01 * loss_c

        loss_mae = self.loss_func(logits, label)
        # loss = loss_mae + loss_c
        loss = loss_mae
        self.optimizer_E.zero_grad()
        self.optimizer_C.zero_grad()
        loss.backward()
        self.optimizer_C.step()
        self.optimizer_E.step()
        return loss

def __():
    '''
    # the finetuned CAV-MAE model
    class CAVMAEFT(nn.Module):
        def __init__(self, label_dim, img_size=224, audio_length=1024, patch_size=16, in_chans=3,
                    embed_dim=768, modality_specific_depth=11, num_heads=12, mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, tr_pos=True):
            super().__init__()
            timm.models.vision_transformer.Block = Block
            print('Use norm_pix_loss: ', norm_pix_loss)

            timm.models.vision_transformer.PatchEmbed = PatchEmbed
            timm.models.vision_transformer.Block = Block

            self.patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim)
            self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

            self.patch_embed_a.num_patches = int(audio_length * 128 / 256)
            print('Number of Audio Patches: {:d}, Visual Patches: {:d}'.format(self.patch_embed_a.num_patches, self.patch_embed_v.num_patches))

            self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

            self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
            self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding

            self.blocks_a = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(modality_specific_depth)])
            self.blocks_v = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(modality_specific_depth)])
            self.blocks_u = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(12 - modality_specific_depth)])

            self.norm_a = norm_layer(embed_dim)
            self.norm_v = norm_layer(embed_dim)
            self.norm = norm_layer(embed_dim)

            self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, label_dim))

            self.initialize_weights()

            print('Audio Positional Embedding Shape:', self.pos_embed_a.shape)
            print('Visual Positional Embedding Shape:', self.pos_embed_v.shape)

        def get_patch_num(self, input_shape, stride):
            test_input = torch.zeros(1, 1, input_shape[0], input_shape[1])
            test_proj = torch.nn.Conv2d(1, 4, kernel_size=(16, 16), stride=(stride, stride))
            test_output = test_proj(test_input)
            print(test_output.shape)
            return test_output.shape[2], test_output[3], test_output[2] * test_output[2]

        def initialize_weights(self):
            pos_embed_a = get_2d_sincos_pos_embed(self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
            self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))

            pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
            self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))

            w = self.patch_embed_a.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            w = self.patch_embed_v.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            torch.nn.init.normal_(self.modality_a, std=.02)
            torch.nn.init.normal_(self.modality_v, std=.02)

            self.apply(self._init_weights)

        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                # we use xavier_uniform following official JAX ViT:
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        def forward(self, a, v, mode):
            # multi-modal fine-tuning, our default method for fine-tuning
            if mode == 'multimodal':
                a = a.unsqueeze(1)
                a = a.transpose(2, 3)
                a = self.patch_embed_a(a)
                a = a + self.pos_embed_a
                a = a + self.modality_a

                v = self.patch_embed_v(v)
                v = v + self.pos_embed_v
                v = v + self.modality_v

                for blk in self.blocks_a:
                    a = blk(a)

                for blk in self.blocks_v:
                    v = blk(v)

                x = torch.cat((a, v), dim=1)

                for blk in self.blocks_u:
                    x = blk(x)
                x = self.norm(x)

                x = x.mean(dim=1)
                x = self.mlp_head(x)
                return x

            # finetune with only audio (and inference with only audio when the model is finetuned with only audio)
            elif mode == 'audioonly':
                a = a.unsqueeze(1)
                a = a.transpose(2, 3)
                a = self.patch_embed_a(a)
                a = a + self.pos_embed_a
                a = a + self.modality_a

                for blk in self.blocks_a:
                    a = blk(a)

                # note here uses the 'a' normalization, it is used in both training and inference, so it is fine
                for blk in self.blocks_u:
                    a = blk(a, 'a')
                a = self.norm_a(a)
                x = a.mean(dim=1)
                x = self.mlp_head(x)
                return x

            # finetune with only image (and inference with only audio when the model is finetuned with only image)
            elif mode == 'videoonly':
                v = self.patch_embed_v(v)
                v = v + self.pos_embed_v
                v = v + self.modality_v

                for blk in self.blocks_v:
                    v = blk(v)

                # note here uses the 'v' normalization, it is used in both training and inference, so it is fine
                for blk in self.blocks_u:
                    v = blk(v, 'v')
                v = self.norm_v(v)
                x = v.mean(dim=1)
                x = self.mlp_head(x)
                return x

            # used in case that the model is finetuned with both modality, but in inference only audio is given
            elif mode == 'missingaudioonly':
                a = a.unsqueeze(1)
                a = a.transpose(2, 3)
                a = self.patch_embed_a(a)
                a = a + self.pos_embed_a
                a = a + self.modality_a

                for blk in self.blocks_a:
                    a = blk(a)

                # two forward passes to the block_u, one with modality-specific normalization, another with unified normalization
                u = a
                for blk in self.blocks_u:
                    u = blk(u) # note here use unified normalization
                u = self.norm(u)
                u = u.mean(dim=1)

                for blk in self.blocks_u:
                    a = blk(a, 'a') # note here use modality-specific normalization
                a = self.norm_a(a)
                a = a.mean(dim=1)

                # average the output of the two forward passes
                x = (u + a) / 2
                x = self.mlp_head(x)
                return x

            # used in case that the model is fine-tuned with both modality, but in inference only image is given
            elif mode == 'missingvideoonly':
                v = self.patch_embed_v(v)
                v = v + self.pos_embed_v
                v = v + self.modality_v

                for blk in self.blocks_v:
                    v = blk(v)

                # two forward passes to the block_u, one with modality-specific normalization, another with unified normalization
                u = v
                for blk in self.blocks_u:
                    u = blk(u) # note here use unified normalization
                u = self.norm(u)
                u = u.mean(dim=1)

                for blk in self.blocks_u:
                    v = blk(v, 'v') # note here use modality-specific normalization
                v = self.norm_v(v)
                v = v.mean(dim=1)

                # average the output of the two forward passes
                x = (u + v) / 2
                x = self.mlp_head(x)
                return x

        # for retrieval
        def forward_feat(self, a, v, mode='av'):

            # return both audio and visual
            if mode == 'av':
                a = a.unsqueeze(1)
                a = a.transpose(2, 3)
                a = self.patch_embed_a(a)
                a = a + self.pos_embed_a
                a = a + self.modality_a

                v = self.patch_embed_v(v)
                v = v + self.pos_embed_v
                v = v + self.modality_v

                for blk in self.blocks_a:
                    a = blk(a)

                for blk in self.blocks_v:
                    v = blk(v)

                for blk in self.blocks_u:
                    a = blk(a, 'a')
                a = self.norm_a(a)

                for blk in self.blocks_u:
                    v = blk(v, 'v')

                v = self.norm_v(v)
                return a, v

            # return only audio
            if mode == 'a':
                a = a.unsqueeze(1)
                a = a.transpose(2, 3)
                a = self.patch_embed_a(a)
                a = a + self.pos_embed_a
                a = a + self.modality_a

                for blk in self.blocks_a:
                    a = blk(a)

                for blk in self.blocks_u:
                    a = blk(a, 'a')

                a = self.norm_a(a)
                return a
    '''
    pass