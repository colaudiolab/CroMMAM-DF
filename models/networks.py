import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch
from torch import nn
import math
from einops import rearrange

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(netG, img_size=224, audio_length=128, num_frames=16, patch_size=16, in_chans=3, tubelet_size=2,
                 embed_dim=768, modality_specific_depth=11, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=True, tr_pos=False,
                 init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    if netG == 'Encoder':
        net = Encoder(img_size=img_size, num_frames=num_frames, tubelet_size=tubelet_size, audio_size=audio_length, patch_size=patch_size, in_chans=in_chans,
                 embed_dim=embed_dim, modality_specific_depth=11, num_heads=12,
                 mlp_ratio=4., norm_layer=norm_layer, norm_pix_loss=norm_pix_loss, tr_pos=False)
    elif netG == 'Decoder':
        net = Decoder(img_size=img_size, num_frames=num_frames, tubelet_size=tubelet_size, patch_size=patch_size,
            embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
            norm_layer=norm_layer)
    elif netG == 'transformer_video':
        net = CViT()
    elif netG == 'transformer_audio':
        net = CAiT()
    
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    # return net
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(netD, dim, num_classes=2, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    if netD == 'Multimodal_Classifer':
        net = Multimodal_Classifer(dim=dim, num_classes=num_classes)
    else:
        raise NotImplementedError('Discrimter model name [%s] is not recognized' % netD)
    
    return init_net(net, init_type, init_gain, gpu_ids)

class ResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

        return self.model(input)


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):

        out = x + self.conv_block(x)
        return out


class UnetGenerator(nn.Module):


    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):

        super(UnetGenerator, self).__init__()


        super(UnetGenerator, self).__init__()

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)

        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer


    def forward(self, input, mode=None):

        return self.model(input, mode=mode)


class UnetSkipConnectionBlock(nn.Module):


    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, model_split=0):

        self.innernc = inner_nc
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x, mode = None, audio_feat = None):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):


    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):

        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

# import torch.functional as F

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

class CViT(nn.Module):
    def __init__(self, image_size=224, patch_size=7, channels=512,
                 dim=1024, depth=6, heads=8, mlp_dim=2048):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        self.pattern_matrix = nn.Parameter(torch.randn(1, 32, 512))
        self.latent_matrix = nn.Parameter(torch.randn(1, 49, 32))
        self.pos_embedding = nn.Parameter(torch.randn(32, 1, dim))

        # self.to_embedding = nn.Linear(32*dim, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
        )

    def forward(self, img, mask=None):
        p = self.patch_size
        x = self.features(img)
        y = rearrange(x, 'b c (h p1) (w p2) -> b (c h w) (p1 p2)', p1=p, p2=p)
        y = torch.matmul(self.pattern_matrix, y)
        y = torch.matmul(y, self.latent_matrix)
        y = rearrange(y, '(b p1) h w -> b p1 (h w)', p1=1)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, y), 1)
        shape = x.shape[0]
        x += self.pos_embedding[0:shape]
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

class CAiT(nn.Module):
    def __init__(self, patch_size=7, dim=1024, depth=6, heads=8, mlp_dim=2048):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2)),
            nn.BatchNorm2d(96, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(2, 2)),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(256, 256, kernel_size=(3, 3)),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3, 3)),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3, 3)),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 2)),
        )

        self.patch_size = patch_size
        self.pattern_matrix = nn.Parameter(torch.randn(1, 32, 256))
        self.latent_matrix = nn.Parameter(torch.randn(1, 40, 32))
        self.pos_embedding = nn.Parameter(torch.randn(32, 1, dim))
        # self.to_embedding = nn.Linear(32 * dim, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
        )

    def forward(self, img, mask=None):
        p = self.patch_size
        x = self.features(img)
        y = rearrange(x, 'b c (h p1) (w p2) -> b (c h w) (p1 p2)', p1=8, p2=5)
        y = torch.matmul(self.pattern_matrix, y)
        y = torch.matmul(y, self.latent_matrix)
        y = rearrange(y, '(b p1) h w -> b p1 (h w)', p1=1)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, y), 1)
        shape = x.shape[0]
        x += self.pos_embedding[0:shape]
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


#------------------------------------------------------------------------------------------------------------------
# add by myself
import timm
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import Mlp
from .pos_embed import get_2d_sincos_pos_embed
from .MARLIN_module import PatchEmbedding3d, Attention, SinCosPositionalEmbedding

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_a = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_a = norm_layer(dim)
        self.norm2_v = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, modality=None):
        if modality == None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif modality == 'a':
            x = x + self.drop_path(self.attn(self.norm1_a(x)))
            x = x + self.drop_path(self.mlp(self.norm2_a(x)))
        elif modality == 'v':
            x = x + self.drop_path(self.attn(self.norm1_v(x)))
            x = x + self.drop_path(self.mlp(self.norm2_v(x)))
        return x

'''
Class CrossAttention and CrossAttentionBlock are borrowed from
https://github.com/IBM/CrossViT/blob/main/models/crossvit.py
'''
class CrossAttention(nn.Module):
    def __init__(self, in_dim, out_dim,in_q_dim,hid_q_dim):
        super(CrossAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_q_dim = in_q_dim #新增
        self.hid_q_dim = hid_q_dim #新增
        # 定义查询、键、值三个线性变换
        self.query = nn.Linear(in_q_dim, hid_q_dim, bias=False) #变化
        self.key = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)
        
    def forward(self, q, y):
        # 对输入进行维度变换，为了方便后面计算注意力分数
        batch_size = q.shape[0]   # batch size
        num_queries = q.shape[1]  # 查询矩阵中的元素个数
        num_keys = y.shape[1]     # 键值矩阵中的元素个数
        q = self.query(q)  # 查询矩阵
        y = self.key(y)    # 键值矩阵
        # 计算注意力分数
        attn_scores = q @ y.transpose(-2, -1) / (self.out_dim ** 0.5)  # 计算注意力分数，注意力分数矩阵的大小为 batch_size x num_queries x num_keys x num_keys
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)  # 对注意力分数进行 softmax 归一化
        # 计算加权和
        V = self.value(y)  # 通过值变换得到值矩阵 V
        output = torch.bmm(attn_weights, V)  # 计算加权和，output 的大小为 batch_size x num_queries x num_keys x out_dim
        return output


class Encoder(nn.Module):
    """
    CAV-MAE encoder
    """
    def __init__(self, img_size=224, num_frames=16, tubelet_size=2, audio_size=128, patch_size=16, in_chans=3,
                 embed_dim=768, modality_specific_depth=11, num_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, tr_pos=False):
        super().__init__()

        # the encoder part
        self.patch_embed_a = PatchEmbed(img_size = audio_size, patch_size = patch_size, in_chans=1, embed_dim = embed_dim)
        # self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed_v = PatchEmbedding3d(
            input_size=(3, num_frames, img_size, img_size),
            patch_size=(tubelet_size, patch_size, patch_size),
            embedding=embed_dim
            )

        # overide the timm package
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        audio_size = to_2tuple(audio_size)
        self.patch_embed_a.num_patches = int(audio_size[0] * audio_size[1] // patch_size // patch_size)
        self.patch_embed_v.num_patches = (img_size // patch_size) * (img_size // patch_size) * (num_frames // tubelet_size)
        # print('Number of Audio Patches: {:d}, Visual Patches: {:d}'.format(self.patch_embed_a.num_patches, self.patch_embed_v.num_patches))

        # self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        # self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        self.pos_embed_a = SinCosPositionalEmbedding((self.patch_embed_a.num_patches, embed_dim), dropout_rate=0.)
        self.pos_embed_v = SinCosPositionalEmbedding((self.patch_embed_v.num_patches, embed_dim), dropout_rate=0.)

        self.cls_token_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # audio-branch
        self.blocks_a = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        # visual-branch
        self.blocks_v = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        # cross attention fuse branch
        self.blocks_cross_a = CrossAttention(in_dim=embed_dim, out_dim=embed_dim, in_q_dim=embed_dim, hid_q_dim=embed_dim)
        self.blocks_cross_v = CrossAttention(in_dim=embed_dim, out_dim=embed_dim, in_q_dim=embed_dim, hid_q_dim=embed_dim)
        # self.blocks_cross_a = CrossAttentionBlock(dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, has_mlp=False)
        # self.blocks_cross_v = CrossAttentionBlock(dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, has_mlp=False)

        # independent normalization layer for audio, visual, and audio-visual
        self.norm_a, self.norm_v = norm_layer(embed_dim), norm_layer(embed_dim)
        
        self.norm_pix_loss = norm_pix_loss

        # self.initialize_weights()
        self.apply(self._init_weights)

        print('Audio Positional Embedding Shape:', self.pos_embed_a.input_shape)
        print('Visual Positional Embedding Shape:', self.pos_embed_v.input_shape)
 
    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding, opt the cls token, add by myself
        pos_embed_a = get_2d_sincos_pos_embed(self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=True)
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))

        pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=True)
        self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)

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
        mask = torch.gather(mask, dim=1, index=ids_restore).bool()

        return x_masked, mask, ids_restore

    def forward_features(self, a, v):
        for blk in self.blocks_a:
            ca = blk(a)
        ca = self.norm_a(ca)

        for blk in self.blocks_v:
            cv = blk(v)
        cv = self.norm_v(cv)

        return ca, cv

    def fusion(self, f_a, f_v):
        # cross attention
        fusion_feat = torch.cat((f_a, f_v), dim=1)
        return fusion_feat

    def forward(self, a, v, tube_mask, mask_ratio_a=0.75, mask_mode='unstructured'):
        # embed patches
        a = a.transpose(2, 3)
        a = self.patch_embed_a(a)
        a = self.pos_embed_a(a)

        v = self.patch_embed_v(v)
        v = self.pos_embed_v(v)
        
        if mask_ratio_a > 0:
            # by default, we always use unstructured masking
            if mask_mode == 'unstructured':
                a_unmasked, mask_a, ids_restore_a = self.random_masking_unstructured(a, mask_ratio_a)
            # in ablation study, we tried time/freq/tf masking. mode in ['freq', 'time', 'tf']
            else:
                a_unmasked, mask_a, ids_restore_a = self.random_masking_structured(a, mask_ratio_a, t=64, f=8, mode=mask_mode)
        else:
            a_unmasked = a
            mask_a = None
            tube_mask = None

        # Tube mask
        B, _, C = v.shape
        v_unmasked = v[~tube_mask].view(B, -1, C) if tube_mask != None else v

        # add cls token
        cls_token_a = self.cls_token_a.expand(B, -1, -1)
        cls_token_v = self.cls_token_v.expand(B, -1, -1)
        a_unmasked = torch.cat((cls_token_a, a_unmasked), dim=1)
        v_unmasked = torch.cat((cls_token_v, v_unmasked), dim=1)
        
        f_a, f_v = self.forward_features(a_unmasked, v_unmasked)

        # cross fusion
        fusion_feat = self.fusion(f_a, f_v)

        # cross attention
        # f_audio = f_a
        # f_video = f_v
        f_audio = self.blocks_cross_a(f_a, fusion_feat)
        f_video = self.blocks_cross_v(f_v, fusion_feat)
        
        # f_audio = f_audio + f_a
        # f_video = f_video + f_v
        # print(f_audio.shape, f_video.shape, fusion_feat.shape)
        # f_a.shape = (batch_size,17,768)
        # f_v.shape = (batch_size,393,768)
        return f_audio, f_video, mask_a, tube_mask, f_a, f_v

class Decoder(nn.Module):
    """
    CAV-MAE decoder
    """
    def __init__(self, img_size=224, num_frames=16, tubelet_size=2, audio_size=(64,128), patch_size=16, in_chans=3,
                 embed_dim=768, modality_specific_depth=11, num_heads=12, decoder_embed_dim=512, decoder_depth=8,
                 decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, tr_pos=False):
        super().__init__()
        # the decoder part
        # overide the timm package
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        audio_size = to_2tuple(audio_size)
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.n_patch_hw_v = img_size // patch_size
        self.num_patch_embed_a = int(audio_size[0] * audio_size[1] // patch_size // patch_size)
        self.num_patch_embed_v = int(self.n_patch_hw_v * self.n_patch_hw_v * (num_frames // tubelet_size))
        output_dim_a = 1 * patch_size ** 2
        output_dim_v = 3 * patch_size ** 2 * tubelet_size

        # Project to lower dimension for the decoder
        self.decoder_embed_a = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_v = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # token used for masking
        self.mask_token_a = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_v = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # self.decoder_modality_a = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # self.decoder_modality_v = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed_a = SinCosPositionalEmbedding((self.num_patch_embed_a, decoder_embed_dim), dropout_rate=0.)  # fixed sin-cos embedding
        self.decoder_pos_embed_v = SinCosPositionalEmbedding((self.num_patch_embed_v, decoder_embed_dim), dropout_rate=0.)  # fixed sin-cos embedding

        self.decoder_blocks_a = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(decoder_depth)])
        self.decoder_blocks_v = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(decoder_depth)])

        self.decoder_norm_a = norm_layer(decoder_embed_dim)
        self.decoder_norm_v = norm_layer(decoder_embed_dim)

        # project channel is different for two modality, use two projection head
        self.decoder_pred_a = nn.Linear(decoder_embed_dim, output_dim_a, bias=True)  # decoder to patch
        self.decoder_pred_v = nn.Linear(decoder_embed_dim, output_dim_v, bias=True)  # decoder to patch

        self.norm_pix_loss = norm_pix_loss

        # self.initialize_weights()
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

    def forward_features(self, a, v, return_token_num_a, return_token_num_v):
        for block in self.decoder_blocks_a:
            a = block(a)
        for block in self.decoder_blocks_v:
            v = block(v)

        if return_token_num_a > 0:
            a = a[:, -return_token_num_a:]
        if return_token_num_v > 0:
            v = v[:, -return_token_num_v:]

        a = self.decoder_norm_a(a)
        v = self.decoder_norm_v(v)
        pred_a = self.decoder_pred_a(a)
        pred_v = self.decoder_pred_v(v)
        # x: (B, N_mask, C)
        return pred_a, pred_v

    def forward(self, audio, video, mask_a, mask_v, mask_ratio_a=0.75, mask_mode='unstructured'):
        a = self.decoder_embed_a(audio)
        v = self.decoder_embed_v(video)

        # refer to VideoMAE modeling_pretrain.py
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        b, n, c = a.shape
        expand_pos_embed_a = self.decoder_pos_embed_a.emb.data.expand(b, -1, -1)
        pos_emb_vis_a = expand_pos_embed_a[~mask_a].view(b, -1, c)
        pos_emb_mask_a = expand_pos_embed_a[mask_a].view(b, -1, c)
        a = torch.cat([a[:,:1, :], a[:,1:,:] + pos_emb_vis_a, self.mask_token_a + pos_emb_mask_a], dim=1)   # cls_token, vis, mask

        b, n, c = v.shape
        expand_pos_embed_v = self.decoder_pos_embed_v.emb.data.expand(b, -1, -1)
        pos_emb_vis_v = expand_pos_embed_v[~mask_v].view(b, -1, c)
        pos_emb_mask_v = expand_pos_embed_v[mask_v].view(b, -1, c)
        v = torch.cat([v[:,:1,:], v[:,1:,:] + pos_emb_vis_v, self.mask_token_v + pos_emb_mask_v], dim=1)# cls_token, vis, mask

        mask_num_a = 0
        mask_num_v = 0
        # mask_num_a = pos_emb_mask_a.shape[1]
        # mask_num_v = pos_emb_mask_v.shape[1]
        pred_a, pred_v = self.forward_features(a, v, return_token_num_a = mask_num_a, return_token_num_v = mask_num_v)

        return pred_a, pred_v

class Multimodal_Classifer(nn.Module):
    def __init__(self, dim, num_classes=2):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, cls_token_a, cls_token_v):
        logits_a = nn.functional.softmax(cls_token_a, dim=-1)
        logits_v = nn.functional.softmax(cls_token_v, dim=-1)
        cls_token_a = logits_a * cls_token_a
        cls_token_v = logits_v * cls_token_v
        cls_toekn = torch.cat((cls_token_a, cls_token_v), dim=1)

        logits = nn.functional.softmax(self.mlp_head(cls_toekn))
        return logits


#------------------------------------------------------------------------------------------------------------------