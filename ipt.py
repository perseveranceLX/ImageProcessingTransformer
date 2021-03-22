""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch.functional import Tensor
import torch.nn as nn
from functools import partial

from torch.nn.modules.activation import Tanhshrink

from timm.models.layers import trunc_normal_

'''
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}
'''

class Ffn(nn.Module):
    # feed forward network layer after attention
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, task_embed=None, level=0):
        N, L, D = x.shape
        qkv = self.qkv(x).reshape(N, L, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # for decoder's task_embedding of different levels of attention layers
        if task_embed != None:
            _N, _H, _L, _D = q.shape
            task_embed = task_embed.reshape(1, _H, _L, _D)
            if level == 1:
                q += task_embed
                k += task_embed
            if level == 2:
                q += task_embed

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(N, L, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, dim, num_heads, ffn_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.ffn = Ffn(in_features=dim, hidden_features=ffn_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class DecoderLayer(nn.Module):
    
    def __init__(self, dim, num_heads, ffn_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn1 = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.attn2 = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm3 = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.ffn = Ffn(in_features=dim, hidden_features=ffn_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, task_embed):
        x = x + self.attn1(self.norm1(x), task_embed=task_embed, level=1)
        x = x + self.attn2(self.norm2(x), task_embed=task_embed, level=2)
        x = x + self.ffn(self.norm3(x))
        return x


class ResBlock(nn.Module):

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=5, stride=1,
                     padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=1,
                     padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class Head(nn.Module):
    """ Head consisting of convolution layers
    Extract features from corrupted images, mapping N3HW images into NCHW feature map.
    """
    def __init__(self, in_channels, out_channels, task_id):
        super(Head, self).__init__()
        assert 0 <= task_id <= 5 
        self.conv1= nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if task_id in [0, 1, 5] else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.resblock = ResBlock(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.resblock(out)

        return out

class PatchEmbed(nn.Module):
    """ Feature to Patch Embedding
    input : N C H W
    output: N num_patch P^2*C
    """
    def __init__(self, patch_size=1, in_channels=64):
        super().__init__()
        self.patch_size = patch_size
        self.dim = self.patch_size ** 2 * in_channels

    def forward(self, x):
        N, C, H, W = ori_shape = x.shape
        
        p = self.patch_size
        num_patches = (H // p) * (W // p)
        out = torch.zeros((N, num_patches, self.dim)).to(x.device)
        #print(f"feature map size: {ori_shape}, embedding size: {out.shape}")
        i, j = 0, 0
        for k in range(num_patches):
            if i + p > W:
                i = 0
                j += p
            out[:, k, :] = x[:, :, i:i+p, j:j+p].flatten(1)
            i += p
        return out, ori_shape

class DePatchEmbed(nn.Module):
    """ Patch Embedding to Feature
    input : N num_patch P^2*C
    output: N C H W
    """
    def __init__(self, patch_size=1, in_channels=64):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = None
        self.dim = self.patch_size ** 2 * in_channels

    def forward(self, x, ori_shape):
        N, num_patches, dim = x.shape
        _, C, H, W = ori_shape
        p = self.patch_size
        out = torch.zeros(ori_shape).to(x.device)
        i, j = 0, 0
        for k in range(num_patches):
            if i + p > W:
                i = 0
                j += p
            out[:, :, i:i+p, j:j+p] = x[:, k, :].reshape(N, C, p, p)
            #out[:, k, :] = x[:, :, i:i+p, j:j+p].flatten(1)
            i += p
        return out


class Tail(nn.Module):
    """ Head consisting of convolution layers
    Extract features from corrupted images, mapping N3HW images into NCHW feature map.
    """
    def __init__(self, task_id, in_channels, out_channels):
        super(Tail, self).__init__()
        assert 0 <= task_id <= 5
        self.task_id = task_id
        # 0, 1 for noise 30, 50; 2, 3, 4 for sr x2, x3, x4, 5 for defog
        if task_id in [0, 1, 5]:
            self.output = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                     padding=1, bias=False)
        elif task_id in [2, 3]:
            self.output = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * task_id * task_id, kernel_size=1, stride=1, padding=0, bias=False),
                nn.PixelShuffle(task_id),
            )
        elif task_id == 4:
            self.output = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * task_id * task_id, kernel_size=1, stride=1, padding=0, bias=False),
                nn.PixelShuffle(2),
                nn.PixelShuffle(2),
            )
        else:
            raise TypeError("Undefined task id")
        
        self.bn1 = nn.BatchNorm2d(out_channels) if task_id in [0, 1, 5] else nn.Identity()
        
    def forward(self, x):
        out = self.output(x)
        #print("task_id:", self.task_id)
        #print("shape of tail's output:", x.shape)
        out = self.bn1(out)
        return out

class ImageProcessingTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=1, in_channels=3, mid_channels=64, num_classes=1000, depth=12,
                 num_heads=8, ffn_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm):
        super(ImageProcessingTransformer, self).__init__()

        self.task_id = None
        self.num_classes = num_classes
        self.embed_dim = patch_size * patch_size * mid_channels
        self.headsets = nn.ModuleList([Head(in_channels, mid_channels, task_id) for task_id in range(6)])
        self.patch_embedding = PatchEmbed(patch_size=patch_size, in_channels=mid_channels)
        self.embed_dim = self.patch_embedding.dim
        if self.embed_dim % num_heads != 0:
            raise RuntimeError("Embedding dim must be devided by numbers of heads")

        self.pos_embed = nn.Parameter(torch.zeros(1, (48 // patch_size) ** 2, self.embed_dim))
        self.task_embed = nn.Parameter(torch.zeros(6, 1, (48 // patch_size) ** 2, self.embed_dim))
        self.encoder = nn.ModuleList([
            EncoderLayer(
                dim=self.embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for _ in range(depth)])
        self.decoder = nn.ModuleList([
            DecoderLayer(
                dim=self.embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for _ in range(depth)])
        #self.norm = norm_layer(self.embed_dim)

        self.de_patch_embedding = DePatchEmbed(patch_size=patch_size, in_channels=mid_channels)
        # tail
        self.tailsets = nn.ModuleList([Tail(id, mid_channels, in_channels) for id in range(6)])

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
    
    def set_task(self, task_id):
        self.task_id = task_id

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        assert 0 <= self.task_id <= 5
        #print("input shape:", x.shape, x.device)
        x = self.headsets[self.task_id](x)
        x, ori_shape = self.patch_embedding(x)
        #print("embedding shape:", x.shape)
        # print(x.device, self.pos_embed.device)
        x = x + self.pos_embed[:, :x.shape[1]]


        for blk in self.encoder:
            x = blk(x)
        for blk in self.decoder:
            x = blk(x, self.task_embed[self.task_id, :, :x.shape[1]])
        x = self.de_patch_embedding(x, ori_shape)
        x = self.tailsets[self.task_id](x)
        #x = self.norm(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

def ipt_base(**kwargs):
    model = ImageProcessingTransformer(
        patch_size=4, depth=12, num_heads=8, ffn_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model
