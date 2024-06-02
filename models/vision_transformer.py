import torch
import torch.nn as nn

import math
from functools import partial
from .utils import trunc_normal_
import numpy as np


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
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
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # print(x.shape)
        # print(self.proj)
        # exit()
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer_Residual(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], residual_interval=3, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.residual_interval = residual_interval
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x, ada_token=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        if ada_token is not None:
            ada_tokens = ada_token.expand(B, -1, -1) # B, p, d
            x = torch.cat((x, ada_tokens), dim=1)

        return self.pos_drop(x)

    def forward(self, x, ada_token=None, use_patches=False):
        x = self.prepare_tokens(x, ada_token)
        for i, blk in enumerate(self.blocks):
            if i % self.residual_interval == 0:  # add a residual connection 
                res = x
            x = blk(x)
            if i % self.residual_interval == 2 and i != 5:  # add the residual connection to the output 
                x = x + res
        x = self.norm(x)

        if use_patches:
            return x[:, 1:]
        else:
            return x[:, 0]


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x, ada_token=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        if ada_token is not None:
            ada_tokens = ada_token.expand(B, -1, -1) # B, p, d
            x = torch.cat((x, ada_tokens), dim=1)

        return self.pos_drop(x)

    def forward(self, x, ada_token=None, use_patches=False):
        x = self.prepare_tokens(x, ada_token)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if use_patches:
            return x[:, 1:]
        else:
            return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

    def get_layers_num_0(self):  
        return self.blocks[:2] # First two layers
    
    def get_layers_num_2(self):  
        return self.blocks[-4:] # Last Four layers
    
    def get_layers_num_2_front6blocks(self):  
        return self.blocks[:6] # First six layers
    
    def get_layers_num_2_last6blocks(self):  
        return self.blocks[-6:] # Last six layers

    def extract_feature(self, x, number_block_per_part, ada_token=None):
        x = self.prepare_tokens(x, ada_token)
        feat = []
        cls_token = []
        patch_and_cls_token = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i+1)%number_block_per_part == 0:
                feat.append(x[:, 1:]) # remove cls token
                cls_token.append(x[:, 0])
                patch_and_cls_token.append(x)
        x = self.norm(x)

        return feat, cls_token, patch_and_cls_token
    
    def get_all_selfattention(self, x, number_block_per_part):
        x = self.prepare_tokens(x)
        attn = []
        for i, blk in enumerate(self.blocks):
            if (i+1)%number_block_per_part == 0:
                attn.append(blk(x, return_attention=True) )
            x = blk(x)
            
        # return attention of the all blocks
        return attn
    
    def get_all_selfattention_with_name(self, x):
        x = self.prepare_tokens(x)
        dict_attn = {}
        for name, blk in self.blocks._modules.items():
        #for i, blk in enumerate(self.blocks):
        #    if (i+1)%number_block_per_part == 0:
        #attn.append(blk(x, return_attention=True) )
            dict_attn[name] = blk(x, return_attention=True)
            x = blk(x)
            
        # return attention of the all blocks
        return dict_attn

def PruneProj(proj, width_ratio): 
    proj_out = proj.weight.data.size(0) #prune
    proj_in = int(proj.weight.data.size(1) // width_ratio) #不prune
    prune_proj = nn.Linear(proj_in, proj_out)
    prune_proj.weight.data = proj.weight.data[:proj_out, :proj_in]
    new_bias = proj.bias.data if proj.bias is not None else None
    if new_bias is not None:
        prune_proj.bias.data = new_bias
    return prune_proj

def transform_attn(hdp_layers_num_heads, num_heads_descendant, hdp_proj_bias):
    hdp_proj_layers = []
    hdp_proj_txt_comps = []

    for i, v in enumerate(hdp_layers_num_heads):
        _layer = nn.Linear(
            hdp_layers_num_heads[i],
            hdp_layers_num_heads[i + 1]
                if i < len(hdp_layers_num_heads) - 1
                else num_heads_descendant,
            bias=hdp_proj_bias,
        )
        hdp_proj_layers.append(_layer)
        hdp_proj_txt_comps.append(f"{hdp_layers_num_heads[i]}{'B' if hdp_proj_bias else ''}")
        if i < len(hdp_layers_num_heads) - 1:
            hdp_proj_layers.append(nn.ReLU())
            hdp_proj_txt_comps.append('relu')
    hdp_proj = nn.Sequential(*hdp_proj_layers)
    hdp_proj_txt = '>'.join(hdp_proj_txt_comps)    
    return hdp_proj, hdp_proj_txt

class Attention_Learngene(nn.Module):
    def __init__(self, inherited_attn, dim, distribution, num_heads, num_heads_learngene, num_heads_descendant, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 hdp=False,
                 hdp_ratios=[],
                 hdp_non_linear=True,):
        super().__init__()
        self.num_heads = num_heads
        self.num_heads_learngene = num_heads_learngene
        self.num_heads_descendant = num_heads_descendant
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = PruneProj(inherited_attn.proj, 1.0*num_heads/num_heads_descendant)
        #self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        assert hdp in ['q', 'k', 'qk', 'qkv'], f'hdp={hdp} | must be one of [q, k, qk, qkv]'
        self.hdp = hdp
        assert isinstance(hdp_ratios, list), 'hdp_ratios must be a list'
        assert len(hdp_ratios) >= 1, 'hdp_ratios must have at least 1 element'
        if not hdp_non_linear:
            assert len(hdp_ratios) == 1, 'if not using `hdp_non_linear`, only 1 layer of hdp projection allowed'
        self.hdp_ratios = hdp_ratios
        self.hdp_layers_num_heads = [int(np.ceil(num_heads_descendant / v)) for v in self.hdp_ratios]
        assert all([isinstance(v, int) and v >= 1 for v in self.hdp_layers_num_heads])
        # print(num_heads_learngene)
        # print(num_heads_descendant)
        # exit()

        # self.hdp_num_heads = self.hdp_layers_num_heads[0]
        self.hdp_proj_bias = bool(hdp_non_linear) # assert self.hdp_num_heads >= 1, '(self-explanatory)'

        inherited_qkv_weight_reshaped = inherited_attn.qkv.weight.view(3, num_heads, head_dim, dim).permute(1, 0, 2, 3)
        # print(inherited_qkv_weight_reshaped.shape)

        if distribution == 'Gaussian-Layer':
            p = torch.randn(num_heads_learngene, num_heads)
            p = torch.softmax(p, dim=1).cuda()
        else:
            print("Uniform")
            p = torch.full((num_heads_learngene, num_heads), 1/num_heads).cuda()
        # Compress the original heads into learngene using Gaussian or Uniform sampling distribution
        transformed_weights = torch.einsum('ij,jklm->iklm', p, inherited_qkv_weight_reshaped)


        transformed_weights = inherited_qkv_weight_reshaped[:num_heads_learngene] 

        transformed_weights = transformed_weights.permute(1, 0, 2, 3)
        # print(transformed_weights.shape)
        self.qkv_learngene = nn.Parameter(transformed_weights.reshape(3 * num_heads_learngene * head_dim, dim))
        # print(HDP_RATIOS)
        # print(num_heads_learngene)
        # print(num_heads)
        # print(self.qkv_learngene.shape)
        self.qk_proj, qk_proj_txt = transform_attn(self.hdp_layers_num_heads, self.num_heads_descendant, self.hdp_proj_bias)
        self.v_proj, v_proj_txt = transform_attn(self.hdp_layers_num_heads, self.num_heads_descendant, self.hdp_proj_bias)
        # print(self.qk_proj)
        # exit()

    def forward(self, x):
        B, N, C = x.shape
        # [B N C] -> [B N Ct] Ct = 3 * num_heads_learngene * head_dim
        qkv = torch.matmul(x, self.qkv_learngene.T) 
        # [B N C] -> [B N Ht D]
        # [B N 3 Ht D] -> [3 B Ht N D]  
        qkv = qkv.reshape(B, N, 3, self.num_heads_learngene, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # [B_ Ht N D] -> 3[B_ Hx N D]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # HDP re-projection, Haxis=-3
        # qk和v分别经过各自的proj扩展
        if self.hdp == 'qkv':
            attn = attn.transpose(-3, -1)
            attn = self.qk_proj(attn)
            attn = attn.transpose(-3, -1)
            # print("attn.shape")
            # print(attn.shape)
            v = v.transpose(-3, -1)
            v = self.v_proj(v)
            # v = self.qk_proj(v)
            v = v.transpose(-3, -1)
            # print("v.shape")
            # print(v.shape)
            # exit()

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block_learngene(nn.Module):
    def __init__(self, i_layer, inherited_blocks, inherited_mlp, mlp_inherited_method, distribution, num_heads, num_heads_learngene, num_heads_descendant, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 hdp=False,
                 hdp_ratios=[],
                 hdp_non_linear=True,
                 ):
        super().__init__()
        self.norm1 = inherited_blocks.norm1
        if hdp: 
            #if i_layer >= 6:
            print("Attention_Learngene")
            self.attn = Attention_Learngene(
                    inherited_blocks.attn, dim, distribution=distribution,
                    num_heads=num_heads, num_heads_learngene=num_heads_learngene, num_heads_descendant=num_heads_descendant,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                    hdp=hdp,
                    hdp_ratios=hdp_ratios,
                    hdp_non_linear=hdp_non_linear,
                    )
        else:
            print("direct_attn")
            self.attn = inherited_blocks.attn
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        self.norm2 = inherited_blocks.norm2
        mlp_hidden_dim = int(dim * mlp_ratio)
        if mlp_inherited_method == 'expand':
            # if i_layer >= 6:
            print("expand_ffn")
            self.mlp = inherited_mlp    
        else:
            print("direct_ffn")
            self.mlp = inherited_blocks.mlp
            
    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransMlp(nn.Module):
    def __init__(self, descendant_fc1_weights, descendant_fc1_biases, descendant_fc2_weights, descendant_fc2_biases, 
                in_features, hidden_features=None, act_layer=nn.GELU, drop=0., out_features=None,
                hdp=False,
                hdp_ratios=[],
                hdp_non_linear=True,
                ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # print(descendant_fc1_biases)
        # self.fc1.weight = nn.Parameter(descendant_fc1_weights.T)
        # self.fc1.bias = nn.Parameter(descendant_fc1_biases)
        self.fc1.weight.data = descendant_fc1_weights
        self.fc1.bias.data = descendant_fc1_biases
        self.act = act_layer()
        # print(self.fc1.bias.data)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2.weight.data = descendant_fc2_weights
        self.fc2.bias.data = descendant_fc2_biases
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def transform_mlp(learngene_depth, descendant_depth, hdp_proj_bias):
    hdp_proj_layers = []
    _layer_1 = nn.Linear(
            learngene_depth,
            descendant_depth,
            bias=hdp_proj_bias,
        )
    hdp_proj_layers.append(_layer_1)
    hdp_proj_layers.append(nn.ReLU())
    _layer_2 = nn.Linear(
            descendant_depth,
            descendant_depth,
            bias=hdp_proj_bias,
        )
    hdp_proj_layers.append(_layer_2)
    hdp_proj = nn.Sequential(*hdp_proj_layers)
    return hdp_proj

class DeiT_mixture_learngene(nn.Module):
    """ Vision Transformer """
    def __init__(self, inherited_layer, distribution, ratio, descendant_ratio, mlp_inherited_method, num_heads, num_heads_learngene, num_heads_descendant, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 hdp=False,
                 hdp_ratios=[],
                 hdp_non_linear=True, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = inherited_layer.patch_embed
        # self.patch_embed = PatchEmbed(
        #     img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = inherited_layer.cls_token
        self.pos_embed = inherited_layer.pos_embed
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        fc1_weights = torch.stack([block.mlp.fc1.weight for block in inherited_layer.blocks])
        fc1_biases = torch.stack([block.mlp.fc1.bias for block in inherited_layer.blocks])
        fc2_weights = torch.stack([block.mlp.fc2.weight for block in inherited_layer.blocks])
        fc2_biases = torch.stack([block.mlp.fc2.bias for block in inherited_layer.blocks])

        learngene_depth = int(12/ratio) # int(depth/ratio)
        descendant_depth = int(learngene_depth*descendant_ratio)

        # distribution = 'Uniform'  #'Gaussian'  config参数
        if distribution == 'Gaussian-Layer':
            p = torch.randn(learngene_depth, 12)
            # p = torch.randn(learngene_depth, depth)
            p = torch.softmax(p, dim=1).cuda()
        else: 
            p = torch.full((learngene_depth, 12), 1.0/12).cuda()
            # p = torch.full((learngene_depth, depth), 1.0/depth).cuda()
        
        print(p)

        self.learngene_fc1_weights = torch.einsum('ij,jkl->ikl', p, fc1_weights)
        self.learngene_fc1_biases = torch.einsum('ij,jk->ik', p, fc1_biases)
        self.learngene_fc2_weights = torch.einsum('ij,jkl->ikl', p, fc2_weights)
        self.learngene_fc2_biases = torch.einsum('ij,jk->ik', p, fc2_biases)
        self.mlp_proj = transform_mlp(learngene_depth, descendant_depth, bool(hdp_non_linear)).cuda()
        print(learngene_depth)
        print(descendant_depth)
        print(self.mlp_proj)
        
        # weight torch.Size([2, 384, 96]) bias torch.Size([2, 384])
        descendant_fc1_weights = self.mlp_proj(self.learngene_fc1_weights.transpose(-3, -1)).transpose(-3, -1)
        descendant_fc1_biases = self.mlp_proj(self.learngene_fc1_biases.transpose(-2, -1)).transpose(-2, -1)
        descendant_fc2_weights = self.mlp_proj(self.learngene_fc2_weights.transpose(-3, -1)).transpose(-3, -1)
        descendant_fc2_biases = self.mlp_proj(self.learngene_fc2_biases.transpose(-2, -1)).transpose(-2, -1)


        print(descendant_fc1_weights.shape)

        self.blocks = nn.ModuleList([
            Block_learngene(i_layer=i, inherited_blocks=inherited_layer.blocks[i], 
                inherited_mlp=TransMlp(descendant_fc1_weights[i], descendant_fc1_biases[i], descendant_fc2_weights[i], descendant_fc2_biases[i], embed_dim, int(embed_dim * mlp_ratio), nn.GELU, drop_rate), 
                mlp_inherited_method=mlp_inherited_method, distribution=distribution,
                num_heads=num_heads, num_heads_learngene=num_heads_learngene, num_heads_descendant=num_heads_descendant, 
                dim=embed_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                hdp=hdp,
                hdp_ratios=hdp_ratios,
                hdp_non_linear=hdp_non_linear)
            for i in range(depth)])

        print(self.blocks)

        self.norm = inherited_layer.norm
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x, ada_token=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        if ada_token is not None:
            ada_tokens = ada_token.expand(B, -1, -1) # B, p, d
            x = torch.cat((x, ada_tokens), dim=1)

        return self.pos_drop(x)

    def forward(self, x, ada_token=None, use_patches=False):
        x = self.prepare_tokens(x, ada_token)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # print(x.shape)
        # exit()

        if use_patches:
            return x[:, 1:]
        else:
            return x[:, 0]


def deit_tiny_attn2_ffn6(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=128, depth=6, num_heads=2, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs) 
    return model

def deit_tiny_attn2_ffn9(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=128, depth=9, num_heads=2, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs) 
    return model

def deit_tiny_attn2_ffn12(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=128, depth=12, num_heads=2, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs) 
    return model

def deit_small_attn4_ffn6(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=256, depth=6, num_heads=4, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs) 
    return model

def deit_small_attn4_ffn9(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=256, depth=9, num_heads=4, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs) 
    return model

def deit_small_attn4_ffn12(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs) 
    return model


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_depth2(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=2, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_depth3(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=3, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_depth4(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=4, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_depth6(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=6, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_depth8(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=8, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_depth9(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=9, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_depth2(patch_size=16, num_classes=0, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=2, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_depth3(patch_size=16, num_classes=0, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=3, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_depth4(patch_size=16, num_classes=0, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_depth6(patch_size=16, num_classes=0, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=6, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_depth8(patch_size=16, num_classes=0, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=8, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_depth9(patch_size=16, num_classes=0, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=9, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_depth6_shared(patch_size=16, num_classes=0, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=6, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_depth6_shared_residual(residual_interval=3, patch_size=16, num_classes=0, **kwargs):
    model = VisionTransformer_Residual(residual_interval=residual_interval, 
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=6, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_weight_transform_depth6(patch_size=16, num_classes=0, **kwargs):
    model = WeightTransform_VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=6, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_weight_transform_depth12(patch_size=16, num_classes=0, **kwargs):
    model = WeightTransform_VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_depth3(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=3, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_depth6(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model