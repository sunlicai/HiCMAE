from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange, repeat


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


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
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # me: support window mask
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, mask=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), mask=mask))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x



"""
adapted from https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
"""
# support cross attention
class GeneralAttention(nn.Module):
    def __init__(
            self, dim, context_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.kv = nn.Linear(dim if context_dim is None else context_dim, all_head_dim * 2, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context=None):
        B, T1, C = x.shape
        q_bias, kv_bias = self.q_bias, None
        if self.q_bias is not None:
            kv_bias = torch.cat((torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, T1, self.num_heads, -1).transpose(1,2) # me: (B, H, T1, C//H)
        kv = F.linear(input=x if context is None else context, weight=self.kv.weight, bias=kv_bias)
        _, T2, _ = kv.shape
        kv = kv.reshape(B, T2, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # make torchscript happy (cannot use tensor as tuple), me： (B, H, T2, C//H)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # me: (B, H, T1, T2)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T1, -1) # (B, H, T1, C//H) -> (B, T1, H, C//H) -> (B, T1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



# cross + self
class CSBlock(nn.Module):
    def __init__(self, dim, context_dim, num_heads, num_cross_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, cross_attn_head_dim=None):
        super().__init__()

        self.cross_attn = GeneralAttention(
            dim=dim, context_dim=context_dim, num_heads=num_cross_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=cross_attn_head_dim)
        self.cross_norm1 = norm_layer(dim)
        self.cross_norm2 = norm_layer(context_dim)

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_0 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_0, self.gamma_1, self.gamma_2 = None, None, None

    def forward(self, x, context):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.cross_attn(self.cross_norm1(x), context=self.cross_norm2(context)))
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_0 * self.cross_attn(self.cross_norm1(x), context=self.cross_norm2(context)))
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # me: for more attention types
        self.temporal_seq_len = num_frames // self.tubelet_size
        self.spatial_num_patches = num_patches // self.temporal_seq_len
        self.input_token_size = (num_frames // self.tubelet_size, img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# me: for audio, copied from PatchEmbed_new in AudioMAE
# slightly modified in forward()
class PatchEmbed2D(nn.Module):
    """ Flexible Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)  # with overlapped patches
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        # self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size)  # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h * w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1, 1, img_size[0], img_size[1])).shape

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2) # (B, N, C)
        # x = self.proj(x)  # 32, 1, 1024, 128 -> 32, 768, 101, 12
        # x = x.flatten(2)  # 32, 768, 101, 12 -> 32, 768, 1212
        # x = x.transpose(1, 2)  # 32, 768, 1212 -> 32, 1212, 768
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False, 
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_mean_pooling=True,
                 keep_temporal_dim=False, # do not perform temporal pooling, has higher priority than 'use_mean_pooling'
                 head_activation_func=None, # activation function after head fc, mainly for the regression task
                 attn_type='joint',
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.attn_type = attn_type
        if attn_type == 'joint':
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values
                )
                for i in range(depth)])
        else:
            raise NotImplementedError

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.keep_temporal_dim = keep_temporal_dim

        if head_activation_func is not None:
            if head_activation_func == 'sigmoid':
                self.head_activation_func = nn.Sigmoid()
            elif head_activation_func == 'relu':
                self.head_activation_func = nn.ReLU()
            elif head_activation_func == 'tanh':
                self.head_activation_func = nn.Tanh()
            else:
                raise NotImplementedError
        else: # default
            self.head_activation_func = nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        if num_classes > 0:
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)

        if num_classes > 0:
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'part_tokens'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, _, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        intermediate_features = []

        for blk in self.blocks:
            x = blk(x)
            intermediate_features.append(x)

        x = torch.stack(intermediate_features, dim=2) # (B, N, L, D), L: number of layers, N: tokens

        x = self.norm(x)
        if self.fc_norm is not None:
            if self.keep_temporal_dim:
                x = rearrange(x, 'b (t hw) c -> b c t hw',
                              t=self.patch_embed.temporal_seq_len,
                              hw=self.patch_embed.spatial_num_patches)
                # spatial mean pooling
                x = x.mean(-1) # (B, C, T)
                # temporal upsample: 8 -> 16, for patch embedding reduction
                x = torch.nn.functional.interpolate(
                    x, scale_factor=self.patch_embed.tubelet_size,
                    mode='linear'
                )
                x = rearrange(x, 'b c t -> b t c')
                return self.fc_norm(x)
            else:
                return self.fc_norm(x.mean(1))
        else:
            return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = self.head_activation_func(x)
        if self.keep_temporal_dim:
            x = x.view(x.size(0), -1) # (B,T,C) -> (B,T*C)
        return x



class VisionTransformerEncoder2D(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, use_checkpoint=False,
                 use_learnable_pos_emb=False, init_scale=0., use_mean_pooling=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed2D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=patch_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        if num_classes > 0:
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)

        if num_classes > 0:
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, _, _ = x.size()

        # x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        intermediate_features = []

        # block
        for blk in self.blocks:
            x = blk(x)
            intermediate_features.append(x)

        x = torch.stack(intermediate_features, dim=2) # (B, N, L, D), L: number of layers, N: tokens


        x = self.norm(x)
        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        else:
            return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class VisionTransformerEncoderForFusion(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, embed_dim=768, embed_dim_audio=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 ):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            CSBlock(dim=embed_dim, context_dim=embed_dim_audio, num_heads=num_heads,
                    num_cross_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # audio
        self.blocks_audio = nn.ModuleList([
            CSBlock(dim=embed_dim_audio, context_dim=embed_dim, num_heads=num_heads,
                    num_cross_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
            for i in range(depth)])

        # do not share norm layer
        self.norm_audio = norm_layer(embed_dim_audio)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, x_audio):
        for blk, blk_audio in zip(self.blocks, self.blocks_audio):
            x, x_audio = blk(x, context=x_audio), blk_audio(x_audio, context=x)

        # norm
        x = self.norm(x)

        x_audio = self.norm_audio(x_audio)

        return x, x_audio



class AudioVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 keep_temporal_dim=False, # do not use it for audio-visual
                 attn_type='joint',
                 img_size_audio=(256, 128),  # (T, F)
                 patch_size_audio=16,
                 in_chans_audio=1,
                 embed_dim_audio=768,
                 depth_audio=12,
                 num_heads_audio=12,
                 fusion_depth=2,
                 fusion_num_heads=12,
                 use_mean_pooling=True, # no use for now
                 head_activation_func=None,  # activation function after head fc, mainly for the regression task
                 ):
        super().__init__()
        # video encoder
        self.encoder_depth = depth
        if depth > 0 :
            self.encoder = VisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                num_classes=0, # no head for video encoder
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                init_values=init_values,
                init_scale=init_scale,
                all_frames=all_frames,
                tubelet_size=tubelet_size,
                use_learnable_pos_emb=use_learnable_pos_emb,
                use_mean_pooling=False, # no mean pooling for video encoder
                attn_type=attn_type,
            )
        else:
            self.encoder = None
            print(f"==> Warning: video-specific encoder is not used!!!")

        # audio encoder
        self.encoder_depth_audio = depth_audio
        if depth_audio > 0:
            self.encoder_audio = VisionTransformerEncoder2D(
                img_size=img_size_audio,
                patch_size=patch_size_audio,
                in_chans=in_chans_audio,
                num_classes=0, # no head for audio encoder
                embed_dim=embed_dim_audio,
                depth=depth_audio,
                num_heads=num_heads_audio,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                init_values=init_values,
                use_learnable_pos_emb=use_learnable_pos_emb,
                use_mean_pooling=False, # no mean pooling for audio encoder
            )
        else:
            self.encoder_audio = None
            print(f"==> Warning: audio-specific encoder is not used!!!")

        # fusion encoder
        if fusion_depth > 0 and self.encoder is not None and self.encoder_audio is not None:
            self.encoder_fusion = VisionTransformerEncoderForFusion(
                embed_dim=embed_dim, # for video
                embed_dim_audio=embed_dim_audio, # for audio
                depth=fusion_depth,
                num_heads=fusion_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=0, # no drop path for fusion encoder
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                init_values=init_values,
            )
            self.encoder_depth_fusion = fusion_depth
        else:
            self.encoder_fusion = None
            self.encoder_depth_fusion = 0
            print(f"==> Warning: fusion encoder is not used!!!")

        # head
        fc_num = 2 * (embed_dim + embed_dim_audio)
        self.fc_norm = norm_layer(fc_num)
        self.head = nn.Linear(fc_num, num_classes) if num_classes > 0 else nn.Identity()

        if head_activation_func is not None:
            if head_activation_func == 'sigmoid':
                self.head_activation_func = nn.Sigmoid()
            elif head_activation_func == 'relu':
                self.head_activation_func = nn.ReLU()
            elif head_activation_func == 'tanh':
                self.head_activation_func = nn.Tanh()
            else:
                raise NotImplementedError
        else: # default
            self.head_activation_func = nn.Identity()

        self.video_layer_weights = nn.Parameter(torch.ones(self.encoder_depth) / self.encoder_depth)
        self.audio_layer_weights = nn.Parameter(torch.ones(self.encoder_depth_audio) / self.encoder_depth_audio)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        # return len(self.blocks)
        return max(self.encoder_depth, self.encoder_depth_audio) + \
               self.encoder_depth_fusion

    def get_num_modality_specific_layers(self):
        # return len(self.blocks)
        return max(self.encoder_depth, self.encoder_depth_audio)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token',
                'encoder.pos_embed', 'encoder_audio.pos_embed', 'encoder_fusion.pos_embed',
                'encoder.cls_token', 'encoder_audio.cls_token', 'encoder_fusion.cls_token',
                }

    def forward_features(self, x, x_audio):
        # encoder: video
        x = self.encoder(x) # [B, N, L, C]
        # encoder: audio
        x_audio = self.encoder_audio(x_audio) # [B, N, L, C]

        # unimodal feature
        video_feature = x.mean(dim=1)  # [B, L, C]
        audio_feature = x_audio.mean(dim=1)  # [B, L, C]
        video_feature = torch.sum(video_feature * torch.nn.functional.softmax(self.video_layer_weights, dim=0).view(1, -1, 1), dim=1)
        audio_feature = torch.sum(audio_feature * torch.nn.functional.softmax(self.audio_layer_weights, dim=0).view(1, -1, 1), dim=1)

        # encoder: fusion
        x = x[:,:,-1] # [B, N, C]
        x_audio = x_audio[:,:,-1] # [B, N, C]
        x, x_audio = self.encoder_fusion(x, x_audio)
        fusion_video_feature = x.mean(dim=1)
        fusion_audio_feature = x_audio.mean(dim=1)
        final_feature = torch.cat([video_feature, audio_feature,
                                   fusion_video_feature, fusion_audio_feature],
                                   dim=-1)

        return self.fc_norm(final_feature)


    def forward(self, x, x_audio, save_feature=False):
        x = self.forward_features(x, x_audio) # (B, C)

        if save_feature:
            feature = x

        x = self.head(x)

        x = self.head_activation_func(x)

        if save_feature:
            return x, feature
        else:
            return x # (B, C)



@register_model
def avit_dim256_patch16_160_a256(pretrained=False, **kwargs):
    embed_dim = 256
    num_heads = 4
    patch_size = 16
    model = AudioVisionTransformer(
        # video
        img_size=160,
        patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True,
        # audio
        img_size_audio=(256, 128),  # (T, F)
        patch_size_audio=patch_size, embed_dim_audio=embed_dim, num_heads_audio=num_heads,
        # fusion
        fusion_num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def avit_dim384_patch16_160_a256(pretrained=False, **kwargs):
    embed_dim = 384
    num_heads = 6
    patch_size = 16
    model = AudioVisionTransformer(
        # video
        img_size=160,
        patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True,
        # audio
        img_size_audio=(256, 128),  # (T, F)
        patch_size_audio=patch_size, embed_dim_audio=embed_dim, num_heads_audio=num_heads,
        # fusion
        fusion_num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def avit_dim512_patch16_160_a256(pretrained=False, **kwargs):
    embed_dim = 512
    num_heads = 8
    patch_size = 16
    model = AudioVisionTransformer(
        # video
        img_size=160,
        patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True,
        # audio
        img_size_audio=(256, 128),  # (T, F)
        patch_size_audio=patch_size, embed_dim_audio=embed_dim, num_heads_audio=num_heads,
        # fusion
        fusion_num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
