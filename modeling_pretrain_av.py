import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from modeling_finetune_av import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table, CSBlock
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from einops import rearrange, repeat

from modeling_finetune_av import PatchEmbed2D

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2,
                 use_learnable_pos_emb=False,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

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
        return {'pos_embed', 'cls_token', 'part_tokens'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask, return_intermediate_features=None):
        _, _, T, _, _ = x.shape
        x = self.patch_embed(x)
        
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible

        intermediate_features = []
        for blk in self.blocks:
            x_vis = blk(x_vis)
            intermediate_features.append(self.norm(x_vis))

        x_vis = self.norm(x_vis)

        if return_intermediate_features is None:
            return x_vis, None
        else:
            intermediate_features = [intermediate_features[i] for i in return_intermediate_features]
            return x_vis, intermediate_features

    def forward(self, x, mask, return_intermediate_features=None):
        x, intermediate_features = self.forward_features(x, mask, return_intermediate_features)
        x = self.head(x)
        return x, intermediate_features


class PretrainVisionTransformerEncoder2D(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, use_checkpoint=False,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed2D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=patch_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

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

    def forward_features(self, x, mask, return_intermediate_features=None):
        x = self.patch_embed(x)

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible

        intermediate_features = []

        for blk in self.blocks:
            x_vis = blk(x_vis)
            intermediate_features.append(self.norm(x_vis))

        x_vis = self.norm(x_vis)
        
        if return_intermediate_features is None:
            return x_vis, None
        else:
            intermediate_features = [intermediate_features[i] for i in return_intermediate_features]
            return x_vis, intermediate_features

    def forward(self, x, mask, return_intermediate_features=None):
        x, intermediate_features = self.forward_features(x, mask, return_intermediate_features)
        x = self.head(x)
        return x, intermediate_features


class PretrainVisionTransformerEncoderForFusion(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, embed_dim=768, embed_dim_audio=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 modal_param_sharing=False):
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
        self.modal_param_sharing = modal_param_sharing
        if not modal_param_sharing:
            self.blocks_audio = nn.ModuleList([
                CSBlock(dim=embed_dim_audio, context_dim=embed_dim, num_heads=num_heads,
                        num_cross_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[i], norm_layer=norm_layer,
                        init_values=init_values)
                for i in range(depth)])
        else:
            self.blocks_audio = self.blocks

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


# decoder with hierarchical skip connections
class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2,
                 ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size ** 2 
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                self.blocks.append(
                    Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                        init_values=init_values)
                )
            else:
                self.blocks.append(
                    CSBlock(
                        dim=embed_dim, context_dim=embed_dim, num_heads=num_heads, num_cross_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                        init_values=init_values)
                )

        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

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

    def forward(self, x, return_token_num, x_skip_connects=None):
        for i, blk in enumerate(self.blocks):
            if i == 0:
                x = blk(x)
            else: # hierarchical skip connections
                x = blk(x, context=x_skip_connects[-i])

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x


class PretrainAudioVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=1536,  #  decoder_num_classes=768,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=2,
                 num_classes=0,  # avoid the error from create_fn in timm
                 in_chans=0,  # avoid the error from create_fn in timm
                 # audio
                 img_size_audio=(256, 128),  # (T, F)
                 patch_size_audio=16,
                 encoder_in_chans_audio=1,
                 encoder_embed_dim_audio=768,
                 encoder_depth_audio=12,
                 encoder_num_heads_audio=12,
                 decoder_num_classes_audio=256,
                 decoder_embed_dim_audio=512,
                 decoder_depth_audio=8,
                 decoder_num_heads_audio=8,
                 # fusion
                 encoder_fusion_depth=2,
                 encoder_fusion_num_heads=12,
                 # contrastive learning
                 inter_contrastive_temperature=0.07,
                 # intermediate layers
                 return_intermediate_features=(3, 6, 9),
                 ):
        super().__init__()

        self.encoder_depth = encoder_depth
        self.return_intermediate_features = return_intermediate_features
        if self.return_intermediate_features is not None:
            assert len(self.return_intermediate_features) == decoder_depth - 1, f"Error: wrong layers (len({return_intermediate_features}) != {decoder_depth-1}) for intermediate_features!"
            assert len(self.return_intermediate_features) == decoder_depth_audio - 1, f"Error: wrong layers (len({return_intermediate_features}) != {decoder_depth_audio-1}) for intermediate_features!"
            for idx in self.return_intermediate_features:
                assert idx < encoder_depth and idx < encoder_depth_audio, f"Error: too large layer index ({idx})!"

        # video encoder
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
        )

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
        )

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)


        # audio encoder
        self.encoder_audio = PretrainVisionTransformerEncoder2D(
            img_size=img_size_audio,
            patch_size=patch_size_audio,
            in_chans=encoder_in_chans_audio,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim_audio,
            depth=encoder_depth_audio,
            num_heads=encoder_num_heads_audio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb
        )

        self.decoder_audio = PretrainVisionTransformerDecoder(
            patch_size=patch_size_audio,
            num_patches=self.encoder_audio.patch_embed.num_patches,
            num_classes=decoder_num_classes_audio,
            embed_dim=decoder_embed_dim_audio,
            depth=decoder_depth_audio,
            num_heads=decoder_num_heads_audio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=1/3, # no meaning, just to scape 'assert'
        )

        self.encoder_to_decoder_audio = nn.Linear(encoder_embed_dim_audio, decoder_embed_dim_audio, bias=False)

        self.mask_token_audio = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim_audio))

        self.pos_embed_audio = get_sinusoid_encoding_table(self.encoder_audio.patch_embed.num_patches, decoder_embed_dim_audio)

        trunc_normal_(self.mask_token_audio, std=.02)


        # cross-modal fusion encoder
        self.encoder_fusion = PretrainVisionTransformerEncoderForFusion(
            embed_dim=encoder_embed_dim, # for video
            embed_dim_audio=encoder_embed_dim_audio, # for audio
            depth=encoder_fusion_depth,
            num_heads=encoder_fusion_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
        )

        self.inter_contrastive_temperature = inter_contrastive_temperature

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
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask, x_audio, mask_audio):
        # encoder: video
        x_vis, x_vis_inter_features = self.encoder(x, mask, self.return_intermediate_features) # [B, N_vis, C_e]
        # encoder: audio
        x_vis_audio, x_vis_audio_inter_features = self.encoder_audio(x_audio, mask_audio, self.return_intermediate_features)

        # hcmcl
        logits_per_video, logits_per_audio = [], []
        for x_vis_inter, x_vis_audio_inter in zip(x_vis_inter_features, x_vis_audio_inter_features):
            # pooling
            video_features_inter = x_vis_inter.mean(dim=1)  # (B, C)
            audio_features_inter = x_vis_audio_inter.mean(dim=1)  # (B, C)

            # normalized features
            video_features_inter = video_features_inter / video_features_inter.norm(dim=1, keepdim=True)
            audio_features_inter = audio_features_inter / audio_features_inter.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logits_per_video_inter = video_features_inter @ audio_features_inter.t() / self.inter_contrastive_temperature
            logits_per_audio_inter = logits_per_video_inter.t()

            logits_per_video.append(logits_per_video_inter)
            logits_per_audio.append(logits_per_audio_inter)

        # encoder: fusion
        x_vis, x_vis_audio = self.encoder_fusion(x_vis, x_vis_audio)

        # decoder: video
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accordingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)  # [B, N, C_d]
        x_vis_inter_features = [self.encoder_to_decoder(feature) + pos_emd_vis for feature in x_vis_inter_features]
        x = self.decoder(x_full, pos_emd_mask.shape[1], x_vis_inter_features)  # [B, N_mask, 2 * 3 * 16 * 16]

        # decoder: audio
        x_vis_audio = self.encoder_to_decoder_audio(x_vis_audio) # [B, N_vis, C_d]
        B_audio, N_audio, C_audio = x_vis_audio.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accordingly.
        expand_pos_embed_audio = self.pos_embed_audio.expand(B_audio, -1, -1).type_as(x_audio).to(x_audio.device).clone().detach()
        pos_emd_vis_audio = expand_pos_embed_audio[~mask_audio].reshape(B_audio, -1, C_audio)
        pos_emd_mask_audio = expand_pos_embed_audio[mask_audio].reshape(B_audio, -1, C_audio)
        x_full_audio = torch.cat([x_vis_audio + pos_emd_vis_audio, self.mask_token_audio + pos_emd_mask_audio], dim=1) # [B, N, C_d]
        x_vis_audio_inter_features = [self.encoder_to_decoder_audio(feature) + pos_emd_vis_audio for feature in x_vis_audio_inter_features]
        x_audio = self.decoder_audio(x_full_audio, pos_emd_mask_audio.shape[1], x_vis_audio_inter_features) # [B, N_mask, 1 * 16 * 16]


        return x, x_audio, logits_per_video, logits_per_audio


@register_model
def pretrain_hicmae_dim384_patch16_160_a256(pretrained=False, **kwargs):
    model = PretrainAudioVisionTransformer(
        img_size=160,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=192,
        decoder_num_heads=3,
        # audio
        img_size_audio=(256, 128),  # (T, F)
        patch_size_audio=16,
        encoder_embed_dim_audio=384,
        encoder_num_heads_audio=6,
        decoder_num_classes_audio=256,
        decoder_embed_dim_audio=192,
        decoder_num_heads_audio=3,
        # fusion
        encoder_fusion_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_hicmae_dim512_patch16_160_a256(pretrained=False, **kwargs):
    model = PretrainAudioVisionTransformer(
        img_size=160,
        patch_size=16,
        encoder_embed_dim=512,
        encoder_num_heads=8,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        # audio
        img_size_audio=(256, 128),  # (T, F)
        patch_size_audio=16,
        encoder_embed_dim_audio=512,
        encoder_num_heads_audio=8,
        decoder_num_classes_audio=256,
        decoder_embed_dim_audio=384,
        decoder_num_heads_audio=6,
        # fusion
        encoder_fusion_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
