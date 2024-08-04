# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer.tokenizer_image.VAR_files.quant import VectorQuantizer2

import math
from functools import partial
from tokenizer.tokenizer_image.VAR_files.basic_vae import Decoder, Encoder

@dataclass
class ModelArgs:
    # =========Original VAE code ========
    # codebook_size: int = 16384
    # codebook_embed_dim: int = 8
    # codebook_l2_norm: bool = True
    # codebook_show_usage: bool = True
    # commit_loss_beta: float = 0.25
    # entropy_loss_ratio: float = 0.0
    
    # encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    # decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    # z_channels: int = 256
    # dropout_p: float = 0.0
    # ====================================
    
    
    vocab_size=4096
    z_channels=32
    using_znorm=False       # whether to normalize when computing the nearest neighbors
    beta=0.25               # commitment loss weight
    default_qresi_counts=0  # if is 0: automatically set to len(v_patch_nums)
    v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16) # number of patches for each scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
    quant_resi=0.5          # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x
    share_quant_resi=4      # use 4 \phi layers for K scales: partially-shared \phi
    
    dropout=0.0
    ch=128
    
    quant_conv_ks=3        # quant conv kernel size
    
    # =========Unused VAE VAR Params, models/vqvae.py=========================
    # test_mode=True,
    # ========================================================================


class VQVaeVarModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        
        # ====ORIGINAL VAE code ==========================================================================
        # self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        # self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        # ================================================================================================
        
        ddconfig = dict(
            dropout=config.dropout, ch=config.ch, z_channels=config.z_channels,
            in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
        )
        ddconfig.pop('double_z', None)  # only KL-VAE should use double_z=True
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)

        # ====ORIGINAL VAE code ========================================================================
        # self.quantize = VectorQuantizer2(config.codebook_size, config.codebook_embed_dim, 
        #                                 config.commit_loss_beta, config.entropy_loss_ratio,
        #                                 config.codebook_l2_norm, config.codebook_show_usage)
        # self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        # self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)
        # ==============================================================================================
        
        self.quantize = VectorQuantizer2(
            vocab_size=config.vocab_size, Cvae=config.z_channels, using_znorm=config.using_znorm, beta=config.beta,
            default_qresi_counts=config.default_qresi_counts, v_patch_nums=config.v_patch_nums, quant_resi=config.quant_resi, share_quant_resi=config.share_quant_resi
        )
        
        self.quant_conv = torch.nn.Conv2d(config.z_channels, config.z_channels, config.quant_conv_ks, stride=1, padding=config.quant_conv_ks//2)
        self.post_quant_conv = torch.nn.Conv2d(config.z_channels, config.z_channels, config.quant_conv_ks, stride=1, padding=config.quant_conv_ks//2)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
def VQ_VAE_VAR(**kwargs):
    # ==================================== ORIGINAL VAE CODE =============================================
    # return VQVaeVarModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))
    # ====================================================================================================
    return VQVaeVarModel(ModelArgs(**kwargs))

VQ_VAE_VAR_models = {'VQ-VAE-VAR': VQ_VAE_VAR}