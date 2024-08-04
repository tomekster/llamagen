# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from tokenizer.tokenizer_image.VAR_files.quant import VectorQuantizer2
from tokenizer.tokenizer_image.VAR_files.basic_vae import Decoder, Encoder


@dataclass
class ModelArgs:
    vocab_size=4096
    z_channels=32
    using_znorm=False       # whether to normalize when computing the nearest neighbors
    beta=0.25               # commitment loss weight
    default_qresi_counts=0  # if is 0: automatically set to len(v_patch_nums)
    v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16) # number of patches for each scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
    quant_resi=0.5          # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x
    share_quant_resi=4      # use 4 \phi layers for K scales: partially-shared \phi
    
    dropout=0.0
    
    # TODO: tsternal, changed to 160 to match the downloadable weights
    # ch=128
    ch=160
    
    quant_conv_ks=3        # quant conv kernel size


class VQVaeVarModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        
        ddconfig = dict(
            dropout=config.dropout, ch=config.ch, z_channels=config.z_channels,
            in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
        )
        
        ddconfig.pop('double_z', None)  # only KL-VAE should use double_z=True
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)

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
################################################################################

def VQ_VAE_VAR(**kwargs):
    return VQVaeVarModel(ModelArgs(**kwargs))

VQ_VAE_VAR_models = {'VQ-VAE-VAR': VQ_VAE_VAR}