# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.vision_transformer import Attention, Mlp
from diffusers.models.embeddings import get_2d_rotary_pos_embed

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def apply_rotary_emb(x, cos, sin):
    cls_tok, patch_tok = x[:, :, 0, :], x[:, :, 1:, :]   # CLS + patches
    cls_tok = cls_tok.unsqueeze(2)

    cos, sin = cos[None, None], sin[None, None]
    cos, sin = cos.to(x.device), sin.to(x.device)

    x_real, x_imag = patch_tok.reshape(*patch_tok.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    patch_tok_out = (patch_tok.float() * cos + x_rotated.float() * sin).to(x.dtype)

    return torch.cat([cls_tok, patch_tok_out], dim=2)


class RoPEAttention(Attention):
    def forward(self, x: torch.Tensor, freqs_cis_cos, freqs_cis_sin) -> torch.Tensor:
        B, N, C = x.shape
        # B, N, C -> B, N, 3, H, D -> 3, B, H, N, D
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cis_cos, freqs_cis_sin)
        k = apply_rotary_emb(k, freqs_cis_cos, freqs_cis_sin)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, scale_factor=1.0, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim) * scale_factor)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
        
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, context_dim, attn_drop=0.0, proj_drop=0.0, qk_norm=False, **block_kwargs):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(context_dim, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        if qk_norm:
            self.q_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-6)
            self.k_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, cond):
        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape
        first_dim = B

        q = self.q_linear(x)
        kv = self.kv_linear(cond).view(first_dim, -1, 2, C)
        k, v = kv.unbind(2)
        q = self.q_norm(q).view(first_dim, -1, self.num_heads, self.head_dim)
        k = self.k_norm(k).view(first_dim, -1, self.num_heads, self.head_dim)
        v = v.view(first_dim, -1, self.num_heads, self.head_dim)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = RoPEAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, freqs_cis_cos, freqs_cis_sin):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), freqs_cis_cos, freqs_cis_sin)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x




class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=16,
        patch_size=1,
        in_channels=3,
        out_channels=None,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        base_size=16,
        in_proj=True,
        cls_token_out_dim=None,
        external_cls_token=False,
        context_dim=768,
    ):
        super().__init__()
        input_size = int(input_size)
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.input_size = input_size
        self.external_cls_token = external_cls_token

        self.base_size = base_size
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, base_size**2+1, hidden_size), requires_grad=False)
        self.hidden_size = hidden_size

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        if in_proj:
            self.in_proj = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True),
                                        nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size))
        else:
            self.in_proj = nn.Identity()
        self.out_proj = nn.Sequential(nn.GroupNorm(num_groups=32, num_channels=hidden_size, eps=1e-6, affine=True),
                                    nn.Conv2d(hidden_size, self.out_channels, kernel_size=1, stride=1))
        
        if not external_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.out_proj_cls = nn.Linear(hidden_size, cls_token_out_dim, bias=True)
        else:
            self.cls_token = nn.Linear(context_dim, hidden_size, bias=True)
            
        self.initialize_weights()

        self.cached_pos_embed = [base_size, base_size]

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.base_size, cls_token=True, extra_tokens=1)

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def reset_pos_embed(self, h, w):
        cached_h, cached_w = self.cached_pos_embed[0], self.cached_pos_embed[1]
        if cached_h != h or cached_w != w or len(self.cached_pos_embed) != 5:
            freqs_cis_cos, freqs_cis_sin = get_2d_rotary_pos_embed(self.hidden_size//self.num_heads, ((0, 0), (h, w)), (h, w))

            cls_pos_embed = self.pos_embed[:, 0, :].unsqueeze(0)

            pos_embed = self.pos_embed[:, 1:, :].reshape(self.base_size, self.base_size, self.pos_embed.shape[-1]).permute(2, 0, 1).unsqueeze(0)
            pos_embed = F.interpolate(
                pos_embed, size=(h, w), mode='bicubic', align_corners=False
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2).float()
            pos_embed = torch.cat([cls_pos_embed, pos_embed], dim=1)
            self.cached_pos_embed = [h, w, freqs_cis_cos, freqs_cis_sin, pos_embed]
        else:
            freqs_cis_cos, freqs_cis_sin, pos_embed = self.cached_pos_embed[2], self.cached_pos_embed[3], self.cached_pos_embed[4]
        
        return freqs_cis_cos, freqs_cis_sin, pos_embed

    def forward(self, x, t, context=None, encoder_depth=-1):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        _, _, H, W = x.shape
        x = self.in_proj(x)

        b, c, h, w = x.shape

        freqs_cis_cos, freqs_cis_sin, pos_embed = self.reset_pos_embed(h, w)
        pos_embed = pos_embed.to(x.device)
        
        x = x.reshape(b, c, h * w)
        x = x.transpose(1,2)

        if not self.external_cls_token:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + pos_embed  # (N, T+1, D)
        else:
            x = torch.cat([self.cls_token(context), x], dim=1) + pos_embed  # (N, T+1, D)

        for i, block in enumerate(self.blocks):
            x = block(x, t, freqs_cis_cos, freqs_cis_sin)                      # (N, T, D)


        cls_token = x[:, 0, :]
        x = x[:, 1:, :]
        x = x.transpose(1,2).reshape(b, c, h, w)
        x = F.interpolate(x, size=(H, W), mode='bicubic', align_corners=False)
        
        if not self.external_cls_token:
            return self.out_proj(x), self.out_proj_cls(cls_token)
        else:
            return self.out_proj(x)



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y

#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=1, num_heads=16, **kwargs)

def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def DiT_S(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)
    
DiT_models = {
    'DiT-XL': DiT_XL,
    'DiT-L':  DiT_L, 
    'DiT-B':  DiT_B,
    'DiT-S':  DiT_S,
}