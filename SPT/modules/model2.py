"""
This implementation is completely based on the version used in TabSyn.
The Stage 2 component is identical to the original TabSyn implementation, 

Reference:
[1] https://github.com/amazon-science/tabsyn/blob/main/tabsyn/model.py
"""

from typing import Callable, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from modules.utils import EDMLoss

ModuleType = Union[str, Callable[..., nn.Module]]

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

# 기존 MLPDiffusion
# class MLPDiffusion(nn.Module):
#     def __init__(self, d_in, dim_t = 512):
#         super().__init__()
#         self.dim_t = dim_t

#         self.proj = nn.Linear(d_in, dim_t)

#         self.mlp = nn.Sequential(
#             nn.Linear(dim_t, dim_t * 2),
#             nn.SiLU(),
#             nn.Linear(dim_t * 2, dim_t * 2),
#             nn.SiLU(),
#             nn.Linear(dim_t * 2, dim_t),
#             nn.SiLU(),
#             nn.Linear(dim_t, d_in),
#         )

#         self.map_noise = PositionalEmbedding(num_channels=dim_t)
#         self.time_embed = nn.Sequential(
#             nn.Linear(dim_t, dim_t),
#             nn.SiLU(),
#             nn.Linear(dim_t, dim_t)
#         )
    
#     def forward(self, x, noise_labels, class_labels=None):
#         emb = self.map_noise(noise_labels)
#         emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
#         emb = self.time_embed(emb)
    
#         x = self.proj(x) + emb
#         return self.mlp(x)

# LLM 정보 주입을 위한 새 MLPDiffusion
class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t=512, llm_dim=None):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        # LLM 임베딩 차원을 dim_t로 맞추기 위한 프로젝션 레이어
        if llm_dim is not None:
            self.llm_proj = nn.Sequential(
                nn.Linear(llm_dim, dim_t),
                nn.SiLU(),
                nn.Linear(dim_t, dim_t)
            )
        else:
            self.llm_proj = None

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
    
    def forward(self, x, noise_labels, llm_emb=None, class_labels=None):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        emb = self.time_embed(emb)
    
        x = self.proj(x) + emb

        # LLM 임베딩 주입
        if llm_emb is not None and self.llm_proj is not None:
            x = x + self.llm_proj(llm_emb) # 기존 MLP 벡터에 정보 임베딩 벡터를 더함

        return self.mlp(x)

# 기존 Precond
# class Precond(nn.Module):
#     def __init__(self,
#         denoise_fn,
#         hid_dim,
#         sigma_min = 0,                # Minimum supported noise level.
#         sigma_max = float('inf'),     # Maximum supported noise level.
#         sigma_data = 0.5,              # Expected standard deviation of the training data.
#     ):
#         super().__init__()

#         self.hid_dim = hid_dim
#         self.sigma_min = sigma_min
#         self.sigma_max = sigma_max
#         self.sigma_data = sigma_data
#         self.denoise_fn_F = denoise_fn

#     def forward(self, x, sigma):  

#         x = x.to(torch.float32)

#         sigma = sigma.to(torch.float32).reshape(-1, 1)
#         dtype = torch.float32

#         c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
#         c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
#         c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
#         c_noise = sigma.log() / 4

#         x_in = c_in * x
#         F_x = self.denoise_fn_F((x_in).to(dtype), c_noise.flatten())

#         assert F_x.dtype == dtype
#         D_x = c_skip * x + c_out * F_x.to(torch.float32)
#         return D_x

#     def round_sigma(self, sigma):
#         return torch.as_tensor(sigma)

# 새 Precond
class Precond(nn.Module):
    def __init__(self,
        denoise_fn,
        hid_dim,
        sigma_min = 0,                # Minimum supported noise level.
        sigma_max = float('inf'),     # Maximum supported noise level.
        sigma_data = 0.5,              # Expected standard deviation of the training data.
    ):
        super().__init__()

        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.denoise_fn_F = denoise_fn

    def forward(self, x, sigma, llm_emb=None):  
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        # denoise_fn_F 호출 시 llm_emb 전달
        F_x = self.denoise_fn_F((x_in).to(dtype), c_noise.flatten(), llm_emb=llm_emb)

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

# 기존 Model
# class Model(nn.Module):
#     def __init__(self, denoise_fn, hid_dim, P_mean=-1.2, P_std=1.2, sigma_data=0.5, gamma=5, opts=None, pfgmpp = False):
#         super().__init__()

#         self.denoise_fn_D = Precond(denoise_fn, hid_dim)
#         self.loss_fn = EDMLoss(P_mean, P_std, sigma_data, hid_dim=hid_dim, gamma=5, opts=None)

#     def forward(self, x):

#         loss = self.loss_fn(self.denoise_fn_D, x)
#         return loss.mean(-1).mean()

# 새 Model
class Model(nn.Module):
    def __init__(self, denoise_fn, hid_dim, P_mean=-1.2, P_std=1.2, sigma_data=0.5, gamma=5, opts=None, pfgmpp=False):
        super().__init__()

        self.denoise_fn_D = Precond(denoise_fn, hid_dim)
        self.loss_fn = EDMLoss(P_mean, P_std, sigma_data, hid_dim=hid_dim, gamma=5, opts=None)

    def forward(self, x, llm_emb=None):
        loss = self.loss_fn(self.denoise_fn_D, x, llm_emb=llm_emb)
        return loss.mean(-1).mean()