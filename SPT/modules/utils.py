#%% 
"""
Reference:
[1] https://github.com/amazon-science/tabsyn/blob/main/tabsyn/latent_utils.py
[2] https://github.com/amazon-science/tabsyn/blob/main/tabsyn/diffusion_utils.py
[3] https://github.com/amazon-science/tabsyn/blob/main/tabsyn/sample.py
"""

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""
import importlib
import torch
import numpy as np
import random

import pandas as pd
from modules.model1 import Decoder_model 

# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)   

randn_like=torch.randn_like

SIGMA_MIN=0.002
SIGMA_MAX=80
rho=7
S_churn= 1
S_min=0
S_max=float('inf')
S_noise=1

def recover_data(syn_num, syn_cat, syn_target, info):
    raw_df = np.concatenate([syn_num,syn_cat,syn_target],axis=1)
    num_col_idx = info['num_col_idx']
    cat_col_idx = [col for col in info['cat_col_idx'] if col not in info['target_col_idx']]
    target_col_idx = info['target_col_idx']

    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}
    syn_df = pd.DataFrame()

    for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
        if i in set(num_col_idx):
            syn_df[i] = raw_df[:, idx_mapping[i]]
        elif i in set(cat_col_idx):
            syn_df[i] = raw_df[:, idx_mapping[i]]
        else:
            syn_df[i] = raw_df[:,idx_mapping[i]]

    return syn_df

# 기존 noise_sample
# def noise_sample(net, num_samples, dim, noise_coef, num_steps = 50, device = 'cuda:0'):
#     # 'noise_coef' is set to a value greater than 0 when adding noise in latent space. 
#     latents = torch.randn([num_samples, dim], device=device) * (1+noise_coef)

#     step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    
#     sigma_min = max(SIGMA_MIN, net.sigma_min) 
#     sigma_max = min(SIGMA_MAX, net.sigma_max)

#     t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
#                 sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
#     t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

#     x_next = latents.to(torch.float32) * t_steps[0]

#     with torch.no_grad():
#         for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
#             x_next = sample_step(net, num_steps, i, t_cur, t_next, x_next)

#     return x_next

# 새 noise_sample
def noise_sample(net, num_samples, dim, noise_coef, num_steps = 50, device = 'cuda:0', llm_emb=None):
    # 'noise_coef' is set to a value greater than 0 when adding noise in latent space. 
    latents = torch.randn([num_samples, dim], device=device) * (1+noise_coef)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    
    sigma_min = max(SIGMA_MIN, net.sigma_min) 
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float32) * t_steps[0]

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_next = sample_step(net, num_steps, i, t_cur, t_next, x_next, llm_emb=llm_emb)

    return x_next

# 기존 sample_step
# def sample_step(net, num_steps, i, t_cur, t_next, x_next):

#     x_cur = x_next
#     # Increase noise temporarily.
#     gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
#     t_hat = net.round_sigma(t_cur + gamma * t_cur) 
#     x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
#     # Euler step.

#     denoised = net(x_hat, t_hat).to(torch.float32)
#     d_cur = (x_hat - denoised) / t_hat
#     x_next = x_hat + (t_next - t_hat) * d_cur

#     # Apply 2nd order correction.
#     if i < num_steps - 1:
#         denoised = net(x_next, t_next).to(torch.float32)
#         d_prime = (x_next - denoised) / t_next
#         x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

#     return x_next

# 새 sample_step
def sample_step(net, num_steps, i, t_cur, t_next, x_next, llm_emb=None):

    x_cur = x_next
    # Increase noise temporarily.
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur) 
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
    
    # Euler step.
    denoised = net(x_hat, t_hat, llm_emb=llm_emb).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction.
    if i < num_steps - 1:
        denoised = net(x_next, t_next, llm_emb=llm_emb).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Implementations of loss function are adapted from the version used in TabSyn.
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, denosie_fn, data, labels, augment_pipe=None):
        rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
        n = torch.randn_like(y) * sigma
        D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()
    
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100, D=128, N=3072, opts=None):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.D = D
        self.N = N
        print(f"In VE loss: D:{self.D}, N:{self.N}")

    def __call__(self, denosie_fn, data, labels = None, augment_pipe=None, stf=False, pfgmpp=False, ref_data=None):
        if pfgmpp:

            # N, 
            rnd_uniform = torch.rand(data.shape[0], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)

            r = sigma.double() * np.sqrt(self.D).astype(np.float64)
            # Sampling form inverse-beta distribution
            samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2.,
                                          size=data.shape[0]).astype(np.double)

            samples_norm = np.clip(samples_norm, 1e-3, 1-1e-3)

            inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
            inverse_beta = torch.from_numpy(inverse_beta).to(data.device).double()
            # Sampling from p_r(R) by change-of-variable
            samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
            samples_norm = samples_norm.view(len(samples_norm), -1)
            # Uniformly sample the angle direction
            gaussian = torch.randn(data.shape[0], self.N).to(samples_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
            # Construct the perturbation for x
            perturbation_x = unit_gaussian * samples_norm
            perturbation_x = perturbation_x.float()

            sigma = sigma.reshape((len(sigma), 1, 1, 1))
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = perturbation_x.view_as(y)
            D_yn = denosie_fn(y + n, sigma, labels,  augment_labels=augment_labels)
        else:
            rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = torch.randn_like(y) * sigma
            D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)

        loss = weight * ((D_yn - y) ** 2)
        return loss

# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

# 기존 EDMLoss
# class EDMLoss:
#     def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, hid_dim = 100, gamma=5, opts=None):
#         self.P_mean = P_mean
#         self.P_std = P_std
#         self.sigma_data = sigma_data
#         self.hid_dim = hid_dim
#         self.gamma = gamma
#         self.opts = opts


#     def __call__(self, denoise_fn, data):

#         rnd_normal = torch.randn(data.shape[0], device=data.device)
#         sigma = (rnd_normal * self.P_std + self.P_mean).exp()

#         weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

#         y = data
#         n = torch.randn_like(y) * sigma.unsqueeze(1)
#         D_yn = denoise_fn(y + n, sigma)
    
#         target = y
#         loss = weight.unsqueeze(1) * ((D_yn - target) ** 2)

#         return loss

# 새 EDMLoss
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, hid_dim = 100, gamma=5, opts=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hid_dim = hid_dim
        self.gamma = gamma
        self.opts = opts


    def __call__(self, denoise_fn, data, llm_emb=None):

        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)
        D_yn = denoise_fn(y + n, sigma, llm_emb=llm_emb)
    
        target = y
        loss = weight.unsqueeze(1) * ((D_yn - target) ** 2)

        return loss



@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse, config, device):
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    n_num_feat = len(num_col_idx)
    print("number of numerical features",n_num_feat)

    model_module = importlib.import_module('modules.model1')
    importlib.reload(model_module)
    ### MODEL INIT
    vae = model_module.Model_VAE(
        num_layers = config['num_layers'],  
        d_numerical = config["d_numerical"], 
        categories = config['categories'], 
        d_token = config['d_token'],
        var = config['var'],
        factor=config["factor"],
        n_head = config["n_head"],
        bias = config['bias']).to(device) 
    pre_decoder = Decoder_model(
        num_layers=config["num_layers"],
        d_numerical=config["d_numerical"],
        categories=config["categories"],
        d_token=config["d_token"],
        n_head=config["n_head"],
        factor=config["factor"]
    )
    vae.load_state_dict(torch.load(info["model_dir"], map_location=device))
    vae.eval()
    pre_decoder.load_weights(vae)
    pre_decoder.eval()

    syn_data = syn_data.reshape(syn_data.shape[0], -1, config['d_token'])
    norm_input = pre_decoder(torch.tensor(syn_data))
    x_hat_num, x_hat_cat = norm_input
    
    syn_cat = []
    for pred in x_hat_cat:
        syn_cat.append(pred.argmax(dim = -1))

    syn_num = x_hat_num.cpu().detach().numpy()
    syn_cat = torch.stack(syn_cat).t().cpu().numpy()
    
    syn_num = num_inverse(syn_num)
    syn_cat = cat_inverse(syn_cat)
    syn_target = syn_cat[:, len(cat_col_idx)-1:]
    syn_cat = syn_cat[:, :len(cat_col_idx)-1]

    return syn_num, syn_cat, syn_target

def process_invalid_id(syn_cat, min_cat, max_cat):
    syn_cat = np.clip(syn_cat, min_cat, max_cat)

    return syn_cat
    