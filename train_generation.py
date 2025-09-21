
import functools

import time
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import open3d as o3d
from torch.utils.tensorboard import SummaryWriter

import argparse
import math
from tqdm import tqdm
from torch.distributions import Normal
from pprint import pprint

from utils.file_utils import *
from utils.visualize import *

from checkpoint import checkpoint
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from positional_encodings import *

import torch.distributed as dist

'''
some utils
'''
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotate(vertices, faces):
    '''
    vertices: [numpoints, 3]
    '''
    M = rotation_matrix([0, 1, 0], np.pi / 2).transpose()
    N = rotation_matrix([1, 0, 0], -np.pi / 4).transpose()
    K = rotation_matrix([0, 0, 1], np.pi).transpose()

    v, f = vertices[:,[1,2,0]].dot(M).dot(N).dot(K), faces[:,[1,2,0]]
    return v, f

def norm(v, f):
    v = (v - v.min())/(v.max() - v.min()) - 0.5

    return v, f

def getGradNorm(net):
    pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))

    gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()))
    return pNorm, gradNorm


def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and m.weight is not None:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_()
        m.bias.data.fill_(0)

'''
models
'''
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus)*1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min,  torch.ones_like(cdf_min)*1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
    x < 0.001, log_cdf_plus,
    torch.where(x > 0.999, log_one_minus_cdf_min,
             torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta)*1e-12))))
    assert log_probs.shape == x.shape
    return log_probs

class GaussianDiffusion:
    def __init__(self,betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type # eps
        self.model_var_type = model_var_type # fixesmall
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))


    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        x_t = root(alpha_tilde_t)*x_0 + root(1-alpha_tilde_t) * noise
        alpha_tilde_t = cum_product(alpha_t)
        alpha_t = 1 - beta_t
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, denoise_fn, data, t, pa, clip_denoised: bool, return_pred_xstart: bool):

        model_output = denoise_fn(data, t, pa)


        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -.5, .5)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.loss_type)


        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, pa, noise_fn, clip_denoised=False, return_pred_xstart=False):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, pa=pa, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample


    def p_sample_loop(self, denoise_fn, pa, shape, device,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            # t_ = (B,) = [t, t, t, t, ....B times]
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, pa=pa, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t.shape == shape
        return img_t

    def p_sample_loop_trajectory(self, denoise_fn, pa, shape, device, freq,
                                 noise_fn=torch.randn,clip_denoised=True, keep_running=False):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        assert isinstance(shape, (tuple, list))

        total_steps =  self.num_timesteps if not keep_running else len(self.betas)

        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [img_t]
        for t in reversed(range(0,total_steps)):

            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, pa=pa, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised,
                                  return_pred_xstart=False)
            if t % freq == 0 or t == total_steps-1:
                imgs.append(img_t)

        assert imgs[-1].shape == shape
        return imgs

    '''losses'''

    def _vb_terms_bpd(self, denoise_fn, data_start, data_t, t, pa, clip_denoised: bool, return_pred_xstart: bool):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=data_start, x_t=data_t, t=t)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, data=data_t, t=t, pa=pa, clip_denoised=clip_denoised, return_pred_xstart=True)
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = kl.mean(dim=list(range(1, len(data_start.shape)))) / np.log(2.)

        return (kl, pred_xstart) if return_pred_xstart else kl

    def p_losses(self, denoise_fn, data_start, t, pangles, noise=None):
        """
        Training loss calculation
        """
        B, D, N = data_start.shape
        assert t.shape == torch.Size([B])

        # noise is not None by default. Else: sample noise for each point and every point cloud
        if noise is None:
            noise = torch.randn(data_start.shape, dtype=data_start.dtype, device=data_start.device)

        assert noise.shape == data_start.shape and noise.dtype == data_start.dtype

        # Diffuse the data
        # data_t = diffused data at time t
        data_t = self.q_sample(x_start=data_start, t=t, noise=noise)

        if self.loss_type == 'mse':
 
            # predict the noise instead of x_start. seems to be weighted naturally like SNR
            eps_recon = denoise_fn(data_t, t, pangles)
            assert data_t.shape == data_start.shape
            assert eps_recon.shape == torch.Size([B, D, N])
            assert eps_recon.shape == data_start.shape

            eps_noise = True
            if eps_noise:
                losses = ((noise - eps_recon)**2).mean(dim=list(range(1, len(data_start.shape))))
            else:
                x_recon = self._predict_xstart_from_eps(data_t, t=t, eps=eps_recon) 
                losses = ((data_start - x_recon)**2).mean(dim=list(range(1, len(data_start.shape))))

        elif self.loss_type == 'kl':
            losses = self._vb_terms_bpd(
                denoise_fn=denoise_fn, data_start=data_start, data_t=data_t, t=t, pa=pangles,
                clip_denoised=False, return_pred_xstart=False)
        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == torch.Size([B])
        return losses

    '''debug'''

    def _prior_bpd(self, x_start):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(T-1)
            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance,
                                 mean2=torch.tensor([0.]).to(qt_mean), logvar2=torch.tensor([0.]).to(qt_log_variance))
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.)

    def calc_bpd_loop(self, denoise_fn, x_start, pa, clip_denoised=True):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps

            vals_bt_, mse_bt_= torch.zeros([B, T], device=x_start.device), torch.zeros([B, T], device=x_start.device)
            for t in reversed(range(T)):

                t_b = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(t)
                # Calculate VLB term at the current timestep
                new_vals_b, pred_xstart = self._vb_terms_bpd(
                    denoise_fn, data_start=x_start, data_t=self.q_sample(x_start=x_start, t=t_b), t=t_b,
                    pa=pa, clip_denoised=clip_denoised, return_pred_xstart=True)
                # MSE for progressive prediction loss
                assert pred_xstart.shape == x_start.shape
                new_mse_b = ((pred_xstart-x_start)**2).mean(dim=list(range(1, len(x_start.shape))))
                assert new_vals_b.shape == new_mse_b.shape ==  torch.Size([B])
                # Insert the calculated term into the tensor of all terms
                mask_bt = t_b[:, None]==torch.arange(T, device=t_b.device)[None, :].float()
                vals_bt_ = vals_bt_ * (~mask_bt) + new_vals_b[:, None] * mask_bt
                mse_bt_ = mse_bt_ * (~mask_bt) + new_mse_b[:, None] * mask_bt
                assert mask_bt.shape == vals_bt_.shape == vals_bt_.shape == torch.Size([B, T])

            prior_bpd_b = self._prior_bpd(x_start)
            total_bpd_b = vals_bt_.sum(dim=1) + prior_bpd_b
            assert vals_bt_.shape == mse_bt_.shape == torch.Size([B, T]) and \
                   total_bpd_b.shape == prior_bpd_b.shape ==  torch.Size([B])
            return total_bpd_b.mean(), vals_bt_.mean(), prior_bpd_b.mean(), mse_bt_.mean()


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3).to(device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width).to(device=device, dtype=dtype)
        self.attention = QKVMultiheadSelfAttention(device=device, dtype=dtype, heads=heads, n_ctx=n_ctx)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)
        return x


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads

        self.c_proj = nn.Linear(width, width).to(device=device, dtype=dtype)
        self.attention = QKVMultiheadCrossAttention(device=device, dtype=dtype, heads=heads, n_ctx=n_ctx)
        
        init_linear(self.c_proj, init_scale)

    def forward(self, x, cond_k, cond_v):
        x = checkpoint(self.attention, (x, cond_k, cond_v), (), True)
        x = self.c_proj(x)
        return x

class MLP(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, width: int, init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4).to(device=device, dtype=dtype) #, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width).to(device=device, dtype=dtype) #, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):

        o1 = self.c_fc(x)
        o2 = self.gelu(o1)
        o3 = self.c_proj(o2)

        return o3


class QKVMultiheadSelfAttention(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, heads: int, n_ctx: int):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        # B, N+82, embed_dim*3
        bs, n_ctx, width = qkv.shape
        
        # embed_dim//heads
        attn_ch = width // self.heads // 3
        
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        
        # B, N+82, heads, embed_dim*3//heads [8, 5082, 2, 24]
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        
        # q, k, v [8, 5082, 2, 8], [8, 5082, 2, 8], [8, 5082, 2, 8]
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        
        # B, heads, N+82, N+82 ==> 8, 2, 5082, 5082
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return_ele = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return return_ele


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, heads: int, n_ctx: int):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, q, k, v):
        # B, N+1, embed_dim ==> 8, 5001, 16
        bs, n_ctx, width = q.shape

        # B, num_units, embed_dim ==> 8, 81, 16
        bs, n_units, width = k.shape

        # embed_dim//heads ==> 16 // 2 = 8
        attn_ch = width // self.heads # // 3

        scale = 1 / math.sqrt(math.sqrt(attn_ch))

        # q, k, v: [8, 5001, 2, 8] [8, 81, 2, 8] [8, 81, 2, 8]
        q = q.view(bs, n_ctx, self.heads, -1)
        k = k.view(bs, n_units, self.heads, -1)
        v = v.view(bs, n_units, self.heads, -1)
 
        # B, heads, N+1, num_units//3(joints) ==> 8, 2, 5001, 27
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards

        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)

        # B, N+1, embed_dim ==> 8, 5001, 16
        return_ele = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return return_ele


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
    ):
        super().__init__()

        self.self_attn = MultiheadSelfAttention(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )

        self.cross_attn = MultiheadCrossAttention(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )

        self.ln_1 = nn.LayerNorm(width).to(device=device, dtype=dtype)
        self.ln_3 = nn.LayerNorm(width).to(device=device, dtype=dtype)
        self.mlp2 = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_4 = nn.LayerNorm(width).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, cond_k: torch.Tensor, cond_v: torch.Tensor):

        # Self Attention
        x = x + self.self_attn(self.ln_1(x))
        # x = x + self.mlp1(self.ln_2(x))

        # Cross Attention
        x = x + self.cross_attn(self.ln_3(x), cond_k, cond_v)
        x = x + self.mlp2(self.ln_4(x))

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
    ):
        super().__init__()
        self.n_ctx = n_ctx

        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, cond_k: torch.tensor, cond_v: torch.tensor):
        for block in self.resblocks:
            x = block(x, cond_k, cond_v)
        return x


class PointDiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        input_channels: int = 3,
        output_channels: int = 3,
        n_ctx: int = 1024,
        width: int = 512,
        layers: int = 4, # 12
        heads: int = 2, # 8
        init_scale: float = 0.25,
        time_token_cond: bool = False,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.time_token_cond = time_token_cond
        self.time_embed = MLP(
            device=device, dtype=dtype, width=width, init_scale=init_scale * math.sqrt(1.0 / width)
        )
        self.ln_pre = nn.LayerNorm(width).to(device=device, dtype=dtype) #, device=device, dtype=dtype)
        self.ln_pre_cond_k = nn.LayerNorm(width).to(device=device, dtype=dtype)
        self.ln_pre_cond_v = nn.LayerNorm(width).to(device=device, dtype=dtype)
        self.backbone = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + int(time_token_cond),
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_post = nn.LayerNorm(width).to(device=device, dtype=dtype) #, device=device, dtype=dtype)
        self.input_proj = nn.Linear(input_channels, width).to(device=device, dtype=dtype) #, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels).to(device=device, dtype=dtype) #, device=device, dtype=dtype)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def batched_custom_point_encoding(self, d_model, points):
        """
            Returns B, N, hidden_size 
        """

        assert d_model % 3 == 0       
 
        B, N, ch = points.shape
        
        points = points.reshape(B*N, ch)

        pe = torch.zeros(points.shape[0], d_model).to(points)
        
        d_model = int(d_model / 3)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model)).to(points)
        
        pos_h = points[:, 0].unsqueeze(1)
        pos_w = points[:, 1].unsqueeze(1)
        pos_l = points[:, 2].unsqueeze(1)
        
        pe[:, 0:d_model:2] = torch.sin(pos_h * div_term)
        pe[:, 1:d_model:2] = torch.cos(pos_h * div_term)
        pe[:, d_model:2*d_model:2] = torch.sin(pos_w * div_term)
        pe[:, d_model + 1:2*d_model:2] = torch.cos(pos_w * div_term)
        pe[:, d_model*2::2] = torch.sin(pos_l * div_term)
        pe[:, d_model*2 + 1::2] = torch.cos(pos_l * div_term)
        
        # Return the reshaped batched tensor
        return pe.reshape(B, N, -1)


    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :return: an [N x C' x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(self.timestep_embedding(t, self.backbone.width))
        return self._forward_with_cond(x, [(t_embed, self.time_token_cond)])

    def _forward_with_cond(
        self, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]]
    ) -> torch.Tensor:
        # x: B, C, N
        # cond_as_token: [((B, embed_dim), True), ((B, num_units, embed_dim), True)]
        
        # B, N, embed_dim
        h = self.input_proj(x.permute(0, 2, 1))  # NCL -> NLC

        for emb, as_token in cond_as_token:
            if not as_token:
                h = h + emb[:, None]

        # extra_tokens[0] ==> B, 1, embed_dim
        # extra_tokens[1] ==> B, 1, embed_dim
        # extra_tokens[2] ==> B, 1, embed_dim
        extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ]
        # concatenate the time embedding to the input
        # B, N+1, embed_dim
        if len(extra_tokens):
            h = torch.cat([extra_tokens[0]] + [h], dim=1)

        # B, N+1, embed_dim
        h = self.ln_pre(h)
        cond_k = self.ln_pre_cond_k(extra_tokens[1])
        cond_v = self.ln_pre_cond_v(extra_tokens[2])

        # B, N+1, embed_dim
        h = self.backbone(h, cond_k, cond_v)

        # B, N+1, embed_dim
        h = self.ln_post(h)

        # B, N, embed_dim
        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens[0:1]) :]

        # B, N, 3
        h = self.output_proj(h)

        return h.permute(0, 2, 1) # B, 3, N


class CLIPImageGridPointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 10000,
        input_channels: int = 3,
        output_channels: int = 3,
        embed_dim: int = 64,
        layers: int = 12,
        heads: int = 8,
        time_token_cond: bool = False,
        token_cond: bool = False,
        pose_format: str = 'joints',
        d_model: int = 6,
        input_encoding: int = 0,
    ):

        if pose_format == 'angles':
            num_units = 57
        elif pose_format == 'joints':
            num_units = 81
        else:
            raise NotImplementedError

        super().__init__(device=device, dtype=dtype, n_ctx=n_ctx, width=embed_dim, layers=layers, heads=heads,
                         time_token_cond=time_token_cond, input_channels=input_channels, output_channels=output_channels)
        self.n_ctx = n_ctx
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.input_encoding = input_encoding

        self.cond_proj_k1 = nn.Linear(self.d_model, self.embed_dim).to(device=device, dtype=dtype)
        self.cond_proj_v1 = nn.Linear(self.d_model, self.embed_dim).to(device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        pangles: torch.Tensor #Optional[Iterable[ImageType]] = None,
    ):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :param images: a batch of images to condition on.
        :param embeddings: a batch of CLIP latent grids to condition on.
        :return: an [N x C' x T] tensor.
        """
        # assert images is not None or embeddings is not None, "must specify images or embeddings"
        # assert images is None or embeddings is None, "cannot specify both images and embeddings"

        # B, C, N ==> typically 16, 3, 10000
        assert x.shape[-1] == self.n_ctx

        # B, embed_dim
        t_embed = self.timestep_embedding(t, self.embed_dim)
        t_embed = self.time_embed(t_embed)

        # B, 27, 3
        pangles = pangles.reshape(pangles.shape[0], -1, 3)

        d_model = self.d_model
        pangles = self.batched_custom_point_encoding(d_model, pangles)
        
        pangles_k = self.cond_proj_k1(pangles) # (B, 27, 3) --> (B, 27, 256)
        pangles_v = self.cond_proj_v1(pangles) # (B, 27, 3) --> (B, 27, 256)


        # Positionally encode the inputs as well
        if self.input_encoding:
            x = self.batched_custom_point_encoding(d_model, x.permute(0, 2, 1)).permute(0, 2, 1)

        cond = [(t_embed, self.time_token_cond), (pangles_k, True), (pangles_v, True)]
        return self._forward_with_cond(x, cond)


class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type: str):
        super(Model, self).__init__()
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type)

        self.model = CLIPImageGridPointDiffusionTransformer(device=torch.device('cuda'),dtype=torch.float32, 
                                                            n_ctx=args.npoints, embed_dim=args.embed_dim,
                                                            layers=args.layers, heads=args.heads,
                                                            input_channels=args.in_ch, output_channels=args.out_ch, 
                                                            time_token_cond=True, token_cond=True,
                                                            pose_format=args.pose_format, d_model=args.d_model,
                                                            input_encoding=args.input_encoding)
        

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, pa, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt =  self.diffusion.calc_bpd_loop(self._denoise, x0, pa, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt':mse_bt
        }

    def get_timestep_embedding(self, timesteps, device):
        assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)

        # emb.shape = [B, 64]
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb

    def _denoise(self, data, t, pangles):
        """
            data: diffused data at time step t
            t: time steps
        """
        B, D, N = data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t, pangles)

        assert out.shape == torch.Size([B, D, N])
        return out

    def get_loss_iter(self, data, pangles, noises=None):

        B, D, N = data.shape

        #if weighted_sampling:
        #    # print ("using weighted time sampling")
        #    # linear weighted random sampling
        #    # weights = np.linspace(0, 1, self.diffusion.num_timesteps, endpoint=False)
        #    # weights = np.flip(weights)
        #    # weighted as 0 --> 5 * p(1000)
        #    weights = np.linspace(start=1.0, stop=5.0, num=self.diffusion.num_timesteps, endpoint=True) ** (-1) 
        #    weights /= weights.sum()
        #    weights = np.ascontiguousarray(weights)
        #    weights = torch.from_numpy(weights).float().to(data.device)
        #    t = torch.multinomial(weights, B, replacement=True).to(data.device)

        # sample 1 timestep from num_of_diffusion_steps for each point cloud
        # else:
        # print ("using UNweighted time sampling")
        # t = (B,)
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, pangles=pangles, noise=noises)
        assert losses.shape == t.shape == torch.Size([B])

        return losses

    def gen_samples(self, shape, pa, device, noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False):
        return self.diffusion.p_sample_loop(self._denoise, pa=pa, shape=shape, device=device, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running)

    def gen_sample_traj(self, shape, pa, device, freq, noise_fn=torch.randn,
                    clip_denoised=True,keep_running=False):
        return self.diffusion.p_sample_loop_trajectory(self._denoise, pa=pa, shape=shape, device=device, noise_fn=noise_fn, 
                                                       freq=freq,
                                                       clip_denoised=clip_denoised,
                                                       keep_running=keep_running)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)


def get_poses(pose_lines, pose_format, category):

    out_pa = []
    for pl in pose_lines:
        da = pl.split(" ")
        da_new = list(filter(None, da))
        da_new_np = np.array(da_new, dtype=np.float32)
        out_pa.append(da_new_np[np.newaxis, ...])

    out_pa = torch.from_numpy(np.concatenate(out_pa)[:, 1:]).float()

    if pose_format == 'joints':
        if category == "Subject001":
            out_pa = out_pa.reshape((-1, 27, 3)) - out_pa.reshape((-1, 27, 3))[:,14:15,:]
            out_pa = out_pa.reshape((-1, 81))
        elif category == "Subject002":
            out_pa = out_pa.reshape((-1, 23, 3)) - out_pa.reshape((-1, 23, 3))[:,3:4,:]
            out_pa = out_pa.reshape((-1, 69))

    return out_pa 


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)

    elif schedule_type == 'warm0.1':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    elif schedule_type == 'warm0.2':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    elif schedule_type == 'warm0.5':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    elif schedule_type == 'cosine':
        betas = betas_for_alpha_bar(
            time_num,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    elif schedule_type == 'quadratic':
        betas = np.linspace(b_start ** 0.5, b_end ** 0.5, time_num, dtype=np.float64) ** 2

    elif schedule_type == 'cubic':
        betas = np.linspace(b_start ** (1./3), b_end ** (1./3), time_num, dtype=np.float64) ** 3

    elif schedule_type == 'cubic_warm0.9':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.9)
        betas[:warmup_time] = np.linspace(b_start ** (1./3), b_end ** (1./3), warmup_time, dtype=np.float64) ** 3

    elif schedule_type == 'four':
        betas = np.linspace(b_start ** 0.25, b_end ** 0.25, time_num, dtype=np.float64) ** 4

    else:
        raise NotImplementedError(schedule_type)
    return betas


def collate_fn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)


def get_dataset(args, output_dir, outf_syn):

    if args.category in ['Subject002', 'Subject001',  'Subject003', 'Subject004']: 
        from datasets.humans_data import ShapeNet15kPointClouds, VizData
    else:
        from datasets.resynth_data import ShapeNet15kPointClouds, VizData

    tr_dataset = ShapeNet15kPointClouds(root_dir=args.dataroot,
        categories=[args.category], split='train',
        tr_sample_size=args.npoints,
        te_sample_size=args.npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        normalize_trans=args.normtrans,
        normalize_az=args.normaz,
        random_subsample=True, ntrain=args.ntrain,
        pose_format=args.pose_format,
        output_dir=output_dir)

    viz_dataset = VizData(opt=args,
        outf_syn=outf_syn, norm_az=args.normaz)

    return tr_dataset, viz_dataset


def get_dataloader(opt, train_dataset, test_dataset=None):

    if opt.distribution_type == 'multi':
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=opt.rank
        )
        if test_dataset is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=opt.world_size,
                rank=opt.rank
            )
        else:
            test_sampler = None
    else:
        train_sampler = None
        test_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=train_sampler,
                                                   shuffle=train_sampler is None, num_workers=int(opt.workers), drop_last=True)

    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.bs,sampler=test_sampler,
                                                   shuffle=False, num_workers=int(opt.workers), drop_last=False)
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader, train_sampler, test_sampler


def train(gpu, opt, output_dir, noises_init):

    set_seed(opt)
    logger = setup_logging(output_dir)

    should_diag = True
    if should_diag:
        outf_syn, = setup_output_subdirs(output_dir, 'syn' + '_' + opt.category)

    if opt.distribution_type == 'multi':
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])

        base_rank =  opt.rank * opt.ngpus_per_node
        opt.rank = base_rank + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

        opt.bs = int(opt.bs / opt.ngpus_per_node)
        opt.workers = 0


    ''' data '''

    num_joints = {'Subject001': 23, 'Subject002': 23, 'Subject003': 23, 'Subject004': 23}
    opt.njoints = num_joints[opt.category]


    train_dataset, viz_dataset = get_dataset(opt, output_dir, outf_syn)
    dataloader, viz_dataloader, train_sampler, viz_sampler = get_dataloader(opt, train_dataset, viz_dataset)

    if opt.input_encoding:
        opt.in_ch = opt.d_model

    '''
    create networks
    '''

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)

    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    if opt.distribution_type == 'multi':  # Multiple processes, single GPU per process
        print ("multi ", gpu)
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model.multi_gpu_wrapper(_transform_)


    elif opt.distribution_type == 'single':
        def _transform_(m):
            return nn.parallel.DataParallel(m)
        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)

    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError('distribution_type = multi | single | None')

    if should_diag:
        logger.info(opt)

    if opt.optimiz == 0:
        print ("Optimizer: Using Adam")
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.decay, betas=(opt.beta1, 0.999))
    elif opt.optimiz == 1:
        print ("Optimizer: Using AdamW")
        optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.decay, betas=(opt.beta1, 0.999))
    else:
        raise NotImplementedError

    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_gamma)

    if opt.model != '' and opt.job_id and opt.task_id:# and gpu == 0:
       print ("opt.model ", os.path.join(opt.model, opt.job_id))
       if opt.restore_id is None or int(opt.task_id) > 1:
          opt.restore_id = opt.job_id 
       opt.model = natsorted(glob.glob(os.path.join(opt.model, opt.restore_id) + '/*.pth'))    
       print ("opt.model ", opt.model)

       if len(opt.model) > 0:
           opt.model = opt.model[-1]
       else:
           opt.model = ''
       print ("opt.model ", opt.model)
    elif opt.model != '' and opt.restore_id:
        opt.model = natsorted(glob.glob(os.path.join(opt.model, opt.restore_id) + '/*.pth'))
        if len(opt.model) > 0:
           opt.model = opt.model[-1]
        else:
           opt.model = ''
        print ("opt.model ", opt.model)
    else:
        opt.model = ''

    if opt.model != '':
        ckpt = torch.load(opt.model)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])

    if opt.model != '':
        start_epoch = torch.load(opt.model)['epoch'] + 1
    else:
        start_epoch = 0

    def new_x_chain(x, num_chain):
        return torch.randn(num_chain, *x.shape[1:], device=x.device)


    s, e, i = opt.ntrain

    weighted_sampling = bool(opt.wt_smpl)
    if weighted_sampling:
        print ("using weighted time sampling")
    else:
        print ("using UNweighted time sampling")
    
    writer = SummaryWriter(log_dir=output_dir)

    for epoch in tqdm(range(start_epoch, opt.niter + 1)):

        if opt.distribution_type == 'multi':
            train_sampler.set_epoch(epoch)

        lr_scheduler.step(epoch)

        for i, data in enumerate(dataloader):

            x = data['train_points'].transpose(1,2)
            pa = data['train_pose']
            mean = data['mean']
            std = data['std']
            
            if opt.train_normals and opt.train_colors:
                raise NotImplementedError

            if opt.train_normals:
                normals = data['train_normals'].transpose(1, 2)
                x = torch.cat([x, normals], dim=1)
            elif opt.train_colors:
                colors = data['train_colors'].transpose(1, 2)
                x = torch.cat([x, colors], dim=1)

            '''
            train diffusion
            '''

            if opt.distribution_type == 'multi' or (opt.distribution_type is None and gpu is not None):
                x = x.cuda(gpu)
                pa = pa.cuda(gpu)

            elif opt.distribution_type == 'single':
                x = x.cuda()
                pa = pa.cuda()


            loss = model.get_loss_iter(x, pa, None).mean()

            optimizer.zero_grad()

            loss.backward()

 
            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

            optimizer.step()

            if i % opt.print_freq == 0 and should_diag:

                logger.info('[{:>3d}/{:>3d}][{:>3d}/{:>3d}]    loss: {:>10.4f},    '
                             .format(
                        epoch, opt.niter, i, len(dataloader),loss.item(),
                        ))

        netpNorm, netgradNorm = getGradNorm(model)
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("netpNorm/train", netpNorm, epoch)
        writer.add_scalar("netgradNorm/train", netgradNorm, epoch)

        if epoch % opt.diagIter == 0 and should_diag and gpu == 0:

            logger.info('Diagnosis:')

            x_range = [x.min().item(), x.max().item()]
            kl_stats = model.all_kl(x, pa)
            logger.info('      [{:>3d}/{:>3d}]    '
                         'x_range: [{:>10.4f}, {:>10.4f}],   '
                         'total_bpd_b: {:>10.4f},    '
                         'terms_bpd: {:>10.4f},  '
                         'prior_bpd_b: {:>10.4f}    '
                         'mse_bt: {:>10.4f}  '
                .format(
                epoch, opt.niter,
                *x_range,
                kl_stats['total_bpd_b'].item(),
                kl_stats['terms_bpd'].item(), kl_stats['prior_bpd_b'].item(), kl_stats['mse_bt'].item()
            ))

            writer.add_scalar("x_min/train", x_range[0], epoch*len(dataloader))
            writer.add_scalar("x_max/train", x_range[1], epoch*len(dataloader))


        if epoch % opt.saveIter == 0:

            if should_diag:


                save_dict = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }

                torch.save(save_dict, '%s/epoch_%d.pth' % (output_dir, epoch))


            if opt.distribution_type == 'multi':
                dist.barrier()
                map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
                model.load_state_dict(
                    torch.load('%s/epoch_%d.pth' % (output_dir, epoch), map_location=map_location)['model_state'])


        if epoch % opt.vizIter == 0 and should_diag and epoch != 0:
            logger.info('Generation: eval')

            model.eval()
            with torch.no_grad():

                if opt.pose_format == 'joints' and opt.normtrans:
                    raise NotImplementedError

                if epoch % opt.vizIter == 0:

                    if opt.distribution_type == 'multi':
                        viz_sampler.set_epoch(epoch)

                    for ii, ddata in enumerate(viz_dataloader):

                        tr_out_pose = ddata['train_pose']
                        te_out_pose = ddata['test_pose']
                        tr_joint_poses = ddata['train_joints']
                        te_joint_poses = ddata['test_joints']

                        mean_tr = mean[0:1].expand(tr_out_pose.shape[0], -1, -1).cpu().numpy()
                        mean_te = mean[0:1].expand(te_out_pose.shape[0], -1, -1).cpu().numpy()
                        std_tr = std[0:1].expand(tr_out_pose.shape[0], -1, -1).cpu().numpy()
                        std_te = std[0:1].expand(te_out_pose.shape[0], -1, -1).cpu().numpy()

                        if opt.pose_format == 'joints':

                            tr_out_pose = (tr_out_pose - mean_tr) / std_tr
                            te_out_pose = (te_out_pose - mean_te) / std_te

                            tr_out_pose = tr_out_pose.reshape(-1, opt.njoints*3)
                            te_out_pose = te_out_pose.reshape(-1, opt.njoints*3)

                        tr_out_pose = tr_out_pose.float()
                        te_out_pose = te_out_pose.float()
                            
                        
                        if opt.distribution_type == 'multi' or (opt.distribution_type is None and gpu is not None):
                            tr_out_pose = tr_out_pose.cuda(gpu)
                            te_out_pose = te_out_pose.cuda(gpu)
                        elif opt.distribution_type == 'single':
                            tr_out_pose = tr_out_pose.cuda()
                            te_out_pose = te_out_pose.cuda()

                        x_gen_eval = model.gen_samples(new_x_chain(x, tr_out_pose.shape[0]).shape, tr_out_pose, x.device, clip_denoised=False)
                        x_gen_test = model.gen_samples(new_x_chain(x, te_out_pose.shape[0]).shape, te_out_pose, x.device, clip_denoised=False)

                        x_gen_list = model.gen_sample_traj(new_x_chain(x, 1).shape, tr_out_pose[0:1], x.device, freq=40, clip_denoised=False)
                        x_gen_all = torch.cat(x_gen_list, dim=0)

                        gen_stats = [x_gen_eval.mean(), x_gen_eval.std()]
                        gen_eval_range = [x_gen_eval.min().item(), x_gen_eval.max().item()]

                        logger.info('      [{:>3d}/{:>3d}]  '
                                     'eval_gen_range: [{:>10.4f}, {:>10.4f}]     '
                                     'eval_gen_stats: [mean={:>10.4f}, std={:>10.4f}]      '
                            .format(
                            epoch, opt.niter,
                            *gen_eval_range, *gen_stats,
                        ))

                        x_gen_eval = x_gen_eval.transpose(1, 2)
                        x_gen_test = x_gen_test.transpose(1, 2)

                        writer.add_scalar("x_gen_eval_min/eval", gen_eval_range[0], epoch*len(dataloader))
                        writer.add_scalar("x_gen_eval_max/eval", gen_eval_range[1], epoch*len(dataloader))

                        if opt.train_normals:
                            x_gen_eval_save = torch.cat([x_gen_eval[:,:,:3].cpu()*std_tr+mean_tr, x_gen_eval[:,:,3:].cpu()], dim=2).numpy()
                            x_gen_test_save = torch.cat([x_gen_test[:,:,:3].cpu()*std_te+mean_te, x_gen_test[:,:,3:].cpu()], dim=2).numpy()
                        elif opt.train_colors:
                            x_gen_eval_save = torch.cat([x_gen_eval[:,:,:3].cpu()*std_tr+mean_tr, (x_gen_eval[:,:,3:].cpu()+1.0) / 2.0], dim=2).numpy()
                            x_gen_test_save = torch.cat([x_gen_test[:,:,:3].cpu()*std_te+mean_te, (x_gen_test[:,:,3:].cpu()+1.0) / 2.0], dim=2).numpy()
                        else:
                            x_gen_eval_save = (x_gen_eval.cpu()*std_tr+mean_tr).numpy()
                            x_gen_test_save = (x_gen_test.cpu()*std_te+mean_te).numpy()

                        save_obj(x_gen_eval_save, outf_syn, epoch, "train_"+str(ii), tr_joint_poses.numpy(), opt.train_colors, opt.train_normals)
                        save_obj(x_gen_test_save, outf_syn, epoch, "test_"+str(ii), te_joint_poses.numpy(), opt.train_colors, opt.train_normals)

                        visualize_pointcloud_batch('%s/epoch_%03d_samples_eval.png' % (outf_syn, epoch),
                                                   x_gen_eval, None, None,
                                                   None)

                        visualize_pointcloud_batch('%s/epoch_%03d_samples_eval_all.png' % (outf_syn, epoch),
                                                   x_gen_all.transpose(1, 2), None,
                                                   None,
                                                   None)

                        visualize_pointcloud_batch('%s/epoch_%03d_x.png' % (outf_syn, epoch), x.transpose(1, 2), None,
                                                   None,
                                                   None)


            logger.info('Generation: train')
            model.train()

    writer.flush()

    dist.destroy_process_group()


def main():
    opt = parse_args()
    if opt.category == 'airplane':
        opt.beta_start = 1e-5
        opt.beta_end = 0.008
        opt.schedule_type = 'warm0.1'

    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    dir_id = os.path.dirname(__file__)

    if opt.dir_id:
        dir_id = opt.dir_id
    else:
        dir_id = os.path.dirname(__file__)

    if opt.job_id is None:
        output_dir = get_output_dir(dir_id, exp_id)
    else:
        output_dir = os.path.join(dir_id, 'output/' + exp_id, opt.job_id) 
        os.makedirs(output_dir, exist_ok=True)

    copy_source(__file__, output_dir)


    ''' workaround '''
    if opt.ntrain is not None:
        opt.ntrain = [int(ele) for ele in opt.ntrain.split(":")]

    len_train = (opt.ntrain[1] - opt.ntrain[0] - 1) // opt.ntrain[2] + 1

    if opt.init_noise is not None:
        noises_init = torch.from_numpy(np.load(opt.init_noise)).float()
        noises_init = noises_init[:len_train]
    else:
        noises_init = torch.randn(len_train, opt.npoints, opt.in_ch)


    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    if opt.distribution_type == 'multi':
        opt.ngpus_per_node = torch.cuda.device_count()
        print ("ngpus_per_node ", opt.ngpus_per_node)
        opt.world_size = opt.ngpus_per_node * opt.world_size
        print ("world size ", opt.world_size)
        mp.spawn(train, nprocs=opt.ngpus_per_node, args=(opt, output_dir, noises_init))
    else:
        train(opt.gpu, opt, output_dir, noises_init)



def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='ShapeNetCore.v2.PC15k/')
    parser.add_argument('--category', default='Subject001')
    parser.add_argument('--dir_id', default=None)
    parser.add_argument('--job_id', default=None)
    parser.add_argument('--task_id', default=None)
    parser.add_argument('--restore_id', default=None)
    parser.add_argument('--init_noise', default=None)

    parser.add_argument('--bs', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, default=8, help='workers')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')

    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', type=int, default=2048)
    parser.add_argument('--ntrain', default=None)
    parser.add_argument('--train_normals', type=int, default=0)
    parser.add_argument('--train_colors', type=int, default=0)
    parser.add_argument('--normtrans', default=0, help='normalize with translation of pc')
    parser.add_argument('--normaz', type=int, default=0, help='normalize with translation of pc')
    parser.add_argument('--input_encoding', type=int, default=0, help='to encode input into a positonal encoding')
    
    parser.add_argument('--wt_smpl', type=int, default=0, help='weighted time step sampling/penalty')

    '''model'''
    parser.add_argument('--pv_blocks', type=str, default="4")
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', type=int, default=1000)

    #params
    parser.add_argument('--attention', default=True)
    parser.add_argument('--pose_format', default='angles')
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=6)
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=3)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')

    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--optimiz', type=int, default=0, help='optimizer: 0 adam, 1 adamw')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--decay', type=float, default=0.01, help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float, default=None, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float, default=1.0, help='lr decay for EBM')

    parser.add_argument('--model', default='', help="path to model (to continue training)")


    '''distributed'''
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distribution_type', default='single', choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    '''eval'''
    parser.add_argument('--saveIter', type=int, default=50, help='unit: epoch')
    parser.add_argument('--diagIter', type=int, default=50, help='unit: epoch')
    parser.add_argument('--vizIter', type=int, default=25, help='unit: epoch')
    parser.add_argument('--print_freq', type=int, default=50, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')


    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    print ("Launching.....")
    main()
