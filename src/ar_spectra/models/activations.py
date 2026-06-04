import torch
from torch import nn

def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 0.000000001)) * (torch.sin(x * alpha) ** 2)

class SnakeBeta(nn.Module):
    """
    Snake activation with trainable beta parameter.
    Adapted from https://github.com/NVIDIA/BigVGAN/blob/main/activations.py
    """
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x):
        # Line up with x to [B, C, T] or [B, C, H, W]
        alpha = self.alpha.unsqueeze(0)
        beta = self.beta.unsqueeze(0)
        
        while alpha.ndim < x.ndim:
            alpha = alpha.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
            
        return snake_beta(x, alpha, beta)
