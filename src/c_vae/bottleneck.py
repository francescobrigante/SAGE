# ===================================================================================
# Complex Gaussian VAE bottleneck: reparameterization trick and KL divergence
# 
# It implements improper complex normal posterior CN(μ, diag(σ), diag(c)) 
#   with μ mean, diag(σ) variance, diag(c) pseudo-variance
#   following Nakashika (Interspeech 2020).
# Optional flag to compute it as Cholesky decomposition enforcing additional constraints.
# ===================================================================================

import torch
from ar_spectra.models.bottlenecks import VAEBottleneck

SMALL_EPSILON: float = 1e-8  # numerical floor: guards log(0) in early training only


def reparametrize(mu: torch.Tensor, sigma: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Sampling coefficients:
        K_R = (σ + c) / √(2(σ + c_R))  =  l₁₁ + i·l₂₁      (complex)
        K_I = sqrt[ (σ²-|c|²) / (2(σ+c_R)) ] =  l₂₂             (real, positive)

    Sampling formula (Nakashika 2020):
        z = μ + K_R ⊙ ε_R + i·K_I ⊙ ε_I,    ε_R, ε_I ~ N(0, I_m) i.i.d. real

    Note: σ + c_R > 0 is guaranteed when σ > |c|
    The .clamp(min=SMALL_EPSILON) guards against NaN near the constraint boundary only.

    Args:
        mu:    complex mean, shape (B, m, ...)
        sigma: real variance, shape (B, m, ...), must satisfy σ > 0
        c:     complex pseudo-variance, shape (B, m, ...), ideally |c| < σ

    Returns:
        z: sampled latent, shape (B, m, ...) complex
    """
    # σ + c_R denominator clamped to avoid numerical issues
    denominator = (sigma + c.real).clamp(min=SMALL_EPSILON)

    K_R = (sigma + c) / (2.0 * denominator).sqrt()

    # clamp to avoid NaN results
    K_I = ((sigma**2 - c.abs()**2).clamp(min=SMALL_EPSILON) / (2.0 * denominator)).sqrt()

    eps_R = torch.randn_like(sigma)
    eps_I = torch.randn_like(sigma)

    return mu + K_R * eps_R + 1j * (K_I * eps_I)


def get_kl(mu: torch.Tensor, sigma: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Closed-form KL divergence (Nakashika 2020):
        D_KL = μ^H μ  +  Σ_k [ σ_k - 1 - 1/2*log(σ_k² - |c_k|²) ]

    Returns:
        KL divergence scalar (summed over m, averaged over batch and spatial dims)
    """

    # μ^H μ = |μ|² elementwise
    mu_x2 = mu.real**2 + mu.imag**2

    # log determinant log(σ² - |c|²) with clamp
    log_det = torch.log((sigma**2 - c.abs()**2).clamp(min=SMALL_EPSILON))

    kl_per_dim = mu_x2 + sigma - 1.0 - 0.5 * log_det
    return kl_per_dim.sum(dim=1).mean()


def get_cholesky_kl(mu: torch.Tensor, l11: torch.Tensor, l21: torch.Tensor, l22: torch.Tensor) -> torch.Tensor:
    """
    KL divergence in Cholesky factor space, used in Mode B (apply_cholesky_constraints=True).
    Substituting σ = l₁₁² + l₂₁² + l₂₂² and σ² - |c|² = 4·l₁₁²·l₂₂² into get_kl gives:
        D_KL = ‖μ‖²  +  Σ_k [ l₁₁² + l₂₁² + l₂₂² - 1 - log(2·l₁₁·l₂₂) ]

    Structurally analogous to the real-VAE KL = ½(σ² + μ² - 1 - log σ²).
    Prior corresponds to l₁₁ = l₂₂ = 1/√2, l₂₁ = 0, μ = 0 → KL=0.

    Because l₁₁, l₂₂ > 0 is enforced by exp(·) in the encoder, σ²-|c|² = 4l₁₁²l₂₂² > 0
    so no clamp needed, instead + SMALL_EPSILON is added for numerical stability.

    Args:
        mu:  complex mean
        l11: first diagonal Cholesky factor = Re(K_R), must be > 0
        l21: off-diagonal Cholesky factor = Im(K_R), unconstrained
        l22: second diagonal Cholesky factor = K_I, must be > 0

    Returns:
        KL divergence scalar
    """
    mu_x2 = mu.real**2 + mu.imag**2

    kl_per_dim = l11**2 + l21**2 + l22**2 - 1.0 - torch.log(2.0 * l11 * l22 + SMALL_EPSILON)
    return (mu_x2 + kl_per_dim).sum(dim=1).mean()


class ComplexVAEBottleneck(VAEBottleneck):
    """
    Expects a complex tensor with 3 x latent_channels along dim=1.
    Works in two modes:

        A) apply_cholesky_constraints = False:

            Predicted parameters: μ (complex), log_σ (used only Re), c (complex)
            The constraint σ > |c| is not explicitly enforced: KL term -1/2*log(σ²-|c|²)
            acts as a natural barrier function (→ +∞ as |c| → σ).

        B) apply_cholesky_constraints = True:

            Predicted parameters: μ (complex), log(l₁₁), l₂₁, log(l₂₂)
            σ > |c| is guaranteed by construction whenever l₁₁,l₂₂ > 0,
            which is ensured by l₁₁ = exp(·) > 0 and l₂₂ = exp(·) > 0.
    """

    # Encoder must output 3 channels per latent channel: [mu | slot2 | slot3]
    encoder_channel_multiplier: int = 3

    def __init__(self, apply_cholesky_constraints: bool = False):
        super().__init__()
        self.apply_cholesky_constraints = apply_cholesky_constraints

    def encode(self, x: torch.Tensor, return_info: bool = False, **kwargs) -> tuple[torch.Tensor, dict] | torch.Tensor:

        assert torch.is_complex(x), "ComplexVAEBottleneck expects a complex-valued input tensor"
        assert x.shape[1] % 3 == 0, f"ComplexVAEBottleneck expects channels divisible by 3 [μ|slot2|slot3], got {x.shape[1]}"
        
        m = x.shape[1] // 3
        mu = x[:, :m]
        slot_2 = x[:, m:2*m]
        slot_3 = x[:, 2*m:]

        # Mode A
        if not self.apply_cholesky_constraints:
            # exp slot2 which is log_variance and get real part
            sigma = torch.exp(slot_2.real)
            c = slot_3

            kl = get_kl(mu, sigma, c)

        # Mode B
        else:
            # Decode Cholesky factors from the encoder slots (log-parameterized diagonals):
            l11 = torch.exp(slot_2.real) # > 0
            l21 = slot_2.imag            # unconstrained
            l22 = torch.exp(slot_3.real) # > 0

            # Recover (σ, c) from Cholesky factors:
            # Key identity: σ² - |c|² = 4·l₁₁²·l₂₂² > 0
            sigma = l11**2 + l21**2 + l22**2
            c = torch.complex(l11**2 - l21**2 - l22**2, 2.0 * l11 * l21)

            kl = get_cholesky_kl(mu, l11, l21, l22)

        # Sampling
        z = reparametrize(mu, sigma, c)
        info = {"kl": kl}
        return (z, info) if return_info else z

    # decode()inherited from VAEBottleneck