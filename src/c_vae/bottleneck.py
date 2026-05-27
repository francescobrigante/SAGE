# ===================================================================================
# Complex Gaussian VAE bottleneck: reparameterization trick and KL divergence
#
# Supports two posterior families:
#   - Proper (C=0):   CN(μ, diag(γ), 0) -> circular, simplest, real-VAE analogue
#   - Improper:       CN(μ, diag(σ), diag(c)) -> full complex Gaussian (Nakashika 2020)
#     with optional Cholesky parameterization to enforce σ > |c| by construction.
# ===================================================================================

import torch
import torch.nn.functional as F
from ar_spectra.models.bottlenecks import VAEBottleneck

SMALL_EPSILON: float = 1e-8  # numerical floor that guards log(0)


def reparametrize(mu: torch.Tensor, sigma: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
    """
    Complex Gaussian reparameterization trick.

    - When c is None (proper mode), `sigma` carries the strictly positive variance `gamma` directly:
        std per real component = √(gamma/2)
        z = μ + std ⊙ ε_R + i · std ⊙ ε_I,    ε_R, ε_I ~ N(0, I_m) i.i.d. real

    - When c is provided (improper mode), sampling follows Nakashika formula (Interspeech 2020):
        z = μ + K_R ⊙ ε_R + i·K_I ⊙ ε_I,    ε_R, ε_I ~ N(0, I_m) i.i.d. real
        K_R = (σ + c) / √(2(σ + c_R))  =  l₁₁ + i·l₂₁           (complex)
        K_I = sqrt[ (σ²-|c|²) / (2(σ+c_R)) ] =  l₂₂             (real, positive)     

    Note: σ + c_R > 0 is guaranteed when σ > |c|.
    The .clamp(min=SMALL_EPSILON) guards against NaN near the constraint boundary only.

    Args:
        mu:    complex mean, shape (B, m, ...)
        sigma: real variance, shape (B, m, ...), must satisfy σ > 0
        c:     complex pseudo-variance, shape (B, m, ...), ideally |c| < σ

    Returns:
        z: sampled latent, shape (B, m, ...) complex
    """

    # Proper mode
    if c is None:
        std = (sigma / 2.0).sqrt()                          # std per real component
        eps_R = torch.randn_like(std)
        eps_I = torch.randn_like(std)
        return mu + std * eps_R + 1j * (std * eps_I)

    # Improper mode (Nakashika 2020)
    denominator = (sigma + c.real).clamp(min=SMALL_EPSILON)     # σ + c_R denominator clamped to avoid numerical issues
    K_R = (sigma + c) / (2.0 * denominator).sqrt()
    K_I = ((sigma**2 - c.abs()**2).clamp(min=SMALL_EPSILON) / (2.0 * denominator)).sqrt()
    eps_R = torch.randn_like(sigma)
    eps_I = torch.randn_like(sigma)
    return mu + K_R * eps_R + 1j * (K_I * eps_I)


def get_proper_kl(mu: torch.Tensor, gamma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    KL divergence for the proper (C=0) posterior CN(μ, diag(γ), 0) vs CN(0, I, 0):
        D_KL = Σᵢ (γᵢ + |μᵢ|² - 1 - log γᵢ)

    Returns:
        kl_scalar:   KL divergence scalar (summed over m, averaged over batch and spatial dims)
        kl_per_dim:  per-element KL tensor, shape (B, m, ...)
    """
    mu_x2 = mu.real**2 + mu.imag**2
    log_gamma = torch.log(gamma)
    kl_per_dim = gamma + mu_x2 - 1.0 - log_gamma
    return kl_per_dim.sum(dim=1).mean(), kl_per_dim


def get_kl(mu: torch.Tensor, sigma: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-form KL divergence (Nakashika 2020):
        D_KL = μ^H μ  +  Σ_k [ σ_k - 1 - 1/2*log(σ_k² - |c_k|²) ]

    Returns:
        kl_scalar:   KL divergence scalar (summed over m, averaged over batch and spatial dims)
        kl_per_dim:  per-element KL tensor, shape (B, m, ...)
    """
    # μ^H μ = |μ|² elementwise
    mu_x2 = mu.real**2 + mu.imag**2

    # log determinant log(σ² - |c|²) = log(σ-|c|) + log(σ+|c|)
    gap1 = (sigma - c.abs()).clamp(min=SMALL_EPSILON)   # σ - |c|
    gap2 =  sigma + c.abs()                             # σ + |c|,  no clamp needed
    log_det = torch.log(gap1) + torch.log(gap2)         # log(σ² - |c|²)

    kl_per_dim = mu_x2 + sigma - 1.0 - 0.5 * log_det
    return kl_per_dim.sum(dim=1).mean(), kl_per_dim




def get_cholesky_kl(mu: torch.Tensor, l11: torch.Tensor, l21: torch.Tensor, l22: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        kl_scalar:   KL divergence scalar
        kl_per_dim:  per-element KL tensor, shape (B, m, ...)
    """
    mu_x2 = mu.real**2 + mu.imag**2

    kl_per_dim = mu_x2 + l11**2 + l21**2 + l22**2 - 1.0 - torch.log(2.0 * l11) - torch.log(l22)
    return kl_per_dim.sum(dim=1).mean(), kl_per_dim


class ComplexVAEBottleneck(VAEBottleneck):
    """
    Complex Gaussian VAE bottleneck.
    Expects a complex tensor with (parameters_to_predict x latent_channels) along dim=1.
    Operates in four modes:

    1) proper = True (Circular Mode):
        Predicted parameters: μ (complex), γ (real only)
        Posterior: CN(μ, diag(γ), 0)
        Strict positivity of γ is ensured by Softplus + SMALL_EPSILON.

    2) proper = False, apply_cholesky_constraints = False, apply_spectral_parameterization = False (DEFAULT):
        Predicted parameters: μ (complex), σ (softplus(slot_2.real) > 0; slot_2.imag unused), c (complex from slot_3)
        The constraint σ > |c| IS enforced by construction: c = σ·tanh(|slot_3|)·∠slot_3,
            so |c| = σ·tanh(|slot_3|) < σ always. The clamp in get_kl is a numerical safeguard only.

    3) proper = False, apply_cholesky_constraints = True:
        Predicted parameters: μ (complex), l₁₁ = softplus(slot_2.real) > 0, l₂₁ = slot_2.imag, l₂₂ = softplus(|slot_3|) > 0
        The constraint σ > |c| is guaranteed by construction whenever l₁₁,l₂₂ > 0,
            which is ensured by l₁₁ = softplus(·) > 0 and l₂₂ = softplus(·) > 0.
        Both components of slot_2 carry gradient (l₁₁ via Re, l₂₁ via Im); l₂₂ uses |slot_3| so both Re and Im of slot_3 contribute.

    4) proper = False, apply_spectral_parameterization = True:
        Predicted parameters: μ (complex), λ₁ = softplus(slot_2.real), λ₂ = softplus(slot_2.imag), θ = slot_3.real
        Eigendecomposition of the augmented 2×2 covariance: σ = λ₁ + λ₂, c = (λ₁ − λ₂)e^{2iθ}
        The constraint σ > |c| is guaranteed by construction (triangle inequality: |λ₁−λ₂| ≤ λ₁+λ₂).
        No clamping needed: log_det = log(4λ₁λ₂) is always finite for λᵢ > 0.
    """

    def __init__(
        self,
        apply_cholesky_constraints: bool = False,
        apply_spectral_parameterization: bool = False,
        proper: bool = False,
    ):
        # 2 parameters for proper mode, 3 for improper mode
        super().__init__(parameters_to_predict=2 if proper else 3)
        self.apply_cholesky_constraints      = apply_cholesky_constraints       # improper Cholesky mode
        self.apply_spectral_parameterization = apply_spectral_parameterization  # improper spectral mode
        self.proper = proper

    def encode(self, x: torch.Tensor, return_info: bool = False, **kwargs) -> tuple[torch.Tensor, dict] | torch.Tensor:

        assert torch.is_complex(x), "ComplexVAEBottleneck expects a complex-valued input tensor"
        m = x.shape[1] // self.parameters_to_predict
        mu = x[:, :m]

        # ────────────────────────────────── Proper mode (C=0) ──────────────────────────────────
        # ⚠ YAML config model: set model.parameters_to_predict=2 and model.bottleneck.proper=true
        if self.proper:
            assert x.shape[1] % 2 == 0, (f"ComplexVAEBottleneck (proper) expects channels divisible by 2 [μ|γ], "f"got {x.shape[1]}")

            slot_2 = x[:, m:]
            sigma = F.softplus(slot_2.real) + SMALL_EPSILON
            c     = None
            kl, kl_per_dim = get_proper_kl(mu, sigma)

        # ────────────────────────────────── Improper mode (Spectral) ──────────────────────────────────
        # ⚠ YAML: set model.parameters_to_predict=3 and model.bottleneck.apply_spectral_parameterization=true
        elif self.apply_spectral_parameterization:
            assert x.shape[1] % 3 == 0, (
                f"ComplexVAEBottleneck (spectral) expects channels divisible by 3 [μ|λ₁λ₂|θ], "
                f"got {x.shape[1]}"
            )
            slot_2 = x[:, m:2*m]
            slot_3 = x[:, 2*m:]

            lambda1 = F.softplus(slot_2.real) + SMALL_EPSILON  # eigenvalue 1, > 0
            lambda2 = F.softplus(slot_2.imag) + SMALL_EPSILON  # eigenvalue 2, > 0 (slot_2.imag was previously wasted)
            theta   = slot_3.real                               # rotation angle, unconstrained

            sigma = lambda1 + lambda2                           # total variance: σ = λ₁ + λ₂
            c = torch.complex(                                  # pseudo-variance: c = (λ₁−λ₂)e^{2iθ}
                (lambda1 - lambda2) * torch.cos(2.0 * theta),
                (lambda1 - lambda2) * torch.sin(2.0 * theta),
            )

            # log(4λ₁λ₂) always finite for λᵢ > 0 — no clamp needed
            log_det    = torch.log(4.0 * lambda1 * lambda2)
            mu_x2      = mu.real**2 + mu.imag**2
            kl_per_dim = mu_x2 + sigma - 1.0 - 0.5 * log_det
            kl         = kl_per_dim.sum(dim=1).mean()

        # ────────────────────────────────── Improper mode (Default) ──────────────────────────────────
        # ⚠ YAML: set model.parameters_to_predict=3
        elif not self.apply_cholesky_constraints:
            assert x.shape[1] % 3 == 0, (f"ComplexVAEBottleneck expects channels divisible by 3 [μ|slot2|slot3], "f"got {x.shape[1]}")
            slot_2 = x[:, m:2*m]
            slot_3 = x[:, 2*m:]

            sigma  = F.softplus(slot_2.real.clamp(max=10.0)) + SMALL_EPSILON  # clamp prevents bf16 overflow → sigma=inf → KL=inf−inf=NaN
            c_mag = slot_3.abs()
            c = sigma * torch.tanh(c_mag) * slot_3 / (c_mag + SMALL_EPSILON)
            kl, kl_per_dim = get_kl(mu, sigma, c)

        # ────────────────────────────────── Improper mode (Cholesky) ──────────────────────────────────
        # ⚠ YAML: set model.parameters_to_predict=3 and model.bottleneck.apply_cholesky_constraints=true
        else:
            assert x.shape[1] % 3 == 0, (f"ComplexVAEBottleneck expects channels divisible by 3 [μ|slot2|slot3], "f"got {x.shape[1]}")
            slot_2 = x[:, m:2*m]
            slot_3 = x[:, 2*m:]

            # Decode Cholesky factors from the encoder slots (softplus-parameterized diagonals):
            # slot_2: l11 = Re (positive diagonal), l21 = Im (off-diagonal, unconstrained)
            # slot_3: l22 uses |slot_3| so both Re and Im contribute; phase of slot_3 is redundant but carries gradient
            l11 = F.softplus(slot_2.real) + SMALL_EPSILON  # > 0
            l21 = slot_2.imag                              # unconstrained real
            l22 = F.softplus(slot_3.abs()) + SMALL_EPSILON # > 0; uses both Re and Im of slot_3

            # Recover (σ, c) from Cholesky factors:
            # Key identity: σ² - |c|² = 4·l₁₁²·l₂₂² > 0
            sigma = l11**2 + l21**2 + l22**2
            c = torch.complex(l11**2 - l21**2 - l22**2, 2.0 * l11 * l21)
            kl, kl_per_dim = get_cholesky_kl(mu, l11, l21, l22)

        # Per-channel KL diagnostics: average over batch and spatial dims → shape (m,)
        # Used to detect posterior collapse (active units where KL_j > 0.1).
        # `active_units` is computed lazily by the consumer to avoid a CUDA→CPU
        # sync on every encode forward (was: `int(... .sum().item())`).
        kl_diag_dims = tuple([0] + list(range(2, kl_per_dim.dim())))
        kl_per_channel = kl_per_dim.detach().mean(dim=kl_diag_dims)           # (m,) — m = true latent channels

        z = reparametrize(mu, sigma, c)
        info = {
            "kl": kl,
            "kl_per_channel": kl_per_channel,
        }
        return (z, info) if return_info else z

    # decode() inherited from VAEBottleneck