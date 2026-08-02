# ===============
# Verifica numerica dell'identità del parallelogramma sulla ComplexMSE del repo:
# |M_hat-M|^2 + |S_hat-S|^2 = 1/2 (|L_hat-L|^2 + |R_hat-R|^2)  (conv. M=(L+R)/2)
# e uguaglianza esatta (loss + gradienti) con la convenzione unitaria /sqrt(2).
# Inoltre: la non-commutatività tra power-norm (alpha=0.65) e la decomposizione M/S.
# ===============
import sys
sys.path.insert(0, "/Users/francesco/Desktop/C-VAE")
sys.path.insert(0, "/Users/francesco/Desktop/C-VAE/src")
import torch

from ar_spectra.training.losses.spectral import ComplexMSE
from ar_spectra.training.pre_transform import PowerMagnitudeTransform

torch.manual_seed(0)
B, F, T = 2, 128, 32

def cac(L, R):
    # grouped layout [Re_L, Re_R, Im_L, Im_R] come dataloader / pre_transform
    return torch.cat([L.real, R.real, L.imag, R.imag], dim=1)

L  = torch.randn(B,1,F,T) + 1j*torch.randn(B,1,F,T)
R  = torch.randn(B,1,F,T) + 1j*torch.randn(B,1,F,T)
Lh = (L + 0.3*(torch.randn_like(L.real)+1j*torch.randn_like(L.real))).clone()
Rh = (R + 0.3*(torch.randn_like(R.real)+1j*torch.randn_like(R.real))).clone()

loss = ComplexMSE(reduction="mean")

lr  = loss(cac(Lh,Rh), cac(L,R))

# convenzione /2 (quella della nota)
M,  S  = (L+R)/2,  (L-R)/2
Mh, Sh = (Lh+Rh)/2,(Lh-Rh)/2
ms_half = loss(cac(Mh,Sh), cac(M,S))

# convenzione unitaria /sqrt(2)
s2 = torch.tensor(2.0).sqrt()
Mu,  Su  = (L+R)/s2,  (L-R)/s2
Muh, Suh = (Lh+Rh)/s2,(Lh-Rh)/s2
ms_unit = loss(cac(Muh,Suh), cac(Mu,Su))

print(f"LR-MSE                      = {lr.item():.10f}")
print(f"MS-MSE (conv /2)            = {ms_half.item():.10f}   ratio LR/MS = {(lr/ms_half).item():.6f}  (atteso 2)")
print(f"MS-MSE (conv /sqrt2)        = {ms_unit.item():.10f}   ratio LR/MS = {(lr/ms_unit).item():.6f}  (atteso 1)")

# --- identità dei gradienti ---
pred = cac(Lh,Rh).requires_grad_(True)
g_lr = torch.autograd.grad(loss(pred, cac(L,R)), pred)[0]

pred2 = cac(Lh,Rh).requires_grad_(True)
Lh2 = torch.complex(pred2[:,0:1], pred2[:,2:3]); Rh2 = torch.complex(pred2[:,1:2], pred2[:,3:4])
ms_loss = loss(cac((Lh2+Rh2)/s2,(Lh2-Rh2)/s2), cac(Mu,Su))
g_ms = torch.autograd.grad(ms_loss, pred2)[0]
print(f"max |grad_LR - grad_MS(unit)| = {(g_lr-g_ms).abs().max().item():.3e}  (atteso ~0)")

# --- power_norm NON commuta con M/S ---
pn = PowerMagnitudeTransform(alpha=0.65, beta=0.35)
comp_then_ms = (pn.transform(cac(L,R))[:, [0]] ) # placeholder non usato, calcolo sotto in complesso
def compress(X):
    mag = X.abs(); unit = X / (mag + 1e-8)
    return unit * 0.35 * mag.pow(0.65)
lhs = (compress(L) - compress(R)) / s2      # S della rappresentazione compressa
rhs = compress((L - R) / s2)                 # compressione della S vera
rel = (lhs - rhs).abs().mean() / rhs.abs().mean()
print(f"rel. diff  S(compress(L,R)) vs compress(S) = {rel.item():.4f}  (>0 => non commutano)")

# --- energia relativa: quanto 'costa' azzerare S sotto la MSE ---
# se il modello predice Lh=Rh=M (collasso mono perfetto), la loss residua e' esattamente ||S||^2
mono = cac(M, M)
collapse_cost = loss(mono, cac(L,R))
total = loss(torch.zeros_like(mono), cac(L,R))
print(f"costo del collasso mono / costo del silenzio = {(collapse_cost/total).item():.4f}  (= E_S/E_tot)")
