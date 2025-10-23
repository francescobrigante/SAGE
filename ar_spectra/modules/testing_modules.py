# test_pad_complex.py
import torch
import torch.nn.functional as F

def supported_complex_dtypes():
    dtypes = []
    if hasattr(torch, "complex32"):
        dtypes.append(torch.complex32)
    dtypes += [torch.complex64, torch.complex128]
    return dtypes

def real_dtype_for(cdtype: torch.dtype) -> torch.dtype:
    if cdtype == getattr(torch, "complex32", None):
        return torch.float16
    if cdtype == torch.complex64:
        return torch.float32
    if cdtype == torch.complex128:
        return torch.float64
    raise ValueError(f"Unsupported complex dtype: {cdtype}")

def make_complex(shape, cdtype: torch.dtype):
    rdtype = real_dtype_for(cdtype)
    real = torch.randn(shape, dtype=rdtype)
    imag = torch.randn(shape, dtype=rdtype)
    return torch.complex(real, imag)  # produce direttamente il cdtype corretto

def try_pad(x, padding, mode, value=0):
    # value=0 è sicuro anche per complessi
    return F.pad(x, padding, mode=mode, value=value)

def run_one_dtype(cdtype: torch.dtype):
    print(f"\n=== Testing dtype: {cdtype} ===")
    # 1D: shape (..., W); padding (left, right)
    x1 = make_complex((2, 3, 8), cdtype)
    pads1 = (2, 2)  # riflette/replica ok: 2 < 8
    # 2D: shape (..., H, W); padding (w_left, w_right, h_left, h_right)
    x2 = make_complex((2, 3, 8, 10), cdtype)
    pads2 = (2, 2, 2, 2)  # 2 < H=8 e 2 < W=10
    # 3D: shape (..., D, H, W); padding (wL,wR,hL,hR,dL,dR)
    x3 = make_complex((2, 3, 6, 8, 10), cdtype)
    pads3 = (2, 2, 2, 2, 1, 1)  # 1 < D=6, 2 < H=8, 2 < W=10

    modes = ("constant", "reflect", "replicate", "circular")
    tests = [
        ("1D", x1, pads1),
        ("2D", x2, pads2),
        ("3D", x3, pads3),
    ]

    for name, x, pads in tests:
        for mode in modes:
            try:
                y = try_pad(x, pads, mode)
                print(f"[OK] {name:>2} {mode:<9} -> out.shape={tuple(y.shape)}, out.dtype={y.dtype}")
            except Exception as e:
                print(f"[FAIL] {name:>2} {mode:<9} -> {type(e).__name__}: {e}")

def main():
    print(f"torch.__version__ = {torch.__version__}")
    dtypes = supported_complex_dtypes()
    print("Complex dtypes found:", dtypes)
    for dt in dtypes:
        run_one_dtype(dt)

if __name__ == "__main__":
    main()
