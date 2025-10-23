import torch
import torch.nn as nn

class ComplexLinearHalf(nn.Module):
    def __init__(self, in_f, out_f, bias=False, dtype=torch.float16, device="cuda"):
        super().__init__()
        self.Wr = nn.Parameter(torch.empty(out_f, in_f, device=device, dtype=dtype))
        self.Wi = nn.Parameter(torch.empty(out_f, in_f, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.Wr); nn.init.kaiming_uniform_(self.Wi)
        self.bias = bias
        if bias:
            self.br = nn.Parameter(torch.zeros(out_f, device=device, dtype=torch.float32))
            self.bi = nn.Parameter(torch.zeros(out_f, device=device, dtype=torch.float32))
            self.bias = True

    def forward(self, x):  # x: complex, real/imag in half
        xr, xi = x.real.to(self.Wr.dtype), x.imag.to(self.Wr.dtype)
        # accumulo in fp32 per stabilità
        yr = (self.Wr @ xr.T - self.Wi @ xi.T).to(torch.float32).T
        yi = (self.Wr @ xi.T + self.Wi @ xr.T).to(torch.float32).T
        if self.bias:
            yr = yr + self.br
            yi = yi + self.bi
        return torch.complex(yr, yi)
