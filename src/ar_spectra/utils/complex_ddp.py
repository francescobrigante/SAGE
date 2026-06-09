# =============================================================================
# Complex DDP Compatibility Transform
# Allows models with torch.complex64 parameters/buffers to be trained with
# PyTorch DistributedDataParallel over NCCL (which does not support ComplexFloat).
# =============================================================================

import torch
import torch.nn as nn

def make_complex_model_ddp_compatible(module: nn.Module):
    """
    Dynamically transforms complex parameters and buffers into real views (float32)
    in-place. This allows DistributedDataParallel (DDP) to serialize and sync them
    using standard collective ops (like NCCL), which crash on complex tensors.
    
    A forward pre-hook is registered to automatically restore the complex tensor
    views immediately before module forward passes, ensuring the inner logic
    continues operating on complex numbers seamlessly.
    
    Args:
        module (nn.Module): The root PyTorch module (e.g., AutoEncoder)
    """
    for name, submodule in module.named_modules():
        sub_complex_params = {}
        sub_complex_buffers = {}
        
        # 1) Handle Parameters
        for p_name, p in list(submodule.named_parameters(recurse=False)):
            if p.is_complex():
                # Remove the complex parameter so DDP ignores it
                delattr(submodule, p_name)
                # Create a float32 representation of shape (..., 2)
                real_p = nn.Parameter(torch.view_as_real(p.data).contiguous(), requires_grad=p.requires_grad)
                mangled_name = f"{p_name}_real_view"
                submodule.register_parameter(mangled_name, real_p)
                sub_complex_params[p_name] = mangled_name
                
        # 2) Handle Buffers
        for b_name, b in list(submodule.named_buffers(recurse=False)):
            if b is not None and b.is_complex():
                delattr(submodule, b_name)
                mangled_name = f"{b_name}_real_view"
                submodule.register_buffer(mangled_name, torch.view_as_real(b).contiguous())
                sub_complex_buffers[b_name] = mangled_name

        if sub_complex_params or sub_complex_buffers:
            # Inject pre-forward hook to assign the complex view back
            def pre_forward_hook(m, inputs, _p=sub_complex_params, _b=sub_complex_buffers):
                for orig_name, mangled_name in _p.items():
                    real_p = getattr(m, mangled_name)
                    # This adds a pure Tensor attribute, NOT a Parameter, so DDP ignores it
                    setattr(m, orig_name, torch.view_as_complex(real_p))
                for orig_name, mangled_name in _b.items():
                    real_b = getattr(m, mangled_name)
                    setattr(m, orig_name, torch.view_as_complex(real_b))
            
            # Register the hook (with_kwargs=False ensures compatibility across PyTorch versions).
            # prepend=True: this restore-hook MUST run before any other pre-forward hook that
            # consumes the complex param (e.g. ComplexWeightNorm._recompute_weight reads
            # `weight_v`). Otherwise that hook sees a stale, pre-`.to(device)` CPU view and
            # builds a CPU weight → "weight CPU / input CUDA" crash + grads not reaching the
            # real-view parameter.
            submodule.register_forward_pre_hook(pre_forward_hook, with_kwargs=False, prepend=True)
            
            # Trigger the hook once manually. This ensures the complex attributes exist
            # in __dict__ immediately (e.g., for initialization logic or weight inspection
            # that might occur before the first forward pass).
            pre_forward_hook(submodule, None)
