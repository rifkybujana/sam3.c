"""
tools/_cpu_patches.py - Shared CUDA->CPU shims for dumping SAM 3 /
SAM 3.1 reference outputs on CPU-only machines.

The upstream sam3 package hardcodes CUDA in several tensor factories
and imports triton (CUDA-only). This module provides:

  install_triton_stub()   - Make `import triton` succeed without the
                            kernels ever running.
  install_cuda_redirect() - Redirect device="cuda" / torch.device("cuda")
                            to CPU in tensor factories and .cuda() ops,
                            and force has_triton_package() false for
                            torch._inductor.
  install_addmm_act_fp32() - Replace sam3.perflib.fused.addmm_act with a
                             CPU/fp32-preserving implementation (must be
                             called AFTER sam3 is imported).

Call install_triton_stub() and install_cuda_redirect() BEFORE importing
any sam3 submodule. Call install_addmm_act_fp32() after.
"""
import sys
import types

import torch
from torch.utils import _triton as _torch_triton


def install_triton_stub():
    _torch_triton.has_triton_package = lambda: False
    _torch_triton.has_triton = lambda: False

    if "triton" in sys.modules:
        return

    class _TritonStubType:
        pass

    triton_stub = types.ModuleType("triton")
    triton_stub.jit = lambda f=None, **kw: (
        f if callable(f) else (lambda g: g)
    )
    triton_stub.heuristics = lambda *a, **kw: (lambda f: f)
    triton_stub.autotune = lambda *a, **kw: (lambda f: f)

    triton_lang_stub = types.ModuleType("triton.language")
    for _name in (
        "dtype", "tensor", "pointer_type", "void", "int1",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64",
        "bfloat16", "block_type", "constexpr",
    ):
        setattr(triton_lang_stub, _name, _TritonStubType)
    triton_stub.language = triton_lang_stub
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = triton_lang_stub


def install_cuda_redirect():
    if torch.cuda.is_available():
        return

    def _cpu_redirect(kwargs):
        dev = kwargs.get("device")
        if dev is None:
            return
        if isinstance(dev, torch.device) and dev.type == "cuda":
            kwargs["device"] = torch.device("cpu")
        elif isinstance(dev, str) and dev.startswith("cuda"):
            kwargs["device"] = "cpu"

    for _fn in ("zeros", "ones", "empty", "arange", "randn", "rand",
                "full", "linspace", "eye", "tensor", "as_tensor"):
        _orig = getattr(torch, _fn)

        def _wrap(orig):
            def _wrapped(*a, **kw):
                _cpu_redirect(kw)
                return orig(*a, **kw)
            return _wrapped
        setattr(torch, _fn, _wrap(_orig))

    torch.Tensor.cuda = lambda self, *a, **kw: self
    torch.nn.Module.cuda = lambda self, *a, **kw: self


def install_addmm_act_fp32():
    import sam3.perflib.fused as _fused

    def _addmm_act_fp32(activation, linear, mat1):
        if torch.is_grad_enabled():
            raise ValueError("Expected grad to be disabled.")
        w = linear.weight.detach()
        b = linear.bias.detach()
        y = torch.nn.functional.linear(mat1, w, b)
        if activation in (torch.nn.functional.relu, torch.nn.ReLU):
            return torch.nn.functional.relu(y)
        if activation in (torch.nn.functional.gelu, torch.nn.GELU):
            return torch.nn.functional.gelu(y)
        raise ValueError(f"Unexpected activation {activation}")

    _fused.addmm_act = _addmm_act_fp32
    import sam3.model.vitdet as _vitdet
    _vitdet.addmm_act = _addmm_act_fp32
