"""Router-free multi-LoRA student: one shared backbone and M LoRA branches.

GPT-2 implements ``c_attn``, ``c_proj`` and ``c_fc`` as ``Conv1D`` (weight stored
as ``[d_in, d_out]``). LLaMA-style blocks use ``nn.Linear`` (weight stored as
``[d_out, d_in]``). ``MultiLoRALayer`` wraps either module and adds the same
low-rank residual, so the base layer's own forward is always the one that
applies the base weights.
"""

import json
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover - older transformers layouts
    from transformers.modeling_utils import Conv1D


_SKIP_MODULE_NAMES = {"lm_head", "wte", "wpe", "embed_tokens", "embed_positions"}
_ARCH_META_KEYS = (
    "num_lora_branches",
    "sreotd_lora_r",
    "sreotd_lora_alpha",
    "sreotd_sigma_b",
)


def _log(message):
    try:
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass
    print(message, flush=True)


def unwrap_model(model):
    """Peel DeepSpeed / DDP wrappers until the underlying ``nn.Module``."""
    seen = set()
    while id(model) not in seen:
        seen.add(id(model))
        inner = getattr(model, "module", None)
        if not isinstance(inner, nn.Module) or inner is model:
            break
        model = inner
    return model


def _io_dims(layer):
    if isinstance(layer, nn.Linear):
        return layer.in_features, layer.out_features
    weight = layer.weight
    if weight.dim() != 2:
        raise TypeError(f"Expected a 2D projection weight, got shape {tuple(weight.shape)}")
    # Conv1D stores weight as [d_in, d_out]. nn.Linear is handled above.
    d_in, d_out = weight.shape
    return int(d_in), int(d_out)


def orthogonal_branch_A(num_branches, rank, d_in, device, dtype):
    """QR init so ``A_i A_j^T = I_r`` if ``i == j`` and ``0`` otherwise.

    A Gaussian matrix ``[d_in, M*r]`` is factored in float32. The transpose of
    the thin Q factor is a set of orthonormal rows, which we split into M blocks
    of ``r`` rows.
    """
    total = num_branches * rank
    if total > d_in:
        raise ValueError(
            f"Orthogonal LoRA init needs M*r={total} <= d_in={d_in}. "
            "Reduce --num-lora-branches or --sreotd-lora-r."
        )
    gaussian = torch.randn(d_in, total, device=device, dtype=torch.float32)
    q, _ = torch.linalg.qr(gaussian, mode="reduced")
    factors = q.transpose(0, 1).contiguous().view(num_branches, rank, d_in)
    return factors.to(dtype=dtype)


class MultiLoRALayer(nn.Module):
    """Base projection plus M LoRA branches with a shared scale ``alpha / r``."""

    def __init__(self, base_layer, num_branches=3, r=16, lora_alpha=16, sigma_b=1e-3):
        super().__init__()
        if not isinstance(base_layer, (nn.Linear, Conv1D)):
            raise TypeError(f"MultiLoRALayer expects nn.Linear or Conv1D, got {type(base_layer)}")
        self.base_layer = base_layer
        self.num_branches = int(num_branches)
        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / float(self.r)
        self.active_branch = None

        d_in, d_out = _io_dims(base_layer)
        self.d_in = d_in
        self.d_out = d_out
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        A = orthogonal_branch_A(self.num_branches, self.r, d_in, device, dtype)
        B = torch.randn(self.num_branches, d_out, self.r, device=device, dtype=torch.float32)
        B = (B * float(sigma_b)).to(dtype=dtype)
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)

    def extra_repr(self):
        return (f"branches={self.num_branches}, r={self.r}, alpha={self.lora_alpha}, "
                f"d_in={self.d_in}, d_out={self.d_out}, scale={self.scaling}")

    def _delta_single(self, x, branch):
        # x @ A[m]^T @ B[m]^T, supporting [..., d_in]
        A_m = self.A[branch]
        B_m = self.B[branch]
        hidden = x @ A_m.transpose(0, 1)
        return hidden @ B_m.transpose(0, 1)

    def _delta_parallel(self, x):
        # x is the repeated batch [B*M, ..., d_in] laid out by repeat_interleave.
        d_in = x.size(-1)
        batch = x.size(0)
        if batch % self.num_branches != 0:
            raise RuntimeError(
                f"MultiLoRALayer expected the batch dimension to be divisible by "
                f"M={self.num_branches} when active_branch is None, got shape {tuple(x.shape)}. "
                "Call set_active_branch(m) for single-branch generation, or "
                "repeat_interleave the inputs by M before the parallel forward."
            )
        B = batch // self.num_branches
        middle = x.shape[1:-1]
        flat = x.reshape(B, self.num_branches, -1, d_in)
        delta = torch.einsum("bmti,mri,mor->bmto", flat, self.A, self.B)
        delta = delta * self.scaling
        return delta.reshape(batch, *middle, self.d_out)

    def forward(self, x):
        base_out = self.base_layer(x)
        if self.active_branch is None:
            delta = self._delta_parallel(x)
        else:
            branch = int(self.active_branch)
            delta = self._delta_single(x, branch) * self.scaling
        if delta.dtype != base_out.dtype:
            delta = delta.to(dtype=base_out.dtype)
        return base_out + delta.reshape_as(base_out)


def _backbone_module(model):
    if hasattr(model, "transformer") and isinstance(model.transformer, nn.Module):
        return model.transformer
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module):
        if hasattr(inner, "layers") or hasattr(inner, "h") or hasattr(inner, "decoder"):
            return inner
        if hasattr(inner, "transformer") and isinstance(inner.transformer, nn.Module):
            return inner.transformer
        nested = getattr(inner, "model", None)
        if isinstance(nested, nn.Module):
            return nested
        return inner
    raise ValueError("Cannot find model.transformer or model.model to attach Multi-LoRA.")


def _replace_projections(module, factory):
    for name, child in list(module.named_children()):
        if name in _SKIP_MODULE_NAMES:
            continue
        if isinstance(child, MultiLoRALayer):
            continue
        if isinstance(child, (nn.Linear, Conv1D)):
            setattr(module, name, factory(child))
        else:
            _replace_projections(child, factory)


def _current_branch(model):
    for module in model.modules():
        if isinstance(module, MultiLoRALayer):
            return module.active_branch
    return None


def set_active_branch(self, branch):
    """Select one LoRA branch, or ``None`` to run all M branches in parallel."""
    if branch is not None:
        branch = int(branch)
        n = int(self._sreotd_num_branches)
        if branch < 0 or branch >= n:
            raise ValueError(f"active branch {branch} is outside 0..{n - 1}")
    for module in self.modules():
        if isinstance(module, MultiLoRALayer):
            module.active_branch = branch


def forward_multi_branch(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
    """Run the shared backbone on M copies of the batch, without ``lm_head``.

    Returns the last hidden state as ``[B, T, M, d]``.
    """
    M = int(self._sreotd_num_branches)
    if input_ids is None:
        raise ValueError("forward_multi_branch requires input_ids")
    bsz = input_ids.size(0)

    def _repeat(tensor):
        if torch.is_tensor(tensor) and tensor.dim() > 0 and tensor.size(0) == bsz:
            return tensor.repeat_interleave(M, dim=0)
        return tensor

    previous = _current_branch(self)
    self.set_active_branch(None)
    try:
        backbone = _backbone_module(self)
        call = {
            "input_ids": _repeat(input_ids),
            "use_cache": False,
            "return_dict": True,
        }
        if attention_mask is not None:
            call["attention_mask"] = _repeat(attention_mask)
        if position_ids is not None:
            call["position_ids"] = _repeat(position_ids)
        for key, value in kwargs.items():
            if key in call:
                continue
            call[key] = _repeat(value)
        outputs = backbone(**call)
        hidden = outputs.last_hidden_state
    finally:
        self.set_active_branch(previous)

    BM, T, d = hidden.shape
    if BM != bsz * M:
        raise RuntimeError(f"Expected hidden batch {bsz * M}, got {BM}")
    return hidden.view(bsz, M, T, d).permute(0, 2, 1, 3).contiguous()


def _bind(model, name, fn):
    setattr(model, name, types.MethodType(fn, model))


def inject_multi_lora(model, args):
    """Replace every backbone projection with a ``MultiLoRALayer``.

    ``lm_head`` is outside the backbone and is left as a full vocabulary head.
    Embeddings (``wte`` / ``wpe``) are not projections and are skipped.
    When ``args.sreotd_freeze_backbone`` is true, only ``A`` and ``B`` train.
    """
    model = unwrap_model(model)
    if any(isinstance(module, MultiLoRALayer) for module in model.modules()):
        return model

    num_branches = int(args.num_lora_branches)
    rank = int(args.sreotd_lora_r)
    alpha = float(args.sreotd_lora_alpha)
    sigma_b = float(args.sreotd_sigma_b)
    backbone = _backbone_module(model)

    def factory(base_layer):
        return MultiLoRALayer(
            base_layer,
            num_branches=num_branches,
            r=rank,
            lora_alpha=alpha,
            sigma_b=sigma_b,
        )

    _replace_projections(backbone, factory)
    wrapped = sum(isinstance(module, MultiLoRALayer) for module in model.modules())
    if wrapped == 0:
        raise RuntimeError("inject_multi_lora found no Linear/Conv1D projections to wrap.")

    freeze = bool(args.sreotd_freeze_backbone)
    for parameter in model.parameters():
        parameter.requires_grad = not freeze
    for module in model.modules():
        if isinstance(module, MultiLoRALayer):
            module.A.requires_grad = True
            module.B.requires_grad = True

    if freeze and getattr(args, "gradient_checkpointing", False) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model._sreotd_num_branches = num_branches
    _bind(model, "set_active_branch", set_active_branch)
    _bind(model, "forward_multi_branch", forward_multi_branch)
    model.set_active_branch(None)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    _log(
        f"SR-EOTD MultiLoRA: wrapped {wrapped} projections | "
        f"M={num_branches} r={rank} alpha={alpha} scale={alpha / rank:.4f} | "
        f"freeze_backbone={freeze} | trainable {trainable}/{total}"
    )
    return model


def project_topk_lm_head(hidden_states, lm_head_weight, topk_indices):
    """Project ``[B, T, M, d]`` hidden states onto the teacher's top-K rows.

    Returns logits ``[B, T, M, K]``.
    """
    if hidden_states.dim() != 4:
        raise ValueError(f"hidden_states must be [B, T, M, d], got {tuple(hidden_states.shape)}")
    if topk_indices.dim() != 3:
        raise ValueError(f"topk_indices must be [B, T, K], got {tuple(topk_indices.shape)}")
    topk_weights = F.embedding(topk_indices, lm_head_weight)
    return torch.einsum("btmd,btkd->btmk", hidden_states, topk_weights)


def get_lm_head_weight(model):
    module = unwrap_model(model)
    head = getattr(module, "lm_head", None)
    if head is not None and hasattr(head, "weight"):
        return head.weight
    embed_out = getattr(module, "embed_out", None)
    if embed_out is not None and hasattr(embed_out, "weight"):
        return embed_out.weight
    raise AttributeError("Student model has no lm_head.weight")


def get_lm_head_bias(model):
    module = unwrap_model(model)
    head = getattr(module, "lm_head", None)
    if head is None:
        return None
    return getattr(head, "bias", None)


def sync_sreotd_architecture_from_checkpoint(args):
    """Match M, r, alpha and sigma_B to a saved SR-EOTD checkpoint when present."""
    path = getattr(args, "model_path", None)
    if not path:
        return None
    meta_path = os.path.join(path, "sreotd_meta.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path) as handle:
        meta = json.load(handle)
    for key in _ARCH_META_KEYS:
        if key not in meta:
            continue
        current = getattr(args, key, None)
        if current != meta[key]:
            _log(f"SR-EOTD checkpoint overrides {key}: {current} -> {meta[key]}")
            setattr(args, key, meta[key])
    return meta


def load_sreotd_weights_if_present(model, args):
    path = getattr(args, "model_path", None)
    if not path:
        return False
    weight_path = os.path.join(path, "sreotd_model.pt")
    if not os.path.isfile(weight_path):
        return False
    try:
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weight_path, map_location="cpu")
    target = unwrap_model(model)
    target.load_state_dict(state, strict=True)
    _log(f"Loaded SR-EOTD weights from {weight_path}")
    return True


def save_sreotd_checkpoint(model, tokenizer, save_dir, args):
    """Save config, tokenizer, the full SR-EOTD state, and the architecture meta."""
    module = unwrap_model(model)
    os.makedirs(save_dir, exist_ok=True)
    module.config.save_pretrained(save_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)
    torch.save(module.state_dict(), os.path.join(save_dir, "sreotd_model.pt"))
    meta = {
        "kd_objective": "sr_eotd",
        "num_lora_branches": args.num_lora_branches,
        "sreotd_lora_r": args.sreotd_lora_r,
        "sreotd_lora_alpha": args.sreotd_lora_alpha,
        "sreotd_sigma_b": args.sreotd_sigma_b,
        "sreotd_freeze_backbone": bool(args.sreotd_freeze_backbone),
        "sreotd_topk": args.sreotd_topk,
        "sreotd_sinkhorn_iters": args.sreotd_sinkhorn_iters,
        "sreotd_eps_min": args.sreotd_eps_min,
        "sreotd_eps_max": args.sreotd_eps_max,
        "sreotd_gamma": args.sreotd_gamma,
        "sreotd_rho": args.sreotd_rho,
        "sreotd_beta": args.sreotd_beta,
        "sreotd_tau": args.sreotd_tau,
    }
    with open(os.path.join(save_dir, "sreotd_meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    _log(f"SR-EOTD checkpoint saved to {save_dir}")


def sreotd_state_only_checkpoint(args):
    """True when ``model_path`` is an SR-EOTD save dir without a HF weight file."""
    path = getattr(args, "model_path", None)
    if getattr(args, "kd_objective", None) != "sr_eotd" or not path:
        return False
    if not os.path.isfile(os.path.join(path, "sreotd_model.pt")):
        return False
    hf_names = (
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    )
    return not any(os.path.isfile(os.path.join(path, name)) for name in hf_names)


def _is_lora_factor(key, value):
    if not torch.is_tensor(value) or value.dim() != 3:
        return False
    return key.endswith(".A") or key.endswith(".B")


def extract_multi_lora_factors(state_dict):
    """A and B only. ``A`` is ``[M, r, d_in]`` and ``B`` is ``[M, d_out, r]``."""
    return {key: value.detach().cpu() for key, value in state_dict.items() if _is_lora_factor(key, value)}


def export_backbone_state_dict(state_dict):
    """Shared backbone weights with Multi-LoRA wrappers removed.

    ``*.base_layer.weight`` becomes the Hugging Face key ``*.weight``.
    """
    exported = {}
    for key, value in state_dict.items():
        if _is_lora_factor(key, value):
            continue
        exported[key.replace(".base_layer.", ".")] = value.detach().cpu()
    return exported


def _merge_base_and_delta(base, delta):
    """Add ``B @ A`` (``[d_out, d_in]``) onto a Linear or Conv1D weight.

    ``nn.Linear`` stores ``[d_out, d_in]``, so the delta is added directly.
    GPT-2 ``Conv1D`` stores ``[d_in, d_out]``, so the delta is transposed.
    """
    base_f = base.detach().float().cpu()
    delta = delta.detach().float().cpu()
    if tuple(base_f.shape) == tuple(delta.shape):
        merged = base_f + delta
    elif tuple(base_f.shape) == (delta.shape[1], delta.shape[0]):
        merged = base_f + delta.transpose(0, 1).contiguous()
    else:
        raise ValueError(
            f"Cannot merge LoRA delta {tuple(delta.shape)} into base weight {tuple(base_f.shape)}"
        )
    return merged.to(dtype=base.dtype)


def export_merged_branch_state_dict(state_dict, branch, scale):
    """Backbone plus ``scale * B_m @ A_m`` written as a plain Hugging Face state dict."""
    exported = export_backbone_state_dict(state_dict)
    branch = int(branch)
    for key, value in state_dict.items():
        if not (key.endswith(".A") and _is_lora_factor(key, value)):
            continue
        prefix = key[:-2]
        factor_b = state_dict[prefix + ".B"]
        base = state_dict[prefix + ".base_layer.weight"]
        if branch < 0 or branch >= value.size(0):
            raise IndexError(f"branch {branch} is outside A shape {tuple(value.shape)} at {key}")
        delta = float(scale) * (factor_b[branch].float() @ value[branch].float())
        exported[prefix + ".weight"] = _merge_base_and_delta(base, delta)
    return exported


def _cosine(left, right):
    left = left.detach().float().flatten().cpu()
    right = right.detach().float().flatten().cpu()
    denom = left.norm() * right.norm()
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(left, right) / denom)


def branch_pair_metrics(A, B, scale):
    """Pairwise ``||A_i A_j^T||_F / sqrt(r)`` and cosine similarity of ``ΔW``."""
    A = A.detach().float().cpu()
    B = B.detach().float().cpu()
    branches, rank, _ = A.shape
    pairs = []
    for left in range(branches):
        for right in range(left, branches):
            gram = A[left] @ A[right].transpose(0, 1)
            orth = float(torch.linalg.norm(gram, ord="fro") / (rank ** 0.5))
            delta_left = float(scale) * (B[left] @ A[left])
            delta_right = float(scale) * (B[right] @ A[right])
            pairs.append({
                "i": left,
                "j": right,
                "A_gram_fro_over_sqrt_r": orth,
                "delta_cosine": _cosine(delta_left, delta_right),
            })
    return pairs


def lora_subspace_orthogonality(model):
    """Per-layer LoRA subspace overlap for the injected ``MultiLoRALayer`` modules."""
    module = unwrap_model(model)
    layers = []
    off_orth, off_cos, diag_orth = [], [], []
    rank = None
    branches = None
    for name, layer in module.named_modules():
        if not isinstance(layer, MultiLoRALayer):
            continue
        rank = int(layer.r)
        branches = int(layer.num_branches)
        pairs = branch_pair_metrics(layer.A, layer.B, layer.scaling)
        for pair in pairs:
            if pair["i"] == pair["j"]:
                diag_orth.append(pair["A_gram_fro_over_sqrt_r"])
            else:
                off_orth.append(pair["A_gram_fro_over_sqrt_r"])
                off_cos.append(pair["delta_cosine"])
        layers.append({"name": name, "pairs": pairs})

    def _mean(values):
        if not values:
            return None
        return float(sum(values) / len(values))

    return {
        "rank": rank,
        "num_branches": branches,
        "num_layers": len(layers),
        "mean_diag_A_gram_fro_over_sqrt_r": _mean(diag_orth),
        "mean_offdiag_A_gram_fro_over_sqrt_r": _mean(off_orth),
        "mean_offdiag_delta_cosine": _mean(off_cos),
        "layers": layers,
    }
