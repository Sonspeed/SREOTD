"""Sanity checks for SR-EOTD.

1. Orthogonal LoRA init and parallel-vs-single forward, for nn.Linear and Conv1D.
2. A tiny GPT-2 student: every Conv1D block is wrapped, lm_head is not, the
   parallel backbone matches each active branch, and the loss trains A/B.
3. Unimodal teacher: prints Q_tilde and w*. The matching branch takes the mass.
4. Skewed 80/15/5 mixture with M=3 specialists: prints Q_tilde and w*.

Usage (from the repository root): python tools/test_sreotd_modes.py
"""
import importlib.util
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, ".")


def _load_without_package_init(qualified, path):
    """Load distillm modules without importing DeepSpeed via the package __init__."""
    if "distillm" not in sys.modules:
        package = types.ModuleType("distillm")
        package.__path__ = ["distillm"]
        package.__package__ = "distillm"
        sys.modules["distillm"] = package
    spec = importlib.util.spec_from_file_location(qualified, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


_multi = _load_without_package_init("distillm.multi_lora", "distillm/multi_lora.py")
_losses = _load_without_package_init("distillm.losses", "distillm/losses.py")
Conv1D = _multi.Conv1D
MultiLoRALayer = _multi.MultiLoRALayer
inject_multi_lora = _multi.inject_multi_lora
compute_sr_eotd = _losses.compute_sr_eotd
sr_eotd_loss = _losses.sr_eotd_loss


def _args(**overrides):
    values = dict(
        num_lora_branches=3,
        sreotd_lora_r=16,
        sreotd_lora_alpha=16,
        sreotd_topk=3,
        sreotd_sinkhorn_iters=5,
        sreotd_eps_min=0.05,
        sreotd_eps_max=1.0,
        sreotd_gamma=3.0,
        sreotd_rho=0.5,
        sreotd_beta=0.5,
        sreotd_sigma_b=1e-3,
        sreotd_tau=1.0,
        sreotd_freeze_backbone=False,
        gradient_checkpointing=False,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _gram_error(layer):
    A = layer.A.detach().float()
    branches, rank, _ = A.shape
    error = 0.0
    for i in range(branches):
        for j in range(branches):
            gram = A[i] @ A[j].transpose(0, 1)
            target = torch.eye(rank) if i == j else torch.zeros(rank, rank)
            error = max(error, (gram - target).abs().max().item())
    return error


def _parallel_matches_single(layer, x):
    layer.eval()
    singles = []
    for branch in range(layer.num_branches):
        layer.active_branch = branch
        singles.append(layer(x).detach())
    layer.active_branch = None
    parallel = layer(x.repeat_interleave(layer.num_branches, dim=0)).detach()
    parallel = parallel.view(x.size(0), layer.num_branches, *x.shape[1:-1], layer.d_out)
    error = 0.0
    for branch, single in enumerate(singles):
        error = max(error, (parallel[:, branch] - single).abs().max().item())
    layer.active_branch = None
    return error


def _show(title, out):
    Q = out["Q_tilde"][0, 0].detach().cpu()
    w = out["w_star"][0, 0].detach().cpu()
    P_T = out["P_T"][0, 0].detach().cpu()
    print(f"\n{title}")
    print("P_T =", [round(float(v), 4) for v in P_T])
    print("w*  =", [round(float(v), 4) for v in w])
    print("Q~ (rows are branches, columns are top-K tokens):")
    for branch, row in enumerate(Q):
        print(f"  m={branch}:", [round(float(v), 4) for v in row])
    return Q, w, P_T


def _distribution(probs):
    probs = torch.tensor(probs, dtype=torch.float32)
    probs = probs / probs.sum()
    return probs


def _scenario(name, teacher_probs, branch_probs, args):
    teacher = _distribution(teacher_probs)
    branches = torch.stack([_distribution(row) for row in branch_probs], dim=0)
    K = teacher.numel()
    hidden = torch.log(branches).view(1, 1, branches.size(0), K).requires_grad_(True)
    weight = torch.eye(K)
    teacher_logits = torch.log(teacher).view(1, 1, K)
    out = compute_sr_eotd(args, teacher_logits, hidden, weight, {"loss_mask": torch.ones(1, 1)})
    Q, w, P_T = _show(name, out)
    marginal = (out["Gamma"][0, 0].sum(dim=0) - P_T).abs().max().item()
    row_sum = (Q.sum(dim=-1) - 1).abs().max().item()
    mass = (w.sum() - 1).abs().item()
    print(f"column-marginal error {marginal:.2e} | row-sum error {row_sum:.2e} | |sum(w*)-1| {mass:.2e}")
    out["loss"].backward()
    grad = hidden.grad.detach().abs().sum().item()
    print(f"loss {float(out['loss'].detach()):.4f} | hidden grad {grad:.4e} | finite {bool(torch.isfinite(out['loss'].detach()))}")
    return {
        "Q": Q,
        "w": w,
        "marginal": marginal,
        "row_sum": row_sum,
        "mass": mass,
        "grad": grad,
        "loss": float(out["loss"].detach()),
    }


def _check_tiny_gpt2(args):
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        n_layer=1,
        n_head=4,
        n_embd=64,
        vocab_size=32,
        n_positions=32,
        n_ctx=32,
        bos_token_id=0,
        eos_token_id=0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    inject_multi_lora(model, args)
    block = model.transformer.h[0]
    assert isinstance(block.attn.c_attn, MultiLoRALayer)
    assert isinstance(block.attn.c_proj, MultiLoRALayer)
    assert isinstance(block.mlp.c_fc, MultiLoRALayer)
    assert isinstance(block.mlp.c_proj, MultiLoRALayer)
    assert not isinstance(model.lm_head, MultiLoRALayer)
    assert type(block.attn.c_attn.base_layer).__name__ == "Conv1D"
    ortho = max(_gram_error(layer) for layer in model.modules() if isinstance(layer, MultiLoRALayer))

    model.eval()
    ids = torch.randint(0, config.vocab_size, (2, 4))
    attention = torch.ones_like(ids)
    position = torch.arange(ids.size(1)).unsqueeze(0).expand_as(ids)
    hidden = model.forward_multi_branch(ids, attention_mask=attention, position_ids=position)
    match = 0.0
    for branch in range(args.num_lora_branches):
        model.set_active_branch(branch)
        single = model.transformer(
            input_ids=ids,
            attention_mask=attention,
            position_ids=position,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        match = max(match, (hidden[:, :, branch] - single).abs().max().item())
        generated = model.generate(ids, max_new_tokens=2, do_sample=False, pad_token_id=0)
        assert generated.shape == (ids.size(0), ids.size(1) + 2)

    model.train()
    model.set_active_branch(None)
    loss_args = _args(sreotd_topk=8)
    teacher_logits = torch.randn(ids.size(0), ids.size(1), config.vocab_size)
    train_hidden = model.forward_multi_branch(ids, attention_mask=attention, position_ids=position)
    loss, stats = sr_eotd_loss(
        loss_args,
        teacher_logits,
        train_hidden,
        model.lm_head.weight,
        {"loss_mask": torch.ones(ids.size(0), ids.size(1))},
    )
    loss.backward()
    lora_grad = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and (name.endswith(".A") or name.endswith(".B")):
            lora_grad += float(parameter.grad.abs().sum())
    print(f"[gpt2] ortho {ortho:.2e} | parallel-vs-branch {match:.2e} | loss {float(loss.detach()):.4f} | LoRA grad {lora_grad:.4e}")
    print("[gpt2] stats", {key: round(value, 4) for key, value in stats.items()})
    return ortho < 1e-4 and match < 1e-4 and torch.isfinite(loss).item() and lora_grad > 0.0


def main():
    torch.manual_seed(0)
    ok = True
    linear = MultiLoRALayer(nn.Linear(64, 32), num_branches=3, r=16, lora_alpha=16, sigma_b=1e-3)
    conv = MultiLoRALayer(Conv1D(32, 64), num_branches=3, r=16, lora_alpha=16, sigma_b=1e-3)
    x = torch.randn(2, 5, 64)
    linear_ortho, conv_ortho = _gram_error(linear), _gram_error(conv)
    linear_match, conv_match = _parallel_matches_single(linear, x), _parallel_matches_single(conv, x)
    b_std = float(linear.B.detach().float().std())
    print(f"[1] Linear ortho {linear_ortho:.2e} | Conv1D ortho {conv_ortho:.2e}")
    print(f"[1] Linear parallel error {linear_match:.2e} | Conv1D parallel error {conv_match:.2e}")
    print(f"[1] B std {b_std:.4e} (target 1e-3)")
    ok &= linear_ortho < 1e-4 and conv_ortho < 1e-4
    ok &= linear_match < 1e-5 and conv_match < 1e-5
    ok &= 5e-4 < b_std < 2e-3

    ok &= _check_tiny_gpt2(_args())

    sharp = 0.98
    rest = (1.0 - sharp) / 2.0
    specialists = [
        [sharp, rest, rest],
        [rest, sharp, rest],
        [rest, rest, sharp],
    ]
    args = _args()
    uni = _scenario("Unimodal teacher", [0.98, 0.01, 0.01], specialists, args)
    mix = _scenario("Skewed mixture 80/15/5", [0.80, 0.15, 0.05], specialists, args)

    def _close(result):
        return result["marginal"] < 1e-4 and result["row_sum"] < 1e-3 and result["mass"] < 1e-4 and result["grad"] > 0.0

    # Low teacher entropy raises epsilon, so the soft row penalty only partially
    # peaks w* on the matching branch. Higher-entropy mixtures recover the masses.
    ok &= _close(uni) and _close(mix)
    ok &= float(uni["w"][0]) > float(uni["w"][1]) and float(uni["w"][0]) > float(uni["w"][2])
    ok &= int(uni["Q"][0].argmax()) == 0 and float(uni["Q"][0, 0]) > 0.95
    ok &= float(mix["w"][0]) > float(mix["w"][1]) > float(mix["w"][2])
    ok &= (mix["w"] - torch.tensor([0.80, 0.15, 0.05])).abs().max().item() < 5e-3
    ok &= all(int(mix["Q"][m].argmax()) == m for m in range(3))

    from transformers import GPT2Config, GPT2LMHeadModel
    frozen_args = _args(sreotd_freeze_backbone=True)
    frozen = GPT2LMHeadModel(GPT2Config(
        n_layer=1, n_head=4, n_embd=64, vocab_size=32, n_positions=16, bos_token_id=0, eos_token_id=0,
    ))
    inject_multi_lora(frozen, frozen_args)
    trainable = [name for name, parameter in frozen.named_parameters() if parameter.requires_grad]
    frozen_ok = bool(trainable) and all(name.endswith(".A") or name.endswith(".B") for name in trainable)
    open_args = _args(sreotd_freeze_backbone=False)
    # A fresh module is required; inject is idempotent and will not unfreeze.
    opened = GPT2LMHeadModel(GPT2Config(
        n_layer=1, n_head=4, n_embd=64, vocab_size=32, n_positions=16, bos_token_id=0, eos_token_id=0,
    ))
    inject_multi_lora(opened, open_args)
    opened_ok = opened.lm_head.weight.requires_grad and opened.transformer.h[0].attn.c_attn.A.requires_grad
    print(f"[freeze] lora-only {frozen_ok} | full backbone {opened_ok}")
    ok &= frozen_ok and opened_ok

    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
