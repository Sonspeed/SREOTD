"""Checks for SR-EOTD mode diagnostics, LoRA merges, and eval-artifact layout.

Usage (from the repository root): python tools/test_sreotd_diagnostics.py
"""
import importlib.util
import os
import sys
import tempfile
import types

import torch

sys.path.insert(0, ".")


def _load(qualified, path):
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


multi = _load("distillm.multi_lora", "distillm/multi_lora.py")
losses = _load("distillm.losses", "distillm/losses.py")
diag = _load("distillm.sreotd_diagnostics", "distillm/sreotd_diagnostics.py")
hub_spec = importlib.util.spec_from_file_location("hf_hub_manager", "tools/hf_hub_manager.py")
hub = importlib.util.module_from_spec(hub_spec)
hub_spec.loader.exec_module(hub)


class _Args:
    num_lora_branches = 3
    sreotd_lora_r = 4
    sreotd_lora_alpha = 4
    sreotd_topk = 4
    sreotd_sinkhorn_iters = 5
    sreotd_eps_min = 0.05
    sreotd_eps_max = 1.0
    sreotd_gamma = 3.0
    sreotd_rho = 0.5
    sreotd_beta = 0.5
    sreotd_tau = 1.0


def test_regions_and_conservation():
    teacher = torch.tensor([[[0.70, 0.10, 0.10, 0.10], [0.25, 0.25, 0.25, 0.25]]])
    entropy = torch.tensor([[0.05, 0.80]])
    weights = torch.tensor([[[0.50, 0.30, 0.20], [1.0 / 3, 1.0 / 3, 1.0 / 3]]])
    target = teacher.unsqueeze(2).expand(1, 2, 3, 4).contiguous()
    probs = target.clone()
    probs[0, 1, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    probs[0, 1, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    probs[0, 1, 2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    mask = torch.tensor([[True, True]])
    top1 = torch.zeros(1, 2, 3, dtype=torch.long)
    top1[0, 1, 1] = 9
    report = diag.mode_collapse_diagnostics(probs, weights, target, teacher, entropy, mask, top1)
    low = report["regions"]["low"]
    high = report["regions"]["high"]
    medium = report["regions"]["medium"]
    assert low["num_tokens"] == 1
    assert high["num_tokens"] == 1
    assert medium["num_tokens"] == 0
    assert medium["pairwise_jsd_mean"] is None
    assert low["pairwise_jsd_mean"] < 1e-6
    assert low["pairwise_tv_distance"] < 1e-6
    assert low["barycentric_conservation_error"] < 1e-5
    assert low["top1_agreement_rate"] == 1.0
    assert high["top1_agreement_rate"] == 0.0
    assert high["pairwise_jsd_mean"] > 0.1
    branch = torch.tensor([0.5, 0.3, 0.2])
    expected_eff = torch.exp(-(branch * (branch + 1e-8).log()).sum()).item()
    assert abs(low["effective_num_branches"] - expected_eff) < 1e-5
    assert abs(low["branch_mass_std"] - branch.std(unbiased=False).item()) < 1e-5
    print("regions ok", {name: report["regions"][name]["num_tokens"] for name in ("low", "medium", "high")})


def test_merge_transpose_and_orthogonality():
    branches, rank, d_in, d_out = 2, 2, 4, 6
    factor_a = torch.randn(branches, rank, d_in)
    factor_b = torch.randn(branches, d_out, rank)
    scale = 0.5
    conv = {
        "block.c_attn.A": factor_a,
        "block.c_attn.B": factor_b,
        "block.c_attn.base_layer.weight": torch.zeros(d_in, d_out),
        "block.c_attn.base_layer.bias": torch.ones(d_out),
        "lm_head.weight": torch.randn(3, 3),
    }
    merged = multi.export_merged_branch_state_dict(conv, 0, scale)
    delta = scale * (factor_b[0] @ factor_a[0])
    assert merged["block.c_attn.weight"].shape == (d_in, d_out)
    assert torch.allclose(merged["block.c_attn.weight"].float(), delta.transpose(0, 1))
    assert torch.equal(merged["block.c_attn.bias"], torch.ones(d_out))
    assert "block.c_attn.A" not in merged
    assert torch.equal(merged["lm_head.weight"], conv["lm_head.weight"])

    linear = {
        "block.fc.A": factor_a,
        "block.fc.B": factor_b,
        "block.fc.base_layer.weight": torch.zeros(d_out, d_in),
    }
    merged_linear = multi.export_merged_branch_state_dict(linear, 1, scale)
    expected = scale * (factor_b[1] @ factor_a[1])
    assert merged_linear["block.fc.weight"].shape == (d_out, d_in)
    assert torch.allclose(merged_linear["block.fc.weight"].float(), expected)

    ortho = multi.orthogonal_branch_A(3, 4, 32, "cpu", torch.float32)
    pairs = multi.branch_pair_metrics(ortho, torch.randn(3, 8, 4), 1.0)
    for pair in pairs:
        if pair["i"] == pair["j"]:
            assert abs(pair["A_gram_fro_over_sqrt_r"] - 1.0) < 1e-4
        else:
            assert abs(pair["A_gram_fro_over_sqrt_r"]) < 1e-4
    print("merge and orthogonality ok")


def test_pad_and_forks():
    stacked = diag.pad_time([torch.ones(2, 3), torch.ones(4, 3)], pad_value=0)
    assert tuple(stacked.shape) == (2, 4, 3)
    assert float(stacked[0, 2:].sum()) == 0.0
    mask = diag.pad_time([torch.ones(2, dtype=torch.bool), torch.ones(3, dtype=torch.bool)], pad_value=False)
    assert mask.dtype == torch.bool
    assert not bool(mask[0, 2])

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def decode(self, ids, skip_special_tokens=False):
            return ",".join(str(int(token)) for token in ids)

    forks = diag.build_fork_snapshots(
        _Tokenizer(),
        input_ids=torch.tensor([[0, 4, 5, 6]]),
        topk_indices=torch.tensor([[[7, 8], [9, 10], [11, 12], [13, 14]]]),
        P_T=torch.full((1, 4, 2), 0.5),
        P_m=torch.full((1, 4, 3, 2), 0.5),
        Q_tilde=torch.full((1, 4, 3, 2), 0.5),
        w_star=torch.full((1, 4, 3), 1.0 / 3),
        H_hat=torch.tensor([[0.1, 0.2, 0.95, 0.4]]),
        loss_mask=torch.tensor([[False, True, True, True]]),
        sample_ids=[4],
        limit=1,
        top_tokens=2,
    )
    assert len(forks) == 1
    assert forks[0]["position"] == 2
    assert forks[0]["sample_id"] == 4
    assert forks[0]["prefix"] == "4,5"
    assert forks[0]["candidates"][0]["token"] == "11"
    assert len(forks[0]["w_star"]) == 3
    print("pad and forks ok")


def test_loss_keys_and_artifact_discovery():
    teacher = torch.randn(1, 2, 20)
    hidden = torch.randn(1, 2, 3, 8)
    weight = torch.randn(20, 8)
    out = losses.compute_sr_eotd(_Args(), teacher, hidden, weight, {"loss_mask": torch.ones(1, 2)})
    assert out["teacher_topk_logits"].shape == (1, 2, 4)
    assert out["student_topk_logits"].shape == (1, 2, 3, 4)
    assert out["Gamma"].shape == (1, 2, 3, 4)
    jsd = diag.token_pairwise_jsd(out["P_m"], torch.tensor([[1.0, 0.0]]))
    assert jsd.shape == (1,)
    with tempfile.TemporaryDirectory() as root:
        for name in ("dolly", "self_inst"):
            folder = os.path.join(root, name)
            os.makedirs(folder)
            with open(os.path.join(folder, "branch_generations.jsonl"), "w") as handle:
                handle.write('{"sample_id": 0}\n')
        found = hub._benchmark_dirs(root)
        assert [name for name, _ in found] == ["dolly", "self_inst"]
        card = hub._dataset_card("owner/repo-eval-data", ["dolly"])
        assert "sreotd_logits_and_modes.pt" in card
        assert "owner/repo-eval-data" in card
    print("loss keys and artifact discovery ok")


if __name__ == "__main__":
    test_regions_and_conservation()
    test_merge_transpose_and_orthogonality()
    test_pad_and_forks()
    test_loss_keys_and_artifact_discovery()
    print("all diagnostics checks passed")
