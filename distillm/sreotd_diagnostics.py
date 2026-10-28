"""Token-level mode-collapse diagnostics for SR-EOTD evaluation.

The quantities here are computed on the teacher top-K support after the same
damped log-Sinkhorn plan used by the training loss. They do not change that loss.
"""

import math

import torch


def pad_time(tensors, pad_value=0):
    """Right-pad variable-length ``[T, ...]`` tensors and stack them to ``[N, T, ...]``."""
    if not tensors:
        raise ValueError("pad_time expected at least one tensor")
    max_t = max(int(tensor.size(0)) for tensor in tensors)
    padded = []
    for tensor in tensors:
        if tensor.size(0) == max_t:
            padded.append(tensor)
            continue
        extra = max_t - int(tensor.size(0))
        filler = tensor.new_full((extra,) + tuple(tensor.shape[1:]), pad_value)
        padded.append(torch.cat([tensor, filler], dim=0))
    return torch.stack(padded, dim=0)


def pairwise_jsd_and_tv(probs):
    """Mean pairwise JSD and total variation on the last distribution axis.

    ``probs`` is ``[N, T, M, K]``. Both results are ``[N, T]``.
    """
    branches = int(probs.size(-2))
    jsd = torch.zeros(probs.shape[:2], device=probs.device, dtype=torch.float32)
    tv = torch.zeros_like(jsd)
    if branches < 2:
        return jsd, tv
    pairs = 0
    for left in range(branches):
        for right in range(left + 1, branches):
            p = probs[:, :, left].float()
            q = probs[:, :, right].float()
            mid = 0.5 * (p + q)
            left_kl = (p * (torch.log(p + 1e-8) - torch.log(mid + 1e-8))).sum(dim=-1)
            right_kl = (q * (torch.log(q + 1e-8) - torch.log(mid + 1e-8))).sum(dim=-1)
            jsd = jsd + 0.5 * (left_kl + right_kl)
            tv = tv + 0.5 * (p - q).abs().sum(dim=-1)
            pairs += 1
    return jsd / pairs, tv / pairs


def token_pairwise_jsd(probs, loss_mask):
    """Response-token mean of the pairwise JSD. Returns ``[N]``."""
    jsd, _ = pairwise_jsd_and_tv(probs)
    mask = loss_mask.to(device=jsd.device, dtype=jsd.dtype)
    if mask.dim() == 3:
        mask = mask.squeeze(-1)
    denom = mask.sum(dim=-1).clamp_min(1.0)
    return (jsd * mask).sum(dim=-1) / denom


def _finite(value):
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _masked_mean(value, region):
    if int(region.sum()) == 0:
        return None
    return _finite(value[region].float().mean())


def mode_collapse_diagnostics(P_m, w_star, Q_tilde, P_T, H_hat, loss_mask, top1_ids):
    """Collapse-vs-diversity stats on low, medium, and high teacher-entropy tokens.

    Regions use the normalized teacher entropy on positions where ``loss_mask`` is true:
    low ``H < 0.2``, medium ``0.2 <= H <= 0.5``, high ``H > 0.5``.
    """
    P_m = P_m.float()
    w_star = w_star.float()
    Q_tilde = Q_tilde.float()
    P_T = P_T.float()
    H_hat = H_hat.float()
    mask = loss_mask.bool()
    if mask.dim() == 3:
        mask = mask.squeeze(-1)
    jsd, tv = pairwise_jsd_and_tv(P_m)
    agreement = (top1_ids == top1_ids[..., :1]).all(dim=-1)
    mass_std = w_star.std(dim=-1, unbiased=False)
    weight_entropy = -(w_star * (w_star + 1e-8).log()).sum(dim=-1)
    effective = weight_entropy.exp()
    reconstructed = (w_star.unsqueeze(-1) * Q_tilde).sum(dim=-2)
    conservation = (reconstructed - P_T).abs().amax(dim=-1)

    regions = {
        "low": mask & (H_hat < 0.2),
        "medium": mask & (H_hat >= 0.2) & (H_hat <= 0.5),
        "high": mask & (H_hat > 0.5),
    }
    ranges = {
        "low": [0.0, 0.2],
        "medium": [0.2, 0.5],
        "high": [0.5, 1.0],
    }
    report = {}
    for name, region in regions.items():
        count = int(region.sum().item())
        error = None
        if count:
            error = _finite(conservation[region].max())
        report[name] = {
            "entropy_range": ranges[name],
            "num_tokens": count,
            "pairwise_jsd_mean": _masked_mean(jsd, region),
            "pairwise_tv_distance": _masked_mean(tv, region),
            "top1_agreement_rate": _masked_mean(agreement.float(), region),
            "branch_mass_std": _masked_mean(mass_std, region),
            "effective_num_branches": _masked_mean(effective, region),
            "barycentric_conservation_error": error,
        }
    return {
        "definitions": {
            "pairwise_jsd_mean": "Mean over tokens of the mean JSD between branch pairs on the teacher top-K support.",
            "pairwise_tv_distance": "Mean over tokens of the mean total variation between branch pairs on that support.",
            "top1_agreement_rate": "Fraction of tokens where all branches share the same full-vocabulary top-1 id.",
            "branch_mass_std": "Mean over tokens of the standard deviation of w* across branches.",
            "effective_num_branches": "Mean over tokens of exp(-sum_m w*_m log(w*_m + 1e-8)).",
            "barycentric_conservation_error": "Max over tokens and support tokens of |sum_m w*_m Q_m(v) - P_T(v)|.",
        },
        "num_valid_tokens": int(mask.sum().item()),
        "regions": report,
    }


def _decode_prefix(tokenizer, token_ids):
    ids = [int(token) for token in token_ids]
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is not None:
        while ids and ids[0] == int(pad_id):
            ids.pop(0)
    if not ids:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_fork_snapshots(
    tokenizer,
    input_ids,
    topk_indices,
    P_T,
    P_m,
    Q_tilde,
    w_star,
    H_hat,
    loss_mask,
    sample_ids,
    limit=100,
    top_tokens=10,
):
    """The ``limit`` response positions with the largest teacher entropy."""
    mask = loss_mask.bool()
    scores = H_hat.float().masked_fill(~mask, -1.0)
    valid = int(mask.sum().item())
    count = min(int(limit), valid)
    if count <= 0:
        return []
    _, flat_index = torch.topk(scores.reshape(-1), k=count)
    time = int(H_hat.size(1))
    width = min(int(top_tokens), int(P_T.size(-1)))
    branches = int(w_star.size(-1))
    snapshots = []
    for index in flat_index.tolist():
        sample = int(index // time)
        position = int(index % time)
        prefix_ids = input_ids[sample, :position + 1].tolist()
        candidates = []
        for slot in range(width):
            token_id = int(topk_indices[sample, position, slot])
            token = tokenizer.decode([token_id]) if tokenizer is not None else str(token_id)
            candidates.append({
                "token_id": token_id,
                "token": token,
                "P_T": round(float(P_T[sample, position, slot]), 6),
                "Q_branch": [round(float(Q_tilde[sample, position, branch, slot]), 6) for branch in range(branches)],
                "P_branch": [round(float(P_m[sample, position, branch, slot]), 6) for branch in range(branches)],
            })
        snapshots.append({
            "sample_id": int(sample_ids[sample]),
            "position": position,
            "teacher_norm_entropy": round(float(H_hat[sample, position]), 6),
            "prefix": _decode_prefix(tokenizer, prefix_ids) if tokenizer is not None else "",
            "w_star": [round(float(w_star[sample, position, branch]), 6) for branch in range(branches)],
            "candidates": candidates,
        })
    return snapshots
