import math

import torch
import torch.nn.functional as F

from .multi_lora import project_topk_lm_head


def ab_div(logits, teacher_logits, no_model_batch, alpha, beta):
    """Calculate D^{(alpha, beta)} divergence."""
    log_p = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    log_q = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    eps = 1e-8

    if abs(alpha) < eps and abs(beta) < eps:
        # Case 1: alpha = 0, beta = 0
        divergence = 0.5 * torch.sum((log_q - log_p).pow(2), dim=-1)

    elif abs(alpha) < eps:
        # Case 2: alpha = 0, beta != 0
        safe_log_ratio_div_beta = torch.where(torch.isfinite(log_q - log_p), log_q - log_p, 0.0)
        divergence = torch.sum(torch.exp(beta * log_q) * (beta * safe_log_ratio_div_beta - 1) + torch.exp(beta * log_p), dim=-1) / (beta ** 2)

    elif abs(beta) < eps:
        # Case 3: beta = 0, alpha != 0
        safe_log_ratio_div_alpha = torch.where(torch.isfinite(log_p - log_q), log_p - log_q, 0.0)
        divergence = torch.sum(torch.exp(alpha * log_p) * (alpha * safe_log_ratio_div_alpha - 1) + torch.exp(alpha * log_q), dim=-1) / (alpha ** 2)

    elif abs(alpha + beta) < eps:
        # Case 4: alpha + beta = 0
        safe_log_r = torch.where(torch.isfinite(log_q - log_p), log_q - log_p, 0.0)
        divergence = torch.sum(alpha * safe_log_r + torch.exp(-alpha * safe_log_r) - 1, dim=-1) / (alpha ** 2)

    else:
        # General case
        apb = alpha + beta
        term1 = torch.exp(torch.logsumexp(alpha * log_p + beta * log_q, dim=-1))
        term2 = (alpha / apb) * torch.exp(torch.logsumexp(apb * log_p, dim=-1))
        term3 = (beta / apb) * torch.exp(torch.logsumexp(apb * log_q, dim=-1))
        divergence = - (term1 - term2 - term3) / (alpha * beta)
        
    mask = (no_model_batch["label"] != -100).float()
    
    # Replace any NaN/inf values in the divergence with 0.0 for safe masking
    safe_divergence = torch.where(torch.isfinite(divergence), divergence, 0.0)
    
    # Apply the mask and compute the final mean loss
    masked_sum = (safe_divergence * mask).sum()
    mask_sum = mask.sum()
    
    # Avoid division by zero if the mask is empty
    loss = masked_sum / mask_sum if mask_sum > 0 else masked_sum
    
    return loss



def bdkd(logits, teacher_logits, no_model_batch):
    def entropy(logits):
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        return -torch.sum(probs * log_probs, dim=-1)

    entropy_student = entropy(logits)  # (B, S)
    entropy_teacher = entropy(teacher_logits)  # (B, S)

    weight_student = torch.where(entropy_student > entropy_teacher, 2.0, 1.0)
    weight_teacher = torch.where(entropy_teacher > entropy_student, 2.0, 1.0)

    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss1 = -torch.sum(x * mask.view(-1) * weight_teacher.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss2 = -torch.sum(x * mask.view(-1) * weight_student.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    return distil_loss1 + distil_loss2


def forward_kl(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def reverse_kl(logits, teacher_logits, no_model_batch):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def get_ratio(teacher_logits, logits, mu=0.5):
    # [B, L, V]
    teacher_logits = torch.masked_fill(teacher_logits, torch.isinf(teacher_logits), 0).to(torch.float32)
    logits = torch.masked_fill(logits, torch.isinf(logits), 0).to(torch.float32)

    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32).detach()

    re_teacher_probs, idx = teacher_probs.sort(dim=-1, descending=True)
    re_student_probs = student_probs.gather(dim=-1, index=idx)

    errors = torch.abs(re_teacher_probs - re_student_probs)

    cum_sum = torch.cumsum(re_teacher_probs, dim=-1)  # B,L,V
    mask = cum_sum > mu
    mask[:, :, 0] = False

    s1 = torch.masked_fill(errors, mask, 0.0).sum(dim=-1)
    s2 = torch.masked_fill(errors, ~mask, 0.0).sum(dim=-1)

    return s1 / (s1 + s2), s2 / (s1 + s2)


def get_kl(teacher_logits, logits, inf_mask, mask, ratio=None):
    # ratio: [B,L]
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_prod_probs = torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    teacher_x = torch.sum(teacher_prod_probs, dim=-1).view(-1)

    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)  # [B,L]->[BL]

    if ratio == None:
        distil_loss = torch.sum((teacher_x - x) * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    else:
        distil_loss = torch.sum((teacher_x - x) * ratio.view(-1) * mask.view(-1), dim=0) / torch.sum(mask.view(-1),
                                                                                                     dim=0)
    return distil_loss


def AKL(teacher_logits, logits, no_model_batch):
    inf_mask = torch.isinf(logits)  # [batch, seq, vocab]
    mask = (no_model_batch["label"] != -100).int()  # [batch, seq]

    h_ratio, l_ratio = get_ratio(teacher_logits, logits)
    distil_loss = get_kl(teacher_logits, logits, inf_mask, mask, h_ratio) + get_kl(logits, teacher_logits, inf_mask,
                                                                                   mask, l_ratio)
    return distil_loss




def js_distance(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1 - lam) * teacher_probs + lam * student_probs

    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = lam * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss += (1 - lam) * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    return distil_loss


def wsd(logits, teacher_logits, no_model_batch, lam=0.5):
    for_kl = forward_kl(logits, teacher_logits, no_model_batch)
    rev_kl = reverse_kl(logits, teacher_logits, no_model_batch)
    distil_loss = (1 - lam) * for_kl + lam * rev_kl

    return distil_loss


def tv_distance(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    prod_probs = 0.5 * torch.masked_fill(torch.abs(teacher_probs - student_probs), inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1 - lam) * student_probs
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1 - lam) * teacher_probs + lam * student_probs

    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def f_divergence(q_logits, p_logits, alpha, iw_clip=1e3):
    # assert isinstance(alpha, float)
    inf_mask = torch.isinf(q_logits) | torch.isinf(p_logits)
    q_logits = torch.masked_fill(q_logits, inf_mask, 0)
    p_logits = torch.masked_fill(p_logits, inf_mask, 0)
    q_prob = torch.nn.functional.softmax(q_logits, dim=-1).detach()
    p_prob = torch.nn.functional.softmax(p_logits, dim=-1).detach()
    q_log_prob = torch.nn.functional.log_softmax(q_logits, dim=-1)  # gradient is only backpropagated here

    importance_ratio = p_prob / q_prob
    if abs(alpha) < 1e-3:
        importance_ratio = importance_ratio.clamp(0, iw_clip)
        f = -importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio.log() - 1.0
    elif abs(alpha - 1.0) < 1e-3:
        f = importance_ratio * importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio
    else:
        iw_alpha = torch.pow(importance_ratio, alpha)
        iw_alpha = iw_alpha.clamp(0, iw_clip)
        f = iw_alpha / alpha / (alpha - 1.0)
        f_base = 1.0 / alpha / (alpha - 1.0)
        rho_f = iw_alpha / alpha + f_base

    loss = torch.sum(q_prob * (f - f_base), dim=-1)
    grad_loss = -torch.sum(q_prob * rho_f * q_log_prob, dim=-1)
    return loss, grad_loss


def alphanet(logits, teacher_logits, no_model_batch, alpha, beta):
    loss1 = ab_div(logits, teacher_logits, no_model_batch, alpha, 1 - alpha)
    loss2 = ab_div(logits, teacher_logits, no_model_batch, beta, 1 - beta)
    if loss1 > loss2:
        return loss1
    return loss2


@torch.no_grad()
def sparse_sinkhorn_sequence(
        pi_r,          # (B, T, N)
        pi_theta,      # (B, T, N)
        K,             # (N, N) sparse kernel exp(-C / epsilon)
        Kt,            # transpose of K
        starts: torch.Tensor,
        ends: torch.Tensor,
        epsilon: float = 1.0,
        max_iter: int = 10,
        tol: float = 1e-6,
        chunk_rows: int = 1024,
):
    B, T, N = pi_r.shape
    M = B * T
    device = pi_r.device
    dtype = K.dtype

    tiny = torch.finfo(dtype).tiny
    eps_damp = 1e-12

    pi_r_flat = pi_r.reshape(M, N).to(dtype=dtype)
    pi_theta_flat = pi_theta.reshape(M, N).to(dtype=dtype)

    u = pi_theta_flat.clone()
    v = pi_r_flat.clone()

    active_mask = torch.zeros(M, dtype=torch.bool, device=device)
    for b in range(B):
        s = int(starts[b].item())
        e = min(T, int(ends[b].item()))
        if s < e:
            active_mask[b * T + s:b * T + e] = True

    for _ in range(max_iter):
        active_all = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
        if active_all.numel() == 0:
            break
        for active in active_all.split(chunk_rows):
            u_act = u[active]
            v_act = v[active]

            pi_theta_act = pi_theta_flat[active]
            pi_r_act = pi_r_flat[active]

            support_theta = pi_theta_act > 0
            support_r = pi_r_act > 0

            Kv = torch.sparse.mm(K, v_act.t()).t().clamp_min(tiny)
            u_new = torch.where(support_theta, pi_theta_act / Kv, torch.zeros_like(u_act))

            Ktu = torch.sparse.mm(Kt, u_new.t()).t().clamp_min(tiny)
            v_new = torch.where(support_r, pi_r_act / Ktu, torch.zeros_like(v_act))

            rel_u = (u_new - u_act).abs() / torch.maximum(u_new.abs(), u_act.abs()).clamp_min(eps_damp)
            rel_v = (v_new - v_act).abs() / torch.maximum(v_new.abs(), v_act.abs()).clamp_min(eps_damp)
            delta = torch.maximum(rel_u.max(dim=1).values, rel_v.max(dim=1).values)

            u[active] = u_new
            v[active] = v_new

            active_mask[active[delta <= tol]] = False

    f = u.clamp_min_(tiny).log_().mul_(epsilon)
    f.masked_fill_(pi_theta_flat <= 0, 0.0)
    f_labels = f.reshape(B, T, N)

    torch.cuda.empty_cache()
    return f_labels


@torch.no_grad()
def sparse_sinkhorn_self_sequence(
        p,             # (B, T, N)
        K,             # (N, N) sparse kernel exp(-C / epsilon), symmetric
        starts: torch.Tensor,
        ends: torch.Tensor,
        epsilon: float = 1.0,
        max_iter: int = 10,
        tol: float = 1e-6,
        chunk_rows: int = 1024,
):
    eta = 0.5
    B, T, N = p.shape
    M = B * T
    device = p.device
    tiny = 1e-16

    p_flat = p.view(M, N)

    a = torch.ones(M, N, device=device)

    active_list = [
        torch.arange(starts[b].item(), min(T, ends[b].item()), device=device) + b * T
        for b in range(B)
    ]
    active = torch.cat(active_list, dim=0) if len(active_list) > 0 else torch.empty(0, dtype=torch.long, device=device)

    for _ in range(max_iter):
        if active.numel() == 0:
            break
        keep = []
        for act in active.split(chunk_rows):
            a_act = a[act]
            s = torch.sparse.mm(K, (p_flat[act] * a_act).t()).t()
            a_prop = 1.0 / (s + tiny)
            a_new = torch.exp(
                (1 - eta) * torch.log(a_act + tiny) +
                eta * torch.log(a_prop + tiny)
            )
            delta = ((a_new - a_act).abs() / torch.maximum(a_new.abs(), a_act.abs()).clamp_min(tiny)).max(dim=1).values
            a[act] = a_new
            keep.append(act[delta > tol])
            del a_act, s, a_prop, a_new, delta
        active = torch.cat(keep) if keep else active[:0]

    f_sub = epsilon * torch.log(a + tiny)
    f_labels = f_sub.view(B, T, N)

    del a, f_sub
    torch.cuda.empty_cache()

    return f_labels


def amid(logits, teacher_logits, no_model_batch, args, mat_K=None, mat_Kt=None, student_temp=1.0, **kwargs):

    # Teacher and Studnet distribution
    logits = logits / student_temp

    p = F.softmax(teacher_logits, dim=-1)
    q = F.softmax(logits, dim=-1)
    logp = F.log_softmax(teacher_logits, dim=-1)
    logq = F.log_softmax(logits, dim=-1)


    # Basic
    alpha = args.amid_alpha
    lam = args.amid_lam
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)


    # Assistant distribution
    if lam <= 0.0:
        r = q
        logr = F.log_softmax(logits, dim=-1)
    elif lam >= 1.0:
        r = p
        logr = F.log_softmax(teacher_logits, dim=-1)
    else:
        if alpha >= 1.0:
            logr_unnorm = lam * logp + (1.0 - lam) * logq
            r = F.softmax(logr_unnorm, dim=-1)
            logr = F.log_softmax(logr_unnorm, dim=-1)
        else:
            t1 = math.log(lam)       + 0.5 * (1.0 - alpha) * logp
            t2 = math.log(1.0 - lam) + 0.5 * (1.0 - alpha) * logq

            logr_unnorm = 2.0 / (1.0 - alpha) * torch.logaddexp(t1, t2)
            r = F.softmax(logr_unnorm, dim=-1)
            logr = F.log_softmax(logr_unnorm, dim=-1)
    del alpha, lam


    # Divergence
    div_name = args.amid_div_name
    div_order = args.amid_div_order

    if div_name == "fkl":
        if div_order == "pr":
            prod_probs = torch.masked_fill(p * (logp - logr), inf_mask, 0)
        elif div_order == "qr":
            prod_probs = torch.masked_fill(q * (logq - logr), inf_mask, 0)
        elif div_order == "rp":
            prod_probs = torch.masked_fill(r * (logr - logp), inf_mask, 0)
        elif div_order == "rq":
            prod_probs = torch.masked_fill(r * (logr - logq), inf_mask, 0)
        else:
            raise ValueError(f"Unknown amid-div-order: {div_order}")
        x = torch.sum(prod_probs, dim=-1).view(-1)
        distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
        return distil_loss

    elif div_name == "ab":
        # ab_alpha, ab_beta = args.ab_alpha, args.ab_beta
        ab_alpha, ab_beta = 0.2, 0.7
        apb = ab_alpha + ab_beta

        if div_order == "pr":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logp + ab_beta * logr, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * logp, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * logr, dim=-1))
        elif div_order == "qr":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logq + ab_beta * logr, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * logq, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * logr, dim=-1))
        elif div_order == "rp":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logr + ab_beta * logp, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * logr, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * logp, dim=-1))
        elif div_order == "rq":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logr + ab_beta * logq, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * logr, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * logq, dim=-1))
        else:
            raise ValueError(f"Unknown amid-div-order: {div_order}")
        divergence = - (term1 - term2 - term3) / (ab_alpha * ab_beta)

        safe_divergence = torch.where(torch.isfinite(divergence), divergence, 0.0)

        masked_sum = (safe_divergence * mask).sum()
        mask_sum = mask.sum()
        loss = masked_sum / mask_sum if mask_sum > 0 else masked_sum
        return loss

    elif div_name in ("wass", "sink"):
        # wass: entropy-regularized Wasserstein distance (ablation)
        # sink: Sinkhorn divergence (WASD)
        if div_order != "pr":
            raise NotImplementedError(f"amid-div-name '{div_name}' supports amid-div-order 'pr' only")
        if mat_K is None or mat_Kt is None:
            raise ValueError("Sparse cost kernel (mat_K, mat_Kt) is required for 'wass'/'sink'")

        starts = mask.float().argmax(dim=1)
        ends = mask.size(1) - mask.flip(dims=[1]).float().argmax(dim=1)

        f_pr = sparse_sinkhorn_sequence(
            p, r.detach(), mat_K, mat_Kt, starts, ends,
            epsilon=args.wd_epsilon, max_iter=args.wd_max_iter, tol=args.wd_tol,
            chunk_rows=args.wd_sinkhorn_chunk,
        )
        if div_name == "sink":
            f_rr = sparse_sinkhorn_self_sequence(
                r.detach(), mat_K, starts, ends,
                epsilon=args.wd_epsilon, max_iter=args.wd_max_iter, tol=args.wd_tol,
                chunk_rows=args.wd_sinkhorn_chunk,
            )
            divergence = r * (f_pr - f_rr)
        else:
            divergence = r * f_pr

        safe_divergence = torch.where(torch.isfinite(divergence), divergence, 0.0)
        safe_divergence = safe_divergence.sum(dim=-1)
        masked_sum = (safe_divergence * mask).sum()
        mask_sum = mask.sum()
        loss = masked_sum / mask_sum if mask_sum > 0 else masked_sum
        return args.wd_lambda * loss

    else:
        raise ValueError(f"Unknown amid-div-name: {div_name}")


SREOTD_STAT_KEYS = (
    "H_teacher",
    "H_target_weighted",
    "H_student_weighted",
    "w_max",
    "w_min",
    "jsd_high_H",
    "jsd_low_H",
)


def _masked_mean(value, mask):
    denom = mask.sum()
    if float(denom) == 0.0:
        return value.new_zeros(())
    return (value * mask).sum() / denom


def _sreotd_stats(H_hat, Q_tilde, log_P_m, P_m, w_star, loss_mask, K):
    """Scalar diagnostics. Entropies are normalized by log(K) so they lie in [0, 1]."""
    log_k = math.log(K) if K > 1 else 1.0
    mask = loss_mask
    H_tok = H_hat.squeeze(-1).squeeze(-1)  # [B, T]
    log_Q = torch.log(Q_tilde + 1e-8)
    H_target = -(Q_tilde * log_Q).sum(dim=-1) / log_k
    H_student = -(P_m * log_P_m).sum(dim=-1) / log_k
    H_target_w = (w_star * H_target).sum(dim=-1)
    H_student_w = (w_star * H_student).sum(dim=-1)

    mean_p = 0.5 * (P_m + Q_tilde)
    log_mean = torch.log(mean_p + 1e-8)
    jsd_m = 0.5 * (P_m * (log_P_m - log_mean)).sum(dim=-1)
    jsd_m = jsd_m + 0.5 * (Q_tilde * (log_Q - log_mean)).sum(dim=-1)
    jsd_tok = (w_star * jsd_m).sum(dim=-1)

    high = mask * (H_tok >= 0.5).to(dtype=mask.dtype)
    low = mask * (H_tok < 0.5).to(dtype=mask.dtype)
    w_max = w_star.max(dim=-1).values
    w_min = w_star.min(dim=-1).values

    values = {
        "H_teacher": _masked_mean(H_tok, mask),
        "H_target_weighted": _masked_mean(H_target_w, mask),
        "H_student_weighted": _masked_mean(H_student_w, mask),
        "w_max": _masked_mean(w_max, mask),
        "w_min": _masked_mean(w_min, mask),
        "jsd_high_H": _masked_mean(jsd_tok, high),
        "jsd_low_H": _masked_mean(jsd_tok, low),
    }
    return {key: float(values[key].detach()) for key in SREOTD_STAT_KEYS}


def compute_sr_eotd(args, teacher_logits, student_hidden_states, lm_head_weight, no_model_batch):
    """Semi-relaxed entropic OT decomposition of the teacher top-K distribution.

    Column marginals are matched to the teacher (hard). Row marginals are only
    pulled toward the uniform ``1/M`` with damping ``tau_row = rho / (rho + eps)``.
    The transport plan is a constant; gradients flow through the student
    distributions ``P_m`` only.
    """
    if "loss_mask" not in no_model_batch:
        raise KeyError("sr_eotd_loss expects no_model_batch['loss_mask']")

    K = min(int(args.sreotd_topk), int(teacher_logits.size(-1)))
    tau = float(getattr(args, "sreotd_tau", 1.0))
    eps_min = float(args.sreotd_eps_min)
    eps_max = float(args.sreotd_eps_max)
    gamma = float(args.sreotd_gamma)
    rho = float(args.sreotd_rho)
    beta = float(args.sreotd_beta)
    iters = int(args.sreotd_sinkhorn_iters)
    log_k = math.log(K) if K > 1 else 1.0

    with torch.no_grad():
        # Top-k in the teacher dtype, then upcast the K-slice. Casting the full
        # vocabulary to float32 here would copy [B, T, V] and dominate eval memory.
        z_topk, topk_indices = torch.topk(teacher_logits, k=K, dim=-1)
        z_topk = z_topk.float()
        P_T = torch.softmax(z_topk / tau, dim=-1)
        if K == 1:
            H_hat = torch.zeros(*P_T.shape[:-1], 1, 1, device=P_T.device, dtype=P_T.dtype)
        else:
            token_entropy = -(P_T * torch.log(P_T + 1e-8)).sum(dim=-1, keepdim=True) / log_k
            H_hat = token_entropy.unsqueeze(-1)
        epsilon = eps_min + (eps_max - eps_min) * torch.exp(-gamma * H_hat)
        tau_row = rho / (rho + epsilon)

    z_m = project_topk_lm_head(student_hidden_states, lm_head_weight, topk_indices)
    log_P_m = F.log_softmax(z_m.float(), dim=-1)
    P_m = torch.exp(log_P_m)
    M = P_m.size(-2)
    log_M = math.log(M)

    with torch.no_grad():
        S = torch.log(P_m.detach() + 1e-8) / epsilon
        for _ in range(iters):
            S = S - torch.logsumexp(S, dim=-2, keepdim=True) + torch.log(P_T.unsqueeze(-2) + 10e-9)
            S = S - tau_row * (torch.logsumexp(S, dim=-1, keepdim=True) + log_M)
        S = S - torch.logsumexp(S, dim=-2, keepdim=True) + torch.log(P_T.unsqueeze(-2) + 10e-9)
        Gamma = torch.exp(S)
        w_star = Gamma.sum(dim=-1)
        Q_tilde = Gamma / (w_star.unsqueeze(-1) + 1e-8)

    L_fwd = (Q_tilde * (torch.log(Q_tilde + 1e-8) - log_P_m)).sum(dim=-1)
    L_rev = (P_m * (log_P_m - torch.log(Q_tilde + 1e-8))).sum(dim=-1)
    L_token = (w_star * (L_fwd + H_hat.squeeze(-1) * beta * L_rev)).sum(dim=-1)

    loss_mask = no_model_batch["loss_mask"]
    if loss_mask.dim() == 3:
        loss_mask = loss_mask.squeeze(-1)
    loss_mask = loss_mask.to(device=L_token.device, dtype=L_token.dtype)
    if loss_mask.shape != L_token.shape:
        raise ValueError(
            f"loss_mask shape {tuple(loss_mask.shape)} does not match L_token {tuple(L_token.shape)}"
        )
    denom = loss_mask.sum().clamp_min(1.0)
    loss = (L_token * loss_mask).sum() / denom

    stats = _sreotd_stats(H_hat, Q_tilde, log_P_m.detach(), P_m.detach(), w_star, loss_mask, K)
    return {
        "loss": loss,
        "stats": stats,
        "Q_tilde": Q_tilde.detach(),
        "w_star": w_star.detach(),
        "Gamma": Gamma.detach(),
        "H_hat": H_hat.detach(),
        "P_T": P_T.detach(),
        "P_m": P_m.detach(),
        "topk_indices": topk_indices.detach(),
        "teacher_topk_logits": z_topk.detach(),
        "student_topk_logits": z_m.detach(),
    }


def sr_eotd_loss(args, teacher_logits, student_hidden_states, lm_head_weight, no_model_batch):
    """Weighted forward KL plus entropy-modulated reverse KL on the OT targets."""
    out = compute_sr_eotd(args, teacher_logits, student_hidden_states, lm_head_weight, no_model_batch)
    return out["loss"], out["stats"]
