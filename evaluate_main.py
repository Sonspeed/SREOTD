from data_utils.prompt_datasets import PromptDataset
from transformers import GenerationConfig
import os
import nltk
for pkg in ['punkt',]:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg)

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import json
from utils import print_rank, save_rank, all_gather

from rouge_metric import compute_metrics, per_example_rouge

from distillm.losses import compute_sr_eotd
from distillm.multi_lora import (
    get_lm_head_bias,
    get_lm_head_weight,
    lora_subspace_orthogonality,
    unwrap_model,
)
from distillm.sreotd_diagnostics import (
    build_fork_snapshots,
    mode_collapse_diagnostics,
    pad_time,
    token_pairwise_jsd,
)

torch.set_num_threads(4)


def prepare_dataset_main(args, tokenizer):
    data = {}
    data["test"] = PromptDataset(args, tokenizer, "valid", args.data_dir, args.dev_num)

    return data


def run_model(args, tokenizer, model, dataset: PromptDataset, epoch, device):
    
    collate_fn = dataset.collate
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None
    
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    model.eval()
    
    all_query_ids = []
    all_response_ids = []
    all_lm_losses = []
    
    generation_config = GenerationConfig (
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )

    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=f"Evaluating {args.data_names} ", disable=(dist.get_rank() != 0))):
            if it == 0:
                print_rank("############### Example ###############")
                print_rank(tokenizer.decode(model_batch["input_ids"][0], skip_special_tokens=True))
                print_rank("############### End ###############")
            
            dataset.move_to_device(model_batch, no_model_batch, device)

            all_ids = torch.cat([model_batch["input_ids"], no_model_batch["rest_ids"]], dim=-1)
            input_ids = all_ids[:, :-1]
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            label_ids = all_ids[:, 1:]
            label_ids = torch.masked_fill(label_ids, label_ids==tokenizer.pad_token_id, -100)
            label_ids[:, :model_batch["input_ids"].size(1)-1] = -100  
            if args.model_type in ["gpt2"]:
                position_ids = (torch.cumsum(attention_mask, dim=-1) - 1) * attention_mask
                out = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, return_dict=True)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = out.logits
            loss_mask = (label_ids != -100).float()
            loss_func = nn.CrossEntropyLoss(reduction="none")
            lm_loss = loss_func(logits.view(-1, logits.size(-1)), label_ids.view(-1)).view(label_ids.size())
            lm_loss = torch.sum(lm_loss * loss_mask, -1) / torch.sum(loss_mask, -1)
            all_lm_losses.append(lm_loss)

            query_ids = model_batch["input_ids"]
            max_new_tokens = args.max_length - query_ids.size(1)
            gen_out = model.generate(
                **model_batch,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens
            )
            full_ids = gen_out.sequences
            response_ids = full_ids[:, query_ids.size(1):] # remove prompt (may include start token)
            
            query_ids = F.pad(query_ids, (args.max_prompt_length-query_ids.size(1), 0, 0, 0), value=tokenizer.pad_token_id)
            response_ids = F.pad(response_ids, (0, args.max_length-args.max_prompt_length-response_ids.size(1), 0, 0), value=tokenizer.pad_token_id)
            
            all_query_ids.append(query_ids)
            all_response_ids.append(response_ids)

    all_lm_losses = torch.cat(all_lm_losses)
    mean_lm_loss = all_lm_losses.mean()
    dist.all_reduce(mean_lm_loss, dist.ReduceOp.SUM, group=dp_group)
    mean_lm_loss = mean_lm_loss.item() / dp_world_size
        
    all_query_ids = torch.cat(all_query_ids)
    all_query_ids = all_gather(all_query_ids, dim=1, group=dp_group, world_size=dp_world_size, op="stack")
    all_query_ids = all_query_ids.view(-1, all_query_ids.size(-1))
    all_query_ids = all_query_ids[:len(dataset)]
    
    all_response_ids = torch.cat(all_response_ids)
    all_response_ids = all_gather(all_response_ids, dim=1, group=dp_group, world_size=dp_world_size, op="stack")
    all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
    all_response_ids = all_response_ids[:len(dataset)]
        
    return (
        mean_lm_loss,
        all_query_ids,
        all_response_ids)


def evaluate_main(args, tokenizer, model, dataset: PromptDataset, split, epoch, device, teacher_model=None):
    if getattr(args, "kd_objective", None) == "sr_eotd":
        evaluate_sreotd_benchmark(args, tokenizer, model, dataset, split, epoch, device, teacher_model)
        return

    lm_loss, query_ids, response_ids = run_model(args, tokenizer, model, dataset, epoch, device)
    query_strs = tokenizer.batch_decode(query_ids, skip_special_tokens=True)
    response_strs = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    
    with open(os.path.join(args.save, "preds.txt"), "w") as f:
        for q, r in zip(query_strs, response_strs):
            f.write(q.replace("\n", "<n>") + "\t\t" + r.replace("\n", "<n>") + "\n")

    all_preds = [[]]
    for q, r in zip(query_strs, response_strs):
        all_preds[0].append((q, q + r))
    torch.save(all_preds, os.path.join(args.save, "preds.pt"))

    all_responses = []
    with open(os.path.join(args.save, "answers.jsonl"), "w") as f:
        for p in all_preds[0]:
            q, r = p
            r = r[len(q):]
            idx = r.find("<|endoftext|>")
            if idx >= 0:
                r = r[:idx]
            f.write(json.dumps({
                "text": r.replace("<n>", "\n").strip()
            }) + "\n")
            all_responses.append(r.replace("<n>", "\n").strip())
    
    gen_res = compute_metrics(all_responses, dataset.answers)

    mean_gen_length = np.mean([len(tokenizer.encode(s)) for s in response_strs])

    log_str = f"{split} | name: {args.data_names} | {gen_res} | lm_loss {round(lm_loss, 4)} | avg. gen lenth: {mean_gen_length}"
    print_rank(log_str)
    save_rank(log_str, os.path.join(args.save, "log.txt"))


def _clean_decoded(tokenizer, token_ids):
    texts = tokenizer.batch_decode(token_ids.cpu(), skip_special_tokens=True)
    cleaned = []
    for text in texts:
        cut = text.find("<|endoftext|>")
        if cut >= 0:
            text = text[:cut]
        cleaned.append(text.replace("<n>", "\n").strip())
    return cleaned


def _pairwise_self_bleu(branch_texts):
    """Mean sentence BLEU of each branch against the other branches, on the same prompt."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    branches = len(branch_texts)
    if branches < 2 or not branch_texts[0]:
        return 0.0
    smoother = SmoothingFunction().method1
    scores = []
    for index in range(len(branch_texts[0])):
        tokenized = [branch_texts[branch][index].split() for branch in range(branches)]
        pair_scores = []
        for left in range(branches):
            for right in range(branches):
                if left == right:
                    continue
                hypothesis = tokenized[left]
                reference = tokenized[right]
                if not hypothesis or not reference:
                    pair_scores.append(0.0)
                    continue
                pair_scores.append(sentence_bleu([reference], hypothesis, smoothing_function=smoother))
        scores.append(sum(pair_scores) / len(pair_scores))
    return sum(scores) / len(scores)


def _cross_branch_distinct2(branch_texts):
    """Distinct-2 of the M answers to the same prompt, averaged over prompts."""
    branches = len(branch_texts)
    if branches == 0 or not branch_texts[0]:
        return 0.0
    values = []
    for index in range(len(branch_texts[0])):
        total = 0
        unique = set()
        for branch in range(branches):
            tokens = branch_texts[branch][index].split()
            grams = [tuple(tokens[i:i + 2]) for i in range(len(tokens) - 1)]
            total += len(grams)
            unique.update(grams)
        values.append((len(unique) / total) if total else 0.0)
    return sum(values) / len(values)


def _jsd_and_disagreement(hidden, weight, bias, mask):
    """Token JSD of the M branch distributions and the count of top-1 disagreements."""
    batch, _, branches, _ = hidden.shape
    # Accumulate in fp32. Branch hidden states are fp16/bf16, while the
    # full-vocab JSD below is computed in fp32.
    jsd_sum = torch.zeros(batch, device=hidden.device, dtype=torch.float32)
    disagree_sum = torch.zeros(batch, device=hidden.device, dtype=torch.float32)
    positions = mask.nonzero(as_tuple=False)
    if positions.numel() == 0:
        return jsd_sum, disagree_sum
    rows = positions[:, 0]
    packed = hidden[positions[:, 0], positions[:, 1]].float()
    bias_f = None if bias is None else bias.float()
    weight_f = weight.float()
    for start in range(0, packed.size(0), 64):
        chunk = packed[start:start + 64]
        row_chunk = rows[start:start + 64]
        entropies = []
        top1 = []
        mixture = None
        for branch in range(branches):
            logits = F.linear(chunk[:, branch], weight_f, bias_f)
            log_prob = F.log_softmax(logits, dim=-1)
            prob = log_prob.exp()
            entropies.append(-(prob * log_prob).sum(dim=-1))
            top1.append(logits.argmax(dim=-1))
            mixture = prob if mixture is None else mixture + prob
            del logits, log_prob, prob
        mean_prob = mixture / float(branches)
        mixture_entropy = -(mean_prob * (mean_prob + 1e-12).log()).sum(dim=-1)
        jsd = mixture_entropy - torch.stack(entropies, dim=0).mean(dim=0)
        winners = torch.stack(top1, dim=-1)
        disagree = ~(winners == winners[:, :1]).all(dim=-1)
        jsd_sum.index_add_(0, row_chunk, jsd)
        disagree_sum.index_add_(0, row_chunk, disagree.float())
        del mixture, mean_prob, chunk
    return jsd_sum, disagree_sum


def _pool_hidden(hidden, mask):
    denom = mask.sum(dim=-1).clamp_min(1.0)
    hidden_mean = (hidden.float() * mask[:, :, None, None]).sum(dim=1) / denom[:, None, None]
    last = mask.long().sum(dim=-1).clamp_min(1) - 1
    index = last.view(hidden.size(0), 1, 1, 1).expand(hidden.size(0), 1, hidden.size(2), hidden.size(3))
    hidden_last = hidden.float().gather(dim=1, index=index).squeeze(1)
    return hidden_mean, hidden_last, denom


def _gather_rows(tensor, dataset_size, world_size, group):
    gathered = all_gather(tensor, dim=1, world_size=world_size, group=group, op="stack")
    flat = gathered.reshape(-1, *tensor.shape[1:])
    return flat[:dataset_size].contiguous()


def _pad_token_id(tokenizer):
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    return tokenizer.eos_token_id


def _teacher_forcing_inputs(args, tokenizer, model_batch, no_model_batch):
    """Shared prefix ``x_<t`` for the teacher and every student branch.

    Prompt tokens are masked. The first supervised position predicts the first
    response token, matching ``run_model``.
    """
    all_ids = torch.cat([model_batch["input_ids"], no_model_batch["rest_ids"]], dim=-1)
    input_ids = all_ids[:, :-1]
    pad_id = _pad_token_id(tokenizer)
    attention_mask = (input_ids != pad_id).long()
    label_ids = all_ids[:, 1:].clone()
    label_ids = torch.masked_fill(label_ids, label_ids == pad_id, -100)
    label_ids[:, :model_batch["input_ids"].size(1) - 1] = -100
    loss_mask = (label_ids != -100).float()
    position_ids = None
    if args.model_type in ["gpt2"]:
        position_ids = (torch.cumsum(attention_mask, dim=-1) - 1) * attention_mask
    return input_ids, attention_mask, position_ids, loss_mask


def _analysis_batch_size(args):
    # Teacher logits are [B, T, V]. Keep this pass small so the float top-k copy fits.
    return max(1, min(int(args.eval_batch_size), 2))


def _own_top16(hidden, weight, bias, k=16, token_chunk=16):
    """Full-vocabulary top-16 of each branch, chunked over time to limit [B, T, V]."""
    batch, time, branches, _ = hidden.shape
    vocab = int(weight.size(0))
    take = min(k, vocab)
    logits_out = torch.empty(batch, time, branches, take, device=hidden.device, dtype=torch.float32)
    ids_out = torch.empty(batch, time, branches, take, device=hidden.device, dtype=torch.long)
    weight_f = weight.detach().float()
    bias_f = None if bias is None else bias.detach().float()
    for branch in range(branches):
        states = hidden[:, :, branch].float()
        for start in range(0, time, token_chunk):
            end = min(time, start + token_chunk)
            logits = F.linear(states[:, start:end], weight_f, bias_f)
            values, indices = torch.topk(logits, k=take, dim=-1)
            logits_out[:, start:end, branch] = values
            ids_out[:, start:end, branch] = indices
            del logits
        del states
    if take < k:
        logits_out = F.pad(logits_out, (0, k - take))
        ids_out = F.pad(ids_out, (0, k - take))
    return logits_out, ids_out


def _benchmark_name(args):
    name = (getattr(args, "data_names", None) or "eval").strip()
    for char in (" ", "/", "\\"):
        name = name.replace(char, "_")
    return name.replace("-", "_") or "eval"


def _as_text(reference):
    if isinstance(reference, list):
        return "\n".join(str(item) for item in reference)
    if reference is None:
        return ""
    return str(reference)


def _load_hf_hub():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "hf_hub_manager.py")
    spec = importlib.util.spec_from_file_location("sreotd_hf_hub_manager", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collect_sreotd_features(args, tokenizer, model, dataset, device, teacher_model):
    """Pooled branch embeddings, mean transport weights, entropy, JSD and disagreement."""
    collate_fn = dataset.collate
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=rank, num_replicas=world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=_analysis_batch_size(args), num_workers=args.num_workers, collate_fn=collate_fn)
    student = unwrap_model(model)
    weight = get_lm_head_weight(student)
    bias = get_lm_head_bias(student)
    model.eval()
    if teacher_model is not None:
        teacher_model.eval()

    hidden_mean, hidden_last = [], []
    weights, entropies, pairwise = [], [], []
    jsd_sums, disagree_sums, token_counts = [], [], []

    with torch.no_grad():
        for model_batch, no_model_batch in tqdm(
            dataloader, desc="SR-EOTD embeddings", disable=(dist.get_rank() != 0)
        ):
            dataset.move_to_device(model_batch, no_model_batch, device)
            input_ids, attention_mask, position_ids, loss_mask = _teacher_forcing_inputs(
                args, tokenizer, model_batch, no_model_batch
            )
            hidden = student.forward_multi_branch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            mask = attention_mask.float()
            mean_h, last_h, denom = _pool_hidden(hidden, mask)
            hidden_mean.append(mean_h)
            hidden_last.append(last_h)
            jsd_sum, disagree_sum = _jsd_and_disagreement(hidden, weight, bias, attention_mask.bool())
            jsd_sums.append(jsd_sum)
            disagree_sums.append(disagree_sum)
            token_counts.append(mask.sum(dim=-1))
            if teacher_model is None:
                weights.append(torch.zeros(mean_h.size(0), mean_h.size(1), device=mean_h.device))
                entropies.append(torch.zeros(mean_h.size(0), device=mean_h.device))
                pairwise.append(torch.zeros(mean_h.size(0), device=mean_h.device))
            else:
                teacher_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": False,
                }
                if position_ids is not None:
                    teacher_kwargs["position_ids"] = position_ids
                teacher_logits = teacher_model(**teacher_kwargs).logits
                details = compute_sr_eotd(
                    args,
                    teacher_logits,
                    hidden,
                    weight,
                    {"loss_mask": loss_mask},
                )
                w_star = details["w_star"]
                teacher_entropy = details["H_hat"].view(w_star.size(0), w_star.size(1))
                response_denom = loss_mask.sum(dim=-1).clamp_min(1.0)
                weights.append((w_star * loss_mask.unsqueeze(-1)).sum(dim=1) / response_denom.unsqueeze(-1))
                entropies.append((teacher_entropy * loss_mask).sum(dim=1) / response_denom)
                pairwise.append(token_pairwise_jsd(details["P_m"], loss_mask))
                del teacher_logits, details
            del hidden

    def _cat(chunks):
        return _gather_rows(torch.cat(chunks, dim=0).contiguous(), len(dataset), world_size, None)

    payload = {
        "hidden_mean": _cat(hidden_mean).float().cpu(),
        "hidden_last": _cat(hidden_last).float().cpu(),
        "w_star": _cat(weights).float().cpu(),
        "H_teacher": _cat(entropies).float().cpu(),
        "pairwise_jsd": _cat(pairwise).float().cpu(),
        "teacher_used": teacher_model is not None,
        "num_branches": args.num_lora_branches,
    }
    jsd_sum = _cat(jsd_sums).float()
    disagree_sum = _cat(disagree_sums).float()
    token_count = _cat(token_counts).float()
    denom = token_count.sum().clamp_min(1.0)
    payload["token_jsd"] = float((jsd_sum.sum() / denom).cpu())
    payload["top1_disagreement_rate"] = float((disagree_sum.sum() / denom).cpu())
    return payload


def _gather_diag_rows(local_rows):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        merged = list(local_rows)
    else:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_rows)
        merged = []
        for part in gathered:
            merged.extend(part or [])
    dedup = {}
    for row in merged:
        dedup.setdefault(int(row["idx"]), row)
    return [dedup[idx] for idx in sorted(dedup)]


def collect_sreotd_logits_and_mode_stats(
    args, teacher_model, student_model, dataset, tokenizer, max_samples=256,
):
    """Teacher-forced top-K logits, OT plan, and entropy-region mode diagnostics.

    Teacher and the M student branches read the same prefix. Only the first
    ``max_samples`` examples are stored. Returns ``None`` on non-zero ranks.
    """
    if teacher_model is None:
        if not dist.is_initialized() or dist.get_rank() == 0:
            print_rank("SR-EOTD logit dump skipped: teacher_model is required")
        return None

    limit = min(int(max_samples), len(dataset))
    subset = Subset(dataset, list(range(limit)))
    rank = dist.get_rank() if dist.is_initialized() else 0
    loader_kwargs = {
        "batch_size": _analysis_batch_size(args),
        "num_workers": args.num_workers,
        "collate_fn": dataset.collate,
    }
    if dist.is_initialized():
        world_size = dist.get_world_size()
        loader_kwargs["sampler"] = DistributedSampler(
            subset, shuffle=False, drop_last=False, rank=rank, num_replicas=world_size,
        )
    else:
        loader_kwargs["shuffle"] = False
    dataloader = DataLoader(subset, **loader_kwargs)
    student = unwrap_model(student_model)
    device = next(student.parameters()).device
    weight = get_lm_head_weight(student)
    bias = get_lm_head_bias(student)
    student_model.eval()
    teacher_model.eval()
    local_rows = []

    with torch.no_grad():
        for model_batch, no_model_batch in tqdm(
            dataloader, desc="SR-EOTD logits", disable=(rank != 0)
        ):
            dataset.move_to_device(model_batch, no_model_batch, device)
            input_ids, attention_mask, position_ids, loss_mask = _teacher_forcing_inputs(
                args, tokenizer, model_batch, no_model_batch
            )
            hidden = student.forward_multi_branch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            teacher_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": False,
            }
            if position_ids is not None:
                teacher_kwargs["position_ids"] = position_ids
            teacher_logits = teacher_model(**teacher_kwargs).logits
            details = compute_sr_eotd(
                args, teacher_logits, hidden, weight, {"loss_mask": loss_mask},
            )
            own_logits, own_ids = _own_top16(hidden, weight, bias, k=16)
            packed = {
                "input_ids": input_ids.to(torch.int32).cpu(),
                "topk_indices": details["topk_indices"].to(torch.int32).cpu(),
                "teacher_topk_logits": details["teacher_topk_logits"].half().cpu(),
                "student_topk_logits": details["student_topk_logits"].half().cpu(),
                "student_own_top16_logits": own_logits.half().cpu(),
                "student_own_top16_ids": own_ids.to(torch.int32).cpu(),
                "ot_branch_weights_w": details["w_star"].half().cpu(),
                "teacher_norm_entropy": details["H_hat"].reshape(hidden.size(0), hidden.size(1)).half().cpu(),
                "loss_mask": loss_mask.bool().cpu(),
                "P_T": details["P_T"].half().cpu(),
                "P_m": details["P_m"].half().cpu(),
                "Q_tilde": details["Q_tilde"].half().cpu(),
            }
            gamma = details["Gamma"].half().cpu()
            indices = no_model_batch["idx"].detach().cpu()
            for row_index in range(input_ids.size(0)):
                sample_index = int(indices[row_index])
                row = {key: value[row_index].contiguous() for key, value in packed.items()}
                row["idx"] = sample_index
                row["gamma"] = gamma[row_index].contiguous() if sample_index < 32 else None
                local_rows.append(row)
            del hidden, teacher_logits, details, own_logits, own_ids, gamma

    rows = _gather_diag_rows(local_rows)
    if dist.is_initialized() and dist.get_rank() != 0:
        return None
    if not rows:
        return {"logits": None, "diagnostics": None, "forks": []}

    def _stack(key, pad_value=0):
        return pad_time([row[key] for row in rows], pad_value=pad_value)

    loss_mask = _stack("loss_mask", pad_value=False)
    entropy = _stack("teacher_norm_entropy")
    weights = _stack("ot_branch_weights_w")
    probs = _stack("P_m")
    target = _stack("Q_tilde")
    teacher_probs = _stack("P_T")
    top1_ids = _stack("student_own_top16_ids")[:, :, :, 0]
    diagnostics = mode_collapse_diagnostics(
        probs, weights.float(), target, teacher_probs, entropy.float(), loss_mask, top1_ids,
    )
    gamma_rows = [row["gamma"] for row in rows if row["gamma"] is not None][:32]
    if gamma_rows:
        gamma = pad_time(gamma_rows, pad_value=0)
        if gamma.size(1) < weights.size(1):
            extra = weights.size(1) - int(gamma.size(1))
            filler = gamma.new_zeros((gamma.size(0), extra) + tuple(gamma.shape[2:]))
            gamma = torch.cat([gamma, filler], dim=1)
    else:
        gamma = weights.new_empty(0, weights.size(1), weights.size(2), teacher_probs.size(-1))
    logits = {
        "topk_indices": _stack("topk_indices"),
        "teacher_topk_logits": _stack("teacher_topk_logits"),
        "student_topk_logits": _stack("student_topk_logits"),
        "student_own_top16_logits": _stack("student_own_top16_logits"),
        "student_own_top16_ids": _stack("student_own_top16_ids"),
        "ot_branch_weights_w": weights,
        "teacher_norm_entropy": entropy,
        "loss_mask": loss_mask,
        "ot_plan_gamma_sample": gamma,
    }
    forks = build_fork_snapshots(
        tokenizer,
        _stack("input_ids"),
        logits["topk_indices"],
        teacher_probs,
        probs,
        target,
        weights.float(),
        entropy.float(),
        loss_mask,
        [row["idx"] for row in rows],
    )
    return {"logits": logits, "diagnostics": diagnostics, "forks": forks}


def evaluate_sreotd_benchmark(args, tokenizer, model, dataset: PromptDataset, split, epoch, device, teacher_model=None):
    """Per-branch ROUGE-L, cross-branch diversity, and reusable merge artifacts."""
    student = unwrap_model(model)
    branches = args.num_lora_branches
    branch_texts = []
    prompts = None
    lm_losses = []
    answers = list(dataset.answers) if hasattr(dataset, "answers") else None

    for branch in range(branches):
        student.set_active_branch(branch)
        lm_loss, query_ids, response_ids = run_model(args, tokenizer, model, dataset, epoch, device)
        texts = _clean_decoded(tokenizer, response_ids)
        branch_texts.append(texts)
        lm_losses.append(lm_loss)
        if prompts is None:
            prompts = _clean_decoded(tokenizer, query_ids)
    student.set_active_branch(None)

    count = min(len(texts) for texts in branch_texts)
    if answers is not None:
        count = min(count, len(answers))
    prompts = prompts[:count]
    branch_texts = [texts[:count] for texts in branch_texts]
    references = answers[:count] if answers is not None else None

    per_branch = []
    rouge_values = []
    exact_values = []
    if references is not None:
        for texts in branch_texts:
            scores = per_example_rouge(texts, references)
            per_branch.append(scores)
            rouge_values.append(100.0 * (sum(scores) / len(scores) if scores else 0.0))
            exact_values.append(compute_metrics(texts, references).get("exact_match", 0.0))
        width = min(len(scores) for scores in per_branch)
        best = 100.0 * sum(max(per_branch[b][i] for b in range(branches)) for i in range(width)) / max(width, 1)
        mean_rouge = sum(rouge_values) / len(rouge_values)
    else:
        best = 0.0
        mean_rouge = 0.0

    self_bleu = _pairwise_self_bleu(branch_texts)
    distinct2 = _cross_branch_distinct2(branch_texts)
    features = _collect_sreotd_features(args, tokenizer, model, dataset, device, teacher_model)
    diag_samples = int(getattr(args, "sreotd_diag_samples", 256))
    diagnostics_payload = collect_sreotd_logits_and_mode_stats(
        args, teacher_model, model, dataset, tokenizer, max_samples=diag_samples,
    )

    metrics = {
        "split": split,
        "data_names": args.data_names,
        "num_examples": count,
        "rougeL_mean": round(mean_rouge, 4),
        f"rougeL_best_of_{branches}": round(best, 4),
        "pairwise_self_bleu": self_bleu,
        "distinct_2": distinct2,
        "token_jsd": features["token_jsd"],
        "top1_disagreement_rate": features["top1_disagreement_rate"],
        "teacher_used_for_weights": features["teacher_used"],
    }
    for branch, value in enumerate(rouge_values):
        metrics[f"rougeL_branch_{branch}"] = round(value, 4)
    for branch, value in enumerate(exact_values):
        metrics[f"exact_match_branch_{branch}"] = value
    for branch, value in enumerate(lm_losses):
        metrics[f"lm_loss_branch_{branch}"] = round(value, 4)

    header = f"{'Branch':<28}{'ROUGE-L':>12}{'EM':>12}{'LM loss':>12}"
    lines = [
        f"SR-EOTD {args.data_names} ({count} examples)",
        header,
    ]
    for branch in range(branches):
        rouge_l = metrics.get(f"rougeL_branch_{branch}", 0.0)
        exact = metrics.get(f"exact_match_branch_{branch}", 0.0)
        lm_loss = metrics.get(f"lm_loss_branch_{branch}", 0.0)
        lines.append(f"{'Branch ' + str(branch):<28}{rouge_l:12.4f}{exact:12.4f}{lm_loss:12.4f}")
    lines.append(f"{'Mean-of-' + str(branches):<28}{mean_rouge:12.4f}")
    lines.append(f"{'Best-of-' + str(branches) + ' (Oracle / Pass@' + str(branches) + ')':<28}{best:12.4f}")
    lines.append(f"Pairwise Self-BLEU: {self_bleu:.4f}")
    lines.append(f"Distinct-2: {distinct2:.4f}")
    lines.append(f"Token-level JSD: {features['token_jsd']:.4f}")
    lines.append(f"Top-1 Disagreement Rate: {features['top1_disagreement_rate']:.4f}")
    table = "\n".join(lines)
    print_rank(table)
    save_rank(table, os.path.join(args.save, "log.txt"))

    benchmark = _benchmark_name(args)
    artifact_dir = os.path.join(args.save, "sreotd_eval_artifacts", benchmark)
    if dist.get_rank() == 0:
        os.makedirs(artifact_dir, exist_ok=True)
        orthogonality = lora_subspace_orthogonality(model)
        diagnostics = None if diagnostics_payload is None else diagnostics_payload.get("diagnostics")
        if diagnostics is not None:
            metrics["mode_collapse"] = {
                name: {
                    "num_tokens": region["num_tokens"],
                    "pairwise_jsd_mean": region["pairwise_jsd_mean"],
                    "top1_agreement_rate": region["top1_agreement_rate"],
                    "effective_num_branches": region["effective_num_branches"],
                    "barycentric_conservation_error": region["barycentric_conservation_error"],
                }
                for name, region in diagnostics["regions"].items()
            }
        if orthogonality.get("mean_offdiag_A_gram_fro_over_sqrt_r") is not None:
            metrics["lora_A_offdiag_fro"] = orthogonality["mean_offdiag_A_gram_fro_over_sqrt_r"]
            metrics["lora_delta_offdiag_cosine"] = orthogonality["mean_offdiag_delta_cosine"]
        hidden_rows = min(count, int(features["hidden_mean"].size(0)))
        with open(os.path.join(artifact_dir, "branch_generations.jsonl"), "w") as handle:
            for index in range(count):
                row = {
                    "sample_id": index,
                    "prompt": prompts[index],
                    "ground_truth": _as_text(references[index]) if references is not None else "",
                }
                rouge_here = []
                for branch in range(branches):
                    row[f"response_branch_{branch}"] = branch_texts[branch][index]
                    if references is not None:
                        rouge_l = round(100.0 * per_branch[branch][index], 4)
                        row[f"rougeL_branch_{branch}"] = rouge_l
                        rouge_here.append(rouge_l)
                if rouge_here:
                    row[f"rougeL_best_of_{branches}"] = max(rouge_here)
                if index < hidden_rows and features["teacher_used"]:
                    row["w_star"] = [round(float(value), 6) for value in features["w_star"][index].tolist()]
                    row["mean_teacher_entropy"] = round(float(features["H_teacher"][index]), 6)
                    row["mean_pairwise_jsd"] = round(float(features["pairwise_jsd"][index]), 6)
                else:
                    row["w_star"] = [0.0] * branches
                    row["mean_teacher_entropy"] = None
                    row["mean_pairwise_jsd"] = None
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(os.path.join(artifact_dir, "eval_summary_metrics.json"), "w") as handle:
            json.dump(metrics, handle, indent=2)
        with open(os.path.join(artifact_dir, "lora_subspace_orthogonality.json"), "w") as handle:
            json.dump(orthogonality, handle, indent=2)
        diagnostics_path = os.path.join(artifact_dir, "mode_collapse_diagnostics.json")
        if diagnostics is None:
            diagnostics = {
                "error": "teacher_model_path was not set, so token-level OT diagnostics were not computed.",
            }
        with open(diagnostics_path, "w") as handle:
            json.dump(diagnostics, handle, indent=2)
        forks = [] if diagnostics_payload is None else diagnostics_payload.get("forks") or []
        with open(os.path.join(artifact_dir, "high_entropy_fork_snapshots.json"), "w") as handle:
            json.dump(forks, handle, ensure_ascii=False, indent=2)
        torch.save(
            {
                "hidden_mean": features["hidden_mean"].half(),
                "hidden_last": features["hidden_last"].half(),
                "w_star": features["w_star"].half(),
                "H_teacher": features["H_teacher"].half(),
                "teacher_used": features["teacher_used"],
                "num_branches": features["num_branches"],
                "hidden_size": int(features["hidden_mean"].size(-1)),
            },
            os.path.join(artifact_dir, "branch_embeddings_and_weights.pt"),
        )
        if diagnostics_payload is not None and diagnostics_payload.get("logits") is not None:
            torch.save(diagnostics_payload["logits"], os.path.join(artifact_dir, "sreotd_logits_and_modes.pt"))
        print_rank(f"SR-EOTD eval artifacts saved to {artifact_dir}")
        if getattr(args, "push_to_hf", False):
            repo = getattr(args, "hf_dataset_repo_id", None)
            if not repo:
                raise ValueError("--push-to-hf requires --hf-dataset-repo-id or --hf-model-repo-id")
            _load_hf_hub().push_sreotd_eval_data_to_hub(
                os.path.dirname(artifact_dir),
                repo,
                token=getattr(args, "hf_token", None),
                private=bool(getattr(args, "hf_private", False)),
                benchmarks=[benchmark],
            )
    dist.barrier()
