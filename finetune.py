import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed

import random
import json
from tqdm import tqdm
import math
import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig)

from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from arguments import get_args

from data_utils.lm_datasets import LMTrainDataset
from utils import get_optimizer_params, get_optimizer_params_peft, print_args, initialize
from utils import print_rank, get_rank
from utils import save_rank
from utils import all_gather
from utils import load_parallel, save_parallel
from utils import get_tokenizer, get_model

from distillm import forward_kl, reverse_kl, js_distance, tv_distance, ab_div, AKL, alphanet, bdkd
from distillm import skewed_forward_kl, skewed_reverse_kl
from distillm import SampleGenerator, ReplayBuffer
from distillm import sr_eotd_loss, SREOTD_STAT_KEYS
from distillm.multi_lora import (
    inject_multi_lora,
    load_sreotd_weights_if_present,
    sync_sreotd_architecture_from_checkpoint,
    unwrap_model,
    get_lm_head_weight,
    save_sreotd_checkpoint,
)

from rouge_metric import compute_metrics, per_example_rouge

from peft import PeftModel

torch.set_num_threads(4)


def get_teacher_model(args, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    if args.model_parallel:
        raise NotImplementedError
    else:
        config.is_model_parallel = False
        try:
            model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.float16)
        except:
            model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.float32)
            model = model.half()

        if args.peft is not None and args.teacher_peft_path is not None:
            if args.peft == "lora":
                model = PeftModel.from_pretrained(model, args.teacher_peft_path)
                model = model.merge_and_unload()
            else:
                raise NotImplementedError
        else:
            if dist.get_rank() == 0:
                print(' > number of parameters: {}'.format(sum([p.nelement() for p in model.parameters()])), flush=True)

    model.eval()

    return model


def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    lora_only = getattr(args, "kd_objective", None) == "sr_eotd" and getattr(args, "sreotd_freeze_backbone", False)
    if args.peft is not None or lora_only:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)

    # Use AdamW.
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.total_iters, eta_min=args.lr_min)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_iters, num_training_steps=args.total_iters, power=0.5)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler


def setup_model_and_optimizer(args, ds_config, device, set_optim=True):
    # get the model
    model = get_model(args, device)
    if getattr(args, "kd_objective", None) == "sr_eotd":
        sync_sreotd_architecture_from_checkpoint(args)
        inject_multi_lora(model, args)
        load_sreotd_weights_if_present(model, args)
    # get the optimizer and lr_scheduler
    if set_optim:
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer, lr_scheduler = None, None

    model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, optimizer=optimizer, args=args, lr_scheduler=lr_scheduler, mpu=None, config_params=ds_config)

    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler


def prepare_dataset(args, tokenizer):
    data = {}
    rng_sample = random.Random(args.seed)
    if args.do_train:
        data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["train"]))
        data["dev"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    elif args.do_eval:
        data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    else:
        raise ValueError("Do train and do eval must set one")

    # pre-trained dataset
    if args.do_train and args.lm_data_dir is not None:
        data["pt_train"] = LMTrainDataset(args, tokenizer, args.lm_data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["pt_train"]))
    return data


def sreotd_token_ce(args, model, model_batch, labels):
    """Standard next-token CE on all M branches of a repeated batch."""
    base = unwrap_model(model)
    base.set_active_branch(None)
    branches = args.num_lora_branches
    bsz = model_batch["input_ids"].size(0)
    repeated = {}
    for key, value in model_batch.items():
        if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == bsz:
            repeated[key] = value.repeat_interleave(branches, dim=0)
        else:
            repeated[key] = value
    rep_labels = labels.repeat_interleave(branches, dim=0)
    outputs = model(**repeated, return_dict=True, use_cache=False)
    logits = outputs.logits
    return nn.CrossEntropyLoss(ignore_index=-100)(
        logits.float().view(-1, logits.size(-1)),
        rep_labels.view(-1),
    )


def pt_loss(args, model, model_batch, no_model_batch):
    if getattr(args, "kd_objective", None) == "sr_eotd":
        return sreotd_token_ce(args, model, model_batch, no_model_batch["label"])
    outputs = model(**model_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), no_model_batch["label"].view(-1))
    return lm_loss


def _reduce_stat_means(stats, device, world_size, group):
    keys = list(SREOTD_STAT_KEYS)
    values = torch.tensor([float(stats[key]) for key in keys], device=device, dtype=torch.float32)
    dist.all_reduce(values, dist.ReduceOp.SUM, group=group)
    values = values / world_size
    return {key: float(values[i]) for i, key in enumerate(keys)}


def symmetrize_and_selfloop(coo: torch.sparse_coo_tensor, self_val: float = 1.0):
    coo = coo.coalesce()
    idx = coo.indices()
    v = coo.values()
    L = coo.size(0)

    i_idx, j_idx = idx[0], idx[1]
    sym_i = torch.cat([i_idx, j_idx], dim=0)
    sym_j = torch.cat([j_idx, i_idx], dim=0)
    sym_v = torch.cat([v, v], dim=0)

    self_idx = torch.arange(L, device=v.device)
    self_v = torch.full((L,), self_val, device=v.device)

    all_i = torch.cat([sym_i, self_idx], dim=0)
    all_j = torch.cat([sym_j, self_idx], dim=0)
    all_v = torch.cat([sym_v, self_v], dim=0)

    coords = torch.stack([all_i, all_j], dim=1)
    unique_coords, inv = torch.unique(coords, dim=0, return_inverse=True)
    max_v = torch.zeros(unique_coords.size(0), device=v.device)
    max_v = max_v.scatter_reduce(0, inv, all_v, reduce='amax', include_self=True)

    return torch.sparse_coo_tensor(unique_coords.t(), max_v, (L, L), device=v.device)


def get_token_embeddings(args, model, teacher_model):
    if args.wd_cost_emb == "teacher":
        if 'openllama' in args.ckpt_name:
            embeddings = teacher_model.model.embed_tokens.weight
        elif 'gpt2' in args.ckpt_name:
            embeddings = teacher_model.transformer.wte.weight
        else:
            raise NotImplementedError(f"Unknown model family for --wd-cost-emb: {args.ckpt_name}")
    elif args.wd_cost_emb == "student":
        if 'openllama' in args.ckpt_name:
            embeddings = model.module.base_model.model.model.embed_tokens.weight
        elif 'gpt2' in args.ckpt_name:
            embeddings = model.transformer.wte.weight
        else:
            raise NotImplementedError(f"Unknown model family for --wd-cost-emb: {args.ckpt_name}")
    else:
        raise NotImplementedError(f"Unknown --wd-cost-emb: {args.wd_cost_emb}")
    return embeddings.data.detach().float()


def build_sparse_cost(args, embeddings):
    N = embeddings.shape[0]
    k = args.wd_cost_dim
    block_size = args.wd_block_size

    if args.wd_cost_type == 'cos':
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1, eps=1e-8)

    rows, cols, vals = [], [], []
    for start in tqdm(range(0, N, block_size), desc="Building sparse cost matrix", disable=(get_rank() != 0)):
        end = min(start + block_size, N)
        block = embeddings[start:end]
        if args.wd_cost_type == 'L2':
            dists = torch.cdist(block, embeddings, p=2)
        elif args.wd_cost_type == 'cos':
            dists = 1 - block @ embeddings.T
        else:
            raise NotImplementedError(f"Unknown --wd-cost-type: {args.wd_cost_type}")
        distances, indices = torch.topk(dists, k + 1, largest=False)
        distances = distances[:, 1:]  # drop self
        indices = indices[:, 1:]
        rows.append(torch.arange(start, end, device=embeddings.device).repeat_interleave(k))
        cols.append(indices.reshape(-1))
        vals.append(distances.reshape(-1))

    indices = torch.stack([torch.cat(rows), torch.cat(cols)], dim=0).long()
    costs = torch.cat(vals).float()
    return indices.cpu(), costs.cpu()


def get_sparse_cost_kernel(args, model, teacher_model, device):
    emb_model_name = args.teacher_ckpt_name if args.wd_cost_emb == "teacher" else args.ckpt_name
    cost_dir = os.path.join(args.base_path, args.wd_cost_dir)
    tag = f"{emb_model_name}_{args.wd_cost_emb}_{args.wd_cost_type}_k{args.wd_cost_dim}"
    cost_save_path = os.path.join(cost_dir, f"cost_{tag}.pt")
    indices_save_path = os.path.join(cost_dir, f"indices_{tag}.pt")

    if not os.path.exists(cost_save_path):
        if get_rank() == 0:
            embeddings = get_token_embeddings(args, model, teacher_model)
            indices, costs = build_sparse_cost(args, embeddings)
            os.makedirs(cost_dir, exist_ok=True)
            torch.save(costs, cost_save_path)
            torch.save(indices, indices_save_path)
        if dist.is_initialized():
            dist.barrier()
    print_rank(f"Loading sparse cost matrix from {cost_save_path}")

    costs = torch.load(cost_save_path).to(device)
    indices = torch.load(indices_save_path).to(device)
    N = teacher_model.config.vocab_size

    K_vals = torch.exp(-costs / args.wd_epsilon)
    mat_K = torch.sparse_coo_tensor(indices, K_vals, size=(N, N))
    mat_K = symmetrize_and_selfloop(mat_K).coalesce()
    mat_Kt = mat_K.transpose(0, 1).coalesce()
    return mat_K, mat_Kt


def get_distil_loss(args, tokenizer, model, teacher_model, model_batch, no_model_batch, logits, epoch, mat_K=None, mat_Kt=None):
    with torch.no_grad():
        teacher_model.eval()
        teacher_outputs = teacher_model(**model_batch, use_cache=False)
        teacher_logits = teacher_outputs.logits
    if args.model_parallel:
        raise NotImplementedError
    else:
        if "sfkl" in args.type:
            distil_loss = skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "srkl" in args.type:
            distil_loss = skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "jsd" in args.type:
            distil_loss = js_distance(logits, teacher_logits, no_model_batch)
        elif "tvd" in args.type:
            distil_loss = tv_distance(logits, teacher_logits, no_model_batch)
        elif "fkl" in args.type or args.type == "kd":
            distil_loss = forward_kl(logits, teacher_logits, no_model_batch)
        elif "rkl" in args.type:
            distil_loss = reverse_kl(logits, teacher_logits, no_model_batch)
        elif "ab" in args.type:
            distil_loss = ab_div(logits, teacher_logits, no_model_batch, args.ab_alpha, args.ab_beta)
        elif "bdkd" in args.type:
            distil_loss = bdkd(logits, teacher_logits, no_model_batch)
        elif "alphanet" in args.type:
            distil_loss = alphanet(logits, teacher_logits, no_model_batch, args.ab_alpha, args.ab_beta)
        elif "akl" in args.type:
            distil_loss = AKL(logits, teacher_logits, no_model_batch)
        elif "amid" in args.type:
            from distillm import amid
            distil_loss = amid(logits, teacher_logits, no_model_batch, args, epoch=epoch, mat_K=mat_K, mat_Kt=mat_Kt, student_temp=args.student_temp)

        else:
            raise ValueError(f"Distillation type {args.type} is not supported yet.")
    return distil_loss


def get_teacher_lm_loss(args, tokenizer, model, teacher_model, model_batch):
    with torch.no_grad():
        t_gen_out = teacher_model.generate(**model_batch, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, max_length=args.max_length,
                                           top_k=0, top_p=1, temperature=1.0, do_sample=True, return_dict_in_generate=True, output_scores=False)

    full_ids = t_gen_out.sequences

    input_ids = full_ids[:, :-1]
    mask = (input_ids != tokenizer.pad_token_id).long()
    labels = full_ids[:, 1:]
    labels = torch.masked_fill(labels, mask == 0, -100)
    labels[:, :model_batch["input_ids"].size(1) - 1] = -100
    loss_mask = (labels != -100).float()

    new_batch = {"input_ids": input_ids, "attention_mask": mask}

    if args.model_type in ["gpt2"]:
        position_ids = torch.cumsum(mask, dim=-1) - 1
        position_ids = torch.masked_fill(position_ids, mask == 0, 0)
        new_batch["position_ids"] = position_ids

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    outputs = model(**new_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    return lm_loss


def _latest_sreotd_ckpt(save_dir):
    best = os.path.join(save_dir, "best_ckpt")
    if os.path.isfile(os.path.join(best, "sreotd_model.pt")):
        return best
    if not os.path.isdir(save_dir):
        return None
    steps = []
    for name in os.listdir(save_dir):
        if name.isdigit() and os.path.isfile(os.path.join(save_dir, name, "sreotd_model.pt")):
            steps.append(int(name))
    if not steps:
        return None
    return os.path.join(save_dir, str(max(steps)))


def _load_hf_hub():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "hf_hub_manager.py")
    spec = importlib.util.spec_from_file_location("sreotd_hf_hub_manager", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _consider_sreotd_best(args, model, tokenizer):
    """Keep the dev checkpoint with the highest Best-of-M ROUGE-L (or lowest loss)."""
    if getattr(args, "kd_objective", None) != "sr_eotd":
        return
    score = getattr(args, "sreotd_last_val_score", None)
    if score is None or score != score:
        return
    previous = getattr(args, "sreotd_best_val_score", None)
    improved = previous is None or score > previous
    if not improved:
        return
    args.sreotd_best_val_score = score
    if dist.get_rank() == 0:
        best_dir = os.path.join(args.save, "best_ckpt")
        save_sreotd_checkpoint(model, tokenizer, best_dir, args)
        metrics = getattr(args, "sreotd_last_val_metrics", None)
        if metrics:
            with open(os.path.join(best_dir, "val_metrics.json"), "w") as handle:
                json.dump(metrics, handle, indent=2)
        print_rank(f"New SR-EOTD best checkpoint ({score:.4f}) -> {best_dir}")
    if dist.is_initialized():
        dist.barrier()


def _push_sreotd_model_if_requested(args):
    if getattr(args, "kd_objective", None) != "sr_eotd" or not getattr(args, "push_to_hf", False):
        return
    if dist.is_initialized() and dist.get_rank() != 0:
        dist.barrier()
        return
    repo = getattr(args, "hf_model_repo_id", None)
    if not repo:
        raise ValueError("--push-to-hf during training requires --hf-model-repo-id")
    ckpt_dir = _latest_sreotd_ckpt(args.save)
    if ckpt_dir is None:
        raise FileNotFoundError(f"No SR-EOTD checkpoint under {args.save}")
    metrics = None
    metrics_path = os.path.join(ckpt_dir, "val_metrics.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path) as handle:
            metrics = json.load(handle)
    _load_hf_hub().push_sreotd_model_to_hub(
        ckpt_dir,
        repo,
        token=getattr(args, "hf_token", None),
        private=bool(getattr(args, "hf_private", False)),
        export_merged_branches=bool(getattr(args, "hf_export_merged_branches", False)),
        metrics=metrics,
    )
    if dist.is_initialized():
        dist.barrier()


def finetune(args, tokenizer: AutoTokenizer, model: deepspeed.DeepSpeedEngine, optimizer: AdamW, lr_scheduler, dataset, device, teacher_model=None):
    print_rank("Start Fine-tuning")

    # print_inspect(model, '*')
    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(dataset['train'], sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["train"].collate)

    if "pt_train" in dataset:
        pt_sampler = DistributedSampler(dataset["pt_train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
        pt_train_dataloader = DataLoader(dataset['pt_train'], sampler=pt_sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["pt_train"].collate)
        pt_train_iter = iter(pt_train_dataloader)

    student_generator = SampleGenerator(args, tokenizer)

    step, global_step = 1, 1
    total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0
    sreotd_window = {key: 0.0 for key in SREOTD_STAT_KEYS}
    sreotd_window_steps = 0

    adaptive_threshold = args.init_threshold if "adaptive" in args.type else None
    # prev_avg_loss, _ = evaluate(args, tokenizer, model, dataset["dev"], "dev", 0, device, adaptive_threshold)
    prev_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev", 0, device, adaptive_threshold)
    _consider_sreotd_best(args, model, tokenizer)
    replay_buffer = ReplayBuffer(args)

    total_res = []

    if args.amid_div_name in ['wass', 'sink'] and getattr(args, "kd_objective", None) != "sr_eotd":
        mat_K, mat_Kt = get_sparse_cost_kernel(args, model, teacher_model, device)
    else:
        mat_K, mat_Kt = None, None

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        model.train()
        train_bar = tqdm(
            train_dataloader,
            desc=f"Train epoch {epoch + 1}/{args.epochs}",
            disable=(get_rank() != 0),
            dynamic_ncols=True,
            leave=True,
        )
        for it, (model_batch, no_model_batch, gen_data) in enumerate(train_bar):
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)

            if args.lm_data_dir is not None:
                try:
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                    # pt_model_batch, pt_no_model_batch, pt_gen_data = pt_train_iter.next()
                except:
                    pt_train_iter = iter(pt_train_dataloader)
                    # pt_model_batch, pt_no_model_batch, pt_gen_data = pt_train_iter.next()
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)

                dataset["pt_train"].move_to_device(pt_model_batch, pt_no_model_batch, pt_gen_data, device)

            torch.cuda.synchronize()
            st_time = time.time()
            if args.teacher_model_path is not None:
                # # sampling ratio:
                if "adaptive" in args.type:
                    samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)
                if "adaptive" in args.type:
                    if args.replay_ratio == "constant":
                        samp_threshold = adaptive_threshold * 0.5
                    elif args.replay_ratio == "increasing":
                        samp_threshold = adaptive_threshold * global_step / args.total_iters
                    else:
                        samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)

                # data generation
                if args.student_gen and getattr(args, "kd_objective", None) == "sr_eotd":
                    unwrap_model(model).set_active_branch(0)
                if args.student_gen:
                    r = np.random.uniform(0, 1)
                    if "mixed" in args.type and r < args.mixed_alpha:
                        model_batch = student_generator.run_sample(model, gen_data)
                        no_model_batch["label"] = model_batch.pop("no_model_batch")

                        replay_buffer.move_to_memory(model_batch, no_model_batch)
                        model_batch, no_model_batch = replay_buffer.sample()
                        model_batch, no_model_batch = replay_buffer.move_to_device(model_batch, no_model_batch, device)

                    elif "adaptive" in args.type and (r < samp_threshold or (r < adaptive_threshold and len(replay_buffer) < args.capacity)):

                        model_batch = student_generator.run_sample(model, gen_data)
                        no_model_batch["label"] = model_batch.pop("no_model_batch")

                        if args.model_type in ["opt"]:
                            model_batch.pop('position_ids')

                        replay_buffer.move_to_memory(model_batch, no_model_batch)

                    elif "adaptive" in args.type and r < adaptive_threshold:
                        model_batch, no_model_batch = replay_buffer.sample()
                        model_batch, no_model_batch = replay_buffer.move_to_device(model_batch, no_model_batch, device)

                    model.train()

            sreotd_stats = None
            if getattr(args, "kd_objective", None) == "sr_eotd":
                if teacher_model is None:
                    raise ValueError("SR-EOTD requires --teacher-model-path.")
                with torch.no_grad():
                    teacher_logits = teacher_model(**model_batch, use_cache=False).logits
                student = unwrap_model(model)
                student_hidden_states = student.forward_multi_branch(**model_batch)
                distil_loss, sreotd_stats = sr_eotd_loss(
                    args,
                    teacher_logits,
                    student_hidden_states,
                    get_lm_head_weight(student),
                    no_model_batch,
                )
                if args.kd_ratio < 1.0:
                    lm_loss = sreotd_token_ce(args, model, model_batch, no_model_batch["label"])
                    loss = (1 - args.kd_ratio) * lm_loss + args.kd_ratio * distil_loss
                else:
                    loss = distil_loss
                # all_reduce below is in-place; keep a distinct tensor for the KD term.
                distil_loss = distil_loss.detach().clone()
            else:
                outputs = model(**model_batch, use_cache=False)

                logits = outputs.logits
                if args.model_parallel:
                    raise NotImplementedError
                else:
                    lm_loss = loss_func(logits.float().view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))

                if teacher_model is not None:
                    distil_loss = get_distil_loss(args, tokenizer, model, teacher_model, model_batch, no_model_batch, logits, epoch, mat_K=mat_K, mat_Kt=mat_Kt)
                    loss = (1 - args.kd_ratio) * lm_loss + args.kd_ratio * distil_loss
                else:
                    loss = lm_loss

            if args.lm_data_dir is not None:
                assert args.lm_coef is not None
                loss += args.lm_coef * pt_loss(args, model, pt_model_batch, pt_no_model_batch)

            model.backward(loss)
            model.step()

            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            global_loss = loss.item() / dp_world_size

            global_distil_loss = 0
            if teacher_model is not None:
                dist.all_reduce(distil_loss, dist.ReduceOp.SUM, group=dp_group)
                global_distil_loss = distil_loss.item() / dp_world_size
                total_distil_loss += global_distil_loss
            if sreotd_stats is not None:
                reduced_stats = _reduce_stat_means(sreotd_stats, loss.device, dp_world_size, dp_group)
                for key, value in reduced_stats.items():
                    sreotd_window[key] += value
                sreotd_window_steps += 1

            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            total_loss += global_loss
            total_time += elapsed_time

            if get_rank() == 0:
                train_bar.set_postfix(
                    loss=f"{global_loss:.4f}",
                    ds_loss=f"{global_distil_loss:.4f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    refresh=False,
                )

            # Logging
            def get_log(log_loss, log_distil_loss, log_time):
                return "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | loss: {:.4f} | ds_loss: {:.4f} | lr: {:.4e} | scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f}".format(
                    epoch,
                    step,
                    args.total_iters * args.gradient_accumulation_steps,
                    global_step,
                    args.total_iters,
                    log_loss,
                    log_distil_loss,
                    lr_scheduler.get_last_lr()[0],
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    elapsed_time,
                    log_time,
                )

            if args.mid_log_num > 0:
                mid_log_step = args.gradient_accumulation_steps // args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                if step % mid_log_step == 0:
                    print_rank(get_log(global_loss, global_distil_loss, 0))

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                log_str = get_log(total_loss / (args.log_interval * args.gradient_accumulation_steps), total_distil_loss / (args.log_interval * args.gradient_accumulation_steps), total_time / (args.log_interval))
                if sreotd_window_steps > 0:
                    stat_str = " | ".join(
                        f"{key}: {sreotd_window[key] / sreotd_window_steps:.4f}" for key in SREOTD_STAT_KEYS
                    )
                    log_str = log_str + " | " + stat_str
                    for key in sreotd_window:
                        sreotd_window[key] = 0.0
                    sreotd_window_steps = 0
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0

            # Checkpointing
            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                save_dir_path = os.path.join(args.save, str(global_step))
                if args.model_parallel:
                    raise NotImplementedError
                else:
                    if dist.get_rank() == 0:
                        os.makedirs(save_dir_path, exist_ok=True)
                        print_rank(f"Model save to {save_dir_path}")
                        if getattr(args, "kd_objective", None) == "sr_eotd":
                            save_sreotd_checkpoint(model, tokenizer, save_dir_path, args)
                        else:
                            tokenizer.save_pretrained(save_dir_path)
                            model.module.save_pretrained(save_dir_path, safe_serialization=False)
                dist.barrier()

            # Evaluation
            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                # curr_avg_loss, cur_res = evaluate(args, tokenizer, model, dataset["dev"], "dev", epoch, device, adaptive_threshold)
                curr_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev", epoch, device, adaptive_threshold)
                _consider_sreotd_best(args, model, tokenizer)
                if "adaptive" in args.type:
                    if curr_avg_loss >= prev_avg_loss + args.loss_eps:
                        adaptive_threshold += 0.1
                        adaptive_threshold = min(adaptive_threshold, 1.0)
                        prev_avg_loss = curr_avg_loss
                # total_res.append([step]+cur_res)
                model.train()

            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1

            if global_step > args.total_iters:
                break

    _push_sreotd_model_if_requested(args)

    ##### Save #####
    # total_res = np.array(total_res)
    # best_res = total_res[np.argmax(total_res[:, -1])]
    # save_rank(f"best | step: {best_res[0]}, avg_loss: {best_res[1]}, exact_match: {best_res[2]}, rougeL: {best_res[3]}", os.path.join(args.save, "log.txt"))
    # np.save(f"{args.save}/total_res#best_{best_res[0]}_{best_res[-1]:.2f}.npy", np.array(total_res))

    return model


def _decode_generated(tokenizer, response_ids):
    return tokenizer.batch_decode(response_ids.cpu(), skip_special_tokens=True)


def _gather_padded_ids(chunks, world_size, group):
    ids = torch.cat(chunks, dim=0)
    ids = all_gather(ids, dim=1, world_size=world_size, group=group, op="stack")
    return ids.view(-1, ids.size(-1))


def evaluate_sreotd(args, tokenizer, model, dataset: LMTrainDataset, split, epoch, device, adaptive_threshold=None):
    """Validation CE plus ROUGE-L of every LoRA branch, the mean, and the oracle best."""
    collate_fn = dataset.collate
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None
    loss_func = nn.CrossEntropyLoss()
    student = unwrap_model(model)
    branches = args.num_lora_branches

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    model.eval()
    all_loss = 0.0
    step = 0
    branch_response_chunks = [[] for _ in range(branches)]

    with torch.no_grad():
        for model_batch, no_model_batch, gen_data in tqdm(dataloader, desc="Evaluating SR-EOTD", disable=(dist.get_rank() != 0)):
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            loss_sum = None
            max_new_tokens = args.max_length - gen_data["input_ids"].size(1)
            for branch in range(branches):
                student.set_active_branch(branch)
                logits = model(**model_batch).logits
                branch_loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
                loss_sum = branch_loss if loss_sum is None else loss_sum + branch_loss
                if args.eval_gen:
                    gen_out = model.generate(**gen_data, generation_config=generation_config, max_new_tokens=max_new_tokens)
                    full_ids = F.pad(gen_out.sequences, (0, args.max_length - gen_out.sequences.shape[1]), value=tokenizer.pad_token_id)
                    response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                    branch_response_chunks[branch].append(response_ids)
            loss = loss_sum / branches
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            all_loss += (loss / dp_world_size).item()
            step += 1

    student.set_active_branch(None)
    branch_texts = []
    if args.eval_gen:
        for branch in range(branches):
            response_ids = _gather_padded_ids(branch_response_chunks[branch], dp_world_size, dp_group)
            branch_texts.append(_decode_generated(tokenizer, response_ids))

    avg_loss = all_loss / max(step, 1)
    selection = -float(avg_loss)
    if get_rank() == 0 and args.eval_gen:
        references = dataset.answers
        per_branch_scores = []
        rouge_values = []
        log_parts = [f"{split} | avg_loss: {avg_loss}"]
        prefix = "val" if split in {"dev", "valid", "val"} else split
        for branch, texts in enumerate(branch_texts):
            texts = texts[:len(references)]
            scores = per_example_rouge(texts, references)
            rouge_l = 100.0 * (sum(scores) / len(scores) if scores else 0.0)
            per_branch_scores.append(scores)
            rouge_values.append(rouge_l)
            log_parts.append(f"{prefix}_rougeL_branch_{branch}: {rouge_l:.4f}")
            eval_dir = os.path.join(args.save, "eval", str(epoch), f"branch_{branch}")
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "answers.jsonl"), "w") as handle:
                for text in texts:
                    handle.write(json.dumps({"text": text}) + "\n")
        width = min((len(scores) for scores in per_branch_scores), default=0)
        best = None
        mean_score = None
        if width > 0:
            best = 100.0 * sum(max(per_branch_scores[b][i] for b in range(branches)) for i in range(width)) / width
            mean_score = sum(rouge_values) / len(rouge_values)
            log_parts.append(f"{prefix}_rougeL_mean: {mean_score:.4f}")
            log_parts.append(f"{prefix}_rougeL_best: {best:.4f}")
            selection = float(best)
        log_str = " | ".join(log_parts)
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
        args.sreotd_last_val_metrics = {
            "split": split,
            "avg_loss": avg_loss,
            "rougeL_mean": None if mean_score is None else round(mean_score, 4),
            "rougeL_best": None if best is None else round(best, 4),
        }
        for branch, value in enumerate(rouge_values):
            args.sreotd_last_val_metrics[f"rougeL_branch_{branch}"] = round(value, 4)
    elif get_rank() == 0:
        log_str = f"{split} | avg_loss: {avg_loss}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
        args.sreotd_last_val_metrics = {"split": split, "avg_loss": avg_loss}

    if dist.is_initialized():
        packed = torch.tensor([selection if get_rank() == 0 else 0.0], dtype=torch.float64, device=device)
        dist.broadcast(packed, src=0)
        selection = float(packed.item())
    args.sreotd_last_val_score = selection
    return avg_loss


def evaluate(args, tokenizer, model, dataset: LMTrainDataset, split, epoch, device, adaptive_threshold=None):
    if getattr(args, "kd_objective", None) == "sr_eotd":
        return evaluate_sreotd(args, tokenizer, model, dataset, split, epoch, device, adaptive_threshold)
    collate_fn = dataset.collate

    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    print_rank("dp size", dp_world_size)

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    model.eval()
    all_loss = 0.0
    step = 0

    all_response_ids = []

    with torch.no_grad():
        for it, (model_batch, no_model_batch, gen_data) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            logits = model(**model_batch).logits
            if args.model_parallel:
                raise NotImplementedError
            else:
                loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))

            max_new_tokens = args.max_length - gen_data["input_ids"].size(1)

            if args.eval_gen:
                gen_out = model.generate(**gen_data, generation_config=generation_config, max_new_tokens=max_new_tokens)

                full_ids = gen_out.sequences

                full_ids = F.pad(full_ids, (0, args.max_length - full_ids.shape[1]), value=tokenizer.pad_token_id)

                response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                all_response_ids.append(response_ids)

            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            loss = loss / dp_world_size
            all_loss += loss.item()
            step += 1

    if args.eval_gen:
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, group=dp_group, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))

        responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)

    if get_rank() == 0:
        if args.eval_gen:
            references = dataset.answers
            responses = responses[:len(references)]

            res = compute_metrics(responses, references)

            eval_dir = os.path.join(args.save, "eval", str(epoch))
            print_rank(eval_dir)
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                for resp in responses:
                    f.write(json.dumps({"text": resp}) + "\n")
        else:
            res = {}

        avg_loss = all_loss / step

        if "adaptive" in args.type:
            log_str = f"{split} | avg_loss: {avg_loss} | {res} | threshold: {adaptive_threshold}"
        else:
            log_str = f"{split} | avg_loss: {avg_loss} | {res}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))

    return all_loss / step # , [avg_loss, res["exact_match"], res["rougeL"]]


def main():
    torch.backends.cudnn.enabled = False

    args = get_args()
    initialize(args)

    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump({key: value for key, value in vars(args).items() if key != "hf_token"}, f)

    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "=" * 30 + f" EXP at {cur_time} " + "=" * 30, os.path.join(args.save, "log.txt"))

    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000

    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0

    ### args.fp32 = not ds_config["fp16"]["enabled"]
    args.fp32 = False

    args.deepspeed_config = None

    # get the tokenizer
    tokenizer = get_tokenizer(args)
    dataset = prepare_dataset(args, tokenizer)

    dp_world_size = dist.get_world_size()

    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)

        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch

        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch

    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)

    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type

    if args.teacher_model_path is not None:
        teacher_model = get_teacher_model(args, device)
    else:
        teacher_model = None

    if args.do_train:
        model = finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model=teacher_model)

    if args.do_eval:
        evaluate(args, tokenizer, model, dataset["test"], "test", 0, device)


if __name__ == "__main__":
    main()
