"""Push SR-EOTD checkpoints and eval artifacts to the Hugging Face Hub.

Model files go to a ``repo_type="model"`` repo. Generations, logits, embeddings,
and mode diagnostics go to a separate ``repo_type="dataset"`` repo.

The dataset viewer reads one parquet config per benchmark. The ``.pt`` tensors
stay under ``raw_artifacts/<benchmark>/`` and are loaded with
``load_sreotd_eval_artifacts_from_hf``.
"""

import argparse
import json
import os
import shutil
import tempfile

import torch


RAW_ARTIFACTS = (
    "sreotd_logits_and_modes.pt",
    "branch_embeddings_and_weights.pt",
    "high_entropy_fork_snapshots.json",
    "mode_collapse_diagnostics.json",
    "lora_subspace_orthogonality.json",
    "eval_summary_metrics.json",
    "branch_generations.jsonl",
)

_TOKENIZER_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "tokenizer.model",
)

LOGITS_SCHEMA = {
    "topk_indices": {"shape": "[N, T, K]", "dtype": "int32", "meaning": "Teacher top-K token ids, K=64."},
    "teacher_topk_logits": {"shape": "[N, T, K]", "dtype": "float16", "meaning": "Teacher logits on that support."},
    "student_topk_logits": {"shape": "[N, T, M, K]", "dtype": "float16", "meaning": "Branch logits on the teacher support."},
    "student_own_top16_logits": {"shape": "[N, T, M, 16]", "dtype": "float16", "meaning": "Largest full-vocabulary logits of each branch."},
    "student_own_top16_ids": {"shape": "[N, T, M, 16]", "dtype": "int32", "meaning": "Token ids of those full-vocabulary logits."},
    "ot_branch_weights_w": {"shape": "[N, T, M]", "dtype": "float16", "meaning": "Sinkhorn row marginal w*."},
    "teacher_norm_entropy": {"shape": "[N, T]", "dtype": "float16", "meaning": "Normalized teacher entropy H_hat on the top-K support."},
    "loss_mask": {"shape": "[N, T]", "dtype": "bool", "meaning": "Response tokens used for the diagnostics."},
    "ot_plan_gamma_sample": {"shape": "[min(N, 32), T, M, K]", "dtype": "float16", "meaning": "Transport plan Gamma* for the first 32 sequences."},
}

EMBEDDINGS_SCHEMA = {
    "hidden_mean": {"shape": "[N, M, d]", "dtype": "float16", "meaning": "Mean-pooled branch hidden states over non-pad tokens."},
    "hidden_last": {"shape": "[N, M, d]", "dtype": "float16", "meaning": "Hidden state at the last non-pad token of each branch."},
    "w_star": {"shape": "[N, M]", "dtype": "float16", "meaning": "Response-token mean of w* for each example."},
    "H_teacher": {"shape": "[N]", "dtype": "float16", "meaning": "Response-token mean of the normalized teacher entropy."},
}


def resolve_hf_token(token=None):
    if token:
        return token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _load_multi_lora():
    import importlib.util
    import sys
    import types
    existing = sys.modules.get("distillm.multi_lora")
    if existing is not None and hasattr(existing, "export_merged_branch_state_dict"):
        return existing
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if "distillm" not in sys.modules:
        package = types.ModuleType("distillm")
        package.__path__ = [os.path.join(root, "distillm")]
        package.__package__ = "distillm"
        __import__("sys").modules["distillm"] = package
    path = os.path.join(root, "distillm", "multi_lora.py")
    spec = importlib.util.spec_from_file_location("distillm.multi_lora", path)
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules["distillm.multi_lora"] = module
    spec.loader.exec_module(module)
    return module


def _read_json(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def _copy_present(src_dir, dst_dir, names):
    os.makedirs(dst_dir, exist_ok=True)
    copied = []
    for name in names:
        src = os.path.join(src_dir, name)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(dst_dir, name))
            copied.append(name)
    return copied


def _load_checkpoint_state(ckpt_dir):
    weight_path = os.path.join(ckpt_dir, "sreotd_model.pt")
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Missing {weight_path}")
    try:
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weight_path, map_location="cpu")
    return state


def _write_hf_folder(path, ckpt_dir, state):
    os.makedirs(path, exist_ok=True)
    torch.save(state, os.path.join(path, "pytorch_model.bin"))
    _copy_present(ckpt_dir, path, _TOKENIZER_FILES)


def _rouge_table(metrics):
    if not metrics:
        return "_No ROUGE metrics were attached to this upload._"
    lines = ["| Metric | Value |", "| --- | ---: |"]
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, (int, float)):
            lines.append(f"| `{key}` | {value} |")
    if len(lines) == 2:
        return "_The metrics file has no numeric scores._"
    return "\n".join(lines)


def _model_card(repo_id, meta, metrics, export_merged):
    branches = (meta or {}).get("num_lora_branches", "M")
    rank = (meta or {}).get("sreotd_lora_r", "r")
    alpha = (meta or {}).get("sreotd_lora_alpha", "alpha")
    merged = ""
    if export_merged:
        merged = (
            "\nMerged branches are plain Hugging Face causal LMs. "
            f"`merged_branches/branch_m` is `phi + (alpha/r) B_m A_m` "
            "(the Conv1D update is the transpose of `B A`).\n\n"
            "```python\n"
            "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
            f"model = AutoModelForCausalLM.from_pretrained({repo_id!r}, subfolder=\"merged_branches/branch_0\")\n"
            f"tokenizer = AutoTokenizer.from_pretrained({repo_id!r})\n"
            "```\n"
        )
    return f"""---
library_name: transformers
tags:
- sreotd
- multi-lora
- knowledge-distillation
---

# {repo_id}

SR-EOTD student: one shared backbone and {branches} LoRA branches (r={rank}, alpha={alpha}).

`pytorch_model.bin` at the repository root is the shared backbone `phi` without the LoRA residual.
`multi_lora_weights.pt` holds every branch factor: `A` is `[M, r, d_in]` and `B` is `[M, d_out, r]`.
The scale applied at merge time is `alpha / r`.

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

backbone = AutoModelForCausalLM.from_pretrained({repo_id!r})
tokenizer = AutoTokenizer.from_pretrained({repo_id!r})
factors = torch.load(
    hf_hub_download({repo_id!r}, "multi_lora_weights.pt"),
    map_location="cpu",
)
```
{merged}
## Validation scores

{_rouge_table(metrics)}
"""


def push_sreotd_model_to_hub(
    ckpt_dir,
    model_repo_id,
    token=None,
    private=False,
    export_merged_branches=False,
    metrics=None,
):
    """Create a model repo and upload the backbone, LoRA factors, and optional merges."""
    from huggingface_hub import HfApi

    if not model_repo_id:
        raise ValueError("hf_model_repo_id is required to push the SR-EOTD model")
    token = resolve_hf_token(token)
    ckpt_dir = os.path.abspath(ckpt_dir)
    state = _load_checkpoint_state(ckpt_dir)
    meta = _read_json(os.path.join(ckpt_dir, "sreotd_meta.json")) or {}
    if metrics is None:
        metrics = _read_json(os.path.join(ckpt_dir, "val_metrics.json"))
    multi = _load_multi_lora()
    alpha = float(meta.get("sreotd_lora_alpha", 16))
    rank = float(meta.get("sreotd_lora_r", 16))
    scale = alpha / rank if rank else 1.0
    factors = multi.extract_multi_lora_factors(state)
    if not factors:
        raise RuntimeError(f"No Multi-LoRA A/B tensors found in {ckpt_dir}")
    backbone = multi.export_backbone_state_dict(state)
    branches = int(meta.get("num_lora_branches") or next(iter(factors.values())).shape[0])

    api = HfApi(token=token)
    api.create_repo(repo_id=model_repo_id, repo_type="model", private=bool(private), exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sreotd_model_") as staging:
        _write_hf_folder(staging, ckpt_dir, backbone)
        torch.save(
            {
                "num_branches": branches,
                "rank": int(rank),
                "lora_alpha": alpha,
                "scaling": scale,
                "state_dict": factors,
            },
            os.path.join(staging, "multi_lora_weights.pt"),
        )
        config = dict(meta)
        config.update({"scaling": scale, "repo_layout": "backbone weights at root; A/B in multi_lora_weights.pt"})
        with open(os.path.join(staging, "sreotd_config.json"), "w") as handle:
            json.dump(config, handle, indent=2)
        if export_merged_branches:
            for branch in range(branches):
                merged = multi.export_merged_branch_state_dict(state, branch, scale)
                folder = os.path.join(staging, "merged_branches", f"branch_{branch}")
                _write_hf_folder(folder, ckpt_dir, merged)
        with open(os.path.join(staging, "README.md"), "w") as handle:
            handle.write(_model_card(model_repo_id, meta, metrics, export_merged_branches))
        api.upload_folder(
            folder_path=staging,
            repo_id=model_repo_id,
            repo_type="model",
            token=token,
            commit_message="Upload SR-EOTD backbone, LoRA factors, and merged branches",
        )
    print(f"SR-EOTD model uploaded to https://huggingface.co/{model_repo_id}", flush=True)
    return model_repo_id


def _benchmark_dirs(eval_dir):
    eval_dir = os.path.abspath(eval_dir)
    direct = os.path.join(eval_dir, "branch_generations.jsonl")
    if os.path.isfile(direct):
        return [(os.path.basename(eval_dir), eval_dir)]
    found = []
    if not os.path.isdir(eval_dir):
        raise FileNotFoundError(f"Eval directory does not exist: {eval_dir}")
    for name in sorted(os.listdir(eval_dir)):
        folder = os.path.join(eval_dir, name)
        if os.path.isdir(folder) and os.path.isfile(os.path.join(folder, "branch_generations.jsonl")):
            found.append((name, folder))
    if not found:
        raise FileNotFoundError(f"No branch_generations.jsonl under {eval_dir}")
    return found


def _load_jsonl(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _preserve_dataset_card(dataset_repo_id, body, token):
    """Keep the Dataset Viewer YAML that ``push_to_hub`` wrote, and replace the prose."""
    from huggingface_hub import hf_hub_download

    existing = ""
    try:
        path = hf_hub_download(
            repo_id=dataset_repo_id,
            filename="README.md",
            repo_type="dataset",
            token=token,
        )
        with open(path) as handle:
            existing = handle.read()
    except Exception:
        existing = ""
    prose = body
    if prose.startswith("---"):
        end = prose.find("\n---", 3)
        if end != -1:
            prose = prose[end + 4:].lstrip("\n")
    if existing.startswith("---"):
        end = existing.find("\n---", 3)
        if end != -1:
            return existing[:end + 4] + "\n\n" + prose
    return prose


def _dataset_card(dataset_repo_id, benchmarks):
    logit_rows = "\n".join(
        f"| `{name}` | `{spec['shape']}` | `{spec['dtype']}` | {spec['meaning']} |"
        for name, spec in LOGITS_SCHEMA.items()
    )
    embed_rows = "\n".join(
        f"| `{name}` | `{spec['shape']}` | `{spec['dtype']}` | {spec['meaning']} |"
        for name, spec in EMBEDDINGS_SCHEMA.items()
    )
    names = ", ".join(f"`{name}`" for name in benchmarks) or "_none in this upload_"
    return f"""# {dataset_repo_id}

SR-EOTD evaluation dump. Each benchmark is a Dataset Viewer config (`split="test"`) built from `branch_generations.jsonl`.
Benchmarks in the latest upload: {names}.

## Generation columns

`sample_id`, `prompt`, `ground_truth`, `response_branch_0`, `response_branch_1`, `response_branch_2`,
`rougeL_branch_*`, `rougeL_best_of_M`, `w_star` (length M), `mean_teacher_entropy`, `mean_pairwise_jsd`.

## `raw_artifacts/<benchmark>/sreotd_logits_and_modes.pt`

| Tensor | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
{logit_rows}

`N` is at most `--sreotd-diag-samples` (default 256). `T` is the padded teacher-forcing length. `M` is the branch count. `K` is 64.

## `raw_artifacts/<benchmark>/branch_embeddings_and_weights.pt`

| Tensor | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
{embed_rows}

Here `N` is the full evaluated split. Hidden states are pooled on non-pad tokens. `w_star` and `H_teacher` are averaged on response tokens (`loss_mask`).

Other files in the same folder: `high_entropy_fork_snapshots.json` (100 highest-entropy positions), `mode_collapse_diagnostics.json` (low / medium / high entropy regions), `lora_subspace_orthogonality.json`, `eval_summary_metrics.json`.

```python
from tools.hf_hub_manager import load_sreotd_eval_artifacts_from_hf
artifacts = load_sreotd_eval_artifacts_from_hf({dataset_repo_id!r}, benchmark="dolly")
logits = artifacts["logits"]["student_topk_logits"]
w_star = artifacts["embeddings"]["w_star"]
```
"""


def push_sreotd_eval_data_to_hub(eval_dir, dataset_repo_id, token=None, private=False, benchmarks=None):
    """Upload parquet generations and the raw `.pt` / JSON diagnostics to a dataset repo."""
    from huggingface_hub import HfApi
    from datasets import Dataset

    if not dataset_repo_id:
        raise ValueError("hf_dataset_repo_id is required to push SR-EOTD eval data")
    token = resolve_hf_token(token)
    selected = _benchmark_dirs(eval_dir)
    if benchmarks:
        wanted = set(benchmarks)
        selected = [item for item in selected if item[0] in wanted]
        if not selected:
            raise FileNotFoundError(f"None of {sorted(wanted)} were found under {eval_dir}")

    api = HfApi(token=token)
    api.create_repo(repo_id=dataset_repo_id, repo_type="dataset", private=bool(private), exist_ok=True)
    uploaded = []
    for name, folder in selected:
        rows = _load_jsonl(os.path.join(folder, "branch_generations.jsonl"))
        if not rows:
            print(f"Skipping empty generations for {name}", flush=True)
            continue
        dataset = Dataset.from_list(rows)
        import inspect
        push_kwargs = {"config_name": name, "split": "test", "token": token}
        if "private" in inspect.signature(dataset.push_to_hub).parameters:
            push_kwargs["private"] = bool(private)
        dataset.push_to_hub(dataset_repo_id, **push_kwargs)
        api.upload_folder(
            folder_path=folder,
            repo_id=dataset_repo_id,
            repo_type="dataset",
            path_in_repo=f"raw_artifacts/{name}",
            allow_patterns=list(RAW_ARTIFACTS),
            token=token,
            commit_message=f"Upload SR-EOTD raw artifacts for {name}",
        )
        uploaded.append(name)
        print(f"Uploaded eval config {name} to {dataset_repo_id}", flush=True)

    card = _preserve_dataset_card(dataset_repo_id, _dataset_card(dataset_repo_id, uploaded), token)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=dataset_repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Update SR-EOTD dataset card",
    )
    print(f"SR-EOTD eval data uploaded to https://huggingface.co/datasets/{dataset_repo_id}", flush=True)
    return dataset_repo_id


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_sreotd_eval_artifacts_from_hf(dataset_repo_id, benchmark="dolly", token=None, cache_dir=None):
    """Download one benchmark's logits, embeddings, and JSON diagnostics.

    Returns a dict with ``logits``, ``embeddings``, ``mode_collapse_diagnostics``,
    ``lora_subspace_orthogonality``, ``eval_summary_metrics``, ``forks``, and ``paths``.
    """
    from huggingface_hub import hf_hub_download

    token = resolve_hf_token(token)
    paths = {}
    loaded = {}
    for name in RAW_ARTIFACTS:
        try:
            paths[name] = hf_hub_download(
                repo_id=dataset_repo_id,
                filename=f"raw_artifacts/{benchmark}/{name}",
                repo_type="dataset",
                token=token,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            paths[name] = None
            loaded[name] = exc
    if paths.get("sreotd_logits_and_modes.pt") is None or paths.get("branch_embeddings_and_weights.pt") is None:
        missing = [name for name in ("sreotd_logits_and_modes.pt", "branch_embeddings_and_weights.pt") if not paths.get(name)]
        raise FileNotFoundError(f"Missing {missing} in {dataset_repo_id} raw_artifacts/{benchmark}")

    def _json(name):
        path = paths.get(name)
        if not path:
            return None
        with open(path) as handle:
            return json.load(handle)

    return {
        "logits": _torch_load(paths["sreotd_logits_and_modes.pt"]),
        "embeddings": _torch_load(paths["branch_embeddings_and_weights.pt"]),
        "forks": _json("high_entropy_fork_snapshots.json"),
        "mode_collapse_diagnostics": _json("mode_collapse_diagnostics.json"),
        "lora_subspace_orthogonality": _json("lora_subspace_orthogonality.json"),
        "eval_summary_metrics": _json("eval_summary_metrics.json"),
        "paths": paths,
    }


def main():
    parser = argparse.ArgumentParser(description="Upload an SR-EOTD checkpoint and its eval artifacts.")
    parser.add_argument("--ckpt-dir", default=None, help="Directory containing sreotd_model.pt")
    parser.add_argument("--eval-dir", default=None, help="sreotd_eval_artifacts directory, or one benchmark folder")
    parser.add_argument("--hf-model-repo-id", default=None)
    parser.add_argument("--hf-dataset-repo-id", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--hf-export-merged-branches", action="store_true")
    args = parser.parse_args()
    token = resolve_hf_token(args.hf_token)
    if args.hf_dataset_repo_id is None and args.hf_model_repo_id:
        args.hf_dataset_repo_id = f"{args.hf_model_repo_id}-eval-data"
    if args.ckpt_dir:
        if not args.hf_model_repo_id:
            raise SystemExit("--hf-model-repo-id is required with --ckpt-dir")
        push_sreotd_model_to_hub(
            args.ckpt_dir,
            args.hf_model_repo_id,
            token=token,
            private=args.hf_private,
            export_merged_branches=args.hf_export_merged_branches,
        )
    if args.eval_dir:
        if not args.hf_dataset_repo_id:
            raise SystemExit("--hf-dataset-repo-id is required with --eval-dir")
        push_sreotd_eval_data_to_hub(
            args.eval_dir,
            args.hf_dataset_repo_id,
            token=token,
            private=args.hf_private,
        )
    if not args.ckpt_dir and not args.eval_dir:
        raise SystemExit("Pass --ckpt-dir and/or --eval-dir")


if __name__ == "__main__":
    main()
