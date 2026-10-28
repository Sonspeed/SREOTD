# SR-EOTD: Semi-Relaxed Entropic OT Decomposition

This repository trains a router-free multi-LoRA student with **Semi-Relaxed Entropic OT Decomposition (SR-EOTD)**.

The student is one shared backbone plus `M` LoRA branches. At each token the teacher distribution is the softmax over its top-`K` logits. A damped log-Sinkhorn plan splits that distribution across the branches: column marginals match the teacher, and row marginals are only pulled toward the uniform `1/M`. The plan is held constant. The loss is a mass-weighted forward KL of each branch toward its OT target, plus an entropy-modulated reverse KL. Gradients flow through the student distributions only.

On GPT-2 the shared backbone stays trainable. For LLaMA, Qwen, and Mistral the backbone is frozen and only the LoRA factors train. Pass `--sreotd-freeze-backbone true` or `false` to override that default.

The training stack follows [MiniLLM](https://github.com/microsoft/LMOps/tree/main/minillm), [AMiD](https://github.com/aailab-kaist/AMiD), [DistiLLM](https://github.com/jongwooko/distillm), and [ABKD](https://github.com/ghwang-s/abkd).

--------------------

## Requirements

Tested environment:

- CUDA 12.8
- Python 3.10
- PyTorch 2.7.1
- NVIDIA RTX PRO 6000 GPU

### Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The same steps are in `install.sh` (`bash install.sh`).

## Datasets

GPT-2 instruction-following runs use [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k). The pretraining setting also uses [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext). Evaluation covers Dolly, Self-Instruct, Vicuna, Super NI, and UnNI, in the same format as [MiniLLM](https://github.com/microsoft/LMOps/tree/main/minillm).

```bash
hf download MiniLLM/dolly --repo-type dataset --local-dir ./data/dolly/
hf download MiniLLM/self-inst --repo-type dataset --local-dir ./data/self-inst/
hf download MiniLLM/Vicuna --repo-type dataset --local-dir ./data/vicuna/
hf download MiniLLM/sinst --repo-type dataset --local-dir ./data/sinst/
hf download MiniLLM/uinst --repo-type dataset --local-dir ./data/uinst/
```

Processed training data:

```bash
hf download MiniLLM/dolly-processed --repo-type dataset --local-dir ./processed_data/dolly/
hf download MiniLLM/openwebtext-processed --repo-type dataset --local-dir ./processed_data/openwebtext/gpt2/512/10M/
```

## Models

```bash
hf download gpt2 --repo-type model --local-dir ./checkpoints/gpt2-base
hf download gpt2-medium --repo-type model --local-dir ./checkpoints/gpt2-medium
hf download gpt2-xl --repo-type model --local-dir ./checkpoints/gpt2-xlarge
```

## Training Pipeline

Scripts are launched from the repository root. The first three arguments are `${BASE_PATH} ${MASTER_PORT} ${GPU_NUM}`.

SR-EOTD reads the teacher from `results/gpt2/train/sft/gpt2-xlarge/`. The pretraining run also reads the initialized student from `results/gpt2/train/init/gpt2-base`. The Hub snapshots below land on those paths. A local SFT or init run (`--type lm`) writes a nested directory under `results/gpt2/train/sft` or `results/gpt2/train/init`; point the distillation script at the checkpoint you want to use.

### Step 1: Teacher Fine-Tuning (SFT)

```bash
bash ./scripts/gpt2/sft/sft_xlarge.sh ./ 2010 1
```

or download the fine-tuned teacher:

```bash
hf download MiniLLM/teacher-gpt2-1.5B --repo-type model --local-dir ./results/gpt2/train/sft/gpt2-xlarge
```

### Step 2: Student Initialization

Used by the pretraining setting. The Dolly-only setting starts from `checkpoints/gpt2-base` and skips this step.

```bash
bash ./scripts/gpt2/init/init_base.sh ./ 2011 1
```

or download the initialized student:

```bash
hf download MiniLLM/init-gpt2-120M --repo-type model --local-dir ./results/gpt2/train/init/gpt2-base
```

### Step 3: Distillation with SR-EOTD

```bash
bash ./scripts/gpt2/sreotd/train_0.1B_1.5B.sh ${BASE_PATH} ${MASTER_PORT} ${GPU_NUM} ${BATCH_SIZE} ${LR}
```

Defaults are port `2012`, `4` GPUs, batch size `32`, and learning rate `0.0001`. Both scripts train GPT-2 XL (1.5B) into GPT-2 Base (0.1B) for 20 epochs with `M=3` LoRA branches, rank `16`, and teacher top-`K` `64`. The backbone stays trainable.

* Dolly KD plus the OpenWebText pretraining loss (`--lm-coef 1`). The student starts from the Step 2 checkpoint. Checkpoints go to `results/gpt2/train/0.1B_1.5B#sreotd/lm_${BATCH_SIZE}_${LR}/`.

  ```bash
  bash ./scripts/gpt2/sreotd/train_0.1B_1.5B.sh ./ 2012 1
  ```

* Dolly KD only. The student starts from `checkpoints/gpt2-base`. Checkpoints go to `results/gpt2/train/0.1B_1.5B#sreotd/nolm_${BATCH_SIZE}_${LR}/`.

  ```bash
  bash ./scripts/gpt2/sreotd/train_0.1B_1.5B_nolm.sh ./ 2013 1
  ```

`--save-interval -1` and `--eval-interval -1` mean one save and one validation pass per epoch. Validation reports the cross-entropy and the ROUGE-L of every branch, the mean, and the oracle best-of-`M`. The directory `best_ckpt/` keeps the epoch with the highest best-of-`M` ROUGE-L. Each checkpoint contains `sreotd_model.pt`, `sreotd_meta.json`, the model config, and the tokenizer. Training logs go to `log.txt` in the run directory, including the SR-EOTD diagnostics `H_teacher`, `H_target_weighted`, `H_student_weighted`, `w_max`, `w_min`, `jsd_high_H`, and `jsd_low_H`.

Older AMiD and WASD entry points remain under `scripts/gpt2/amid` and `scripts/gpt2/wasd`. `--kd-objective` accepts only `sr_eotd`.

#### Key Arguments

- `--kd-objective sr_eotd`: train with SR-EOTD. `--type sreotd` selects this run.
- `--num-lora-branches`: number of LoRA branches `M` (default `3`). `A` is initialized by QR so distinct branches start orthogonal. `B` is Gaussian with stddev `--sreotd-sigma-b`.
- `--sreotd-lora-r`, `--sreotd-lora-alpha`: rank and alpha. The residual scale is `alpha / r`. `M * r` must be at most the projection input size.
- `--sreotd-freeze-backbone`: train only `A` and `B`. Default is false for GPT-2, GPT-J, and OPT, and true for LLaMA, Qwen, and Mistral.
- `--sreotd-topk`: teacher support size `K` (default `64`).
- `--sreotd-tau`: temperature of the teacher softmax on those logits.
- `--sreotd-eps-min`, `--sreotd-eps-max`, `--sreotd-gamma`: adaptive entropic temperature `eps_min + (eps_max - eps_min) * exp(-gamma * H)`, where `H` is the teacher entropy normalized by `log(K)`.
- `--sreotd-sinkhorn-iters`: damped log-Sinkhorn iterations.
- `--sreotd-rho`: soft row-marginal penalty. The row damping is `rho / (rho + eps)`.
- `--sreotd-beta`: weight of the entropy-modulated reverse KL.
- `--kd-ratio`: mix of the OT loss and token cross-entropy on all branches. The scripts use `1.0`, so the OT loss is the distillation term. The pretraining run adds `--lm-coef` times the OpenWebText cross-entropy, also on all branches.
- `--sreotd-diag-samples`: leading evaluation examples stored with token-level logits and OT plans (default `256`).

## Evaluation

Load an SR-EOTD checkpoint with `--kd-objective sr_eotd`. `evaluate.py` then rebuilds the LoRA branches from `sreotd_meta.json` and loads `sreotd_model.pt`. Pass the fine-tuned teacher so the token-level OT diagnostics are computed.

`${CKPT}` is a path under `results/gpt2/train/`, for example `0.1B_1.5B#sreotd/lm_32_0.0001/best_ckpt`.

```bash
CKPT="0.1B_1.5B#sreotd/lm_32_0.0001/best_ckpt"
TEACHER="./results/gpt2/train/sft/gpt2-xlarge"
EXTRA="--kd-objective sr_eotd --teacher-model-path ${TEACHER}"

bash ./scripts/gpt2/eval/eval_main_dolly.sh     ./ 2113 1 "${CKPT}" --seed 10 --eval-batch-size 16 ${EXTRA}
bash ./scripts/gpt2/eval/eval_main_self_inst.sh ./ 2114 1 "${CKPT}" --seed 10 --eval-batch-size 16 ${EXTRA}
bash ./scripts/gpt2/eval/eval_main_vicuna.sh    ./ 2115 1 "${CKPT}" --seed 10 --eval-batch-size 16 ${EXTRA}
bash ./scripts/gpt2/eval/eval_main_sinst.sh     ./ 2116 1 "${CKPT}" --seed 10 --eval-batch-size 16 ${EXTRA}
bash ./scripts/gpt2/eval/eval_main_uinst.sh     ./ 2117 1 "${CKPT}" --seed 10 --eval-batch-size 16 ${EXTRA}
```

Each benchmark prints per-branch ROUGE-L, exact match, and LM loss, plus the mean, the best-of-`M` score, pairwise Self-BLEU, Distinct-2, token-level JSD, and the top-1 disagreement rate. Artifacts are written under the eval save directory in `sreotd_eval_artifacts/<benchmark>/`:

- `eval_summary_metrics.json`
- `branch_generations.jsonl`
- `lora_subspace_orthogonality.json`
- `mode_collapse_diagnostics.json` (low / medium / high teacher-entropy tokens)
- `high_entropy_fork_snapshots.json`
- `branch_embeddings_and_weights.pt`
- `sreotd_logits_and_modes.pt`

Add `--push-to-hf --hf-model-repo-id <user/model>` on a training run to upload the best checkpoint when training finishes. The same flags on evaluation upload that benchmark's artifacts. The dataset repo defaults to `<hf-model-repo-id>-eval-data`. `--hf-export-merged-branches` also uploads one backbone per branch with that LoRA residual merged in. The token is `--hf-token` or the `HF_TOKEN` environment variable.

## Sanity Checks

```bash
python tools/test_sreotd_modes.py
python tools/test_sreotd_diagnostics.py
```

`test_sreotd_modes.py` checks the orthogonal LoRA init, the parallel branch forward, the loss on a tiny GPT-2 student, and the transport weights on a unimodal teacher and on a skewed mixture. `test_sreotd_diagnostics.py` checks the mode-collapse report and the saved logit artifacts.

## Acknowledgements

This codebase builds upon:

- **MiniLLM**: https://github.com/microsoft/LMOps/tree/main/minillm
- **AMiD**: https://github.com/aailab-kaist/AMiD
- **DistiLLM**: https://github.com/jongwooko/distillm
- **ABKD**: https://github.com/ghwang-s/abkd
