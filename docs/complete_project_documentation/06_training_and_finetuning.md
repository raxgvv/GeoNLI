# 06 — Training and Fine-Tuning Investigation

## Executive Summary

GeoNLI contains evidence of **two distinct fine-tuning efforts**:

1. **LoRA adapter** for captioning on Qwen3-VL-8B-Instruct
2. **Full fine-tuning** of Qwen3-VL-8B-Instruct for VQA on VRSBench

Both are **optional** at runtime — the system degrades gracefully when checkpoints are absent.

---

## Evidence Classification

| Claim | Evidence Type |
|---|---|
| VQA model was fine-tuned on VRSBench | [Code] Training scripts + configs |
| LoRA adapter exists for captioning | [Config] adapter_config.json |
| Training used 2×L40S GPUs | [Config] prod_2L40S.yaml comments |
| VQA training ran for 5 epochs | [Config] prod config + trainer_state.json |
| Captioning adapter trained for 1 epoch | [Config] training_state.json |
| No training was run in this repo session | [Unknown] No logs in repo |

---

## 1. VQA Full Fine-Tuning

### Base Model
- **Model:** `Qwen/Qwen3-VL-8B-Instruct` [Config: VQA-fine-tuning/configs/prod_2L40S.yaml:2]
- **Architecture:** Qwen3VLForConditionalGeneration [Code: VQA-fine-tuning/vrsbench_finetune/model.py:11]

### Training Objective
- **Task:** Visual question answering on remote sensing imagery
- **Dataset:** VRSBench (custom dataset) [Config: VQA-fine-tuning/README.md:3]
- **Loss:** Cross-entropy with task weighting [Code: VQA-finetuning/vrsbench_finetune/train.py:23-70]
- **Task weights:** binary=1.0, numeric=1.0, semantic=1.0 [Config: prod_2L40S.yaml:26-29]

### Training Configuration

| Parameter | dev.yaml | prod_2L40S.yaml |
|---|---|---|
| Base model | Qwen/Qwen3-VL-8B-Instruct | Qwen/Qwen3-VL-8B-Instruct |
| Epochs | 3 | 5 |
| Batch size | 4 | 8 |
| Gradient accumulation | 4 | 4 |
| Backbone LR | 5e-6 | 5e-6 |
| LM head LR | 2e-5 | 2e-5 |
| Weight decay | 0.01 | 0.01 |
| Scheduler | cosine | cosine |
| Warmup ratio | 0.03 | 0.03 |
| Mixed precision | bf16 | bf16 |
| Flash attention | true (if available) | true (if available) |
| Gradient checkpointing | enabled | enabled |
| Effective batch size | 4×4=16 | 8×4×2=64 |

### Training Features

| Feature | Evidence |
|---|---|
| Full fine-tuning (not LoRA) | [Code] train.py: model.gradient_checkpointing_enable() only |
| Mixed precision (bf16) | [Code] train.py:103-106 |
| Gradient checkpointing | [Code] model.py:21 |
| Gradient accumulation | [Code] train.py:182-184, 238 |
| Task-weighted loss | [Code] train.py:64-68 |
| Separate LR for backbone vs lm_head | [Code] train.py:160-179 |
| Cosine scheduler with warmup | [Code] train.py:186-191 |
| Checkpoint resume | [Code] train.py:196-211 |

### Checkpoint Evidence

| Artifact | Value | Location |
|---|---|---|
| trainer_state.json | `{"epoch": 4, "step": 10724}` | models/vqa-model/ |
| Shard count | 4 | model.safetensors.index.json |
| Total parameters | 8,767,123,696 | model.safetensors.index.json:3 |
| Total size | ~17.5 GB | model.safetensors.index.json:4 |

**Interpretation:** The checkpoint at step 10724 of epoch 4 suggests training completed (5 epochs planned, stopped at epoch 4). The `checkpoint-2681` referenced in `run_eval.sh` may be an intermediate checkpoint. [Inferred]

### Training Data

- **Source:** VRSBench dataset [Config: VQA-fine-tuning/README.md:3]
- **Format:** JSONL with fields: image, question, task, answer_text [Code: dataset.py:82-89]
- **Task types:** binary, numeric, semantic [Code: preprocess.py:77-96]
- **Image directory:** `../VRSBench/Images_train` [Config: prod_2L40S.yaml:8]
- **Data location:** Not in repository (external) [Unknown]

### Data Preprocessing

1. Load VRSBench JSON annotations [Code: preprocess.py:48-49]
2. Detect task type from answer [Code: preprocess.py:77-96]
3. Binary: "yes"/"no" → task="binary" [Code: preprocess.py:18-22]
4. Numeric: Parse float/scientific notation [Code: preprocess.py:24-45]
5. Semantic: Everything else [Code: preprocess.py:95-96]
6. Tokenize with Qwen3-VL processor [Code: dataset.py:57-62]
7. Mask prompt labels (-100) [Code: dataset.py:79-80]

### Label Generation

Labels are **automatic** from VRSBench annotations:
- Binary answers: "yes" → "Yes", "no" → "No" [Code: preprocess.py:88]
- Numeric answers: Formatted to 1 decimal place [Code: preprocess.py:93]
- Semantic answers: Raw text [Code: preprocess.py:95-96]

---

## 2. Captioning LoRA Adapter

### Base Model
- **Model:** `Qwen/Qwen3-VL-8B-Instruct` [Config: models/captioning-model/model/adapter_config.json:6]

### Adapter Configuration

| Parameter | Value |
|---|---|
| PEFT type | LORA |
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Bias | none |
| Target modules | up_proj, down_proj, k_proj, q_proj, o_proj, gate_proj, v_proj |
| Task type | CAUSAL_LM |

### Training Evidence

| Artifact | Value | Location |
|---|---|---|
| training_state.json | `{"epoch": 1, "global_step": 6426, "tag": "step_6426"}` | models/captioning-model/ |
| Optimizer state | scheduler.bin, random_states_0.pkl | models/captioning-model/state/ |

**Note:** No training scripts for the captioning adapter are present in the repository. The adapter was likely trained separately and only the resulting weights were saved. [Unknown]

### Adapter Compatibility

The adapter targets `Qwen/Qwen3-VL-8B-Instruct`, but the default `ADV_VLM_MODEL_ID` is `Qwen/Qwen3-VL-2B-Instruct`. The code validates this match [Code: models/captioning.py:80-83]. To use the adapter:
- Set `ADV_VLM_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct`
- The processor will be loaded from the base model ID [Code: models/captioning.py:62-63]

---

## 3. Pix2Pix SAR-to-Optical Training

### Architecture
- UNet with 8 encoder blocks (C64→C512) [Code: models/SAR2Optical/src/networks.py:7-43]
- 7 decoder blocks with dropout [Code: models/SAR2Optical/src/networks.py:46-80]

### Checkpoint
- **File:** `pix2pix_gen_180.pth` [Code: backend/services/sar_converter.py:15]
- **Not in repository** (gitignored) [Config: .gitignore:62]
- **Training evidence:** No training logs or scripts for this checkpoint in the repo [Unknown]

---

## 4. SAM3 Training

- SAM3 is used **pretrained only** [Code: models/sam3/runtime_service.py]
- No fine-tuning code for SAM3 in the repository
- The checkpoint is downloaded from `facebook/sam3` [Code: models/sam3/runtime_service.py:69]

---

## 5. Query Router Training

- The Mistral-7B router is used **pretrained only** [Code: backend/services/query_router.py]
- No fine-tuning code for the router
- **Status:** Disabled by default [Code: backend/core/model.py:225]

---

## Training Strategy Assessment

### Why full fine-tuning for VQA?

| Alternative | Why Not Used |
|---|---|
| LoRA/PEFT | Full fine-tuning may provide better performance for domain-specific VQA |
| Prompt engineering | Limited by the model's pretraining distribution |
| Few-shot | Not suitable for 8B model fine-tuning |

**Assessment:** Full fine-tuning on 2×L40S GPUs with 64 effective batch size is a serious training setup. The 5-epoch schedule with cosine decay and separate backbone/head learning rates is a standard, well-designed approach. [Inferred]

### Why LoRA for captioning?

| Alternative | Why Not Used |
|---|---|
| Full fine-tuning | LoRA is more parameter-efficient and sufficient for style transfer |
| No fine-tuning | The base model may not produce concise remote sensing captions |

**Assessment:** LoRA with r=16, alpha=32 targeting all attention and MLP projections is a standard configuration for adapting a VLM to a specific captioning style. [Inferred]

---

## What Cannot Be Verified

| Question | Status |
|---|---|
| Actual training loss values | Not in repository |
| Training hardware used | Inferred from config comments (2×L40S) |
| Training duration | Not recorded in repository |
| Dataset size | Not recorded |
| Evaluation metrics on test set | predictions.jsonl exists but metrics not in repo |
| Whether training completed successfully | trainer_state.json suggests completion |

---

## Reproducing Training

### VQA Fine-tuning

```bash
cd VQA-fine-tuning
pip install -r requirements.txt
python vrsbench_finetune/preprocess.py \
  --input_dir ../VRSBench/Annotations_train \
  --image_dir ../VRSBench/Images_train \
  --out_jsonl data/processed/train.jsonl \
  --numeric_stats data/processed/numeric_stats.json \
  --binary_stats data/processed/binary_stats.json
bash scripts/run_train.sh
```

### VQA Evaluation

```bash
python vrsbench_finetune/eval.py \
  --config configs/prod_2L40S.yaml \
  --checkpoint checkpoints/prod/checkpoint-2681 \
  --out_file predictions.jsonl \
  --save_samples sample_outputs.txt
```

**Note:** The VRSBench dataset is not included in the repository and must be obtained separately. [Unknown]

---

*Next: [Data and Preprocessing](07_data_and_preprocessing.md)*
