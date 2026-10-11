# 12 — Evaluation and Experimental Validation

## Existing Tests

### Unit Tests

| Test File | Purpose | Status |
|---|---|---|
| `backend/tests/test_eval_api.py` | HTTP contract tests (mocked models) | Implemented |
| `backend/tests/test_sam3_postprocessing.py` | SAM3 mask-to-OBB conversion | Implemented |
| `backend/tests/benchmark_grounding.py` | Live SAM3 API benchmark | Implemented |

### Test Coverage

**test_eval_api.py** (8 tests):
1. Health check + caption response schema
2. VQA dispatch + inference error handling
3. Missing query rejection
4. Invalid image error handling
5. Grounding prompt normalization
6. Grounding failure handling
7. Generation timeout handling
8. SAR conversion failure handling

**test_sam3_postprocessing.py** (2 tests):
1. Raw count vs. thresholded detections
2. Low-confidence candidates → valid empty result

---

## Tests That Have Been Executed

| Test | Result | Evidence |
|---|---|---|
| None in this session | Not run | No execution logs |

**Note:** The README states "No real-model inference was run during publication of this repository; claims above reflect code paths and mocked contract tests only." [Doc: README.md:47]

---

## Benchmark Scripts

### `backend/tests/benchmark_grounding.py`

**Purpose:** Repeatable live SAM3 API benchmark on bundled sample images.

**Usage:**
```bash
python -m backend.tests.benchmark_grounding
```

**Output:** JSON records with duration, detection counts, and scores.

**Sample images:** `Extra Resourse/sample1.png`, `Extra Resourse/sample2.png`

**Prompts:** "Find airplanes", "Locate the runway", "Locate the roads", "Find buildings"

---

## Metrics

### VQA Metrics (from eval.py)

| Metric | Task | Formula |
|---|---|---|
| Binary accuracy | Binary | correct / total |
| Binary yes accuracy | Binary | yes_correct / yes_total |
| Binary no accuracy | Binary | no_correct / no_total |
| Numeric MAE | Numeric | mean(abs(pred - ref)) |
| Numeric RMSE | Numeric | sqrt(mean((pred - ref)²)) |
| Numeric score | Numeric | mean(exp(-23 * abs_error / abs(ref))) |
| Semantic BERT-BLEU | Semantic | LP * max(P_n) |

### Evaluation Status

| Metric | Value | Evidence |
|---|---|---|
| VQA accuracy | Not in repository | predictions.jsonl exists but no metrics |
| Caption quality | Not measured | — |
| Grounding IoU | Not measured | — |
| Inference latency | Not measured | — |
| Peak GPU memory | Not measured | — |

---

## What Has NOT Been Measured

| Claim | Status |
|---|---|
| VQA model accuracy | Not in repository |
| Captioning quality with/without LoRA | Not measured |
| Grounding precision/recall | Not measured |
| End-to-end latency | Not measured |
| GPU memory usage | Not measured |
| SAR conversion quality | Not measured |
| Scheduler overhead | Not measured |

---

## Recommended Experimental Plan

### 1. VQA Evaluation

```bash
cd VQA-fine-tuning
python vrsbench_finetune/eval.py \
  --config configs/prod_2L40S.yaml \
  --checkpoint ../models/vqa-model \
  --out_file predictions.jsonl
```

**Expected output:** Binary accuracy, numeric MAE/RMSE, semantic BERT-BLEU.

### 2. Ablation: VQA with shared VLM only

Remove `VQA_MODEL_DIR` env var → system falls back to Qwen3-VL-2B → compare accuracy.

### 3. Grounding Evaluation

```bash
python -m backend.tests.benchmark_grounding
```

Compare detection counts and scores across prompts.

### 4. Latency Measurement

Add timing logs to measure:
- VLM inference time
- SAM3 inference time
- Model swap overhead
- End-to-end response time

### 5. Memory Profiling

Use `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()` to measure peak usage.

---

## Baseline Comparisons

| Comparison | How |
|---|---|
| GeoNLI VQA vs. base Qwen3-VL-2B | Remove VQA_MODEL_DIR |
| GeoNLI captioning vs. base Qwen3-VL-2B | Remove CAPTIONING_MODEL_DIR |
| GeoNLI grounding vs. SAM3 paper | Compare to published SAM3 benchmarks |
| GeoNLI SAR vs. Pix2Pix paper | Compare to published Pix2Pix results |

---

## Logging and Monitoring

### What Is Logged

| Event | Location |
|---|---|
| Request start/end | backend/app.py |
| Inference timing | backend/core/model.py |
| Model placement | models/captioning.py, models/sam3/runtime_service.py |
| SAM3 candidates | models/sam3/unified_inference.py |
| Errors | backend/api/routes.py |

### What Is NOT Monitored

- GPU memory usage over time
- Queue wait times
- Model load times
- Cache hit rates

---

*Next: [Setup and Deployment](13_setup_and_deployment.md)*
