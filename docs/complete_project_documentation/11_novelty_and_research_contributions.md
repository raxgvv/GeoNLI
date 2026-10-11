# 11 — Novelty and Research Contributions

## Executive Summary

GeoNLI is primarily an **engineering integration** project rather than a pure research contribution. Its novelty lies in the practical combination of multiple existing models into a cohesive geospatial reasoning system, along with several engineering innovations for resource-constrained deployment.

---

## Potential Contributions

### 1. Multi-Model Geospatial Reasoning System

| Aspect | Details |
|---|---|
| **Problem** | No single model can handle all geospatial tasks (captioning, VQA, grounding, SAR conversion) |
| **Existing approach** | Separate tools for each task |
| **GeoNLI implementation** | Unified system with task routing |
| **Evidence** | backend/core/model.py — ModelManager routes to different backends |
| **Classification** | Engineering contribution / useful integration |

**Limitations:**
- The routing logic is simple (task_type based), not learned
- No joint training across tasks
- Each model operates independently

---

### 2. GPU Memory Scheduler for Multi-Model Inference

| Aspect | Details |
|---|---|
| **Problem** | Multiple large models (VLM 2B+, SAM3) must share a single 6 GiB GPU |
| **Existing approach** | Load all models on GPU (requires large GPU) or run sequentially (slow) |
| **GeoNLI implementation** | Fair FIFO lock + exclusive CPU↔GPU swapping |
| **Evidence** | backend/services/inference_lock.py:10-112 |
| **Classification** | Engineering contribution |

**Technical Details:**
- `FairReentrantLock`: FIFO reentrant lock prevents starvation [Code: inference_lock.py:10-52]
- `GPUModelScheduler`: Exclusive placement ensures only one model on GPU [Code: inference_lock.py:58-112]
- Models swap CPU↔GPU per task family

**Limitations:**
- Swapping introduces latency (model transfer time)
- Only supports two models (VLM + SAM3)
- No overlapping inference

**Evaluation plan:** Measure latency and memory usage against a baseline that loads all models on GPU simultaneously.

---

### 3. SAM3 Integration for Remote Sensing Grounding

| Aspect | Details |
|---|---|
| **Problem** | Locating objects in satellite imagery from text descriptions |
| **Existing approach** | Manual inspection or fixed-class detectors |
| **GeoNLI implementation** | SAM3 text-guided segmentation → OBB extraction |
| **Evidence** | models/sam3/unified_inference.py:72-123 |
| **Classification** | Useful integration |

**Technical Details:**
- Mask-to-OBB conversion using OpenCV contour detection
- Corner reordering for consistent output format
- NMS for duplicate removal
- Query normalization (singularization, location word removal)

**Limitations:**
- SAM3 not fine-tuned for remote sensing
- CLIP filtering disabled (may reduce precision)
- No confidence calibration

---

### 4. Domain-Specific Fine-Tuning

| Aspect | Details |
|---|---|
| **Problem** | General VLMs perform poorly on remote sensing tasks |
| **GeoNLI implementation** | Fine-tuned Qwen3-VL-8B on VRSBench + LoRA adapter for captioning |
| **Evidence** | VQA-fine-tuning/, models/captioning-model/ |
| **Classification** | Engineering contribution |

**Limitations:**
- Training data not publicly available
- No evaluation metrics in repository
- Captioning adapter training details unknown

---

### 5. SAR-to-Optical Translation

| Aspect | Details |
|---|---|
| **Problem** | SAR imagery is difficult for humans and VLMs to interpret |
| **GeoNLI implementation** | Pix2Pix UNet for SAR→optical conversion |
| **Evidence** | models/SAR2Optical/ |
| **Classification** | Engineering contribution |

**Limitations:**
- Checkpoint not bundled
- Fixed 256×256 output
- No uncertainty estimation

---

## What Is NOT Novel

| Aspect | Reality |
|---|---|
| Model architectures | All models are existing (Qwen3-VL, SAM3, Pix2Pix) |
| Fine-tuning technique | Standard full fine-tuning and LoRA |
| API design | Standard REST API |
| Frontend | Standard React application |

---

## Comparison to Related Systems

| System | Capabilities | Difference from GeoNLI |
|---|---|---|
| GPT-4V | Captioning, VQA | No grounding, no SAR, not self-hosted |
| LLaVA | Captioning, VQA | No grounding, no SAR |
| SAM | Segmentation | No VQA, no captioning |
| GeoNLI | Captioning, VQA, grounding, SAR | All-in-one with GPU scheduler |

---

## Evaluation Plan

### Baseline Comparison

| Metric | How to Measure |
|---|---|
| Caption quality | BLEU, CIDEr, human evaluation |
| VQA accuracy | Accuracy (binary), MAE (numeric), BERT-BLEU (semantic) |
| Grounding quality | IoU, mAP |
| Latency | End-to-end response time |
| Memory | Peak GPU memory usage |

### Ablation Studies

| Component | Ablation | Expected Impact |
|---|---|---|
| LoRA adapter | Remove → base model only | Caption quality decrease |
| Fine-tuned VQA | Remove → shared VLM only | VQA accuracy decrease |
| SAM3 | Remove → no grounding | Grounding unavailable |
| GPU scheduler | Remove → all on GPU | OOM on 6 GiB card |
| Query router | Enable → SAM+VLM path | Numeric accuracy increase |

---

## Questions I Must Be Able to Answer Without Guessing

1. What is the actual VQA accuracy of the fine-tuned model? → Not in repository
2. What is the captioning quality with vs. without LoRA? → Not measured
3. How much does the GPU scheduler improve latency vs. naive approach? → Not measured
4. What is the SAM3 grounding IoU on remote sensing imagery? → Not measured
5. What training data was used for the captioning LoRA? → Not in repository

---

*Next: [Evaluation and Experiments](12_evaluation_and_experiments.md)*
