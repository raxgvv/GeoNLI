# 16 — Presentation and Technical Viva Guide

## A. 30-Second Explanation

"GeoNLI is a geospatial AI system that lets you ask questions about satellite images in plain English. You upload an image and ask things like 'How many buildings are there?' or 'Find the airplanes' — it combines a vision-language model for understanding, a segmentation model for locating objects, and specialized converters for different image types, all orchestrated to run on a single consumer GPU."

---

## B. One-Minute Explanation

"GeoNLI — Geospatial Natural Language Interface — is a system I built for ISRO's Inter-IIT Tech Meet that makes satellite imagery accessible through natural language. Users upload an image and can ask for captions, yes/no questions, numeric measurements like area and perimeter, or locate objects with bounding boxes. The backend orchestrates multiple AI models: a Qwen3-VL vision-language model handles captioning and VQA, SAM3 performs text-guided segmentation for grounding, and a Pix2Pix network converts SAR imagery to optical. The key engineering challenge was fitting all these models on a single 6-gigabyte GPU, which I solved with a custom fair-scheduler that swaps models between CPU and GPU exclusively. The system also includes fine-tuned checkpoints — a LoRA adapter for captioning and a fully fine-tuned Qwen3-VL-8B for VQA on a remote sensing dataset."

---

## C. Five-Minute Technical Walkthrough

1. **Problem** (30s): Satellite imagery analysis requires specialized tools
2. **Solution Overview** (30s): Natural language interface with multi-model backend
3. **Architecture** (60s): React frontend → Flask API → Model manager → VLM/SAM3/Pix2Pix
4. **Key Innovation** (60s): GPU scheduler with fair FIFO lock and exclusive placement
5. **Training** (60s): Full fine-tune of Qwen3-VL-8B on VRSBench + LoRA for captioning
6. **Results** (30s): Three core capabilities working end-to-end

---

## D. 10-Minute Detailed Presentation Outline

1. Introduction and motivation (1 min)
2. System architecture walkthrough (2 min)
3. Deep dive: GPU scheduler (2 min)
4. Deep dive: Model pipelines (2 min)
5. Training methodology (1.5 min)
6. Demo (1 min)
7. Q&A (0.5 min)

---

## E. Slide-by-Slide Structure

| Slide | Title | Key Visual |
|---|---|---|
| 1 | GeoNLI: Geospatial Natural Language Interface | System logo |
| 2 | Problem: Satellite imagery is hard to analyze | GIS screenshot |
| 3 | Solution: Ask questions in natural language | UI screenshot |
| 4 | Architecture | System diagram |
| 5 | GPU Scheduler | Lock + swap diagram |
| 6 | Inference Pipeline | Request flow |
| 7 | Models | Model cards |
| 8 | Training | Loss curves (if available) |
| 9 | Demo | Live demo |
| 10 | Q&A | — |

---

## F. Recommended Architecture Diagrams

1. **High-level system** — Frontend → Backend → Models
2. **GPU scheduler** — Lock + placement flow
3. **Request lifecycle** — Sequence diagram
4. **Training pipeline** — Data → Model → Checkpoint

---

## G. Recommended Code Snippets

1. **GPU Lock** — FairReentrantLock acquire/release
2. **Model routing** — ModelManager.process_all_queries
3. **SAM3 OBB** — mask_to_oriented_bbox
4. **Adapter loading** — PeftModel.from_pretrained

---

## H. Complete Demonstration Script

```
1. Start services: python start_geonli.py
2. Open browser: http://localhost:5173
3. Upload sample image (drag & drop)
4. Captioning: Click "Generate caption" → show result
5. Binary VQA: Select VQA → Binary → "Is there a river?" → show "Yes"
6. Numeric VQA: Select VQA → Numeric → "How many buildings?" → show count
7. Grounding: Select Grounding → "Find water bodies" → show boxes on image
8. SAR: Upload SAR image → show conversion + analysis
```

---

## I. Questions a Reviewer Is Likely to Ask

### Model Selection

**Q: Why Qwen3-VL-2B instead of 8B?**
A: The 2B model fits alongside SAM3 on a 6 GiB GPU with the swapping scheduler. The 8B model is available as an optional fine-tuned checkpoint for VQA.

**Q: Why SAM3 instead of SAM2 or YOLO?**
A: SAM3 provides open-vocabulary text-guided segmentation with oriented bounding boxes, which is ideal for remote sensing where object categories are diverse.

**Q: Why Pix2Pix for SAR conversion?**
A: Pix2Pix is lightweight, fast, and well-suited for paired image-to-image translation. More complex models (diffusion) would be too slow for real-time use.

### Fine-Tuning

**Q: What dataset was used for VQA fine-tuning?**
A: VRSBench, a remote sensing VQA dataset. The training code is in VQA-fine-tuning/.

**Q: How many epochs?**
A: 5 epochs with cosine decay. Checkpoint at step 10724, epoch 4.

**Q: What is the LoRA configuration?**
A: Rank 16, alpha 32, dropout 0.05, targeting all attention and MLP projections.

### Architecture

**Q: How does the GPU scheduler work?**
A: A fair FIFO lock ensures only one model holds GPU memory at a time. Models swap CPU↔GPU exclusively per task.

**Q: What happens if SAM3 fails?**
A: The error is caught and returned as a 503 with a user-friendly message. The request ID allows server-side log correlation.

**Q: Is the system concurrent?**
A: Requests are serialized by the GPU lock. This is a deliberate trade-off for memory-constrained deployment.

### Data

**Q: What image formats are supported?**
A: PNG and JPEG via multipart upload. The backend converts all inputs to RGB.

**Q: What is the spatial resolution used for?**
A: It converts pixel measurements to real-world units (meters, square meters) for numeric VQA.

---

## J. Detailed Model-Selection Questions

**Q: Why not use a single large model for everything?**
A: No single model handles captioning, VQA, segmentation, and image translation well. A multi-model approach allows each task to use the best-suited architecture.

**Q: Why is the query router disabled?**
A: It requires loading a second 7B model, which competes with Qwen3-VL and SAM3 for GPU memory. The router is implemented but disabled to keep the core flows working on 6 GiB.

**Q: What is the evidence for model selection?**
A: The choices are grounded in the code: Qwen3-VL for vision-language, SAM3 for segmentation, Pix2Pix for image translation. Alternatives were likely evaluated during development, but only the selected models are in the repository.

---

## K. Fine-Tuning and Training Questions

**Q: Was the captioning LoRA trained on the same data as VQA?**
A: Unknown. No training scripts for the captioning adapter are in the repository.

**Q: What hardware was used for training?**
A: The prod config is named "2L40S" suggesting 2× NVIDIA L40S GPUs (48 GB each).

**Q: How was the train/val split created?**
A: The split logic is not in the repository. The VRSBench dataset likely has predefined splits.

---

## L. Questions About Datasets and Preprocessing

**Q: What is VRSBench?**
A: A remote sensing VQA dataset. Not publicly available in this repository.

**Q: How are numeric answers formatted?**
A: To one decimal place (e.g., "42.0"). The format is set in preprocess.py:93.

**Q: How are binary answers canonicalized?**
A: "Yes"/"No" with validation that the output contains exactly one of these words.

---

## M. Questions About Architecture and Algorithms

**Q: How does SAM3 convert masks to bounding boxes?**
A: OpenCV contour detection → minimum area rectangle → 4 corner points → clockwise reorder.

**Q: How is the FCC blue band synthesized?**
A: Linear combination: 13.9053 + 0.873393×green − 0.138254×red − 0.100616×NIR.

**Q: How are numeric values computed from bounding boxes?**
A: Area = pixel_area × resolution², Perimeter = pixel_perimeter × resolution, Length = max_side × resolution.

---

## N. Questions About Novelty and Research Contribution

**Q: What is novel about GeoNLI?**
A: The engineering integration of multiple models with a GPU scheduler for memory-constrained deployment. The individual models are existing, but the system design is a practical contribution.

**Q: How does GeoNLI compare to GPT-4V?**
A: GPT-4V is a general-purpose API. GeoNLI is self-hosted, specialized for geospatial tasks, and includes grounding and SAR conversion.

---

## O. Questions About Performance, Evaluation, and Limitations

**Q: What is the inference latency?**
A: Not measured in this repository. The 90-second frontend timeout suggests inference can take tens of seconds.

**Q: What is the GPU memory usage?**
A: Not measured. The scheduler is designed for 6 GiB cards.

**Q: What are the main limitations?**
A: Missing checkpoints, disabled router, no evaluation metrics, single-user design.

---

## P. Questions About GPU Memory and Deployment

**Q: Can GeoNLI run without a GPU?**
A: Yes, but impractically slow. The code supports CPU execution.

**Q: What happens if a checkpoint is missing?**
A: The system degrades gracefully: missing SAM3 → 503 for grounding, missing Pix2Pix → 503 for SAR, missing VQA → falls back to shared VLM.

**Q: How do I deploy on a cloud GPU?**
A: Same as local. Install CUDA, PyTorch, dependencies, and run start_geonli.py.

---

## Q. Difficult Questions That Expose Weaknesses

**Q: The SAM+VLM routed numeric path is dead code. Why include it?**
A: It was implemented for evaluation purposes but disabled by default to avoid loading a second large model. It demonstrates the intended architecture.

**Q: There are no evaluation metrics. How do we know it works?**
A: The contract tests verify the code paths. Real-model evaluation was not performed during repository publication.

**Q: The captioning LoRA targets 8B but the default VLM is 2B. Isn't this a bug?**
A: It's a configuration mismatch. To use the LoRA, set ADV_VLM_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct. The code validates this match and raises an error if mismatched.

**Q: The frontend has a simulation mode. Does this mislead users?**
A: Simulation mode is clearly labeled in the debug panel and is intended for development. It allows frontend work without a running backend.

---

## R. Questions I Must Be Able to Answer Without Guessing

1. **What is the VQA accuracy of the fine-tuned model?** → Run eval.py to get metrics.
2. **What training data was used for captioning LoRA?** → Investigate team records.
3. **What is the actual inference latency?** → Benchmark with real hardware.
4. **What is the peak GPU memory usage?** → Profile with nvidia-smi.
5. **Does the SAM+VLM routed path improve numeric accuracy?** → Enable router and A/B test.
6. **What is the FCC conversion formula derived from?** → It appears to be an empirical fit; original source unknown.
7. **Why is the blue band formula what it is?** → Not documented in code; likely empirical.

---

## S. Technically Correct Answers

| Question | Correct Answer |
|---|---|
| What models are used? | Qwen3-VL-2B (shared), Qwen3-VL-8B (fine-tuned VQA), Qwen3-VL-8B + LoRA (captioning), SAM3 (grounding), Pix2Pix (SAR) |
| Is the router enabled? | No, it's disabled by default |
| What GPU is needed? | 6+ GiB VRAM recommended |
| What is the upload limit? | 16 MB |
| Is there authentication? | No |
| What Python version? | 3.10 |

---

*Next: [Technical Glossary](17_technical_glossary.md)*
