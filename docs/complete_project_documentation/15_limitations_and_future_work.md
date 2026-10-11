# 15 — Limitations, Risks, and Future Work

## Confirmed Limitations

### 1. GPU Memory Constraints

| Aspect | Detail |
|---|---|
| **Evidence** | README states 6 GiB target; scheduler swaps models |
| **Impact** | Only one model on GPU at a time; swapping adds latency |
| **Root cause** | Multiple large models (VLM 2B, SAM3) |
| **Fix** | Use larger GPU or reduce model sizes |
| **Trade-off** | Larger GPU = more cost; smaller models = less capability |

### 2. Missing Checkpoints

| Aspect | Detail |
|---|---|
| **Evidence** | SAM3, Pix2Pix, VQA checkpoints not in repo |
| **Impact** | Grounding and SAR conversion unavailable without manual download |
| **Root cause** | .gitignore excludes model weights |
| **Fix** | Provide download scripts or bundle weights |
| **Trade-off** | Bundling increases repo size |

### 3. Query Router Disabled

| Aspect | Detail |
|---|---|
| **Evidence** | `self.router = None` in ModelManager [Code: backend/core/model.py:225] |
| **Impact** | Numeric VQA uses VLM-only (less accurate for counts) |
| **Root cause** | Would require loading second 7B model |
| **Fix** | Enable router when GPU headroom allows |
| **Trade-off** | Better numeric accuracy vs. more memory |

### 4. No Evaluation Metrics

| Aspect | Detail |
|---|---|
| **Evidence** | No accuracy/latency numbers in repository |
| **Impact** | Cannot compare to baselines or measure improvement |
| **Root cause** | Evaluation not run/published |
| **Fix** | Run eval.py and publish results |
| **Trade-off** | Time to run evaluation |

### 5. SAM3 Filtering Disabled

| Aspect | Detail |
|---|---|
| **Evidence** | CLIP disabled, refiner disabled, VLM agent disabled [Code: runtime_service.py:57,105] |
| **Impact** | May produce more false positives in grounding |
| **Root cause** | Additional models would increase memory/complexity |
| **Fix** | Enable CLIP or VLM filtering |
| **Trade-off** | Better precision vs. more memory |

### 6. Fixed SAR Output Resolution

| Aspect | Detail |
|---|---|
| **Evidence** | Pix2Pix outputs 256×256 [Code: sar_converter.py:41] |
| **Impact** | Large images lose resolution after conversion |
| **Root cause** | Pix2Pix architecture limitation |
| **Fix** | Use tiling or higher-resolution model |
| **Trade-off** | Better quality vs. more compute |

### 7. No Authentication

| Aspect | Detail |
|---|---|
| **Evidence** | No auth middleware in Flask app |
| **Impact** | Anyone with network access can use the API |
| **Root cause** | Not implemented |
| **Fix** | Add API key or OAuth |
| **Trade-off** | Security vs. usability |

### 8. Single-User Design

| Aspect | Detail |
|---|---|
| **Evidence** | Global locks serialize all requests |
| **Impact** | Concurrent users experience queueing |
| **Root cause** | GPU memory limits prevent parallel inference |
| **Fix** | Multi-GPU or request batching |
| **Trade-off** | Throughput vs. cost |

---

## Potential Risks

| Risk | Likelihood | Impact |
|---|---|---|
| SAM3 checkpoint download fails | Medium | Grounding unavailable |
| HF rate limiting | Low | Slow first load |
| GPU OOM on small cards | Medium | Inference fails |
| Model output format changes | Low | Parsing errors |
| Dependency version conflicts | Medium | Installation issues |

---

## Prioritized Roadmap

### Critical Fixes

1. **Add checkpoint download scripts** — automate SAM3/Pix2Pix setup
2. **Add evaluation metrics** — run and publish VQA accuracy
3. **Add health check for models** — verify checkpoints exist at startup

### High-Value Improvements

1. **Enable query router on high-memory GPUs** — improve numeric VQA
2. **Add streaming responses** — show partial captions
3. **Add batch evaluation endpoint** — speed up judge harness
4. **Add request caching** — avoid redundant inference

### Optional Enhancements

1. **Support more image types** — multispectral, hyperspectral
2. **Add model fine-tuning UI** — train without code
3. **Add A/B testing framework** — compare model versions
4. **Add monitoring dashboard** — track latency, memory, usage

---

## Verification of Fixes

| Fix | How to Verify |
|---|---|
| Checkpoint download | Run grounding without manual download |
| Evaluation metrics | eval.py produces accuracy numbers |
| Health check | /geoNLI/health reports model status |
| Router enabled | Numeric VQA uses SAM+VLM path |
| Streaming | Client receives partial results |
| Batch endpoint | Multiple queries in one request |

---

*Next: [Presentation and Viva Guide](16_presentation_and_viva_guide.md)*
