# 18 — Traceability and Coverage Audit

## Requirements-to-Implementation Matrix

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| Image captioning | CaptioningModel + CaptioningBackend | backend/core/model.py:120-137 | Implemented |
| Binary VQA | VQAModel with canonicalization | backend/core/model.py:276-286 | Implemented |
| Numeric VQA | _compute_numeric_value from OBBs | backend/core/model.py:307-344 | Implemented (VLM-only) |
| Semantic VQA | VQAModel direct | backend/core/model.py:170-178 | Implemented |
| Text grounding | SAM3 + OBB extraction | models/sam3/unified_inference.py:72-123 | Implemented |
| FCC conversion | Band-remap algorithm | backend/utils/image_ops.py:34-55 | Implemented |
| SAR conversion | Pix2Pix UNet | models/SAR2Optical/ | Implemented (optional) |
| GPU scheduler | Fair lock + placement | backend/services/inference_lock.py | Implemented |
| Query router | Mistral-7B router | backend/services/query_router.py | Implemented (disabled) |
| Frontend UI | React dashboard | ui/frontend/src/ | Implemented |
| API | Flask endpoints | backend/api/routes.py | Implemented |
| Fine-tuning | VQA + LoRA | VQA-fine-tuning/, models/ | Implemented |

---

## Frontend Components Matrix

| Component | File | Documented | Lines |
|---|---|---|---|
| GeoNLIDashboard | GeoNLIDashboard.jsx | Yes | 435 |
| Stage | components/Stage.jsx | Yes | 241 |
| CommandBar | components/CommandBar.jsx | Yes | 192 |
| SceneInspector | components/SceneInspector.jsx | Yes | 240 |
| SessionStrip | components/SessionStrip.jsx | Yes | 148 |
| CaptionCard | components/CaptionCard.jsx | Yes | 121 |
| LogSheet | components/LogSheet.jsx | Yes | 128 |
| DebugPanel | components/DebugPanel.jsx | Yes | 36 |
| StatusBadge | components/StatusBadge.jsx | Yes | 38 |
| HideUiToggle | components/HideUiToggle.jsx | Yes | 16 |
| ConfirmDialog | components/ConfirmDialog.jsx | Yes | 74 |
| API Client | services/api.js | Yes | 124 |
| App | App.jsx | Yes | 8 |
| main | main.jsx | Yes | 14 |

---

## API Endpoints Matrix

| Endpoint | Method | Documented | Lines |
|---|---|---|---|
| /geoNLI/eval | POST | Yes | 46-339 |
| /geoNLI/health | GET | Yes | 341-343 |

---

## Backend Functions Matrix

| Function | File | Documented | Lines |
|---|---|---|---|
| create_app | backend/app.py | Yes | 50-96 |
| _configure_logging | backend/app.py | Yes | 13-48 |
| analyze | backend/api/routes.py | Yes | 46-339 |
| health | backend/api/routes.py | Yes | 341-343 |
| ModelManager | backend/core/model.py | Yes | 197-459 |
| AdvancedVLMAdapter | backend/core/model.py | Yes | 45-109 |
| CaptioningModel | backend/core/model.py | Yes | 120-137 |
| VQAModel | backend/core/model.py | Yes | 155-194 |
| FairReentrantLock | backend/services/inference_lock.py | Yes | 10-52 |
| GPUModelScheduler | backend/services/inference_lock.py | Yes | 58-112 |
| QueryRouter | backend/services/query_router.py | Yes | 24-400 |
| Sar2OpticalConverter | backend/services/sar_converter.py | Yes | 19-75 |
| run_sam3_grounding | backend/services/sam3_runner.py | Yes | 88-150 |
| load_image_from_url | backend/utils/image_ops.py | Yes | 11-18 |
| fcc_to_rgb_array | backend/utils/image_ops.py | Yes | 34-55 |

---

## Model Wrappers Matrix

| Model | Wrapper | Documented | Lines |
|---|---|---|---|
| Qwen3-VL-2B | CaptioningBackend | Yes | models/captioning.py:32-281 |
| Qwen3-VL-8B VQA | VQABackend | Yes | models/vqa.py:68-157 |
| SAM3 | Sam3GroundingService | Yes | models/sam3/runtime_service.py:45-293 |
| Pix2Pix | Sar2OpticalConverter | Yes | backend/services/sar_converter.py:19-75 |
| Mistral-7B | QueryRouter | Yes | backend/services/query_router.py:24-400 |

---

## Training Scripts Matrix

| Script | Purpose | Documented | Lines |
|---|---|---|---|
| train.py | VQA full fine-tuning | Yes | 285 |
| model.py | Model loading | Yes | 41 |
| dataset.py | Dataset + collator | Yes | 133 |
| preprocess.py | Data preprocessing | Yes | 177 |
| eval.py | Evaluation | Yes | 277 |
| utils.py | Checkpoint utils | Yes | 72 |
| inference.py | Standalone inference | Yes | 136 |

---

## Tests Matrix

| Test | File | Documented | Status |
|---|---|---|---|
| API contract tests | backend/tests/test_eval_api.py | Yes | Implemented |
| SAM3 post-processing | backend/tests/test_sam3_postprocessing.py | Yes | Implemented |
| Grounding benchmark | backend/tests/benchmark_grounding.py | Yes | Implemented |

---

## Documentation Chapters Matrix

| Chapter | File | Status |
|---|---|---|
| 01 Project Overview | 01_project_overview.md | Complete |
| 02 System Architecture | 02_system_architecture.md | Complete |
| 03 Codebase Walkthrough | 03_complete_codebase_walkthrough.md | Complete |
| 04 Line-by-Line Reference | 04_line_by_line_code_reference.md | Complete |
| 05 Models and Selection | 05_models_and_model_selection.md | Complete |
| 06 Training and Fine-Tuning | 06_training_and_finetuning.md | Complete |
| 07 Data and Preprocessing | 07_data_and_preprocessing.md | Complete |
| 08 Inference Pipelines | 08_inference_pipelines.md | Complete |
| 09 Frontend Walkthrough | 09_frontend_walkthrough.md | Complete |
| 10 Backend and API Reference | 10_backend_and_api_reference.md | Complete |
| 11 Novelty and Contributions | 11_novelty_and_research_contributions.md | Complete |
| 12 Evaluation and Experiments | 12_evaluation_and_experiments.md | Complete |
| 13 Setup and Deployment | 13_setup_and_deployment.md | Complete |
| 14 Dependencies | 14_dependencies_and_technology_choices.md | Complete |
| 15 Limitations and Future Work | 15_limitations_and_future_work.md | Complete |
| 16 Presentation and Viva Guide | 16_presentation_and_viva_guide.md | Complete |
| 17 Technical Glossary | 17_technical_glossary.md | Complete |
| 18 Coverage Audit | 18_documentation_coverage_audit.md | Complete |

---

## Files Not Documented Line-by-Line

| File | Reason |
|---|---|
| models/sam3/sam3/ (SAM3 core) | Third-party vendored code |
| models/sam3/eval/ (SAM3 eval) | Third-party, not used in runtime |
| models/sam3/train/ (SAM3 train) | Third-party, not used in runtime |
| models/sam3/scripts/ | Third-party utility scripts |
| models/SAR2Optical/src/ (layers, loss, etc.) | Supporting code, not directly used |
| VQA-fine-tuning/tests/ | Test files, not core logic |
| ui/frontend/dist/ | Generated build output |
| ui/frontend/api/geoNLI/eval.js | Vercel proxy, documented in 09 |
| *.backup, *.before_offload | Backup files, not active |
| Extra Resourse/ | Duplicate scratch content |

---

## Coverage Summary

| Category | Total Files | Documented | Coverage |
|---|---|---|---|
| Backend source | 12 | 12 | 100% |
| Model source | 5 | 5 | 100% |
| Frontend source | 14 | 14 | 100% |
| Training source | 7 | 7 | 100% |
| Tests | 3 | 3 | 100% |
| Launcher scripts | 3 | 3 | 100% |
| **Total** | **44** | **44** | **100%** |

---

## Verification Status

| Check | Status |
|---|---|
| All expected files exist | Verified |
| Markdown links resolve | Verified |
| Mermaid diagrams present | Verified |
| File paths accurate | Verified |
| Line number references | Verified (may drift) |
| Model names match code | Verified |
| Training claims match evidence | Verified |
| No credentials in docs | Verified |
| No binary artifacts in docs | Verified |
| Source code unmodified | Verified |

---

## Remaining Uncertainties

| Question | Status |
|---|---|
| Actual VQA accuracy | Not in repository |
| Captioning quality with/without LoRA | Not measured |
| Training data for captioning LoRA | Not in repository |
| Inference latency | Not measured |
| Peak GPU memory | Not measured |
| SAM3 grounding IoU | Not measured |
| FCC formula derivation | Not documented |

---

*End of documentation package.*
