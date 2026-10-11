# GeoNLI — Complete Project Documentation

**Geospatial Natural Language Interface**
ISRO Inter-IIT Tech Meet 14.0

---

## Reading Order

| # | Document | Purpose |
|---|----------|---------|
| 01 | [Project Overview](01_project_overview.md) | What GeoNLI is, why it exists, what it does |
| 02 | [System Architecture](02_system_architecture.md) | High-level architecture, diagrams, component interactions |
| 03 | [Complete Codebase Walkthrough](03_complete_codebase_walkthrough.md) | File-by-file explanation of every source file |
| 04 | [Line-by-Line Code Reference](04_line_by_line_code_reference.md) | Detailed line-by-line explanations of all active source |
| 05 | [Models and Model Selection](05_models_and_model_selection.md) | Every model identified, why selected, how used |
| 06 | [Training and Fine-Tuning](06_training_and_finetuning.md) | Complete training investigation and evidence |
| 07 | [Data and Preprocessing](07_data_and_preprocessing.md) | Data sources, pipelines, transformations |
| 08 | [Inference Pipelines](08_inference_pipelines.md) | Every user-facing capability end-to-end |
| 09 | [Frontend Walkthrough](09_frontend_walkthrough.md) | React UI architecture and components |
| 10 | [Backend and API Reference](10_backend_and_api_reference.md) | Every endpoint, schema, error handling |
| 11 | [Novelty and Research Contributions](11_novelty_and_research_contributions.md) | What is genuinely novel, evidence-based |
| 12 | [Evaluation and Experiments](12_evaluation_and_experiments.md) | Tests, benchmarks, metrics, gaps |
| 13 | [Setup and Deployment](13_setup_and_deployment.md) | Installation, startup, troubleshooting |
| 14 | [Dependencies and Technology Choices](14_dependencies_and_technology_choices.md) | Every dependency explained |
| 15 | [Limitations and Future Work](15_limitations_and_future_work.md) | Verified limitations, risks, roadmap |
| 16 | [Presentation and Viva Guide](16_presentation_and_viva_guide.md) | 50+ Q&A, demo scripts, defense preparation |
| 17 | [Technical Glossary](17_technical_glossary.md) | All relevant terms explained |
| 18 | [Documentation Coverage Audit](18_documentation_coverage_audit.md) | Traceability matrix, coverage checklist |

---

## Quick Facts

- **Project Root:** `/home/rav/Inter-IIT`
- **Backend:** Flask on port 8000
- **Frontend:** React 19 + Vite 7 + Tailwind CSS 4
- **Primary VLM:** Qwen/Qwen3-VL-2B-Instruct (shared backbone)
- **Optional VQA:** Fine-tuned Qwen3-VL-8B (local checkpoint, ~8.7B params)
- **Optional Captioning:** LoRA adapter on Qwen3-VL-8B (r=16, alpha=32)
- **Grounding:** SAM3 (Segment Anything Model 3)
- **SAR Conversion:** Pix2Pix UNet
- **Query Router:** Mistral-7B-Instruct (disabled by default)
- **GPU Strategy:** Fair FIFO lock + exclusive CPU↔GPU swapping

---

## Key Capabilities

| Capability | Status | Model |
|---|---|---|
| RGB image captioning | Implemented | Qwen3-VL-2B (+ optional LoRA) |
| FCC → RGB conversion | Implemented | Band-remap algorithm |
| SAR → optical conversion | Implemented (optional) | Pix2Pix UNet |
| Binary VQA (yes/no) | Implemented | Qwen3-VL-2B or fine-tuned 8B |
| Numeric VQA (count/area/perimeter/length) | Implemented | SAM3 + spatial resolution |
| Semantic VQA | Implemented | Qwen3-VL-2B or fine-tuned 8B |
| Text-guided grounding | Implemented | SAM3 |
| GPU memory scheduler | Implemented | Fair FIFO lock + placement |

---

## Evidence Legend

Throughout this documentation:

- **[Code]** — Verified directly from source code
- **[Config]** — Verified from configuration or checkpoint metadata
- **[Test]** — Verified by running code
- **[Doc]** — Supported by official model documentation
- **[Inferred]** — Inferred from available evidence
- **[Unknown]** — Not verified from available evidence

---

*Generated from the current repository state. All claims are grounded in actual source code, configuration files, and observable artifacts.*
