# 17 — Technical Glossary and Conceptual Foundations

## Transformer Architecture

| Term | Simple Explanation | Precise Explanation | GeoNLI Usage |
|---|---|---|---|
| **Transformer** | A neural network that processes sequences using attention | Deep learning architecture using self-attention mechanisms without recurrence | Qwen3-VL backbone architecture |
| **Attention** | Mechanism to focus on relevant parts of input | Computes weighted sums of values based on query-key similarity | Used in VLM and SAM3 |
| **Self-Attention** | Input attends to itself | Each position computes attention scores with all other positions | Core of transformer layers |
| **Multi-Head Attention** | Multiple attention computations in parallel | Multiple attention heads with different projection matrices | Qwen3-VL: 32 heads (text), 16 heads (vision) |
| **RoPE** | Position encoding using rotation | Rotary Position Embedding — encodes relative positions via rotation matrices | Qwen3-VL uses M-RoPE with sections [24,20,20] |
| **M-RoPE** | Multi-resolution RoPE | Multi-resolution Rotary Position Embedding for handling variable-length inputs | Configured in VQA checkpoint |

---

## Vision-Language Models

| Term | Simple Explanation | Precise Explanation | GeoNLI Usage |
|---|---|---|---|
| **VLM** | Model that understands images and text | Vision-language model combining visual and linguistic representations | Qwen3-VL-2B/8B |
| **Vision Encoder** | Converts images to numbers | CNN or transformer that processes images into feature embeddings | Qwen3-VL: ViT with 27 layers, hidden_size=1152 |
| **Patch Embedding** | Breaking images into small pieces | Dividing image into fixed-size patches and embedding each as a token | Qwen3-VL: 16×16 pixel patches |
| **Spatial Merge** | Combining nearby patches | Merging adjacent patch tokens to reduce sequence length | Qwen3-VL: merge factor 2 |
| **Image Token** | Text representation of image patches | Special tokens representing image regions in the input sequence | `<\|image_pad\|>` in Qwen3-VL |

---

## Generation and Decoding

| Term | Simple Explanation | Precise Explanation | GeoNLI Usage |
|---|---|---|---|
| **Autoregressive Generation** | Predicting one token at a time | Generating output tokens sequentially, conditioning on previously generated tokens | All text generation in GeoNLI |
| **Tokenization** | Converting text to numbers | Mapping text tokens to integer indices for model input | Qwen tokenizer with vocab_size=151936 |
| **Chat Template** | Format for conversation messages | Structured format with role markers (`<\|im_start\|>`, `<\|im_end\|>`) | Used in captioning.py and vqa.py |
| **Temperature** | Controls randomness in generation | Softmax temperature scaling; lower = more deterministic | Default 0.2 for VLM |
| **Top-P** | Nucleus sampling threshold | Sampling from smallest set of tokens with cumulative probability ≥ p | Default 0.9 |
| **Greedy Decoding** | Always picking the most likely token | Selecting the argmax token at each step | Used when temperature=0 |
| **Beam Search** | Exploring multiple generation paths | Maintaining k highest-probability sequences during generation | Not used (greedy default) |

---

## Visual Question Answering

| Term | Simple Explanation | Precise Explanation | GeoNLI Usage |
|---|---|---|---|
| **VQA** | Answering questions about images | Task of generating natural language answers to questions about visual content | Core capability |
| **Binary VQA** | Yes/no questions | VQA restricted to yes/no answers | Canonicalized to "Yes"/"No" |
| **Numeric VQA** | Questions with number answers | VQA requiring numeric responses (counts, measurements) | Count, area, perimeter, length |
| **Semantic VQA** | Open-ended questions | VQA with free-form text answers | Short answer generation |
| **Grounding** | Locating objects in images | Predicting bounding boxes or masks for referred objects | SAM3 + OBB extraction |

---

## Segmentation and Detection

| Term | Simple Explanation | Precise Explanation | GeoNLI Usage |
|---|---|---|---|
| **Segmentation** | Pixel-level object identification | Classifying each pixel as belonging to an object or background | SAM3 mask generation |
| **Mask** | Binary image of object region | 2D array indicating which pixels belong to an object | SAM3 output masks |
| **Oriented Bounding Box (OBB)** | Rectangle with rotation | 4 corner points defining a rotated rectangle | SAM3 output format |
| **Axis-Aligned Bounding Box (AABB)** | Rectangle without rotation | [x_min, y_min, x_max, y_max] format | NMS approximation |
| **Non-Maximum Suppression (NMS)** | Removing duplicate detections | Suppressing overlapping detections keeping highest score | Applied with IoU threshold 0.6 |
| **IoU** | Overlap between boxes | Intersection over Union — ratio of overlapping area to union area | Used in NMS |
| **Contour** | Boundary of a shape | Curve joining all continuous points along a boundary | OpenCV findContours |
| **Min Area Rect** | Smallest rectangle enclosing a shape | Minimum-area rotated rectangle containing all points | cv2.minAreaRect |

---

## Fine-Tuning and Transfer Learning

| Term | Simple Explanation | Precise Explanation | GeoNLI Usage |
|---|---|---|---|
| **Pre-trained Model** | Model trained on large data before fine-tuning | Neural network with weights learned from large-scale training data | Qwen3-VL-2B/8B, SAM3 |
| **Fine-Tuning** | Adapting pre-trained model to specific task | Further training pre-trained weights on task-specific data | VQA full fine-tune, captioning LoRA |
| **Transfer Learning** | Reusing learned representations | Applying knowledge from one task to another | All models use transfer learning |
| **LoRA** | Efficient fine-tuning with small matrices | Low-Rank Adaptation — adds trainable low-rank matrices to frozen weights | Captioning adapter (r=16, alpha=32) |
| **PEFT** | Parameter-efficient fine-tuning | Family of methods to fine-tune with fewer trainable parameters | LoRA is a PEFT method |
| **Full Fine-Tuning** | Updating all model weights | Training all parameters on task-specific data | VQA checkpoint |
| **Adapter** | Small added module | Additional trainable layers/weights added to frozen model | LoRA adapter in captioning-model/ |
| **Gradient Checkpointing** | Trading compute for memory | Recomputing activations during backward pass instead of storing them | Enabled in VQA training |
| **Mixed Precision** | Using lower-precision numbers | Training with fp16/bf16 to reduce memory and increase speed | bf16 in VQA training |

---

## GPU and Compute

| Term | Simple Explanation | Precise Explanation | GeoNLI Usage |
|---|---|---|---|
| **CUDA** | NVIDIA GPU computing platform | Parallel computing platform for NVIDIA GPUs | Required for GPU inference |
| **VRAM** | GPU memory | Video Random Access Memory — memory on GPU | 6+ GiB recommended |
| **bf16** | Brain float 16 | 16-bit floating point format with 8-bit exponent | Used for VLM on CUDA |
| **fp16** | Half-precision float | 16-bit floating point format with 5-bit exponent | Fallback for non-bf16 GPUs |
| **fp32** | Single-precision float | 32-bit floating point format | Used for CPU inference |
| **Autocast** | Automatic mixed precision | Automatically casting operations to appropriate precision | Used in VLM generation |
| **Model Swapping** | Moving models between devices | Transferring model weights between CPU and GPU | GPU scheduler core mechanism |
| **FIFO Lock** | First-in-first-out lock | Lock where waiters acquire in order of arrival | FairReentrantLock |
| **Reentrant Lock** | Lock that can be acquired multiple times by same thread | Lock tracking acquisition depth | Prevents deadlock in nested calls |

---

## Remote Sensing

| Term | Simple Explanation | Precise Explanation | GeoNLI Usage |
|---|---|---|---|
| **EO** | Earth Observation | Acquiring information about Earth's surface via satellite imagery | Application domain |
| **SAR** | Radar-based imaging | Synthetic Aperture Radar — active microwave imaging | SAR2Optical conversion |
| **Optical** | Visible-light imaging | Traditional photography/satellite imagery in visible spectrum | Standard input |
| **FCC** | False-color composite | Image combining non-visible bands (eIR, R, G) displayed as RGB | NIR, R, G bands |
| **NIR** | Near-infrared | Wavelength range just beyond visible red | Band 0 in FCC |
| **Spatial Resolution** | Ground size per pixel | Distance on ground represented by one pixel | Used for metric conversion |
| **Multispectral** | Multiple wavelength bands | Imaging in several discrete wavelength bands | Not directly supported |

---

## API and Web

| Term | Simple Explanation | Precise Explanation | GeoNLI Usage |
|---|---|---|---|
| **REST API** | Web interface using HTTP methods | Representational State Transfer — stateless HTTP API | `/geoNLI/eval`, `/geoNLI/health` |
| **Endpoint** | URL for API access | Specific URL path handling requests | `/geoNLI/eval` |
| **Multipart** | Multiple data in one request | Content-Type for sending files + form data | Frontend upload format |
| **JSON** | JavaScript Object Notation | Lightweight data interchange format | API request/response format |
| **CORS** | Cross-Origin Resource Sharing | Browser security mechanism for cross-origin requests | Enabled for frontend access |
| **Blueprint** | Flask component organizer | Way to organize related routes | `api_bp` blueprint |
| **ContextVar** | Context-local variable | Python variable bound to execution context | Request ID propagation |

---

## Evaluation Metrics

| Term | Simple Explanation | Precise Explanation | GeoNLI Usage |
|---|---|---|---|
| **Accuracy** | Correct predictions / total | Classification correctness | Binary VQA metric |
| **MAE** | Mean Absolute Error | Mean of absolute differences between predicted and actual | Numeric VQA metric |
| **RMSE** | Root Mean Square Error | Square root of mean squared differences | Numeric VQA metric |
| **BERT-BLEU** | Semantic similarity score | Score based on BERT embeddings and n-gram matching | Semantic VQA metric |
| **mAP** | Mean Average Precision | Area under precision-recall curve | Grounding metric (not used) |

---

*Next: [Documentation Coverage Audit](18_documentation_coverage_audit.md)*
