"""Routing utilities powered by a lightweight Mistral-7B policy."""
from __future__ import annotations

import ast
import logging
import math
import threading
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    route: str = "VLM_ONLY"
    numeric_type: str = "NONE"


class QueryRouter:
    """LLM-based router that mirrors the provided prompt logic."""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: str = "auto",
        *,
        temperature: float = 0.1,
        max_new_tokens: int = 80,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        logger.info("Loading router model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self._lock = threading.Lock()
        logger.info("Router model loaded successfully!")

    # ------------------------------------------------------------------
    # Prompt helpers (lifted from the provided specification)
    # ------------------------------------------------------------------
    def _binary_prompt(self, query: str) -> str:
        return f"""You are a routing assistant for binary remote sensing questions.
Your job is to decide whether the question should be answered by one of two pipelines: VLM_ONLY or SAM_PLUS_VLM.

You must return only one of these options.

Routing rules:

Choose VLM_ONLY when:

The answer depends on global visual understanding

Overall texture, land cover, color, tone, general appearance

Scene-level reasoning (urban vs rural, farmland vs forest)

High-level judgment that does not require precise object locations

No need to isolate individual objects or measure distances

Choose SAM_PLUS_VLM when:

Answer requires detecting specific objects (court, ship, vehicle, helipad, tanks, etc.)

Validating object presence inside a region

Checking adjacency or proximity (“near”, “close to”, “inside”, “along”, “next to”)

Checking positional relations between two objects

Confirming presence of objects near water bodies, roads, ports, airports, compounds

Segmentation or bounding steps clearly improves the answer

If unsure, think:

“Would object segmentation or spatial filtering make this more reliable?”

If yes → SAM_PLUS_VLM
If no → VLM_ONLY

Do not output explanations. Only the pipeline name.

Few-shot examples:

Example 1
Question: Is the terrain mostly brown?
Output:
VLM_ONLY

Example 2
Question: Is there a basketball court in the scene?
Output:
SAM_PLUS_VLM

Example 3
Question: Is this an urban area?
Output:
VLM_ONLY

Example 4
Question: Is there a vehicle near the river?
Output:
SAM_PLUS_VLM

Example 5
Question: Are there dense forests on the left half of the image?
Output:
VLM_ONLY

Example 6
Question: Is a helipad located on top of any building?
Output:
SAM_PLUS_VLM

Example 7
Question: Is any vessel docked at the coastline?
Output:
SAM_PLUS_VLM

Example 8
Question: Are most rooftops red?
Output:
VLM_ONLY

Example 9
Question: Is there a transmission tower close to the highway?
Output:
SAM_PLUS_VLM

Final output rule:

When I provide a binary question, output only: VLM_ONLY or SAM_PLUS_VLM

Nothing else.

Question: {query}
Output:"""

    def _numeric_prompt(self, query: str) -> str:
        return f"""You are a routing assistant for numeric remote sensing questions.
Your job is to decide:

1. Which processing pipeline to use:

   * VLM_ONLY
   * SAM_PLUS_VLM

2. If SAM_PLUS_VLM is used, which numeric task type it is:

   * COUNT
   * LENGTH
   * AREA
   * PERIMETER

You must output a single Python dictionary with this exact structure:

{{ "route": "...", "numeric_type": "..." }}

Allowed values:

* "route" must be either "VLM_ONLY" or "SAM_PLUS_VLM".
* "numeric_type" must be one of:

  * "NONE"      (only when route = "VLM_ONLY")
  * "COUNT"
  * "LENGTH"
  * "AREA"
  * "PERIMETER"

No other keys are allowed.
Do not output explanations or any extra text.
Return only the Python dictionary.

---

### Routing rules

Choose route = "VLM_ONLY" and numeric_type = "NONE" when:

* The answer can be derived from overall scene-level reasoning.
* The question involves large, easily identifiable features, for example runways, rivers, bridges, canals.
* Counting does not require precise object localization, region filtering or adjacency checks.
* The question is generic and does not target a specific object category or region.
  Example: “How many unique objects are present in the image?”

Choose route = "SAM_PLUS_VLM" when the question is clearly about one of these four numeric types:

1. **COUNT**

   * Counting specific objects (cars, planes, ships, tanks, buildings, greenhouses, etc.), especially:

     * inside a region or boundary
     * near a river, road, coastline, runway, or compound
   * Then set:
     route = "SAM_PLUS_VLM" and numeric_type = "COUNT".

2. **LENGTH**

   * Length of a linear structure:

     * runway length
     * bridge span
     * diameter of storage tank
     * road segment length
     * canal section length
   * Then set:
     route = "SAM_PLUS_VLM" and numeric_type = "LENGTH".

3. **AREA**

   * Area of a region:

     * farmland patch
     * swimming pool
     * lake or reservoir
     * apron or courtyard
   * Then set:
     route = "SAM_PLUS_VLM" and numeric_type = "AREA".

4. **PERIMETER**

   * Perimeter or boundary length:

     * fenced compound
     * forest patch boundary
     * outline of a storage tank
   * Then set:
     route = "SAM_PLUS_VLM" and numeric_type = "PERIMETER".

If the query does not fit exactly into COUNT, LENGTH, AREA or PERIMETER, then route to VLM_ONLY with numeric_type = "NONE".

---

### Few-shot examples

**Example 1**
Question: How many runways are visible in the scene?
Reasoning: Large obvious features, global view.
Output:

{{ "route": "VLM_ONLY", "numeric_type": "NONE" }}

**Example 2**
Question: How many airplanes are on the runway apron?
Reasoning: Counting specific objects in a region.
Output:

{{ "route": "SAM_PLUS_VLM", "numeric_type": "COUNT" }}

**Example 3**
Question: How many rivers flow through the area?
Reasoning: Global structural count.
Output:

{{ "route": "VLM_ONLY", "numeric_type": "NONE" }}

**Example 4**
Question: How many ships are docked at the port?
Reasoning: Counting specific objects in a spatially constrained area.
Output:

{{ "route": "SAM_PLUS_VLM", "numeric_type": "COUNT" }}

**Example 5**
Question: How many greenhouses exist inside the farmland boundary?
Reasoning: Region-filtered counting.
Output:

{{ "route": "SAM_PLUS_VLM", "numeric_type": "COUNT" }}

**Example 6**
Question: How many distinct bridges cross the main river?
Reasoning: Few large obvious structures, global view.
Output:

{{ "route": "VLM_ONLY", "numeric_type": "NONE" }}

**Example 7**
Question: How many vehicles are parked along the perimeter road?
Reasoning: Counting specific objects along a region.
Output:

{{ "route": "SAM_PLUS_VLM", "numeric_type": "COUNT" }}

**Example 8**
Question: How many unique objects are present in the image?
Reasoning: Vague, not tied to specific object or region.
Output:

{{ "route": "VLM_ONLY", "numeric_type": "NONE" }}

**Example 9**
Question: What is the area of the swimming pool region?
Reasoning: Area of a single object/region.
Output:

{{ "route": "SAM_PLUS_VLM", "numeric_type": "AREA" }}

**Example 10**
Question: What is the length of the runway?
Reasoning: Length of a linear structure.
Output:

{{ "route": "SAM_PLUS_VLM", "numeric_type": "LENGTH" }}

**Example 11**
Question: What is the perimeter of the fenced industrial compound?
Reasoning: Perimeter of a region.
Output:

{{ "route": "SAM_PLUS_VLM", "numeric_type": "PERIMETER" }}

**Example 12**
Question: How many large storage tanks can be seen at the facility?
Reasoning: If they are few, large, and obvious in global view, VLM-only is acceptable.
Output:

{{ "route": "VLM_ONLY", "numeric_type": "NONE" }}

---

Question: {query}
Output only the Python dictionary:"""

    # ------------------------------------------------------------------
    # Core routing
    # ------------------------------------------------------------------
    def route(self, query: str, task_type: str) -> RouteDecision:
        task = (task_type or "semantic").lower()
        if task == "semantic":
            return RouteDecision()
        if task not in {"binary", "numeric"}:
            logger.warning("Unknown task type '%s'; defaulting to VLM_ONLY", task_type)
            return RouteDecision()

        prompt = self._binary_prompt(query) if task == "binary" else self._numeric_prompt(query)
        answer = self._run_llm(prompt)

        if task == "binary":
            line = (answer.splitlines()[0] if answer else "").strip().upper()
            if "SAM_PLUS_VLM" in line:
                return RouteDecision(route="SAM_PLUS_VLM")
            if "VLM_ONLY" in line:
                return RouteDecision(route="VLM_ONLY")
            logger.warning("Unclear binary routing output '%s'; defaulting to VLM_ONLY", line)
            return RouteDecision()

        decision_dict = self._parse_numeric_output(answer)
        route = decision_dict.get("route", "VLM_ONLY")
        numeric_type = decision_dict.get("numeric_type", "NONE")
        route = route if route in {"VLM_ONLY", "SAM_PLUS_VLM"} else "VLM_ONLY"
        if route == "VLM_ONLY":
            numeric_type = "NONE"
        elif numeric_type not in {"COUNT", "LENGTH", "AREA", "PERIMETER"}:
            numeric_type = "COUNT"
        return RouteDecision(route=route, numeric_type=numeric_type)

    def _run_llm(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        do_sample = self.temperature > 0
        with torch.no_grad():
            with self._lock:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=max(self.temperature, 0.0),
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

        generated_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def _parse_numeric_output(self, answer: str) -> dict:
        clean = answer.replace("```python", "").replace("```json", "").replace("```", "").strip()
        try:
            parsed = ast.literal_eval(clean)
            if isinstance(parsed, dict):
                return parsed
            logger.warning("Numeric router output is not a dict: %s", parsed)
        except Exception as exc:
            logger.warning("Failed to parse numeric routing output '%s': %s", clean, exc)
        return {"route": "VLM_ONLY", "numeric_type": "NONE"}


# ----------------------------------------------------------------------
# Deterministic geometry helpers for numeric outputs
# ----------------------------------------------------------------------

def get_largest_side(corners: List[List[float]], scale: float = 1.0) -> float:
    """Return the largest side length for a 4-point polygon."""
    if len(corners) != 4:
        raise ValueError("Expected exactly 4 corners")

    side_lengths = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        distance = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        side_lengths.append(distance)

    return max(side_lengths) * scale


def get_perimeter(corners: List[List[float]], scale: float = 1.0) -> float:
    """Return the perimeter for a 4-point polygon."""
    if len(corners) != 4:
        raise ValueError("Expected exactly 4 corners")

    perimeter = 0.0
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        distance = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        perimeter += distance

    return perimeter * scale


def get_area(num_pixels: float, scale: float = 1) -> float:
    """Compute area from the instance pixel count and scale."""
    return float(num_pixels) * scale
