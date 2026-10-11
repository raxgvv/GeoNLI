"""Repeatable live SAM3 API benchmark on bundled sample images.

Run after starting the backend:
    python -m backend.tests.benchmark_grounding

The test uses the backend process so it does not load a duplicate SAM3 model.
Each result is a separate request; raw/post-filter counts and confidence values
are returned by the instrumented API and detailed stage times are in geonli.log.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
SAMPLES = [ROOT / "Extra Resourse" / "sample1.png", ROOT / "Extra Resourse" / "sample2.png"]
PROMPTS = ("Find airplanes", "Locate the runway", "Locate the roads", "Find buildings")


def _post(url: str, image_path: Path, query: str) -> tuple[int, dict]:
    boundary = f"----GeoNLI{uuid.uuid4().hex}"
    image_bytes = image_path.read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{image_path.name}\"\r\nContent-Type: image/png\r\n\r\n".encode(),
            image_bytes,
            f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"task_type\"\r\n\r\ngrounding\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"query\"\r\n\r\n{query}\r\n".encode(),
            f"--{boundary}--\r\n".encode(),
        ]
    )
    request = Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urlopen(request, timeout=180) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/geoNLI/eval")
    parser.add_argument("--images", nargs="*", type=Path, default=SAMPLES)
    parser.add_argument("--prompts", nargs="*", default=PROMPTS)
    args = parser.parse_args()

    for image_path in args.images:
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        for prompt in args.prompts:
            started = time.monotonic()
            try:
                status, payload = _post(args.url, image_path, prompt)
                instances = payload.get("filtered_instances", [])
                record = {
                    "image": image_path.name,
                    "prompt": prompt,
                    "status": status,
                    "request_id": payload.get("request_id"),
                    "duration_s": round(time.monotonic() - started, 3),
                    "raw_detections": payload.get("raw_detection_count"),
                    "post_filter_detections": len(payload.get("bboxes", [])),
                    "scores": [round(float(item.get("score", 0)), 4) for item in instances],
                    "bboxes": payload.get("bboxes", []),
                    "error": payload.get("error"),
                }
            except (OSError, URLError, TimeoutError, ValueError) as exc:
                record = {
                    "image": image_path.name,
                    "prompt": prompt,
                    "status": "request_failed",
                    "duration_s": round(time.monotonic() - started, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
