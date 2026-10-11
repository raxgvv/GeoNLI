"""Small HTTP contract tests that do not load heavyweight models."""
import io
import unittest
from unittest.mock import patch

from PIL import Image

from backend.app import create_app
from models.sam3.runtime_service import Sam3GroundingService


def png_bytes():
    image = Image.new("RGB", (8, 8), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class EvalApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = create_app().test_client()

    def setUp(self):
        self.image = png_bytes()

    def post_task(self, task="captioning", query="", vqa_type="semantic", image_type="RGB"):
        return self.client.post(
            "/geoNLI/eval",
            data={
                "image": (io.BytesIO(self.image), "sample.png"),
                "task_type": task,
                "query": query,
                "vqa_type": vqa_type,
                "image_type": image_type,
                "spatial_resolution_m": "1.0",
            },
            content_type="multipart/form-data",
        )

    def test_health_and_caption_response_schema(self):
        self.assertEqual(self.client.get("/geoNLI/health").status_code, 200)
        fake_model = type("FakeModel", (), {
            "process_all_queries": lambda *_args, **_kwargs: {
                "caption": {"answer": "Test caption"}
            }
        })()
        with patch("backend.api.routes.get_model", return_value=fake_model):
            response = self.post_task()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["answer"], "Test caption")
        self.assertTrue(response.json["timestamp"])

    def test_vqa_dispatch_and_inference_error(self):
        fake_model = type("FakeModel", (), {
            "process_all_queries": lambda *_args, **_kwargs: {
                "attr_binary": {"answer": "Yes"}
            }
        })()
        with patch("backend.api.routes.get_model", return_value=fake_model):
            response = self.post_task("vqa", "Are airplanes visible?", "binary")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["answer"], "Yes")

        failed_model = type("FailedModel", (), {
            "process_all_queries": lambda *_args, **_kwargs: {
                "caption": {"error": "CUDA out of memory"}
            }
        })()
        with patch("backend.api.routes.get_model", return_value=failed_model):
            response = self.post_task()
        self.assertEqual(response.status_code, 500)
        self.assertIn("GPU memory", response.json["error"])
        self.assertNotIn("details", response.json)

    def test_missing_query_is_rejected(self):
        response = self.post_task("grounding")
        self.assertEqual(response.status_code, 400)
        self.assertIn("query is required", response.json["error"])

    def test_invalid_image_returns_actionable_error(self):
        response = self.client.post(
            "/geoNLI/eval",
            data={
                "image": (io.BytesIO(b"not an image"), "broken.png"),
                "task_type": "captioning",
                "image_type": "RGB",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Could not decode the uploaded image", response.json["error"])
        self.assertNotIn("BytesIO", response.json["error"])

    def test_grounding_prompts_reduce_to_detector_labels(self):
        self.assertEqual(Sam3GroundingService._grounding_phrase("where are airplanes?"), "airplane")
        self.assertEqual(Sam3GroundingService._grounding_phrase("Locate the roads"), "road")
        self.assertEqual(Sam3GroundingService._grounding_phrase("Find water bodies"), "water body")

    def test_grounding_failure_is_not_reported_as_a_valid_empty_result(self):
        with patch("backend.api.routes.run_sam3_grounding", side_effect=RuntimeError("SAM3 forward failed")):
            response = self.post_task("grounding", "Locate the runway")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json["success"])
        self.assertIn("request_id", response.json)
        self.assertNotIn("No groundings found", response.json["error"])

    def test_generation_timeout_is_returned_as_a_gateway_timeout(self):
        failed_model = type("SlowModel", (), {
            "process_all_queries": lambda *_args, **_kwargs: {
                "caption": {"error": "generation time limit reached", "error_type": "TimeoutError"}
            }
        })()
        with patch("backend.api.routes.get_model", return_value=failed_model):
            response = self.post_task()
        self.assertEqual(response.status_code, 504)
        self.assertIn("time limit", response.json["error"])

    def test_sar_conversion_failure_is_actionable(self):
        with patch("backend.api.routes.convert_sar_to_rgb", side_effect=FileNotFoundError("Set SAR2OPTICAL_CHECKPOINT to the trained weights")):
            response = self.post_task(image_type="SAR")
        self.assertEqual(response.status_code, 503)
        self.assertIn("SAR2OPTICAL_CHECKPOINT", response.json["error"])


if __name__ == "__main__":
    unittest.main()
