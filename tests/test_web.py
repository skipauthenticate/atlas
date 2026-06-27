from pathlib import Path
from tempfile import TemporaryDirectory
import importlib
import os
import unittest

from fastapi.testclient import TestClient


class WebTests(unittest.TestCase):
    def test_html_pages_render_with_current_starlette_signature(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["ATLAS_VOICE_DATA_DIR"] = str(root / "data")
            os.environ["ATLAS_VOICE_MODELS_DIR"] = str(root / "models")
            os.environ["ATLAS_VOICE_HF_CACHE"] = str(root / "cache" / "huggingface")
            os.environ["ATLAS_VOICE_STUB_MODE"] = "true"

            import atlas_voice.web.app as web_app

            web_app = importlib.reload(web_app)
            web_app.settings.ensure_directories()
            web_app.db.initialize()
            recording_id = web_app.db.create_recording(root / "audio.wav", title="Test Audio")
            web_app.db.update_recording(recording_id, status="done")
            web_app.db.replace_segments(
                recording_id,
                [
                    {
                        "start": 0,
                        "end": 1,
                        "speaker": "SPEAKER_00",
                        "text": "Atlas Voice transcript",
                    }
                ],
            )
            web_app.db.save_summary(recording_id, "Atlas Voice summary", model="test")

            client = TestClient(web_app.app)

            dashboard = client.get("/")
            detail = client.get(f"/recordings/{recording_id}")
            search = client.get("/search", params={"q": "Atlas"})

            self.assertEqual(dashboard.status_code, 200)
            self.assertIn("Test Audio", dashboard.text)
            self.assertEqual(detail.status_code, 200)
            self.assertIn("SPEAKER_00", detail.text)
            self.assertEqual(search.status_code, 200)
            self.assertIn("[Atlas]", search.text)
            self.assertNotIn("&lt;mark&gt;", search.text)


if __name__ == "__main__":
    unittest.main()
