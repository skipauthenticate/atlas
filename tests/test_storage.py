from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from atlas_voice.config import Settings
from atlas_voice.database import Database
from atlas_voice.storage import FileStorage


def make_test_settings(root: Path) -> Settings:
    return Settings(
        host="127.0.0.1",
        port=8787,
        data_dir=root / "data",
        models_dir=root / "models",
        hf_cache_dir=root / "cache" / "huggingface",
        whisperx_model="large-v3-turbo",
        whisperx_device="cpu",
        whisperx_compute_type="int8",
        pyannote_model="pyannote/speaker-diarization-community-1",
        hf_token=None,
        llm_base_url="http://localhost:8080/v1/chat/completions",
        llm_model="test",
        llm_temperature=0.2,
        llm_max_tokens=100,
        stub_mode=True,
    )


class StorageTests(unittest.TestCase):
    def test_ingest_detects_duplicate_sha256(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = make_test_settings(root)
            settings.ensure_directories()
            db = Database(settings.db_path)
            db.initialize()
            storage = FileStorage(settings, db)

            first_source = root / "first.wav"
            second_source = root / "second.wav"
            first_source.write_bytes(b"same-content")
            second_source.write_bytes(b"same-content")

            first_id = db.create_recording(first_source)
            second_id = db.create_recording(second_source)

            first_result = storage.ingest_source(first_id)
            second_result = storage.ingest_source(second_id)
            second_recording = db.get_recording(second_id)

            self.assertFalse(first_result["duplicate"])
            self.assertTrue(second_result["duplicate"])
            self.assertEqual(second_recording["duplicate_of"], first_id)
            self.assertEqual(second_recording["status"], "duplicate")


if __name__ == "__main__":
    unittest.main()
