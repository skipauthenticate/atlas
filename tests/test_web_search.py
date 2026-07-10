from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

from atlas_voice.config import Settings
from atlas_voice.web_search import search_web, web_search_query, web_search_requested


class WebSearchTests(unittest.TestCase):
    def test_routes_explicit_and_time_sensitive_queries_only(self) -> None:
        self.assertTrue(web_search_requested("Search the web for Jetson release notes"))
        self.assertTrue(web_search_requested("What's the latest JetPack release?"))
        self.assertTrue(web_search_requested("What is the weather today?"))
        self.assertFalse(web_search_requested("Help me plan the current project"))
        self.assertFalse(web_search_requested("Tell me a short joke"))
        self.assertEqual(web_search_query("Please search the web for JetPack 7.2"), "JetPack 7.2")

    def test_disabled_search_returns_status_without_network(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = _settings(Path(tmp), web_search_enabled=False)
            with patch("atlas_voice.web_search._get_http_client") as client:
                result = search_web("latest JetPack", settings)

        self.assertEqual(result.error, "disabled")
        self.assertEqual(result.results, ())
        client.assert_not_called()

    def test_searxng_results_are_sanitized_bounded_and_cached(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = _settings(
                Path(tmp),
                web_search_enabled=True,
                web_search_max_results=2,
                web_search_cache_ttl_seconds=60,
            )
            response = _response(
                {
                    "results": [
                        {
                            "title": "<b>NVIDIA</b> Jetson",
                            "url": "https://www.nvidia.com/jetson",
                            "content": "<em>Fast</em> edge AI",
                            "engine": "duckduckgo",
                        },
                        {
                            "title": "Duplicate",
                            "url": "https://www.nvidia.com/jetson",
                            "content": "duplicate",
                        },
                        {
                            "title": "Unsafe",
                            "url": "file:///etc/passwd",
                            "content": "ignore",
                        },
                        {
                            "title": "Docs",
                            "url": "https://docs.nvidia.com/jetson",
                            "content": "Official docs",
                            "publishedDate": "2026-06-01",
                        },
                    ]
                }
            )
            with patch("atlas_voice.web_search._cache", OrderedDict()):
                with patch("atlas_voice.web_search._get_http_client") as client:
                    client.return_value.get.return_value = response
                    first = search_web("Jetson unique-cache-query", settings)
                    second = search_web("Jetson unique-cache-query", settings)

        self.assertEqual([item.title for item in first.results], ["NVIDIA Jetson", "Docs"])
        self.assertEqual(first.results[0].snippet, "Fast edge AI")
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(client.return_value.get.call_count, 1)

    def test_provider_failure_is_non_fatal_and_redacted_to_a_short_error(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = _settings(Path(tmp), web_search_enabled=True)
            with patch("atlas_voice.web_search._cache", OrderedDict()):
                with patch("atlas_voice.web_search._get_http_client") as client:
                    client.return_value.get.side_effect = TimeoutError(
                        "provider timed out"
                    )
                    result = search_web("latest timeout-only-query", settings)
                    cached = search_web("latest timeout-only-query", settings)

        self.assertEqual(result.results, ())
        self.assertIn("TimeoutError", result.error or "")
        self.assertLessEqual(len(result.error or ""), 240)
        self.assertTrue(cached.cached)
        self.assertEqual(client.return_value.get.call_count, 1)

    def test_http_failure_never_persists_the_query_url(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = _settings(Path(tmp), web_search_enabled=True)
            request = httpx.Request(
                "GET",
                "https://search.invalid/search?q=private-spoken-query",
            )
            response = httpx.Response(503, request=request)
            failure = httpx.HTTPStatusError(
                "failed for private-spoken-query",
                request=request,
                response=response,
            )
            with patch("atlas_voice.web_search._cache", OrderedDict()):
                with patch("atlas_voice.web_search._get_http_client") as client:
                    client.return_value.get.side_effect = failure
                    result = search_web("private-spoken-query", settings)

        self.assertEqual(
            result.error,
            "HTTPStatusError: search provider returned HTTP 503",
        )
        self.assertNotIn("private-spoken-query", result.error or "")


def _response(payload: dict[str, object]) -> object:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return payload

    return Response()


def _settings(root: Path, **overrides: object) -> Settings:
    values = {
        "host": "127.0.0.1",
        "port": 8787,
        "data_dir": root / "data",
        "models_dir": root / "models",
        "hf_cache_dir": root / "cache" / "huggingface",
        "whisperx_model": "tiny.en",
        "whisperx_device": "cpu",
        "whisperx_compute_type": "int8",
        "pyannote_model": "pyannote/speaker-diarization-community-1",
        "hf_token": None,
        "llm_base_url": "http://127.0.0.1:8080/v1/chat/completions",
        "llm_model": "qwen-local",
        "llm_temperature": 0.2,
        "llm_max_tokens": 1200,
        "stub_mode": True,
    }
    values.update(overrides)
    return Settings(**values)


if __name__ == "__main__":
    unittest.main()
