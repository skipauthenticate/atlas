from __future__ import annotations

import html
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlparse

from .config import Settings


MAX_WEB_QUERY_CHARS = 400
MAX_WEB_SNIPPET_CHARS = 360
MAX_WEB_TITLE_CHARS = 180
MAX_WEB_CACHE_ENTRIES = 128
NEGATIVE_CACHE_TTL_SECONDS = 3.0

_EXPLICIT_WEB_PATTERNS = (
    re.compile(r"\b(?:search|browse)\s+(?:the\s+)?(?:web|internet|online)\b", re.I),
    re.compile(r"\b(?:look|check)\s+(?:it\s+)?up\b", re.I),
    re.compile(r"\b(?:google|web\s+search)\b", re.I),
    re.compile(r"\bfind\s+(?:this|that|it|information|info|sources?)\s+online\b", re.I),
)
_ALWAYS_FRESH_TERMS = re.compile(
    r"\b(?:breaking|forecast|headlines?|latest|news|online|weather)\b",
    re.I,
)
_FRESHNESS_CUES = re.compile(r"\b(?:current(?:ly)?|now|today|tonight|this\s+week)\b", re.I)
_FRESHNESS_SUBJECTS = re.compile(
    r"\b(?:ceo|exchange\s+rate|forecast|law|leader|president|price|release|"
    r"schedule|score|software|standings|stock|version|weather)\b",
    re.I,
)
_QUERY_PREFIXES = (
    re.compile(r"^\s*(?:please\s+)?(?:search|browse)\s+(?:the\s+)?(?:web|internet)\s+for\s+", re.I),
    re.compile(r"^\s*(?:please\s+)?(?:look|check)\s+(?:it\s+)?up\s*[:,-]?\s*", re.I),
    re.compile(r"^\s*(?:please\s+)?web\s+search\s*[:,-]?\s*", re.I),
)
_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str
    source: str = ""
    published_at: str | None = None

    def to_public_dict(self) -> dict[str, str | None]:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "published_at": self.published_at,
        }


@dataclass(frozen=True)
class WebSearchResponse:
    query: str
    provider: str
    results: tuple[WebSearchResult, ...]
    latency_ms: int
    cached: bool = False
    error: str | None = None


_cache_lock = threading.Lock()
_cache: OrderedDict[str, tuple[float, WebSearchResponse]] = OrderedDict()
_http_client_lock = threading.Lock()
_http_client: Any | None = None


def web_search_requested(query: str) -> bool:
    """Return true when a normal voice turn needs current public information."""
    text = _clean_text(query)
    if not text:
        return False
    if any(pattern.search(text) for pattern in _EXPLICIT_WEB_PATTERNS):
        return True
    if _ALWAYS_FRESH_TERMS.search(text):
        return True
    return bool(_FRESHNESS_CUES.search(text) and _FRESHNESS_SUBJECTS.search(text))


def web_search_query(query: str) -> str:
    text = _clean_text(query)
    for pattern in _QUERY_PREFIXES:
        text = pattern.sub("", text, count=1)
    return text[:MAX_WEB_QUERY_CHARS].strip() or _clean_text(query)[:MAX_WEB_QUERY_CHARS]


def search_web(query: str, settings: Settings) -> WebSearchResponse:
    cleaned_query = web_search_query(query)
    provider = settings.web_search_provider.strip().lower()
    started = time.perf_counter()
    if not cleaned_query:
        return WebSearchResponse("", provider, (), 0, error="empty_query")
    if not settings.web_search_enabled:
        return WebSearchResponse(cleaned_query, provider, (), 0, error="disabled")
    if provider not in {"brave", "searxng"}:
        return WebSearchResponse(cleaned_query, provider, (), 0, error="unsupported_provider")

    cache_key = _cache_key(cleaned_query, settings)
    cached = _get_cached(cache_key)
    if cached is not None:
        return replace(cached, cached=True, latency_ms=_elapsed_ms(started))

    try:
        if provider == "searxng":
            results = _search_searxng(cleaned_query, settings)
        else:
            results = _search_brave(cleaned_query, settings)
        response = WebSearchResponse(
            query=cleaned_query,
            provider=provider,
            results=tuple(results[: settings.web_search_max_results]),
            latency_ms=_elapsed_ms(started),
        )
    except Exception as exc:  # noqa: BLE001 - web search is a best-effort voice input.
        response = WebSearchResponse(
            query=cleaned_query,
            provider=provider,
            results=(),
            latency_ms=_elapsed_ms(started),
            error=_safe_error(exc),
        )

    if response.results:
        _put_cached(cache_key, response, settings.web_search_cache_ttl_seconds)
    elif response.error:
        _put_cached(
            cache_key,
            response,
            min(settings.web_search_cache_ttl_seconds, NEGATIVE_CACHE_TTL_SECONDS),
        )
    return response


def _search_searxng(query: str, settings: Settings) -> list[WebSearchResult]:
    client = _get_http_client()
    response = client.get(
        settings.web_search_base_url,
        params={
            "q": query,
            "format": "json",
            "categories": "general",
            "safesearch": "1",
        },
        headers={"Accept": "application/json"},
        timeout=settings.web_search_timeout,
    )
    response.raise_for_status()
    payload = response.json()
    raw_results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(raw_results, list):
        return []
    results: list[WebSearchResult] = []
    seen_urls: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        result = _result_from_payload(
            title=item.get("title"),
            url=item.get("url"),
            snippet=item.get("content"),
            source=item.get("engine"),
            published_at=item.get("publishedDate") or item.get("pubdate"),
        )
        if result is None or result.url in seen_urls:
            continue
        seen_urls.add(result.url)
        results.append(result)
        if len(results) >= settings.web_search_max_results:
            break
    return results


def _search_brave(query: str, settings: Settings) -> list[WebSearchResult]:
    if not settings.web_search_api_key:
        raise RuntimeError("ATLAS_VOICE_WEB_SEARCH_API_KEY is required for Brave Search")
    client = _get_http_client()
    response = client.get(
        settings.web_search_base_url,
        params={
            "q": query,
            "count": settings.web_search_max_results,
            "safesearch": "moderate",
            "text_decorations": "false",
            "result_filter": "web",
        },
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": settings.web_search_api_key,
        },
        timeout=settings.web_search_timeout,
    )
    response.raise_for_status()
    payload = response.json()
    web = payload.get("web") if isinstance(payload, dict) else None
    raw_results = web.get("results") if isinstance(web, dict) else None
    if not isinstance(raw_results, list):
        return []
    results: list[WebSearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        result = _result_from_payload(
            title=item.get("title"),
            url=item.get("url"),
            snippet=item.get("description"),
            source=item.get("profile", {}).get("long_name")
            if isinstance(item.get("profile"), dict)
            else None,
            published_at=item.get("page_age") or item.get("age"),
        )
        if result is not None:
            results.append(result)
    return results


def _result_from_payload(
    *,
    title: Any,
    url: Any,
    snippet: Any,
    source: Any,
    published_at: Any,
) -> WebSearchResult | None:
    clean_url = _clean_url(url)
    if not clean_url:
        return None
    hostname = (urlparse(clean_url).hostname or "").removeprefix("www.")
    clean_title = _clean_text(title)[:MAX_WEB_TITLE_CHARS] or hostname
    clean_snippet = _clean_text(snippet)[:MAX_WEB_SNIPPET_CHARS]
    if not clean_title and not clean_snippet:
        return None
    return WebSearchResult(
        title=clean_title,
        url=clean_url,
        snippet=clean_snippet,
        source=_clean_text(source)[:80] or hostname,
        published_at=_clean_text(published_at)[:80] or None,
    )


def _clean_url(value: Any) -> str:
    text = str(value or "").strip()
    try:
        parsed = urlparse(text)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    if parsed.username or parsed.password:
        return ""
    return text[:2000]


def _clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    return _WHITESPACE.sub(" ", _HTML_TAG.sub(" ", text)).strip()


def _cache_key(query: str, settings: Settings) -> str:
    return "\x1f".join(
        (
            settings.web_search_provider.strip().lower(),
            settings.web_search_base_url.strip(),
            query.casefold(),
            str(settings.web_search_max_results),
        )
    )


def _get_http_client() -> Any:
    import httpx

    global _http_client
    with _http_client_lock:
        if _http_client is None or _http_client.is_closed:
            _http_client = httpx.Client(
                follow_redirects=False,
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            )
        return _http_client


def close_web_search_client() -> None:
    global _http_client
    with _http_client_lock:
        client = _http_client
        _http_client = None
    if client is not None:
        client.close()


def _get_cached(key: str) -> WebSearchResponse | None:
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(key)
        if cached is None:
            return None
        expires_at, response = cached
        if expires_at <= now:
            _cache.pop(key, None)
            return None
        _cache.move_to_end(key)
        return response


def _put_cached(key: str, response: WebSearchResponse, ttl_seconds: float) -> None:
    if ttl_seconds <= 0:
        return
    with _cache_lock:
        _cache[key] = (time.monotonic() + ttl_seconds, response)
        _cache.move_to_end(key)
        while len(_cache) > MAX_WEB_CACHE_ENTRIES:
            _cache.popitem(last=False)


def _safe_error(exc: Exception) -> str:
    name = type(exc).__name__
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return f"{name}: search provider returned HTTP {status_code}"
    lowered_name = name.casefold()
    if "timeout" in lowered_name:
        return f"{name}: search provider timed out"
    if "json" in lowered_name or "decode" in lowered_name:
        return f"{name}: search provider returned malformed data"
    if isinstance(exc, RuntimeError):
        return f"{name}: search provider configuration error"
    return f"{name}: search provider request failed"


def _elapsed_ms(started: float) -> int:
    return max(round((time.perf_counter() - started) * 1000), 0)
