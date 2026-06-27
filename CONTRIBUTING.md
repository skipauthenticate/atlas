# Contributing

Atlas Voice is intended to be local-first and private by default. Keep changes
focused on that operating model unless a feature explicitly opts into networked
or shared use.

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
python -m unittest discover -s tests
```

For worker development with real ASR and diarization:

```bash
pip install -e ".[worker,dev]"
```

## Guidelines

- Do not commit `.env`, data, model weights, cache directories, or transcripts
  that contain private audio content.
- Prefer small, retryable pipeline changes with explicit artifacts.
- Keep localhost-only behavior as the default for the web service.
- Add tests for queue transitions, timestamp merging, export behavior, and any
  processing step that changes stored transcript or summary data.
