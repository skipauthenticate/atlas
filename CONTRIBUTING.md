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
scripts/check-open-source-ready.sh
```

For worker development with real ASR and diarization:

```bash
pip install -e ".[worker,dev]"
```

## Guidelines

- Do not commit `.env`, data, model weights, cache directories, or transcripts
  that contain private audio content.
- Keep examples portable. Use relative paths like `models/llm/model.gguf` or
  placeholders like `/path/to/model.gguf`; do not commit paths from a personal
  workstation.
- Do not include real tokens, API keys, service URLs that expose private
  infrastructure, or private recording metadata in fixtures, docs, or tests.
- Prefer small, retryable pipeline changes with explicit artifacts.
- Keep localhost-only behavior as the default for the web service.
- Add tests for queue transitions, timestamp merging, export behavior, and any
  processing step that changes stored transcript or summary data.

## Security Reports

Do not file public issues with exploit details or private recordings. Follow
`SECURITY.md` for vulnerability reporting and use public issues only for
non-sensitive coordination.
