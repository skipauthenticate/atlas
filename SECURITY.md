# Security Policy

Atlas Voice is designed for local-first processing of private recordings. Please
report vulnerabilities privately when possible and avoid sharing private audio,
transcripts, tokens, or environment files in public issues.

## Reporting a Vulnerability

Use GitHub private vulnerability reporting if it is enabled for this repository.
If it is not enabled, open a public issue titled `Security contact request`
without exploit details, credentials, logs, recordings, or transcripts. A
maintainer can then arrange a private reporting channel.

Include this information when it is safe to share privately:

- Affected version or commit.
- Deployment mode: Docker Compose, local source install, or systemd user units.
- Impacted component: web UI, worker, storage, model integration, or scripts.
- Reproduction steps with synthetic or redacted data.
- Any suggested mitigation.

## Supported Versions

This project is early-stage. Security fixes are handled on the default branch
and released from there.

## Privacy-Sensitive Data

Do not attach real recordings, transcripts, summaries, `.env` files, model
tokens, API keys, or local database files to public issues or pull requests.
