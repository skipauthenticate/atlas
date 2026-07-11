from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY_ROOT / "scripts" / "run-local-service.sh"
UNIT_TEMPLATE = REPOSITORY_ROOT / "deploy" / "systemd" / "atlas-voice-llm.service.in"


class LocalLlmServiceTests(unittest.TestCase):
    def _fixture(self, directory: str) -> tuple[Path, Path, Path]:
        root = Path(directory)
        scripts = root / "scripts"
        config = root / "config"
        scripts.mkdir()
        config.mkdir()

        launcher = scripts / LAUNCHER.name
        shutil.copy2(LAUNCHER, launcher)
        preset = config / "voice-models.example.ini"
        preset.write_text("[qwen3.5-9b]\nmodel = torch.gguf\nload-on-startup = true\n")

        capture = root / "argv.bin"
        fake_server = root / "llama-server"
        fake_server.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'printf \'%s\\0\' "$PWD" "$@" >"$CAPTURE_PATH"\n'
        )
        fake_server.chmod(0o755)
        return launcher, fake_server, capture

    @staticmethod
    def _environment() -> dict[str, str]:
        env = os.environ.copy()
        for name in (
            "ATLAS_LLAMA_SERVER",
            "ATLAS_LLAMA_MODELS_PRESET",
            "ATLAS_LLAMA_HOST",
            "ATLAS_LLAMA_PORT",
        ):
            env.pop(name, None)
        return env

    def test_llm_mode_execs_one_model_offline_router_from_repository_root(self) -> None:
        with TemporaryDirectory() as tmp:
            launcher, fake_server, capture = self._fixture(tmp)
            env = self._environment()
            env.update(
                ATLAS_LLAMA_SERVER=str(fake_server),
                ATLAS_LLAMA_HOST="127.0.0.9",
                ATLAS_LLAMA_PORT="18080",
                CAPTURE_PATH=str(capture),
                LLAMA_API_KEY="must-not-appear-in-argv",
            )

            result = subprocess.run(
                [str(launcher), "llm"],
                cwd="/",
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            values = capture.read_bytes().rstrip(b"\0").decode().split("\0")
            self.assertEqual(
                values,
                [
                    tmp,
                    "--models-preset",
                    "config/voice-models.example.ini",
                    "--models-max",
                    "1",
                    "--host",
                    "127.0.0.9",
                    "--port",
                    "18080",
                    "--no-webui",
                    "--offline",
                    "--metrics",
                ],
            )
            self.assertNotIn("must-not-appear-in-argv", " ".join(values))

    def test_llm_mode_requires_an_absolute_executable_server_path(self) -> None:
        with TemporaryDirectory() as tmp:
            launcher, _, _ = self._fixture(tmp)
            env = self._environment()

            missing = subprocess.run(
                [str(launcher), "llm"],
                cwd="/",
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(missing.returncode, 1)
            self.assertIn("ATLAS_LLAMA_SERVER must be set", missing.stderr)

            env["ATLAS_LLAMA_SERVER"] = "relative/llama-server"
            relative = subprocess.run(
                [str(launcher), "llm"],
                cwd="/",
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(relative.returncode, 1)
            self.assertIn("must be an absolute path", relative.stderr)

    def test_llm_mode_rejects_an_invalid_port_before_exec(self) -> None:
        with TemporaryDirectory() as tmp:
            launcher, fake_server, capture = self._fixture(tmp)
            env = self._environment()
            env.update(
                ATLAS_LLAMA_SERVER=str(fake_server),
                ATLAS_LLAMA_PORT="8080 --api-key leaked",
                CAPTURE_PATH=str(capture),
            )

            result = subprocess.run(
                [str(launcher), "llm"],
                cwd="/",
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("must be an integer", result.stderr)
            self.assertFalse(capture.exists())

    def test_launcher_and_unit_template_are_static_safe(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        launcher = LAUNCHER.read_text()
        unit = UNIT_TEMPLATE.read_text()
        preset = (REPOSITORY_ROOT / "config" / "voice-models.example.ini").read_text()

        self.assertIn("--models-max 1", launcher)
        self.assertIn("--no-webui", launcher)
        self.assertIn("--offline", launcher)
        self.assertIn("--metrics", launcher)
        self.assertNotIn('"--api-key"', launcher)
        self.assertIn("WorkingDirectory=@ATLAS_VOICE_ROOT@", unit)
        self.assertIn(
            "ExecStart=@ATLAS_VOICE_ROOT@/scripts/run-local-service.sh llm",
            unit,
        )
        self.assertNotIn("--api-key", unit)
        self.assertNotIn("version=1", preset.replace(" ", ""))
        self.assertNotIn("[default]", preset)


if __name__ == "__main__":
    unittest.main()
