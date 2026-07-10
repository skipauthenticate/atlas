from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from atlas_voice.audio import build_ffmpeg_command, normalize_audio


class AudioTests(unittest.TestCase):
    def test_build_ffmpeg_command_normalizes_to_16k_mono_wav(self) -> None:
        command = build_ffmpeg_command(Path("input.m4a"), Path("output.wav"))

        self.assertEqual(command[0], "ffmpeg")
        self.assertIn("-ac", command)
        self.assertEqual(command[command.index("-ac") + 1], "1")
        self.assertIn("-ar", command)
        self.assertEqual(command[command.index("-ar") + 1], "16000")
        self.assertIn("-nostdin", command)
        self.assertEqual(command[command.index("-c:a") + 1], "pcm_s16le")
        self.assertEqual(command[command.index("-sample_fmt") + 1], "s16")
        self.assertEqual(command[-2:], ["wav", "output.wav"])

    def test_cleanup_modes_add_only_the_requested_filter_chain(self) -> None:
        plain = build_ffmpeg_command(Path("input.wav"), Path("output.wav"))
        adaptive = build_ffmpeg_command(
            Path("input.wav"),
            Path("output.wav"),
            cleanup_mode="adaptive",
        )
        enhanced = build_ffmpeg_command(
            Path("input.wav"),
            Path("output.wav"),
            cleanup_mode="enhanced",
        )

        self.assertNotIn("-af", plain)
        self.assertIn("dynaudnorm", adaptive[adaptive.index("-af") + 1])
        self.assertIn("afftdn", enhanced[enhanced.index("-af") + 1])
        with self.assertRaisesRegex(ValueError, "Unknown audio cleanup mode"):
            build_ffmpeg_command(
                Path("input.wav"),
                Path("output.wav"),
                cleanup_mode="aggressive",
            )

    def test_normalize_audio_atomically_replaces_destination(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.wav"
            output = root / "output.wav"
            source.write_bytes(b"source")
            output.write_bytes(b"old")

            def fake_run(command, **kwargs):
                Path(command[-1]).write_bytes(b"normalized")
                self.assertEqual(kwargs["timeout"], 12)
                self.assertTrue(kwargs["capture_output"])
                self.assertTrue(kwargs["text"])

            with mock.patch("subprocess.run", side_effect=fake_run) as run_mock:
                normalize_audio(source, output, timeout_seconds=12)

            self.assertEqual(output.read_bytes(), b"normalized")
            self.assertNotEqual(Path(run_mock.call_args.args[0][-1]), output)
            self.assertEqual(list(root.glob(".output.wav.*.tmp")), [])

    def test_cleanup_failure_retries_without_filters(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.wav"
            output = root / "output.wav"
            source.write_bytes(b"source")
            calls = []

            def fake_run(command, **_kwargs):
                calls.append(command)
                temporary_path = Path(command[-1])
                if len(calls) == 1:
                    temporary_path.write_bytes(b"partial")
                    raise subprocess.CalledProcessError(
                        1,
                        command,
                        stderr="No such filter: afftdn",
                    )
                temporary_path.write_bytes(b"plain")

            with mock.patch("subprocess.run", side_effect=fake_run):
                normalize_audio(source, output, cleanup_mode="enhanced")

            self.assertEqual(len(calls), 2)
            self.assertIn("-af", calls[0])
            self.assertNotIn("-af", calls[1])
            self.assertEqual(output.read_bytes(), b"plain")
            self.assertEqual(list(root.glob(".output.wav.*.tmp")), [])

    def test_normalization_failure_preserves_existing_destination_and_cleans_temp(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.wav"
            output = root / "output.wav"
            source.write_bytes(b"source")
            output.write_bytes(b"old")
            error = subprocess.CalledProcessError(
                1,
                ["ffmpeg"],
                stderr="invalid audio stream",
            )

            with mock.patch("subprocess.run", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, "invalid audio stream"):
                    normalize_audio(source, output)

            self.assertEqual(output.read_bytes(), b"old")
            self.assertEqual(list(root.glob(".output.wav.*.tmp")), [])

    def test_normalization_timeout_preserves_existing_destination(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.wav"
            output = root / "output.wav"
            source.write_bytes(b"source")
            output.write_bytes(b"old")

            with mock.patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(["ffmpeg"], 3),
            ):
                with self.assertRaisesRegex(RuntimeError, "timed out after 3 seconds"):
                    normalize_audio(source, output, timeout_seconds=3)

            self.assertEqual(output.read_bytes(), b"old")
            self.assertEqual(list(root.glob(".output.wav.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
