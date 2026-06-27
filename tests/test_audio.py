from pathlib import Path
import unittest

from atlas_voice.audio import build_ffmpeg_command


class AudioTests(unittest.TestCase):
    def test_build_ffmpeg_command_normalizes_to_16k_mono_wav(self) -> None:
        command = build_ffmpeg_command(Path("input.m4a"), Path("output.wav"))

        self.assertEqual(command[0], "ffmpeg")
        self.assertIn("-ac", command)
        self.assertEqual(command[command.index("-ac") + 1], "1")
        self.assertIn("-ar", command)
        self.assertEqual(command[command.index("-ar") + 1], "16000")
        self.assertEqual(command[-2:], ["wav", "output.wav"])


if __name__ == "__main__":
    unittest.main()
