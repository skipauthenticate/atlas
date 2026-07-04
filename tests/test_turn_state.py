import math
import unittest
from array import array

from atlas_voice.turn_state import RealtimeTurnState


class RealtimeTurnStateTests(unittest.TestCase):
    def test_audio_buffer_commit_and_clear(self) -> None:
        state = RealtimeTurnState(sample_rate=24000, channels=1)

        self.assertEqual(state.append_audio(b"abc"), 3)
        self.assertEqual(state.append_audio(b"de"), 5)
        self.assertEqual(state.buffered_audio_bytes, 5)
        self.assertEqual(state.commit_audio(), b"abcde")
        self.assertEqual(state.buffered_audio_bytes, 0)

        state.append_audio(b"again")
        state.clear_audio()

        self.assertEqual(state.commit_audio(), b"")

    def test_audio_buffer_tracks_browser_media_type(self) -> None:
        state = RealtimeTurnState(sample_rate=24000, channels=1)

        state.append_audio(b"webm", media_type="audio/webm;codecs=opus")
        payload, media_type = state.commit_audio_with_media_type()

        self.assertEqual(payload, b"webm")
        self.assertEqual(media_type, "audio/webm")
        self.assertIsNone(state.audio_media_type)

    def test_session_audio_format_updates_from_event(self) -> None:
        state = RealtimeTurnState(sample_rate=24000, channels=1)

        state.update_audio_format({"input_audio_sample_rate": 16000, "channels": 2})

        self.assertEqual(state.sample_rate, 16000)
        self.assertEqual(state.channels, 2)

        state.update_audio_format({"sample_rate": 8000})

        self.assertEqual(state.sample_rate, 8000)
        self.assertEqual(state.channels, 2)

    def test_realtime_vad_detects_speech_then_end_of_turn(self) -> None:
        state = RealtimeTurnState(sample_rate=16000, channels=1)
        speech = _pcm_tone(16000, 0.2, amplitude=7000)
        silence = _pcm_silence(16000, 0.15)

        speech_result = state.update_realtime_vad(
            speech,
            enabled=True,
            energy_threshold=1000,
            min_speech_ms=100,
            silence_duration_ms=100,
        )
        silence_result = state.update_realtime_vad(
            silence,
            enabled=True,
            energy_threshold=1000,
            min_speech_ms=100,
            silence_duration_ms=100,
        )

        self.assertTrue(speech_result.speech_started)
        self.assertFalse(speech_result.end_of_turn)
        self.assertFalse(silence_result.speech_started)
        self.assertTrue(silence_result.end_of_turn)
        self.assertGreaterEqual(silence_result.silence_ms, 100)

    def test_realtime_vad_ignores_container_audio(self) -> None:
        state = RealtimeTurnState(sample_rate=16000, channels=1)

        result = state.update_realtime_vad(
            b"webm-container",
            enabled=True,
            media_type="audio/webm",
        )

        self.assertFalse(result.analyzed)
        self.assertFalse(result.end_of_turn)

    def test_streaming_transcript_accumulates_and_clears_with_audio(self) -> None:
        state = RealtimeTurnState(sample_rate=24000, channels=1)

        self.assertEqual(state.append_streaming_transcript(" Hello", final=False), "Hello")
        self.assertEqual(state.append_streaming_transcript(" Atlas ", final=True), "Hello Atlas")
        self.assertEqual(state.pending_streaming_transcript, "Hello Atlas")
        self.assertEqual(state.pop_streaming_transcript(), "Hello Atlas")
        self.assertIsNone(state.pop_streaming_transcript())

        state.append_streaming_transcript("discard me", final=False)
        state.clear_audio()

        self.assertIsNone(state.pending_streaming_transcript)


    def test_pending_text_and_response_lifecycle(self) -> None:
        state = RealtimeTurnState(sample_rate=24000, channels=1)

        state.set_pending_text("Hello Atlas")
        self.assertTrue(state.has_pending_text)
        self.assertEqual(state.pop_pending_text(), "Hello Atlas")
        self.assertFalse(state.has_pending_text)
        self.assertIsNone(state.pop_pending_text())

        state.begin_response("resp_1")
        self.assertEqual(state.active_response_id, "resp_1")
        self.assertTrue(state.response_in_progress)
        state.cancel_response()
        self.assertTrue(state.cancel_requested)
        state.complete_response()

        self.assertIsNone(state.active_response_id)
        self.assertFalse(state.response_in_progress)
        self.assertFalse(state.cancel_requested)


def _pcm_tone(sample_rate: int, seconds: float, *, amplitude: int) -> bytes:
    samples = array("h")
    for index in range(int(sample_rate * seconds)):
        value = int(amplitude * math.sin(2 * math.pi * 440 * index / sample_rate))
        samples.append(value)
    return samples.tobytes()


def _pcm_silence(sample_rate: int, seconds: float) -> bytes:
    samples = array("h", [0] * int(sample_rate * seconds))
    return samples.tobytes()


if __name__ == "__main__":
    unittest.main()
