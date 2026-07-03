import unittest

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

    def test_session_audio_format_updates_from_event(self) -> None:
        state = RealtimeTurnState(sample_rate=24000, channels=1)

        state.update_audio_format({"input_audio_sample_rate": 16000, "channels": 2})

        self.assertEqual(state.sample_rate, 16000)
        self.assertEqual(state.channels, 2)

        state.update_audio_format({"sample_rate": 8000})

        self.assertEqual(state.sample_rate, 8000)
        self.assertEqual(state.channels, 2)

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


if __name__ == "__main__":
    unittest.main()
