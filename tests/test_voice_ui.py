import re
from pathlib import Path
import unittest


class VoiceUiStaticTests(unittest.TestCase):
    @staticmethod
    def _last_css_block(css: str, selector: str) -> str:
        matches = list(re.finditer(rf"{re.escape(selector)}\s*{{(?P<body>.*?)}}", css, flags=re.S))
        if not matches:
            raise AssertionError(f"Missing CSS selector: {selector}")
        return max((match.group("body") for match in matches), key=len)

    @staticmethod
    def _script_between(script: str, start: str, end: str) -> str:
        start_index = script.find(start)
        if start_index == -1:
            raise AssertionError(f"Missing script marker: {start}")
        end_index = script.find(end, start_index)
        if end_index == -1:
            raise AssertionError(f"Missing script marker: {end}")
        return script[start_index:end_index]

    def test_voice_page_uses_shared_shell_and_focused_workspace_order(self) -> None:
        template = Path("atlas_voice/web/templates/voice.html").read_text()
        base = Path("atlas_voice/web/templates/base.html").read_text()

        self.assertIn("{% block body_class %}voice-page{% endblock %}", template)
        self.assertIn('data-layout="atlas-voice-workbench"', template)
        self.assertIn('<aside class="sidebar"', base)
        self.assertIn('id="app-settings-dialog"', base)
        self.assertNotIn('<nav class="voice-rail"', template)
        ordered = [
            'class="voice-appbar"',
            'class="voice-tool-tabs"',
            'class="voice-session-layout"',
            'class="voice-main"',
            'class="voice-transcript-panel"',
            'class="voice-transport"',
        ]
        positions = [template.index(marker) for marker in ordered]
        self.assertEqual(positions, sorted(positions))

    def test_voice_ui_uses_clean_white_black_theme_without_purple(self) -> None:
        css = Path("atlas_voice/web/static/app.css").read_text().lower()

        for purple in ("#5867d8", "#3743a4", "#9eabff", "#c8d0ff", "#6865b8"):
            self.assertNotIn(purple, css)
        self.assertIn("color-scheme: light", css)
        self.assertIn("--surface: #ffffff", css)
        self.assertIn("--ink: #171717", css)
        self.assertIn("--accent: #171717", css)
        self.assertIn("--voice-accent: #ff5b35", css)
        self.assertIn("background: #ffffff", self._last_css_block(css, ".voice-workbench"))

    def test_voice_page_uses_local_lucide_assets_without_reference_branding(self) -> None:
        template = Path("atlas_voice/web/templates/voice.html").read_text().lower()
        base = Path("atlas_voice/web/templates/base.html").read_text().lower()
        production = template + base + Path("atlas_voice/web/static/voice.js").read_text().lower()

        self.assertNotIn("elevenlabs", production)
        self.assertNotIn("eleven labs", production)
        referenced = set(re.findall(r'/static/icons/([^"?]+\.svg)', template + base))
        self.assertGreaterEqual(len(referenced), 12)
        for name in referenced:
            self.assertTrue(Path("atlas_voice/web/static/icons", name).is_file(), name)

    def test_voice_bubble_is_canvas_driven_by_input_and_output_audio(self) -> None:
        template = Path("atlas_voice/web/templates/voice.html").read_text()
        script = Path("atlas_voice/web/static/voice.js").read_text()

        self.assertIn("data-voice-bubble", template)
        self.assertIn('data-voice-state="idle"', template)
        self.assertIn("class VoiceBubble", script)
        self.assertIn("createMediaElementSource", script)
        self.assertIn("getByteTimeDomainData", script)
        self.assertIn("bubble.setLevel", script)
        for state in ("listening", "thinking", "synthesizing", "speaking"):
            self.assertIn(f"'{state}'", script)

    def test_browser_capture_streams_pcm_for_server_vad(self) -> None:
        script = Path("atlas_voice/web/static/voice.js").read_text()

        self.assertIn("navigator.mediaDevices.getUserMedia", script)
        self.assertIn("AudioWorkletNode", script)
        self.assertIn("createScriptProcessor", script)
        self.assertIn("downsampleToPCM16", script)
        self.assertIn("echoCancellation: true", script)
        self.assertIn("noiseSuppression: true", script)
        self.assertIn("input_audio_buffer.append", script)
        self.assertIn("input_audio_buffer.commit", script)
        self.assertIn("input_audio_buffer.clear", script)
        self.assertIn("media_type: 'audio/pcm'", script)
        self.assertNotIn("new MediaRecorder", script)
        self.assertNotIn("audio/webm", script)

    def test_voice_client_handles_automatic_turn_and_barge_in_events(self) -> None:
        script = Path("atlas_voice/web/static/voice.js").read_text()

        for event_type in (
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
            "input_audio_buffer.committed",
            "conversation.item.input_audio_transcription.done",
            "response.cancelled",
            "response.interrupted",
        ):
            self.assertIn(event_type, script)
        self.assertIn("clearPlayback()", script)

    def test_pause_and_private_have_distinct_exclusive_capture_semantics(self) -> None:
        script = Path("atlas_voice/web/static/voice.js").read_text()
        transition = self._script_between(
            script,
            "const setCaptureMode =",
            "const startCall =",
        )
        capture_controls = self._script_between(
            script,
            "const updateCaptureControls =",
            "const setCaptureMode =",
        )
        prompt_state = self._script_between(
            script,
            "const setPromptAvailability =",
            "refreshPromptAvailability = setPromptAvailability;",
        )
        presentation = self._script_between(
            script,
            "function restoreCapturePresentation()",
            "function revokeCurrentAudioUrl()",
        )
        pause_handler = self._script_between(
            script,
            "controls.get('pause')?.addEventListener",
            "controls.get('private')?.addEventListener",
        )
        private_handler = self._script_between(
            script,
            "controls.get('private')?.addEventListener",
            "controls.get('interrupt')?.addEventListener",
        )

        self.assertIn("await setCaptureMode('paused')", pause_handler)
        self.assertIn("await setCaptureMode('private')", private_handler)
        self.assertIn("const leavingCurrentMode = captureMode === nextMode", transition)
        self.assertIn("resumeMicrophoneAfterCaptureMode", transition)
        self.assertIn("captureMode = 'open'", transition)
        self.assertIn("await startMicrophone()", transition)
        self.assertIn("Resuming microphone", transition)
        self.assertIn("stopPcmCapture({ commit: nextMode === 'paused' })", transition)
        self.assertIn("sendRealtimeEvent({ type: 'input_audio_buffer.clear' })", transition)
        self.assertIn("const isPaused = captureMode === 'paused'", capture_controls)
        self.assertIn("const isPrivate = captureMode === 'private'", capture_controls)
        self.assertIn("togglePressed(controls.get('pause'), isPaused)", capture_controls)
        self.assertIn("togglePressed(controls.get('private'), isPrivate)", capture_controls)
        self.assertIn("'Resume microphone'", capture_controls)
        self.assertIn("'Leave private mode'", capture_controls)
        self.assertIn("const privateMode = captureMode === 'private'", prompt_state)
        self.assertIn("const blocked = !enabled || !callActive || privateMode", prompt_state)
        self.assertIn("input.disabled = blocked", prompt_state)
        self.assertIn("attachButton.disabled = uploading || privateMode", prompt_state)
        self.assertIn("Mic paused; typing available", presentation)
        self.assertIn("Private; all input blocked", presentation)
        self.assertNotIn("let paused =", script)
        self.assertNotIn("let privateMode =", script)

    def test_interrupt_cancels_response_without_dropping_live_microphone(self) -> None:
        script = Path("atlas_voice/web/static/voice.js").read_text()
        handler = self._script_between(
            script,
            "controls.get('interrupt')?.addEventListener",
            "controls.get('play')?.addEventListener",
        )

        self.assertIn("response.cancel", handler)
        self.assertIn("clearPlayback()", handler)
        self.assertNotIn("stopPcmCapture", handler)

    def test_call_controls_have_explicit_lifecycle_and_reset(self) -> None:
        script = Path("atlas_voice/web/static/voice.js").read_text()
        template = Path("atlas_voice/web/templates/voice.html").read_text()
        end_call = self._script_between(script, "const endCall =", "handleSocketClosed")

        self.assertIn('data-transport-action="start-call"', template)
        self.assertIn('data-transport-action="end-call"', template)
        self.assertIn("if (!enabled || callActive) return", script)
        self.assertIn("socket.close(1000, 'call ended')", end_call)
        self.assertIn("stopPcmCapture({ commit: false })", end_call)
        self.assertIn("stopTimer()", end_call)
        self.assertIn("clearPlayback()", end_call)
        self.assertIn("callActive = false", end_call)

    def test_transport_uses_stable_icon_controls_and_volume_slider(self) -> None:
        template = Path("atlas_voice/web/templates/voice.html").read_text()
        css = Path("atlas_voice/web/static/app.css").read_text()
        block = self._last_css_block(css, ".voice-transport")

        for action in ("mic", "pause", "private", "interrupt", "play"):
            self.assertRegex(
                template,
                rf'data-transport-action="{action}"[^>]+aria-label="[^"]+"[^>]+title="[^"]+"',
            )
        self.assertIn('type="range"', template)
        self.assertIn("grid-template-columns", block)
        self.assertIn("min-height: 72px", block)

    def test_voice_layout_has_stable_desktop_and_mobile_dimensions(self) -> None:
        css = Path("atlas_voice/web/static/app.css").read_text()
        workbench = self._last_css_block(css, ".voice-workbench")
        bubble = self._last_css_block(css, ".voice-bubble")
        transcript_panel = self._last_css_block(
            css, '.voice-workbench[data-active-view="transcript"] .voice-transcript-panel'
        )

        self.assertIn("height: 100dvh", workbench)
        self.assertIn("grid-template-rows:", workbench)
        self.assertIn("var(--atlas-control-touch)", workbench)
        self.assertIn("auto", workbench)
        self.assertIn("aspect-ratio: 1", bubble)
        self.assertIn("display: grid", transcript_panel)
        self.assertIn("width: 100%", transcript_panel)
        self.assertIn('.voice-workbench [role="tabpanel"][hidden]', css)
        self.assertIn("width: min(920px, 100%)", css)
        self.assertIn("@media (max-width: 760px)", css)
        self.assertIn('.voice-workbench[data-active-view="transcript"] .voice-main', css)
        self.assertIn("grid-template-columns: 12px minmax(0, 1fr)", css)
        self.assertNotIn('[data-transport-action="private"] {\n    display: none;', css)

    def test_voice_states_and_transcript_are_accessible(self) -> None:
        template = Path("atlas_voice/web/templates/voice.html").read_text()
        script = Path("atlas_voice/web/static/voice.js").read_text()

        self.assertIn('aria-live="polite"', template)
        self.assertIn('aria-label="Realtime call"', template)
        self.assertIn('role="tablist"', template)
        self.assertIn('aria-orientation="horizontal"', template)
        self.assertIn('role="tab"', template)
        self.assertIn('id="voice-live-tab"', template)
        self.assertIn('id="voice-transcript-tab"', template)
        self.assertIn('aria-controls="voice-live-panel"', template)
        self.assertIn('aria-controls="voice-conversation"', template)
        self.assertIn('role="tabpanel"', template)
        self.assertIn('aria-labelledby="voice-live-tab"', template)
        self.assertIn('aria-labelledby="voice-transcript-tab"', template)
        self.assertIn('data-transcript-stream hidden', template)
        self.assertEqual(template.count('aria-describedby="voice-capture-help"'), 2)
        self.assertIn("Pause stops the microphone but keeps typing available.", template)
        self.assertIn("Private discards capture and blocks all input.", template)
        self.assertIn('aria-label="Assistant volume"', template)

        for key in ("ArrowRight", "ArrowLeft", "Home", "End"):
            self.assertIn(f"event.key === '{key}'", script)
        self.assertIn("panel.hidden = !selected", script)
        self.assertIn("item.tabIndex = selected ? 0 : -1", script)
        self.assertIn("activateVoiceView(voiceTabs[nextIndex], { focus: true })", script)

    def test_voice_profiles_are_accessible_session_local_and_persistent(self) -> None:
        template = Path("atlas_voice/web/templates/voice.html").read_text()
        script = Path("atlas_voice/web/static/voice.js").read_text()
        css = Path("atlas_voice/web/static/app.css").read_text()
        profile_markup = template.split('data-voice-profile-control', 1)[1].split(
            'class="voice-session-layout"', 1
        )[0]

        self.assertIn("voice_settings.voice_profiles", template)
        self.assertIn("voice_settings.default_voice_profile", template)
        self.assertIn('role="radiogroup"', profile_markup)
        self.assertIn('role="radio"', profile_markup)
        self.assertIn('aria-checked=', profile_markup)
        self.assertIn('aria-live="polite"', profile_markup)
        self.assertIn("data-voice-profile-status", profile_markup)
        self.assertNotIn("data-effective-voice-model", profile_markup)
        self.assertNotIn("qwen27", profile_markup)
        for profile_id, name in (("light", "Light"), ("torch", "Torch"), ("fire", "Fire")):
            self.assertIn(f"'id': '{profile_id}'", template)
            self.assertIn(f"'name': '{name}'", template)
        self.assertIn("voice-profile-{{ profile.id }}-description", profile_markup)
        self.assertIn('data-profile-icon="{{ profile.icon }}"', profile_markup)
        for icon in ("feather.svg", "flame.svg", "flame-kindling.svg"):
            self.assertIn(icon, template)

        self.assertNotIn("quality_tier", profile_markup)
        self.assertNotIn("base_url", profile_markup)
        self.assertNotIn("realtime_host", profile_markup)
        self.assertIn('name="quality_tier"', template)
        self.assertIn("body.append('quality_tier'", script)

        self.assertIn("atlas.voice.profile", script)
        self.assertIn("window.localStorage.getItem", script)
        self.assertIn("allowedVoiceProfiles.has(stored)", script)
        self.assertIn("window.localStorage.setItem", script)
        self.assertIn("type: 'session.update'", script)
        self.assertIn("voice_profile: profileId", script)
        self.assertIn("payload.type === 'session.created'", script)
        self.assertIn("payload.type === 'response.done'", script)
        self.assertIn("completedResponse.served_model", script)
        self.assertIn("effectiveVoiceModel = servedModel", script)
        self.assertIn("payload.type === 'session.updated'", script)
        self.assertIn("applySessionVoiceProfile(payload.session || payload)", script)
        self.assertIn("safeVoiceModelLabel", script)
        self.assertIn("const basename = label.split(", script)
        self.assertIn("filter(Boolean).pop()", script)
        self.assertIn("' selected for next call'", script)
        self.assertIn("'Switching to '", script)
        self.assertIn("' active'", script)
        self.assertIn("option.disabled = waiting", script)
        self.assertIn(
            "pendingVoiceProfile && pendingVoiceProfile !== effectiveVoiceProfile",
            script,
        )
        self.assertIn("updateModeSelectIcon", script)
        self.assertIn("select.addEventListener('change'", script)
        self.assertIn("Warming local voice", script)
        self.assertNotIn("payload.model || 'voice'", script)

        for selector in (
            ".voice-profile-bar",
            ".voice-profile-segments",
            ".voice-profile-option",
            ".voice-profile-status",
        ):
            self.assertIn(selector, css)
        self.assertIn("@media (max-width: 440px)", css)
        self.assertIn("@media (max-width: 360px)", css)
        self.assertIn("min-height: var(--atlas-control-touch)", css)
        self.assertNotIn(".voice-tool-tabs .voice-profile-option", css)
        self.assertNotIn(".voice-profile-effective", css)

    def test_mode_icons_are_reused_across_processing_surfaces(self) -> None:
        templates = {
            name: Path(f"atlas_voice/web/templates/{name}.html").read_text()
            for name in ("base", "index", "recording", "recordings", "voice")
        }
        design = Path("atlas_voice/web/static/design-system.css").read_text()

        for icon in ("feather.svg", "flame.svg", "flame-kindling.svg"):
            self.assertTrue(Path("atlas_voice/web/static/icons", icon).is_file())
        self.assertIn("profile.icon", templates["base"])
        self.assertIn("profile.icon", templates["recordings"])
        self.assertIn("quality_profile.icon", templates["recording"])
        self.assertIn("default_quality_view.icon", templates["index"])
        self.assertIn("data-mode-select-icon", templates["index"])
        self.assertIn("data-mode-select-icon", templates["voice"])
        self.assertIn(".quality-choice-icon", design)
        self.assertIn(".mode-select-control", design)

    def test_settings_keep_diagnostics_out_of_primary_voice_surface(self) -> None:
        template = Path("atlas_voice/web/templates/voice.html").read_text()
        content, settings = template.split("{% block settings_dialog_extra %}", 1)

        self.assertNotIn("Voice Playground", content)
        self.assertNotIn("Privacy Events", content)
        self.assertIn("Voice Playground", settings)
        self.assertIn("Privacy and sessions", settings)
        self.assertIn("Memory and coaching", settings)
        self.assertIn("data-tts-playground", settings)

    def test_static_assets_are_versioned_and_legacy_gradients_are_absent(self) -> None:
        base = Path("atlas_voice/web/templates/base.html").read_text().lower()
        index = Path("atlas_voice/web/templates/index.html").read_text().lower()
        voice = Path("atlas_voice/web/templates/voice.html").read_text().lower()
        app_css = Path("atlas_voice/web/static/app.css").read_text().lower()
        design_css = Path("atlas_voice/web/static/design-system.css").read_text().lower()
        css = app_css + design_css

        self.assertIn("/static/app.css?v=20260710.3", base)
        self.assertIn("/static/design-system.css?v=20260710.2", base)
        self.assertIn("/static/voice.js?v=20260710.5", index)
        self.assertIn("/static/voice.js?v=20260710.5", voice)
        self.assertRegex(base, r'/static/app\.css\?v=[^"\']+')
        self.assertRegex(base, r'/static/design-system\.css\?v=[^"\']+')
        self.assertRegex(index, r'/static/voice\.js\?v=[^"\']+')
        self.assertRegex(voice, r'/static/voice\.js\?v=[^"\']+')
        for token in ("gradient(", "purple", "violet", "indigo"):
            self.assertNotIn(token, css)

    def test_design_system_foundation_is_loaded_and_documented(self) -> None:
        base = Path("atlas_voice/web/templates/base.html").read_text()
        design = Path("atlas_voice/web/static/design-system.css").read_text()
        documentation = Path("docs/DESIGN_SYSTEM.md").read_text()
        package_config = Path("pyproject.toml").read_text()

        self.assertLess(base.index("/static/app.css"), base.index("/static/design-system.css"))
        for token in (
            "--atlas-color-canvas",
            "--atlas-font-sans",
            "--atlas-space-4",
            "--atlas-radius-control",
            "--atlas-elevation-dialog",
            "--atlas-control-touch",
            "--atlas-motion-standard",
            "--atlas-z-dialog",
            "--bg: var(--atlas-color-canvas-muted)",
        ):
            self.assertIn(token, design)
        for selector in (
            ".quality-choice",
            ".capture-mode-explanation",
            ".recording-upload-dialog",
            ".template-confirmation",
            ".recording-people",
            ".inline-status",
        ):
            self.assertIn(selector, design)
        for phrase in ("Light", "Torch", "Fire", "Plain-language states", "Accessibility"):
            self.assertIn(phrase, documentation)
        self.assertIn("web/static/*.css", package_config)

    def test_settings_use_edge_aligned_drawer_and_icon_close(self) -> None:
        base = Path("atlas_voice/web/templates/base.html").read_text()
        css = Path("atlas_voice/web/static/app.css").read_text()
        drawer = self._last_css_block(css, ".settings-dialog")

        self.assertIn('class="settings-scroll"', base)
        self.assertIn('src="/static/icons/x.svg"', base)
        self.assertNotIn(">Done</button>", base)
        self.assertIn("inset: 0 0 0 auto", drawer)
        self.assertIn("height: 100dvh", drawer)
        self.assertIn("max-width: none", drawer)
        self.assertIn("border-radius: 0", drawer)

    def test_workspace_search_uses_navigation_toolbar(self) -> None:
        base = Path("atlas_voice/web/templates/base.html").read_text()
        search = Path("atlas_voice/web/templates/search.html").read_text()
        css = Path("atlas_voice/web/static/app.css").read_text()
        design = Path("atlas_voice/web/static/design-system.css").read_text()
        toolbar = self._last_css_block(css, ".workspace-topbar")

        self.assertNotIn('class="sidebar-search"', base)
        self.assertIn('class="workspace-topbar"', base)
        self.assertIn('role="search"', base)
        self.assertIn("data-global-search", base)
        self.assertIn('aria-label="Clear search"', base)
        self.assertIn("position: sticky", toolbar)
        self.assertIn("grid-template-columns", toolbar)
        self.assertIn('class="search-main-form"', search)
        self.assertIn('class="search-scope"', search)
        self.assertIn('class="search-group-list"', search)
        self.assertIn('class="search-group"', search)
        self.assertIn('class="search-match"', search)
        self.assertIn("<mark>", search)
        for selector in (
            ".search-main-form",
            ".search-scope",
            ".search-group",
            ".search-match",
        ):
            self.assertIn(selector, design)

    def test_dashboard_chat_uses_two_tier_prompt_composer(self) -> None:
        template = Path("atlas_voice/web/templates/index.html").read_text()
        css = Path("atlas_voice/web/static/app.css").read_text()
        script = Path("atlas_voice/web/static/voice.js").read_text()
        composer = self._last_css_block(css, ".dashboard-command-bar")
        prompt = self._last_css_block(
            css, ".dashboard-command-bar textarea.dashboard-prompt-input"
        )

        self.assertIn("<textarea", template)
        self.assertIn('class="dashboard-composer-actions"', template)
        self.assertIn('class="composer-round chat-attach-button"', template)
        self.assertIn('aria-label="Send message"', template)
        self.assertNotIn('class="dashboard-chat-row"', template)
        self.assertIn("width: min(768px, 100%)", composer)
        self.assertIn("min-height: 146px", composer)
        self.assertIn("max-height: 35svh", prompt)
        self.assertIn("resize: none", prompt)
        self.assertIn("border-radius: 50%", css)
        self.assertIn("const resizePromptInput", script)
        self.assertIn("!input.value.trim()", script)
        self.assertIn("event.shiftKey", script)
        self.assertIn("form.requestSubmit()", script)

    def test_dashboard_source_picker_is_accessible_and_stateful(self) -> None:
        template = Path("atlas_voice/web/templates/index.html").read_text()
        css = Path("atlas_voice/web/static/app.css").read_text()
        script = Path("atlas_voice/web/static/voice.js").read_text()
        popover = self._last_css_block(css, ".chat-source-popover")

        self.assertIn('type="hidden" name="source_context"', template)
        self.assertIn('type="hidden" name="recording_id"', template)
        self.assertIn('aria-haspopup="listbox"', template)
        self.assertIn('role="listbox"', template)
        self.assertIn('role="option"', template)
        for value in ("recordings", "uploads", "voice"):
            self.assertIn(f'data-source-value="{value}"', template)
        self.assertNotIn('class="chat-source-select"', template)
        self.assertIn("position: absolute", popover)
        self.assertIn("top: calc(100% + 9px)", popover)
        self.assertIn("opens-up", css)
        for key in ("ArrowDown", "ArrowUp", "Home", "End", "Escape"):
            self.assertIn(f"'{key}'", script)
        self.assertIn("selectSourceOption", script)
        self.assertIn("sourceInput.dispatchEvent", script)
        self.assertIn("type: 'session.update'", script)
        self.assertIn("recording_id: recordingId || null", script)
        self.assertNotIn("body.append('source_context'", script)
        self.assertIn("source_context: sourceInput?.value", script)
        self.assertIn("recording_id: recordingInput?.value", script)

    def test_recordings_have_a_dedicated_library_and_focused_chat_panel(self) -> None:
        base = Path("atlas_voice/web/templates/base.html").read_text()
        dashboard = Path("atlas_voice/web/templates/index.html").read_text()
        library = Path("atlas_voice/web/templates/recordings.html").read_text()
        detail = Path("atlas_voice/web/templates/recording.html").read_text()
        css = Path("atlas_voice/web/static/app.css").read_text()
        recordings_script = Path("atlas_voice/web/static/recordings.js").read_text()

        self.assertIn('href="/recordings"', base)
        self.assertNotIn('id="recordings"', dashboard)
        self.assertNotIn("data-dashboard-recording-list", dashboard)
        self.assertIn("data-recording-context-panel", dashboard)
        self.assertIn("Full transcript", dashboard)
        self.assertIn("data-lazy-transcript", dashboard)
        self.assertIn("data-recording-library", library)
        self.assertIn("/static/recordings.js", library)
        self.assertIn('action="/recording-folders"', library)
        self.assertIn("folder_id", library)
        self.assertIn("Chat about this", detail)
        self.assertIn("note-reader", detail)
        self.assertIn("recording-transcript", detail)
        self.assertIn("data-transcript-endpoint", detail)
        self.assertIn(".recording-context-panel", css)
        self.assertIn(".recordings-layout", css)
        self.assertIn("fetch(endpoint", recordings_script)
        self.assertIn("textContent = text", recordings_script)
        self.assertIn("replaceChildren", recordings_script)

    def test_dashboard_attachment_controls_use_icon_feedback(self) -> None:
        template = Path("atlas_voice/web/templates/index.html").read_text()
        css = Path("atlas_voice/web/static/app.css").read_text()
        script = Path("atlas_voice/web/static/voice.js").read_text()

        self.assertIn("multiple hidden", template)
        self.assertIn("attachButton.dataset.count", script)
        self.assertIn("removeIcon.src = '/static/icons/x.svg'", script)
        self.assertNotIn("remove.textContent = 'Remove'", script)
        self.assertIn(".chat-attach-button.has-files::after", css)
        self.assertIn(".voice-mic-button.is-listening", css)
        self.assertIn(".dashboard-command-bar.is-uploading", css)
        self.assertIn("form.setAttribute('aria-busy', 'true')", script)
        self.assertIn("Upload ${count} file", script)
        self.assertIn("Stop voice input", script)


if __name__ == "__main__":
    unittest.main()
