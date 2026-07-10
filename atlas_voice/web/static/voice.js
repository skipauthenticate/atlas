(() => {
  const DEFAULT_SAMPLE_RATE = 24000;
  const CAPTURE_FRAME_SIZE = 4096;

  function websocketUrl(path, token) {
    const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = new URL(`${scheme}//${window.location.host}${path}`);
    if (token) url.searchParams.set('token', token);
    return url.toString();
  }

  function bytesToBase64(bytes) {
    let binary = '';
    const chunkSize = 0x8000;
    for (let offset = 0; offset < bytes.length; offset += chunkSize) {
      binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
    }
    return btoa(binary);
  }

  function downsampleToPCM16(samples, inputRate, outputRate) {
    if (!samples.length) return new Uint8Array();
    const ratio = inputRate / outputRate;
    const outputLength = Math.max(Math.floor(samples.length / ratio), 1);
    const pcm = new Int16Array(outputLength);

    for (let outputIndex = 0; outputIndex < outputLength; outputIndex += 1) {
      const start = Math.floor(outputIndex * ratio);
      const end = Math.max(Math.floor((outputIndex + 1) * ratio), start + 1);
      let sum = 0;
      let count = 0;
      for (let inputIndex = start; inputIndex < Math.min(end, samples.length); inputIndex += 1) {
        sum += samples[inputIndex];
        count += 1;
      }
      const value = Math.max(-1, Math.min(1, count ? sum / count : samples[start] || 0));
      pcm[outputIndex] = value < 0 ? value * 0x8000 : value * 0x7fff;
    }
    return new Uint8Array(pcm.buffer);
  }

  function audioUrlFromDelta(delta, mediaType) {
    const binary = atob(delta || '');
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return URL.createObjectURL(new Blob([bytes], { type: mediaType || 'audio/wav' }));
  }

  function fileSizeLabel(size) {
    if (!Number.isFinite(size) || size <= 0) return '0 KB';
    if (size < 1024 * 1024) return `${Math.max(Math.round(size / 1024), 1)} KB`;
    return `${(size / (1024 * 1024)).toFixed(size < 10 * 1024 * 1024 ? 1 : 0)} MB`;
  }

  function appendMessage(transcript, role, text) {
    const messageList = transcript.querySelector('[data-transcript-messages]') || transcript;
    messageList.querySelector('[data-transcript-empty]')?.remove();

    const item = document.createElement('div');
    item.className = `voice-message ${role === 'Atlas' ? 'assistant' : 'user'}`;

    const label = document.createElement('span');
    label.textContent = role === 'Atlas' ? 'Atlas' : 'You';

    const body = document.createElement('p');
    body.textContent = text;

    item.append(label, body);
    messageList.append(item);
    item.scrollIntoView({ block: 'nearest', behavior: 'smooth' });

    const count = transcript.querySelector('[data-transcript-count]');
    if (count) count.textContent = String(messageList.querySelectorAll('.voice-message').length);
    return body;
  }

  function appendSources(messageBody, sources) {
    if (!messageBody?.parentElement || !Array.isArray(sources) || !sources.length) return;
    messageBody.parentElement.querySelector('.voice-message-sources')?.remove();
    const list = document.createElement('div');
    list.className = 'voice-message-sources';
    list.setAttribute('aria-label', 'Web sources');
    sources.slice(0, 4).forEach((source) => {
      try {
        const url = new URL(source?.url || '');
        if (!['http:', 'https:'].includes(url.protocol)) return;
        const link = document.createElement('a');
        link.href = url.href;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = source?.title || url.hostname;
        link.title = `${source?.title || url.hostname}, ${url.hostname}`;
        list.append(link);
      } catch (_error) {
        // Ignore malformed provider URLs.
      }
    });
    if (list.childElementCount) messageBody.parentElement.append(list);
  }

  class VoiceBubble {
    constructor(canvas, stage) {
      this.canvas = canvas;
      this.stage = stage;
      this.context = canvas?.getContext('2d') || null;
      this.state = 'idle';
      this.level = 0;
      this.smoothedLevel = 0;
      this.outputAnalyser = null;
      this.outputData = null;
      this.outputContext = null;
      this.outputSource = null;
      this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      this.startedAt = performance.now();
      this.resizeObserver = new ResizeObserver(() => this.resize());
      if (canvas) this.resizeObserver.observe(canvas);
      this.resize();
      this.frame = requestAnimationFrame((time) => this.draw(time));
    }

    resize() {
      if (!this.canvas) return;
      const rect = this.canvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(Math.round(rect.width * ratio), 1);
      const height = Math.max(Math.round(rect.height * ratio), 1);
      if (this.canvas.width !== width || this.canvas.height !== height) {
        this.canvas.width = width;
        this.canvas.height = height;
      }
    }

    setState(state) {
      this.state = state;
      if (this.stage) this.stage.dataset.voiceState = state;
    }

    setLevel(level) {
      this.level = Math.max(0, Math.min(1, Number(level) || 0));
    }

    async attachOutput(audio) {
      if (!audio || this.outputSource) {
        await this.outputContext?.resume();
        return;
      }
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) return;
      this.outputContext = new AudioContextClass({ latencyHint: 'interactive' });
      this.outputSource = this.outputContext.createMediaElementSource(audio);
      this.outputAnalyser = this.outputContext.createAnalyser();
      this.outputAnalyser.fftSize = 256;
      this.outputAnalyser.smoothingTimeConstant = 0.72;
      this.outputData = new Uint8Array(this.outputAnalyser.fftSize);
      this.outputSource.connect(this.outputAnalyser);
      this.outputAnalyser.connect(this.outputContext.destination);
      await this.outputContext.resume();
    }

    outputLevel() {
      if (!this.outputAnalyser || !this.outputData) return 0;
      this.outputAnalyser.getByteTimeDomainData(this.outputData);
      let sum = 0;
      for (const sample of this.outputData) {
        const normalized = (sample - 128) / 128;
        sum += normalized * normalized;
      }
      return Math.min(Math.sqrt(sum / this.outputData.length) * 4.5, 1);
    }

    draw(timestamp) {
      if (!this.context || !this.canvas) return;
      const ctx = this.context;
      const width = this.canvas.width;
      const height = this.canvas.height;
      const centerX = width / 2;
      const centerY = height / 2;
      const scale = Math.min(width, height);
      const time = this.reducedMotion ? 0 : (timestamp - this.startedAt) / 1000;
      const measured = this.state === 'speaking' ? this.outputLevel() : this.level;
      this.smoothedLevel += (measured - this.smoothedLevel) * (measured > this.smoothedLevel ? 0.3 : 0.08);
      const active = ['listening', 'thinking', 'synthesizing', 'speaking'].includes(this.state);
      const energy = active ? Math.max(this.smoothedLevel, this.state === 'thinking' ? 0.1 : 0.035) : 0.018;
      const baseRadius = scale * (this.state === 'speaking' ? 0.315 : 0.3);
      const accent = '#ff5b35';
      const ink = '#171717';

      ctx.clearRect(0, 0, width, height);
      ctx.save();
      ctx.translate(centerX, centerY);

      for (let ring = 3; ring >= 1; ring -= 1) {
        ctx.beginPath();
        const ringRadius = baseRadius + scale * (0.025 * ring + energy * 0.02 * ring);
        for (let point = 0; point <= 96; point += 1) {
          const angle = (point / 96) * Math.PI * 2;
          const harmonic =
            Math.sin(angle * 3 + time * 1.4) * 0.55 +
            Math.sin(angle * 5 - time * 1.05) * 0.28 +
            Math.sin(angle * 2 + time * 0.62) * 0.17;
          const radius = ringRadius + harmonic * scale * energy * (0.018 + ring * 0.004);
          const x = Math.cos(angle) * radius;
          const y = Math.sin(angle) * radius;
          if (point === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.strokeStyle = this.state === 'listening' || this.state === 'speaking'
          ? `rgba(255, 91, 53, ${0.06 + ring * 0.035})`
          : `rgba(23, 23, 23, ${0.025 + ring * 0.025})`;
        ctx.lineWidth = Math.max(scale * 0.004, 2);
        ctx.stroke();
      }

      ctx.beginPath();
      for (let point = 0; point <= 128; point += 1) {
        const angle = (point / 128) * Math.PI * 2;
        const harmonic =
          Math.sin(angle * 3 + time * 1.6) * 0.5 +
          Math.sin(angle * 7 - time * 0.85) * 0.3 +
          Math.sin(angle * 4 + time * 0.42) * 0.2;
        const radius = baseRadius + harmonic * scale * energy * 0.04;
        const x = Math.cos(angle) * radius;
        const y = Math.sin(angle) * radius;
        if (point === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = ink;
      ctx.fill();

      if (this.state !== 'idle' && this.state !== 'error') {
        ctx.beginPath();
        ctx.arc(0, 0, baseRadius + scale * 0.012, -Math.PI * 0.7 + time * 0.45, -Math.PI * 0.18 + time * 0.45);
        ctx.strokeStyle = accent;
        ctx.lineWidth = Math.max(scale * 0.009, 3);
        ctx.lineCap = 'round';
        ctx.stroke();
      }

      ctx.restore();
      this.frame = requestAnimationFrame((nextTime) => this.draw(nextTime));
    }
  }

  window.addEventListener('DOMContentLoaded', () => {
    const root = document.querySelector('[data-voice-console="true"]');
    if (!root) return;

    const enabled = root.dataset.assistantEnabled === 'true';
    const socketPath = root.dataset.websocketPath || '/v1/realtime';
    const realtimeToken = root.dataset.realtimeToken || '';
    const targetSampleRate = Number(root.dataset.audioSampleRate) || DEFAULT_SAMPLE_RATE;
    const form = root.querySelector('[data-voice-prompt]');
    const input = root.querySelector('#voice-prompt-input');
    const submit = root.querySelector('#voice-prompt-submit');
    const transcript = root.querySelector('[data-transcript-stream]');
    const connection = root.querySelector('[data-voice-connection]');
    const stage = root.querySelector('[data-voice-stage]');
    const callTitle = root.querySelector('[data-call-title]');
    const callState = root.querySelector('[data-call-state]');
    const bubbleKicker = root.querySelector('[data-bubble-kicker]');
    const liveCaption = root.querySelector('[data-live-caption]');
    const captureHelp = root.querySelector('[data-capture-help]');
    const chatVoiceToggle = root.querySelector('[data-chat-voice-toggle]');
    const audio = root.querySelector('[data-response-audio]');
    const playbackStatus = root.querySelector('[data-playback-status]');
    const bubble = new VoiceBubble(root.querySelector('[data-voice-bubble]'), stage);

    if (!form || !input || !submit || !transcript) return;

    const fileInput = form.querySelector('[data-chat-file-input]');
    const attachButton = form.querySelector('[data-chat-attach]');
    const attachmentTray = form.querySelector('[data-chat-attachments]');
    const uploadStatus = form.querySelector('[data-chat-upload-status]');
    const uploadOptions = form.querySelector('[data-upload-options]');
    const expectedSpeakers = form.querySelector('[name="expected_main_speakers"]');
    const qualityTier = form.querySelector('[name="quality_tier"]');
    const uploadEndpoint = form.dataset.uploadEndpoint || '/upload';
    const sourceMenu = form.querySelector('[data-chat-source-menu]');
    const sourceInput = form.querySelector('[data-chat-source]');
    const recordingInput = form.querySelector('[data-chat-recording-id]');
    const sourceTrigger = form.querySelector('[data-chat-source-trigger]');
    const sourceList = form.querySelector('[data-chat-source-list]');
    const sourceLabel = form.querySelector('[data-chat-source-label]');
    const sourceTriggerIcon = sourceTrigger?.querySelector('.source-trigger-icon');
    const sourceOptions = Array.from(form.querySelectorAll('[data-source-value]'));
    const pendingEvents = [];
    const playbackQueue = [];
    let socket = null;
    let assistantText = null;
    let playbackEnabled = false;
    let currentAudioUrl = null;
    let selectedFiles = [];
    let uploading = false;
    let callActive = false;
    let callEnding = false;
    let micEnabled = false;
    let captureMode = 'open';
    let resumeMicrophoneAfterCaptureMode = false;
    let captureTransitioning = false;
    let micSpeechPending = false;
    let micStream = null;
    let captureContext = null;
    let captureSource = null;
    let captureNode = null;
    let captureSink = null;
    let workletUrl = null;
    let currentVoiceState = 'idle';
    let handleSocketClosed = () => {};
    let refreshPromptAvailability = () => {};

    const voiceProfileControl = root.querySelector('[data-voice-profile-control]');
    const voiceProfileOptions = Array.from(
      voiceProfileControl?.querySelectorAll('[data-voice-profile]') || [],
    );
    const voiceProfileStatus = voiceProfileControl?.querySelector(
      '[data-voice-profile-status]',
    );
    const effectiveVoiceProfileName = document.querySelector(
      '[data-effective-voice-profile]',
    );
    const effectiveVoiceProfileIcon = document.querySelector(
      '[data-effective-voice-profile-icon]',
    );
    const effectiveVoiceModelName = document.querySelector(
      '[data-effective-voice-model]',
    );
    const voiceProfileStorageKey = 'atlas.voice.profile';
    const voiceProfileMetadata = new Map(
      voiceProfileOptions.map((option) => [
        option.dataset.voiceProfile || '',
        {
          name: option.dataset.profileName || option.textContent.trim(),
          icon: safeModeIconFilename(option.dataset.profileIcon),
        },
      ]),
    );
    const allowedVoiceProfiles = new Map(
      Array.from(voiceProfileMetadata, ([profileId, metadata]) => [profileId, metadata.name]),
    );
    const configuredDefaultVoiceProfile = voiceProfileControl?.dataset.defaultVoiceProfile || '';
    const defaultVoiceProfile = allowedVoiceProfiles.has(configuredDefaultVoiceProfile)
      ? configuredDefaultVoiceProfile
      : (voiceProfileOptions[0]?.dataset.voiceProfile || '');

    function safeVoiceModelLabel(value) {
      const label = String(value || '').trim();
      if (!label || /^[a-z][a-z\d+.-]*:\/\//i.test(label)) return 'Local model';
      const basename = label.split(/[\\/]/).filter(Boolean).pop();
      if (!basename || basename === '.' || basename === '..') return 'Local model';
      return basename.slice(0, 120);
    }

    function safeModeIconFilename(value) {
      const filename = String(value || '').trim().toLowerCase();
      return /^[a-z0-9-]+\.svg$/.test(filename) ? filename : 'flame.svg';
    }

    function updateModeSelectIcon(select) {
      const icon = select.closest('.mode-select-field')?.querySelector('[data-mode-select-icon]');
      const selected = select.selectedOptions?.[0];
      if (!icon || !selected) return;
      icon.src = '/static/icons/' + safeModeIconFilename(selected.dataset.icon);
    }

    function readPreferredVoiceProfile() {
      if (!voiceProfileControl) return null;
      try {
        const stored = window.localStorage.getItem(voiceProfileStorageKey);
        if (allowedVoiceProfiles.has(stored)) return stored;
        if (stored) window.localStorage.removeItem(voiceProfileStorageKey);
      } catch (_error) {
        // Storage can be unavailable in hardened or private browser contexts.
      }
      return null;
    }

    function persistPreferredVoiceProfile(profileId) {
      if (!allowedVoiceProfiles.has(profileId)) return;
      try {
        window.localStorage.setItem(voiceProfileStorageKey, profileId);
      } catch (_error) {
        // The session-local selection still works when persistence is unavailable.
      }
    }

    let preferredVoiceProfile = readPreferredVoiceProfile() || defaultVoiceProfile;
    let effectiveVoiceProfile = defaultVoiceProfile;
    let effectiveVoiceModel = safeVoiceModelLabel(effectiveVoiceModelName?.textContent);
    let pendingVoiceProfile = null;

    function renderVoiceProfileState() {
      if (!voiceProfileControl) return;
      const waiting = Boolean(pendingVoiceProfile && pendingVoiceProfile !== effectiveVoiceProfile);
      voiceProfileOptions.forEach((option) => {
        const selected = option.dataset.voiceProfile === preferredVoiceProfile;
        option.setAttribute('aria-checked', selected ? 'true' : 'false');
        option.tabIndex = selected ? 0 : -1;
        option.disabled = waiting;
      });
      voiceProfileControl.dataset.preferredVoiceProfile = preferredVoiceProfile;
      voiceProfileControl.dataset.effectiveVoiceProfile = effectiveVoiceProfile;
      voiceProfileControl.classList.toggle('is-pending', waiting);
      if (waiting) voiceProfileControl.setAttribute('aria-busy', 'true');
      else voiceProfileControl.removeAttribute('aria-busy');

      const preferredMetadata = voiceProfileMetadata.get(preferredVoiceProfile) || {
        name: 'Voice',
        icon: 'flame.svg',
      };
      const effectiveMetadata = voiceProfileMetadata.get(effectiveVoiceProfile) || {
        name: 'Voice',
        icon: 'flame.svg',
      };
      if (voiceProfileStatus) {
        if (waiting) voiceProfileStatus.textContent = 'Switching to ' + preferredMetadata.name;
        else if (callActive) voiceProfileStatus.textContent = effectiveMetadata.name + ' active';
        else voiceProfileStatus.textContent = preferredMetadata.name + ' selected for next call';
      }
      const diagnosticMetadata = callActive ? effectiveMetadata : preferredMetadata;
      if (effectiveVoiceProfileName) effectiveVoiceProfileName.textContent = diagnosticMetadata.name;
      if (effectiveVoiceProfileIcon) {
        effectiveVoiceProfileIcon.src = '/static/icons/' + diagnosticMetadata.icon;
      }
      if (effectiveVoiceModelName) effectiveVoiceModelName.textContent = effectiveVoiceModel;
    }

    function requestVoiceProfile(profileId, { persist = true, force = false } = {}) {
      if (!allowedVoiceProfiles.has(profileId)) return false;
      if (pendingVoiceProfile && pendingVoiceProfile !== effectiveVoiceProfile) return false;
      preferredVoiceProfile = profileId;
      if (persist) persistPreferredVoiceProfile(profileId);
      if (callActive && (force || profileId !== effectiveVoiceProfile)) {
        pendingVoiceProfile = profileId;
        sendRealtimeEvent({
          type: 'session.update',
          voice_profile: profileId,
        });
      }
      renderVoiceProfileState();
      return true;
    }

    function applySessionVoiceProfile(session, { confirmPreference = false } = {}) {
      if (!voiceProfileControl || !session || typeof session !== 'object') return;
      const profileId = String(session.voice_profile || '').trim().toLowerCase();
      if (allowedVoiceProfiles.has(profileId)) {
        effectiveVoiceProfile = profileId;
        if (confirmPreference) {
          preferredVoiceProfile = profileId;
          persistPreferredVoiceProfile(profileId);
        }
        if (confirmPreference || pendingVoiceProfile === profileId) pendingVoiceProfile = null;
      }
      const model = safeVoiceModelLabel(session.model || session.llm_model);
      if (model !== 'Local model' || !effectiveVoiceModel) effectiveVoiceModel = model;
      renderVoiceProfileState();
    }

    voiceProfileOptions.forEach((option, index) => {
      option.addEventListener('click', () => {
        requestVoiceProfile(option.dataset.voiceProfile || '');
      });
      option.addEventListener('keydown', (event) => {
        let nextIndex = null;
        if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
          nextIndex = (index + 1) % voiceProfileOptions.length;
        } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
          nextIndex = (index - 1 + voiceProfileOptions.length) % voiceProfileOptions.length;
        } else if (event.key === 'Home') {
          nextIndex = 0;
        } else if (event.key === 'End') {
          nextIndex = voiceProfileOptions.length - 1;
        }
        if (nextIndex === null) return;
        event.preventDefault();
        const nextOption = voiceProfileOptions[nextIndex];
        nextOption.focus();
        requestVoiceProfile(nextOption.dataset.voiceProfile || '');
      });
    });

    root.querySelectorAll('[data-mode-select]').forEach((select) => {
      updateModeSelectIcon(select);
      select.addEventListener('change', () => updateModeSelectIcon(select));
    });
    if (preferredVoiceProfile) persistPreferredVoiceProfile(preferredVoiceProfile);
    renderVoiceProfileState();
    const resizePromptInput = () => {
      if (!(input instanceof HTMLTextAreaElement)) return;
      input.style.height = 'auto';
      const maximum = Math.min(window.innerHeight * 0.35, 300);
      input.style.height = `${Math.min(input.scrollHeight, maximum)}px`;
      input.style.overflowY = input.scrollHeight > maximum ? 'auto' : 'hidden';
    };

    input.addEventListener('input', () => {
      resizePromptInput();
      refreshPromptAvailability();
    });
    resizePromptInput();

    input.addEventListener('keydown', (event) => {
      if (
        !(input instanceof HTMLTextAreaElement)
        || event.key !== 'Enter'
        || event.shiftKey
        || event.isComposing
      ) return;
      event.preventDefault();
      if (!submit.disabled) form.requestSubmit();
    });

    const selectedSourceOption = () => (
      sourceOptions.find((option) => option.getAttribute('aria-selected') === 'true')
      || sourceOptions[0]
    );

    const setSourceMenuOpen = (open, { focusSelected = false } = {}) => {
      if (!sourceMenu || !sourceTrigger || !sourceList) return;
      sourceMenu.classList.toggle('is-open', open);
      sourceTrigger.setAttribute('aria-expanded', open ? 'true' : 'false');
      sourceList.hidden = !open;
      if (open) {
        const triggerBounds = sourceTrigger.getBoundingClientRect();
        const menuHeight = sourceList.scrollHeight;
        const spaceBelow = window.innerHeight - triggerBounds.bottom;
        sourceMenu.classList.toggle('opens-up', spaceBelow < menuHeight + 16);
      } else {
        sourceMenu.classList.remove('opens-up');
      }
      if (open && focusSelected) {
        requestAnimationFrame(() => selectedSourceOption()?.focus());
      }
    };

    const selectSourceOption = (option) => {
      if (!option || !sourceInput || !sourceTrigger || !sourceLabel) return;
      const value = option.dataset.sourceValue || '';
      const label = option.dataset.sourceLabel || option.textContent.trim();
      const recordingId = option.dataset.recordingId || '';
      sourceInput.value = value;
      if (recordingInput) recordingInput.value = recordingId;
      sourceLabel.textContent = label;
      sourceTrigger.setAttribute('aria-label', `Select source: ${label}`);
      const optionIcon = option.querySelector('.ui-icon');
      if (sourceTriggerIcon && optionIcon) sourceTriggerIcon.src = optionIcon.src;
      sourceOptions.forEach((item) => {
        item.setAttribute('aria-selected', item === option ? 'true' : 'false');
      });
      sourceInput.dispatchEvent(new Event('change', { bubbles: true }));
      recordingInput?.dispatchEvent(new Event('change', { bubbles: true }));
      sendRealtimeEvent({
        type: 'session.update',
        source_context: value,
        recording_id: recordingId || null,
      });
      const contextPanel = root.querySelector('[data-recording-context-panel]');
      document.body.classList.toggle('chat-recording-focus', Boolean(recordingId));
      if (recordingId) {
        root.classList.add('has-recording-context');
        if (contextPanel) contextPanel.hidden = false;
        window.history.replaceState({}, '', `/?recording=${encodeURIComponent(recordingId)}`);
      } else {
        root.classList.remove('has-recording-context');
        if (contextPanel) contextPanel.hidden = true;
        if (window.location.search.includes('recording=')) {
          window.history.replaceState({}, '', '/');
        }
      }
      setSourceMenuOpen(false);
      sourceTrigger.focus();
    };

    const moveSourceFocus = (direction) => {
      if (!sourceOptions.length) return;
      const activeIndex = sourceOptions.indexOf(document.activeElement);
      const selectedIndex = Math.max(sourceOptions.indexOf(selectedSourceOption()), 0);
      const currentIndex = activeIndex >= 0 ? activeIndex : selectedIndex;
      const nextIndex = (currentIndex + direction + sourceOptions.length) % sourceOptions.length;
      sourceOptions[nextIndex].focus();
    };

    if (sourceMenu && sourceTrigger && sourceList && sourceOptions.length) {
      sourceTrigger.addEventListener('click', () => {
        const open = sourceTrigger.getAttribute('aria-expanded') !== 'true';
        setSourceMenuOpen(open, { focusSelected: open });
      });
      sourceTrigger.addEventListener('keydown', (event) => {
        if (!['ArrowDown', 'ArrowUp'].includes(event.key)) return;
        event.preventDefault();
        setSourceMenuOpen(true, { focusSelected: true });
        if (event.key === 'ArrowUp') {
          requestAnimationFrame(() => moveSourceFocus(-1));
        }
      });
      sourceList.addEventListener('keydown', (event) => {
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
          event.preventDefault();
          moveSourceFocus(event.key === 'ArrowDown' ? 1 : -1);
          return;
        }
        if (event.key === 'Home' || event.key === 'End') {
          event.preventDefault();
          sourceOptions[event.key === 'Home' ? 0 : sourceOptions.length - 1].focus();
          return;
        }
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          selectSourceOption(document.activeElement);
          return;
        }
        if (event.key === 'Escape') {
          event.preventDefault();
          setSourceMenuOpen(false);
          sourceTrigger.focus();
        }
      });
      sourceOptions.forEach((option) => {
        option.addEventListener('click', () => selectSourceOption(option));
      });
      sourceMenu.addEventListener('focusout', () => {
        requestAnimationFrame(() => {
          if (!sourceMenu.contains(document.activeElement)) setSourceMenuOpen(false);
        });
      });
      document.addEventListener('pointerdown', (event) => {
        if (!sourceMenu.contains(event.target)) setSourceMenuOpen(false);
      });
    }

    function setVoiceState(state, caption) {
      currentVoiceState = state;
      bubble.setState(state);
      const labels = {
        idle: ['Local voice', 'Call Atlas'],
        connecting: ['Local voice', 'Connecting'],
        listening: ['Your turn', 'Listening'],
        transcribing: ['Atlas', 'Got it'],
        thinking: ['Atlas', 'Thinking'],
        synthesizing: ['Aiden', 'Finding the voice'],
        speaking: ['Atlas', 'Speaking'],
        paused: ['Call paused', 'Paused'],
        private: ['Local only', 'Private'],
        error: ['Check connection', 'Voice error'],
      };
      const [kicker, title] = labels[state] || labels.idle;
      if (bubbleKicker) bubbleKicker.textContent = kicker;
      if (callTitle) callTitle.textContent = title;
      if (caption && liveCaption) liveCaption.textContent = caption;
      if (callState) {
        callState.textContent = callActive ? `${targetSampleRate / 1000} kHz / ${micEnabled ? 'mic on' : 'mic off'}` : `${targetSampleRate / 1000} kHz / local`;
      }
    }

    function setConnection(state) {
      if (connection) connection.textContent = state;
      root.dataset.connectionState = state;
    }

    function setPlaybackStatus(state) {
      if (playbackStatus) playbackStatus.textContent = state;
    }

    function setCaptureHelp(message) {
      if (captureHelp) captureHelp.textContent = message;
    }

    function restoreCapturePresentation() {
      root.dataset.captureMode = callActive ? captureMode : 'idle';
      if (!callActive) return;
      if (captureMode === 'paused') {
        setConnection('paused');
        setPlaybackStatus('Mic paused; typing available');
        setVoiceState('paused', 'Microphone paused. You can keep typing.');
        setCaptureHelp('Microphone paused. Type a message, or resume to turn the microphone back on.');
        return;
      }
      if (captureMode === 'private') {
        setConnection('private');
        setPlaybackStatus('Private; all input blocked');
        setVoiceState('private', 'Private is on. New input is blocked.');
        setCaptureHelp('Private is on. The current mic buffer was discarded and no new input can be sent.');
        return;
      }
      setConnection(micEnabled ? 'listening' : 'online');
      setPlaybackStatus(micEnabled ? 'Listening' : 'Microphone muted');
      setVoiceState(micEnabled ? 'listening' : 'idle', micEnabled ? 'Ready when you are' : 'Microphone is muted');
      setCaptureHelp(
        micEnabled
          ? 'Microphone on. Pause keeps typing available; Private discards capture and blocks all input.'
          : 'Microphone muted. Typed messages are still available.',
      );
    }

    function revokeCurrentAudioUrl() {
      if (!currentAudioUrl) return;
      URL.revokeObjectURL(currentAudioUrl);
      currentAudioUrl = null;
    }

    function clearPlayback() {
      playbackQueue.splice(0).forEach((url) => URL.revokeObjectURL(url));
      audio?.pause();
      if (audio) audio.removeAttribute('src');
      revokeCurrentAudioUrl();
    }

    function playNextAudio() {
      if (!audio || !playbackEnabled || currentAudioUrl || playbackQueue.length === 0) return;
      currentAudioUrl = playbackQueue.shift();
      audio.src = currentAudioUrl;
      audio.play().catch(() => {
        setPlaybackStatus('Playback blocked');
        setVoiceState('error', 'Browser playback was blocked');
        revokeCurrentAudioUrl();
      });
    }

    audio?.addEventListener('play', () => {
      setPlaybackStatus('Atlas speaking');
      setVoiceState('speaking', assistantText?.textContent || '');
    });

    audio?.addEventListener('ended', () => {
      revokeCurrentAudioUrl();
      if (playbackQueue.length) {
        setPlaybackStatus('Audio queued');
        playNextAudio();
      } else {
        restoreCapturePresentation();
      }
    });

    audio?.addEventListener('error', () => {
      revokeCurrentAudioUrl();
      setPlaybackStatus('Playback error');
      if (playbackQueue.length) playNextAudio();
      else setVoiceState('error', 'Assistant audio could not play');
    });

    function connect() {
      if (!enabled || !callActive) return null;
      if (socket && socket.readyState <= WebSocket.OPEN) return socket;
      socket = new WebSocket(websocketUrl(socketPath, realtimeToken));
      setConnection('connecting');
      setVoiceState('connecting', 'Opening a local session');

      socket.addEventListener('open', () => {
        setConnection('online');
        if (voiceProfileControl && preferredVoiceProfile) {
          requestVoiceProfile(preferredVoiceProfile, { persist: false, force: true });
        }
        while (pendingEvents.length) socket.send(JSON.stringify(pendingEvents.shift()));
      });

      socket.addEventListener('message', (event) => {
        let payload;
        try {
          payload = JSON.parse(event.data);
        } catch (_error) {
          return;
        }

        if (payload.type === 'session.created') {
          setConnection('online');
          applySessionVoiceProfile(payload.session || payload);
          setVoiceState(micEnabled ? 'listening' : 'idle', micEnabled ? 'Ready when you are' : 'Microphone is muted');
          return;
        }
        if (payload.type === 'session.updated') {
          applySessionVoiceProfile(payload.session || payload, { confirmPreference: true });
          return;
        }
        if (payload.type === 'input_audio_buffer.speech_started') {
          micSpeechPending = true;
          setVoiceState('listening', 'Listening...');
          return;
        }
        if (payload.type === 'input_audio_buffer.speech_stopped') {
          setVoiceState('transcribing', 'Finishing your thought');
          return;
        }
        if (payload.type === 'input_audio_buffer.committed') {
          micSpeechPending = false;
          setVoiceState('transcribing', 'Transcribing locally');
          return;
        }
        if (payload.type === 'conversation.item.input_audio_transcription.done') {
          if (payload.text) appendMessage(transcript, 'You', payload.text);
          setVoiceState('thinking', payload.text || 'Thinking');
          return;
        }
        if (payload.type === 'response.created') {
          assistantText = appendMessage(transcript, 'Atlas', '');
          setVoiceState('thinking', 'Thinking');
          return;
        }
        if (payload.type === 'response.retrieval.started') {
          const caption = payload.web
            ? (payload.local ? 'Searching your history and the web' : 'Searching the web')
            : 'Searching your history';
          setPlaybackStatus('Retrieving evidence');
          setVoiceState('thinking', caption);
          return;
        }
        if (payload.type === 'response.retrieval.done') {
          appendSources(assistantText, payload.sources);
          const count = Number(payload.local_hit_count || 0) + Number(payload.web_result_count || 0);
          const caption = count ? `Found ${count} relevant source${count === 1 ? '' : 's'}` : 'No matching sources found';
          setVoiceState('thinking', caption);
          return;
        }
        if (payload.type === 'response.text.delta' && assistantText) {
          assistantText.textContent += payload.delta || '';
          return;
        }
        if (payload.type === 'response.text.done') {
          if (assistantText) assistantText.textContent = payload.text || assistantText.textContent;
          setVoiceState('synthesizing', 'Preparing a natural voice');
          return;
        }
        if (payload.type === 'response.audio.started') {
          setVoiceState('synthesizing', 'Warming local voice');
          return;
        }
        if (payload.type === 'response.audio.delta' && payload.delta) {
          playbackQueue.push(audioUrlFromDelta(payload.delta, payload.media_type));
          setPlaybackStatus('Audio ready');
          playNextAudio();
          return;
        }
        if (payload.type === 'response.audio.done') {
          if (payload.status === 'skipped') {
            restoreCapturePresentation();
          } else if (!currentAudioUrl && playbackQueue.length === 0) {
            setPlaybackStatus('Audio ready');
          }
          return;
        }
        if (payload.type === 'response.audio.failed') {
          setPlaybackStatus('Voice unavailable');
          setVoiceState(micEnabled ? 'listening' : 'error', payload.error?.message || 'Voice synthesis failed');
          return;
        }
        if (payload.type === 'response.interrupted' || payload.type === 'response.cancelled') {
          clearPlayback();
          assistantText = null;
          if (captureMode === 'open') {
            setPlaybackStatus('Interrupted');
            setVoiceState(micEnabled ? 'listening' : 'idle', micEnabled ? 'Listening...' : 'Response stopped');
          } else {
            restoreCapturePresentation();
          }
          return;
        }
        if (payload.type === 'response.failed') {
          assistantText = null;
          setVoiceState('error', payload.error?.message || 'The local model did not respond');
          return;
        }
        if (payload.type === 'response.done') {
          const completedResponse = payload.response || payload;
          const servedModel = safeVoiceModelLabel(completedResponse.served_model);
          if (servedModel !== 'Local model') {
            effectiveVoiceModel = servedModel;
            renderVoiceProfileState();
          }
          assistantText = null;
          if (!currentAudioUrl && playbackQueue.length === 0 && currentVoiceState !== 'speaking') {
            restoreCapturePresentation();
          }
          return;
        }
        if (payload.type === 'response.tool_call.created') {
          appendMessage(transcript, 'Atlas', `Local action: ${payload.tool_call?.name || 'tool'}`);
          return;
        }
        if (payload.type === 'response.tool_call.requires_confirmation') {
          appendMessage(transcript, 'Atlas', `Confirmation needed: ${payload.tool_call?.name || 'local action'}`);
          setVoiceState('idle', 'Waiting for confirmation');
          return;
        }
        if (payload.type === 'error') {
          const message = payload.error?.message || 'Realtime error';
          appendMessage(transcript, 'Atlas', message);
          setVoiceState('error', message);
        }
      });

      socket.addEventListener('close', () => {
        const wasActive = callActive;
        socket = null;
        if (wasActive) handleSocketClosed();
        else setConnection('idle');
      });

      socket.addEventListener('error', () => {
        setConnection('error');
        setVoiceState('error', 'The local realtime service is unavailable');
      });
      return socket;
    }

    function sendRealtimeEvent(message) {
      if (!callActive) return false;
      const activeSocket = connect();
      if (activeSocket?.readyState === WebSocket.OPEN) activeSocket.send(JSON.stringify(message));
      else pendingEvents.push(message);
      return true;
    }

    function handleCaptureSamples(samples, inputRate) {
      if (!micEnabled || !callActive || captureMode !== 'open' || !samples?.length) return;
      let squareSum = 0;
      for (const sample of samples) squareSum += sample * sample;
      const rms = Math.sqrt(squareSum / samples.length);
      bubble.setLevel(Math.min(rms * 5.5, 1));
      if (rms > 0.018) micSpeechPending = true;
      const pcm = downsampleToPCM16(samples, inputRate, targetSampleRate);
      if (!pcm.length) return;
      sendRealtimeEvent({
        type: 'input_audio_buffer.append',
        audio: bytesToBase64(pcm),
        media_type: 'audio/pcm',
      });
    }

    async function startPcmCapture() {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Browser microphone access is unavailable');
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) throw new Error('Browser audio processing is unavailable');

      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      captureContext = new AudioContextClass({ latencyHint: 'interactive' });
      await captureContext.resume();
      captureSource = captureContext.createMediaStreamSource(micStream);
      captureSink = captureContext.createGain();
      captureSink.gain.value = 0;
      captureSink.connect(captureContext.destination);

      if (captureContext.audioWorklet && typeof AudioWorkletNode !== 'undefined') {
        const source = `
          class AtlasPCMProcessor extends AudioWorkletProcessor {
            constructor() {
              super();
              this.buffer = new Float32Array(${CAPTURE_FRAME_SIZE});
              this.offset = 0;
            }
            process(inputs) {
              const input = inputs[0] && inputs[0][0];
              if (!input) return true;
              let cursor = 0;
              while (cursor < input.length) {
                const available = this.buffer.length - this.offset;
                const count = Math.min(available, input.length - cursor);
                this.buffer.set(input.subarray(cursor, cursor + count), this.offset);
                this.offset += count;
                cursor += count;
                if (this.offset === this.buffer.length) {
                  const completed = this.buffer;
                  this.port.postMessage(completed, [completed.buffer]);
                  this.buffer = new Float32Array(${CAPTURE_FRAME_SIZE});
                  this.offset = 0;
                }
              }
              return true;
            }
          }
          registerProcessor('atlas-pcm-processor', AtlasPCMProcessor);
        `;
        workletUrl = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
        await captureContext.audioWorklet.addModule(workletUrl);
        captureNode = new AudioWorkletNode(captureContext, 'atlas-pcm-processor');
        captureNode.port.onmessage = (event) => handleCaptureSamples(event.data, captureContext.sampleRate);
      } else {
        captureNode = captureContext.createScriptProcessor(CAPTURE_FRAME_SIZE, 1, 1);
        captureNode.onaudioprocess = (event) => {
          handleCaptureSamples(event.inputBuffer.getChannelData(0), captureContext.sampleRate);
        };
      }

      captureSource.connect(captureNode);
      captureNode.connect(captureSink);
      sendRealtimeEvent({
        type: 'session.update',
        input_audio_sample_rate: targetSampleRate,
        channels: 1,
        media_type: 'audio/pcm',
        source_context: sourceInput?.value || '',
        recording_id: recordingInput?.value || null,
      });
    }

    async function stopPcmCapture({ commit = true } = {}) {
      micEnabled = false;
      bubble.setLevel(0);
      try {
        captureSource?.disconnect();
        captureNode?.disconnect();
        captureSink?.disconnect();
      } catch (_error) {
        // Nodes may already be disconnected during browser teardown.
      }
      if (captureNode && 'onaudioprocess' in captureNode) captureNode.onaudioprocess = null;
      micStream?.getTracks().forEach((track) => track.stop());
      micStream = null;
      captureSource = null;
      captureNode = null;
      captureSink = null;
      if (captureContext) await captureContext.close().catch(() => {});
      captureContext = null;
      if (workletUrl) URL.revokeObjectURL(workletUrl);
      workletUrl = null;

      if (commit && micSpeechPending) sendRealtimeEvent({ type: 'input_audio_buffer.commit' });
      else sendRealtimeEvent({ type: 'input_audio_buffer.clear' });
      micSpeechPending = false;
    }

    function setUploadStatus(message, state = '') {
      if (!uploadStatus) return;
      uploadStatus.textContent = message || '';
      uploadStatus.dataset.state = message ? state : '';
    }

    function renderAttachments() {
      if (!attachmentTray) return;
      attachmentTray.replaceChildren();
      attachmentTray.hidden = selectedFiles.length === 0;
      const hasFiles = selectedFiles.length > 0;
      if (uploadOptions) uploadOptions.hidden = !hasFiles;
      form.classList.toggle('has-attachments', hasFiles);
      if (attachButton) {
        const count = selectedFiles.length;
        attachButton.classList.toggle('has-files', count > 0);
        attachButton.dataset.count = count > 9 ? '9+' : String(count || '');
        attachButton.setAttribute('aria-label', count ? `Add files, ${count} attached` : 'Attach audio');
        const submitLabel = count ? `Upload ${count} file${count === 1 ? '' : 's'}` : 'Send message';
        submit.setAttribute('aria-label', submitLabel);
        submit.title = submitLabel;
      }
      selectedFiles.forEach((file, index) => {
        const item = document.createElement('span');
        item.className = 'chat-attachment-chip';
        const icon = document.createElement('img');
        icon.className = 'ui-icon';
        icon.src = '/static/icons/paperclip.svg';
        icon.alt = '';
        const name = document.createElement('span');
        name.textContent = `${file.name} (${fileSizeLabel(file.size)})`;
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.setAttribute('aria-label', `Remove ${file.name}`);
        const removeIcon = document.createElement('img');
        removeIcon.className = 'ui-icon';
        removeIcon.src = '/static/icons/x.svg';
        removeIcon.alt = '';
        remove.append(removeIcon);
        remove.addEventListener('click', () => {
          selectedFiles.splice(index, 1);
          renderAttachments();
          setUploadStatus('');
          refreshPromptAvailability();
        });
        item.append(icon, name, remove);
        attachmentTray.append(item);
      });
    }

    function addSelectedFiles(files) {
      const nextFiles = Array.from(files || []);
      if (!nextFiles.length) return;
      selectedFiles = selectedFiles.concat(nextFiles);
      renderAttachments();
      setUploadStatus('');
      refreshPromptAvailability();
    }

    async function uploadSelectedFiles() {
      if (uploading || !selectedFiles.length) return;
      uploading = true;
      form.classList.add('is-uploading');
      form.setAttribute('aria-busy', 'true');
      refreshPromptAvailability();
      const files = selectedFiles.slice();
      appendMessage(transcript, 'You', files.length === 1 ? `Attached ${files[0].name}` : `Attached ${files.length} files`);
      try {
        const uploaded = [];
        for (const file of files) {
          setUploadStatus(`Uploading ${file.name}`, 'loading');
          const body = new FormData();
          body.append('file', file, file.name);
          body.append('expected_main_speakers', expectedSpeakers?.value || '');
          body.append('quality_tier', qualityTier?.value || 'torch');
          const response = await fetch(uploadEndpoint, { method: 'POST', headers: { accept: 'application/json' }, body });
          const contentType = response.headers.get('content-type') || '';
          const payload = contentType.includes('application/json') ? await response.json() : {};
          if (!response.ok) throw new Error(payload.detail || `Upload failed for ${file.name}`);
          uploaded.push(payload);
        }
        selectedFiles = [];
        renderAttachments();
        setUploadStatus(uploaded.length === 1 ? 'Added to pipeline' : `${uploaded.length} files added`, 'success');
        window.setTimeout(() => {
          if (uploadStatus?.dataset.state === 'success') setUploadStatus('');
        }, 3000);
        appendMessage(transcript, 'Atlas', uploaded.length === 1 ? `${files[0].name} is in the pipeline.` : `${uploaded.length} files are in the pipeline.`);
      } catch (error) {
        setUploadStatus(error.message || 'Upload failed', 'error');
        appendMessage(transcript, 'Atlas', error.message || 'Upload failed');
      } finally {
        uploading = false;
        form.classList.remove('is-uploading');
        form.setAttribute('aria-busy', 'false');
        refreshPromptAvailability();
      }
    }

    attachButton?.addEventListener('click', () => fileInput?.click());
    fileInput?.addEventListener('change', () => {
      addSelectedFiles(fileInput.files);
      fileInput.value = '';
    });

    ['dragenter', 'dragover'].forEach((eventName) => {
      form.addEventListener(eventName, (event) => {
        if (!Array.from(event.dataTransfer?.types || []).includes('Files')) return;
        event.preventDefault();
        form.classList.add('is-dragging');
      });
    });
    ['dragleave', 'drop'].forEach((eventName) => {
      form.addEventListener(eventName, (event) => {
        if (eventName === 'drop') {
          event.preventDefault();
          addSelectedFiles(event.dataTransfer?.files);
        }
        form.classList.remove('is-dragging');
      });
    });

    form.addEventListener('submit', (event) => {
      event.preventDefault();
      if (selectedFiles.length) {
        uploadSelectedFiles();
        return;
      }
      const text = input.value.trim();
      if (!text || !enabled || !callActive || input.disabled) return;
      appendMessage(transcript, 'You', text);
      input.value = '';
      resizePromptInput();
      refreshPromptAvailability();
      setVoiceState('thinking', text);
      sendRealtimeEvent({
        type: 'input_text',
        text,
        source_context: sourceInput?.value || '',
        recording_id: recordingInput?.value || null,
      });
    });

    const transport = root.querySelector('[data-voice-transport]');
    if (transport) {
      const controls = new Map(
        Array.from(transport.querySelectorAll('[data-transport-action]')).map((button) => [button.dataset.transportAction, button]),
      );
      const timer = transport.querySelector('[data-voice-timer]');
      const volume = transport.querySelector('[data-transport-volume]');
      const volumeOutput = transport.querySelector('[data-volume-output]');
      const explicitCallControls = controls.has('start-call');
      let sessionStartedAt = null;
      let timerId = null;

      const togglePressed = (button, pressed) => {
        button?.setAttribute('aria-pressed', pressed ? 'true' : 'false');
        button?.classList.toggle('active', pressed);
      };

      const formatElapsed = () => {
        if (!sessionStartedAt) return '00:00';
        const seconds = Math.max(Math.floor((Date.now() - sessionStartedAt) / 1000), 0);
        return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
      };

      const startTimer = () => {
        sessionStartedAt = Date.now();
        if (timer) timer.textContent = '00:00';
        timerId = window.setInterval(() => {
          if (timer) timer.textContent = formatElapsed();
        }, 1000);
      };

      const stopTimer = () => {
        if (timerId) window.clearInterval(timerId);
        timerId = null;
        sessionStartedAt = null;
        if (timer) timer.textContent = '00:00';
      };

      const setControlLabel = (button, label) => {
        if (!button) return;
        button.setAttribute('aria-label', label);
        button.title = label;
      };

      const markMicState = (active) => {
        micEnabled = active;
        const micButton = controls.get('mic');
        togglePressed(micButton, active);
        togglePressed(chatVoiceToggle, active);
        chatVoiceToggle?.classList.toggle('is-listening', active);
        setControlLabel(micButton, active ? 'Mute microphone' : 'Turn on microphone');
        if (chatVoiceToggle) {
          const label = active ? 'Stop voice input' : 'Start voice input';
          setControlLabel(chatVoiceToggle, label);
        }
      };

      const setControlAvailability = () => {
        const captureHeld = captureMode !== 'open' || captureTransitioning;
        controls.get('start-call')?.toggleAttribute('disabled', !enabled || callActive);
        controls.get('end-call')?.toggleAttribute('disabled', !enabled || !callActive);
        controls.get('mic')?.toggleAttribute('disabled', !enabled || !callActive || captureHeld);
        controls.get('pause')?.toggleAttribute('disabled', !enabled || !callActive || captureTransitioning);
        controls.get('private')?.toggleAttribute('disabled', !enabled || !callActive || captureTransitioning);
        ['interrupt', 'play'].forEach((action) => {
          controls.get(action)?.toggleAttribute('disabled', !enabled || !callActive);
        });
        if (volume) volume.disabled = !enabled || !callActive;
      };

      const setPromptAvailability = () => {
        const privateMode = captureMode === 'private';
        const blocked = !enabled || !callActive || privateMode;
        input.disabled = blocked;
        if (fileInput) fileInput.disabled = privateMode;
        submit.disabled = uploading || blocked || (
          selectedFiles.length === 0 && !input.value.trim()
        );
        if (attachButton) attachButton.disabled = uploading || privateMode;
        if (chatVoiceToggle) chatVoiceToggle.disabled = !enabled || uploading || captureMode !== 'open';
        if (sourceTrigger) sourceTrigger.disabled = uploading || privateMode;
        setControlAvailability();
      };
      refreshPromptAvailability = setPromptAvailability;

      const refreshCallClass = () => {
        root.classList.toggle('is-call-active', callActive);
        root.classList.toggle('is-call-idle', !callActive);
      };

      const startMicrophone = async () => {
        if (!callActive || captureMode !== 'open') return false;
        await startPcmCapture();
        markMicState(true);
        setConnection('listening');
        setPlaybackStatus('Listening');
        setVoiceState('listening', 'Ready when you are');
        setCaptureHelp('Microphone on. Pause keeps typing available; Private discards capture and blocks all input.');
        return true;
      };

      const updateCaptureControls = () => {
        const isPaused = captureMode === 'paused';
        const isPrivate = captureMode === 'private';
        togglePressed(controls.get('pause'), isPaused);
        togglePressed(controls.get('private'), isPrivate);
        setControlLabel(controls.get('pause'), isPaused ? 'Resume microphone' : 'Pause microphone');
        setControlLabel(controls.get('private'), isPrivate ? 'Leave private mode' : 'Go private');
        root.dataset.captureMode = callActive ? captureMode : 'idle';
      };

      const setCaptureMode = async (nextMode) => {
        if (!callActive || captureTransitioning) return;
        const leavingCurrentMode = captureMode === nextMode;
        captureTransitioning = true;
        if (!leavingCurrentMode) {
          resumeMicrophoneAfterCaptureMode = micEnabled || resumeMicrophoneAfterCaptureMode;
        }
        setPromptAvailability();
        try {
          if (leavingCurrentMode) {
            const shouldResume = resumeMicrophoneAfterCaptureMode;
            captureMode = 'open';
            resumeMicrophoneAfterCaptureMode = false;
            updateCaptureControls();
            if (shouldResume) {
              setConnection('resuming');
              setPlaybackStatus('Resuming microphone');
              setVoiceState('connecting', 'Turning the microphone back on');
              setCaptureHelp('Resuming the microphone...');
              try {
                await startMicrophone();
              } catch (error) {
                markMicState(false);
                setConnection('mic error');
                setPlaybackStatus('Microphone unavailable');
                setVoiceState('error', error.message || 'Microphone unavailable');
                setCaptureHelp('The microphone could not resume. Typing is still available.');
              }
            } else {
              restoreCapturePresentation();
            }
            return;
          }

          const wasListening = micEnabled;
          captureMode = nextMode;
          updateCaptureControls();
          if (wasListening) {
            markMicState(false);
            await stopPcmCapture({ commit: nextMode === 'paused' });
          } else if (nextMode === 'private') {
            sendRealtimeEvent({ type: 'input_audio_buffer.clear' });
            micSpeechPending = false;
          }
          if (nextMode === 'private') setSourceMenuOpen(false);
          restoreCapturePresentation();
        } finally {
          captureTransitioning = false;
          setPromptAvailability();
        }
      };

      const startCall = async () => {
        if (!enabled || callActive) return;
        callActive = true;
        callEnding = false;
        captureMode = 'open';
        resumeMicrophoneAfterCaptureMode = false;
        captureTransitioning = false;
        playbackEnabled = true;
        pendingEvents.length = 0;
        togglePressed(controls.get('play'), true);
        pendingVoiceProfile = null;
        renderVoiceProfileState();
        updateCaptureControls();
        refreshCallClass();
        setPromptAvailability();
        startTimer();
        setPlaybackStatus('Connecting');
        setCaptureHelp('Starting the microphone. Pause will keep typing available; Private will block all input.');
        await bubble.attachOutput(audio).catch(() => {});
        connect();
        try {
          await startMicrophone();
        } catch (error) {
          markMicState(false);
          setConnection('mic error');
          setPlaybackStatus('Microphone unavailable');
          setVoiceState('error', error.message || 'Microphone unavailable');
          setCaptureHelp('The microphone is unavailable. You can still use typed messages.');
          appendMessage(transcript, 'Atlas', error.message || 'Microphone unavailable');
        }
      };

      const endCall = async ({ fromSocket = false } = {}) => {
        if (!callActive && !fromSocket) return;
        const unexpected = fromSocket && !callEnding;
        callEnding = true;
        pendingEvents.length = 0;
        if (micStream || captureContext) await stopPcmCapture({ commit: false });
        markMicState(false);
        if (!fromSocket && socket?.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ type: 'response.cancel', reason: 'call_ended' }));
          socket.close(1000, 'call ended');
        } else if (!fromSocket && socket?.readyState === WebSocket.CONNECTING) {
          socket.close(1000, 'call ended');
        }
        socket = null;
        callActive = false;
        captureMode = 'open';
        pendingVoiceProfile = null;
        renderVoiceProfileState();
        resumeMicrophoneAfterCaptureMode = false;
        captureTransitioning = false;
        playbackEnabled = false;
        assistantText = null;
        clearPlayback();
        stopTimer();
        ['mic', 'interrupt', 'play'].forEach((action) => togglePressed(controls.get(action), false));
        updateCaptureControls();
        setConnection(unexpected ? 'offline' : 'ready');
        setPlaybackStatus(unexpected ? 'Disconnected' : 'Call idle');
        setVoiceState(unexpected ? 'error' : 'idle', unexpected ? 'Local realtime disconnected' : 'Ready when you are');
        setCaptureHelp(unexpected ? 'The local call disconnected.' : 'Start a call to use the microphone.');
        refreshCallClass();
        setPromptAvailability();
        window.setTimeout(() => { callEnding = false; }, 0);
      };
      handleSocketClosed = () => { endCall({ fromSocket: true }); };

      controls.get('start-call')?.addEventListener('click', () => { startCall(); });
      controls.get('end-call')?.addEventListener('click', () => { endCall(); });

      controls.get('mic')?.addEventListener('click', async () => {
        if (!callActive || captureMode !== 'open') return;
        if (micEnabled) {
          markMicState(false);
          await stopPcmCapture({ commit: true });
          restoreCapturePresentation();
        } else {
          try {
            await startMicrophone();
          } catch (error) {
            markMicState(false);
            setConnection('mic error');
            setPlaybackStatus('Microphone unavailable');
            setVoiceState('error', error.message || 'Microphone unavailable');
            setCaptureHelp('The microphone is unavailable. Typed messages are still available.');
          }
        }
        setPromptAvailability();
      });

      controls.get('pause')?.addEventListener('click', async () => {
        await setCaptureMode('paused');
      });

      controls.get('private')?.addEventListener('click', async () => {
        await setCaptureMode('private');
      });

      controls.get('interrupt')?.addEventListener('click', () => {
        if (!callActive) return;
        if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: 'response.cancel', reason: 'client_interrupt' }));
        clearPlayback();
        assistantText = null;
        if (captureMode === 'open') {
          setPlaybackStatus('Interrupted');
          setVoiceState(micEnabled ? 'listening' : 'idle', micEnabled ? 'Listening...' : 'Response stopped');
        } else {
          restoreCapturePresentation();
        }
        togglePressed(controls.get('interrupt'), true);
        window.setTimeout(() => togglePressed(controls.get('interrupt'), false), 220);
      });

      controls.get('play')?.addEventListener('click', () => {
        if (!callActive) return;
        const button = controls.get('play');
        playbackEnabled = button?.getAttribute('aria-pressed') !== 'true';
        togglePressed(button, playbackEnabled);
        setPlaybackStatus(playbackEnabled ? 'Audio on' : 'Audio muted');
        if (!playbackEnabled) audio?.pause();
        else if (audio && currentAudioUrl && audio.paused) audio.play().catch(() => {});
        else playNextAudio();
      });

      volume?.addEventListener('input', () => {
        if (volumeOutput) volumeOutput.textContent = volume.value;
        if (audio) audio.volume = Number(volume.value) / 100;
      });
      if (audio && volume) audio.volume = Number(volume.value) / 100;

      chatVoiceToggle?.addEventListener('click', () => {
        if (!callActive) startCall();
        else controls.get('mic')?.click();
      });

      if (!explicitCallControls && enabled) {
        callActive = true;
        playbackEnabled = true;
        refreshCallClass();
        connect();
      }
      updateCaptureControls();
      refreshCallClass();
      setPromptAvailability();
    }

    const voiceTabs = Array.from(root.querySelectorAll('[role="tab"][data-voice-view]'));
    const activateVoiceView = (button, { focus = false } = {}) => {
      if (!button || !voiceTabs.includes(button)) return;
      root.dataset.activeView = button.dataset.voiceView || 'live';
      voiceTabs.forEach((item) => {
        const selected = item === button;
        item.setAttribute('aria-selected', selected ? 'true' : 'false');
        item.tabIndex = selected ? 0 : -1;
        const panelId = item.getAttribute('aria-controls');
        const panel = panelId ? document.getElementById(panelId) : null;
        if (panel) panel.hidden = !selected;
      });
      if (focus) button.focus();
    };

    voiceTabs.forEach((button) => {
      button.addEventListener('click', () => activateVoiceView(button));
    });

    root.querySelector('[role="tablist"]')?.addEventListener('keydown', (event) => {
      const currentTab = event.target.closest?.('[role="tab"]');
      const currentIndex = voiceTabs.indexOf(currentTab);
      if (currentIndex < 0) return;
      let nextIndex;
      if (event.key === 'ArrowRight') nextIndex = (currentIndex + 1) % voiceTabs.length;
      else if (event.key === 'ArrowLeft') nextIndex = (currentIndex - 1 + voiceTabs.length) % voiceTabs.length;
      else if (event.key === 'Home') nextIndex = 0;
      else if (event.key === 'End') nextIndex = voiceTabs.length - 1;
      else return;
      event.preventDefault();
      activateVoiceView(voiceTabs[nextIndex], { focus: true });
    });

    const initialVoiceTab = voiceTabs.find((button) => button.getAttribute('aria-selected') === 'true') || voiceTabs[0];
    if (initialVoiceTab) activateVoiceView(initialVoiceTab);

    const setupJsonPlayground = (selector, inputSelector, resultSelector, buildRequest, formatResult) => {
      const section = document.querySelector(selector);
      const playgroundForm = section?.querySelector('form');
      const playgroundInput = section?.querySelector(inputSelector);
      const result = section?.querySelector(resultSelector);
      if (!section || !playgroundForm || !playgroundInput || !result) return;
      playgroundForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const request = buildRequest(playgroundInput);
        if (!request) return;
        result.textContent = 'running';
        try {
          const response = await fetch(section.dataset.endpoint, request);
          const payload = await response.json();
          if (!response.ok) throw new Error(payload.detail || 'Request failed');
          result.textContent = formatResult(payload);
        } catch (error) {
          result.textContent = error.message || 'Request failed';
        }
      });
    };

    setupJsonPlayground(
      '[data-tts-playground]',
      'input[name="tts_text"]',
      '[data-tts-playground-result]',
      (field) => field.value.trim() ? { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text: field.value.trim() }) } : null,
      (payload) => `${payload.provider} / ${payload.latency_ms ?? '?'} ms`,
    );
    setupJsonPlayground(
      '[data-model-playground]',
      'input[name="model_text"]',
      '[data-model-playground-result]',
      (field) => field.value.trim() ? { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text: field.value.trim() }) } : null,
      (payload) => `${payload.text || '(empty)'} (${payload.latency_ms ?? '?'} ms)`,
    );

    const sttSection = document.querySelector('[data-stt-playground]');
    const sttForm = sttSection?.querySelector('form');
    sttForm?.addEventListener('submit', async (event) => {
      event.preventDefault();
      const field = sttForm.querySelector('input[name="stt_audio"]');
      const result = sttSection.querySelector('[data-stt-playground-result]');
      const file = field?.files?.[0];
      if (!file || !result) return;
      result.textContent = 'transcribing';
      const body = new FormData();
      body.append('file', file);
      try {
        const response = await fetch(sttSection.dataset.endpoint, { method: 'POST', body });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || 'STT failed');
        result.textContent = `${payload.text || '(empty)'} (${payload.latency_ms ?? '?'} ms)`;
      } catch (error) {
        result.textContent = error.message || 'STT failed';
      }
    });

    setVoiceState(enabled ? 'idle' : 'error', enabled ? 'Ready when you are' : 'Enable the assistant in settings');
  });
})();
