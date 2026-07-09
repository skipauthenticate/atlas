(() => {
  const DEFAULT_SAMPLE_RATE = 24000;
  const CAPTURE_FRAME_SIZE = 4096;

  function websocketUrl(path) {
    const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${scheme}//${window.location.host}${path}`;
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

  function insertRecordingRow(payload) {
    const list = document.querySelector('[data-dashboard-recording-list]');
    if (!list || !payload?.recording_id) return;
    document.querySelector('[data-recordings-empty]')?.remove();
    list.hidden = false;

    const row = document.createElement('a');
    row.className = 'recording-row';
    row.href = payload.url || `/recordings/${payload.recording_id}`;
    const copy = document.createElement('span');
    const title = document.createElement('strong');
    title.textContent = payload.title || 'Uploaded recording';
    const created = document.createElement('small');
    created.textContent = payload.created_at_display || 'just now';
    copy.append(title, created);
    const status = document.createElement('span');
    status.className = `status status-${payload.status || 'queued'}`;
    status.textContent = payload.status || 'queued';
    row.append(copy, status);
    list.prepend(row);
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
    const chatVoiceToggle = root.querySelector('[data-chat-voice-toggle]');
    const audio = root.querySelector('[data-response-audio]');
    const playbackStatus = root.querySelector('[data-playback-status]');
    const bubble = new VoiceBubble(root.querySelector('[data-voice-bubble]'), stage);

    if (!form || !input || !submit || !transcript) return;

    const fileInput = form.querySelector('[data-chat-file-input]');
    const attachButton = form.querySelector('[data-chat-attach]');
    const attachmentTray = form.querySelector('[data-chat-attachments]');
    const uploadStatus = form.querySelector('[data-chat-upload-status]');
    const uploadEndpoint = form.dataset.uploadEndpoint || '/upload';
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
    let paused = false;
    let privateMode = false;
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
        setPlaybackStatus(micEnabled ? 'Listening' : 'In call');
        setVoiceState(micEnabled ? 'listening' : 'idle', micEnabled ? 'Ready when you are' : 'Microphone is muted');
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
      socket = new WebSocket(websocketUrl(socketPath));
      setConnection('connecting');
      setVoiceState('connecting', 'Opening a local session');

      socket.addEventListener('open', () => {
        setConnection('online');
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
          setVoiceState(micEnabled ? 'listening' : 'idle', micEnabled ? 'Ready when you are' : 'Microphone is muted');
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
          setVoiceState('synthesizing', `Warming ${payload.model || 'voice'}`);
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
            setPlaybackStatus('Text only');
            setVoiceState(micEnabled ? 'listening' : 'idle', 'Ready when you are');
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
          setPlaybackStatus('Interrupted');
          setVoiceState(micEnabled ? 'listening' : 'idle', micEnabled ? 'Listening...' : 'Response stopped');
          return;
        }
        if (payload.type === 'response.failed') {
          assistantText = null;
          setVoiceState('error', payload.error?.message || 'The local model did not respond');
          return;
        }
        if (payload.type === 'response.done') {
          assistantText = null;
          if (!currentAudioUrl && playbackQueue.length === 0 && currentVoiceState !== 'speaking') {
            setVoiceState(micEnabled ? 'listening' : 'idle', 'Ready when you are');
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
      if (!micEnabled || !callActive || paused || privateMode || !samples?.length) return;
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

    function setUploadStatus(message) {
      if (uploadStatus) uploadStatus.textContent = message || '';
    }

    function renderAttachments() {
      if (!attachmentTray) return;
      attachmentTray.replaceChildren();
      attachmentTray.hidden = selectedFiles.length === 0;
      selectedFiles.forEach((file, index) => {
        const item = document.createElement('span');
        item.className = 'chat-attachment-chip';
        const name = document.createElement('span');
        name.textContent = `${file.name} (${fileSizeLabel(file.size)})`;
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.setAttribute('aria-label', `Remove ${file.name}`);
        remove.textContent = 'Remove';
        remove.addEventListener('click', () => {
          selectedFiles.splice(index, 1);
          renderAttachments();
          refreshPromptAvailability();
        });
        item.append(name, remove);
        attachmentTray.append(item);
      });
    }

    function addSelectedFiles(files) {
      const nextFiles = Array.from(files || []);
      if (!nextFiles.length) return;
      selectedFiles = selectedFiles.concat(nextFiles);
      renderAttachments();
      setUploadStatus(selectedFiles.length === 1 ? '1 file ready' : `${selectedFiles.length} files ready`);
      refreshPromptAvailability();
    }

    async function uploadSelectedFiles() {
      if (uploading || !selectedFiles.length) return;
      uploading = true;
      refreshPromptAvailability();
      const files = selectedFiles.slice();
      appendMessage(transcript, 'You', files.length === 1 ? `Attached ${files[0].name}` : `Attached ${files.length} files`);
      try {
        const uploaded = [];
        for (const file of files) {
          setUploadStatus(`Uploading ${file.name}`);
          const body = new FormData();
          body.append('file', file, file.name);
          const response = await fetch(uploadEndpoint, { method: 'POST', headers: { accept: 'application/json' }, body });
          const contentType = response.headers.get('content-type') || '';
          const payload = contentType.includes('application/json') ? await response.json() : {};
          if (!response.ok) throw new Error(payload.detail || `Upload failed for ${file.name}`);
          uploaded.push(payload);
          insertRecordingRow(payload);
        }
        selectedFiles = [];
        renderAttachments();
        setUploadStatus(uploaded.length === 1 ? 'Added to pipeline' : `${uploaded.length} files added`);
        appendMessage(transcript, 'Atlas', uploaded.length === 1 ? `${files[0].name} is in the pipeline.` : `${uploaded.length} files are in the pipeline.`);
      } catch (error) {
        setUploadStatus(error.message || 'Upload failed');
        appendMessage(transcript, 'Atlas', error.message || 'Upload failed');
      } finally {
        uploading = false;
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
      setVoiceState('thinking', text);
      sendRealtimeEvent({ type: 'input_text', text });
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

      const markMicState = (active) => {
        micEnabled = active;
        togglePressed(controls.get('mic'), active);
        togglePressed(chatVoiceToggle, active);
        chatVoiceToggle?.classList.toggle('is-listening', active);
      };

      const setControlAvailability = () => {
        controls.get('start-call')?.toggleAttribute('disabled', !enabled || callActive);
        controls.get('end-call')?.toggleAttribute('disabled', !enabled || !callActive);
        ['mic', 'pause', 'private', 'interrupt', 'play'].forEach((action) => {
          controls.get(action)?.toggleAttribute('disabled', !enabled || !callActive);
        });
        if (volume) volume.disabled = !enabled || !callActive;
      };

      const setPromptAvailability = () => {
        const blocked = !enabled || !callActive || paused || privateMode;
        input.disabled = blocked;
        submit.disabled = uploading || (selectedFiles.length === 0 && blocked);
        if (attachButton) attachButton.disabled = uploading;
        if (chatVoiceToggle) chatVoiceToggle.disabled = !enabled || uploading;
        setControlAvailability();
      };
      refreshPromptAvailability = setPromptAvailability;

      const refreshCallClass = () => {
        root.classList.toggle('is-call-active', callActive);
        root.classList.toggle('is-call-idle', !callActive);
      };

      const startMicrophone = async () => {
        await startPcmCapture();
        markMicState(true);
        setConnection('listening');
        setPlaybackStatus('Listening');
        setVoiceState('listening', 'Ready when you are');
      };

      const startCall = async () => {
        if (!enabled || callActive) return;
        callActive = true;
        callEnding = false;
        paused = false;
        privateMode = false;
        playbackEnabled = true;
        pendingEvents.length = 0;
        togglePressed(controls.get('play'), true);
        togglePressed(controls.get('pause'), false);
        togglePressed(controls.get('private'), false);
        refreshCallClass();
        setPromptAvailability();
        startTimer();
        setPlaybackStatus('Connecting');
        await bubble.attachOutput(audio).catch(() => {});
        connect();
        try {
          await startMicrophone();
        } catch (error) {
          markMicState(false);
          setConnection('mic error');
          setVoiceState('error', error.message || 'Microphone unavailable');
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
        paused = false;
        privateMode = false;
        playbackEnabled = false;
        assistantText = null;
        clearPlayback();
        stopTimer();
        ['mic', 'pause', 'private', 'interrupt', 'play'].forEach((action) => togglePressed(controls.get(action), false));
        setConnection(unexpected ? 'offline' : 'ready');
        setPlaybackStatus(unexpected ? 'Disconnected' : 'Call idle');
        setVoiceState(unexpected ? 'error' : 'idle', unexpected ? 'Local realtime disconnected' : 'Ready when you are');
        refreshCallClass();
        setPromptAvailability();
        window.setTimeout(() => { callEnding = false; }, 0);
      };
      handleSocketClosed = () => { endCall({ fromSocket: true }); };

      controls.get('start-call')?.addEventListener('click', () => { startCall(); });
      controls.get('end-call')?.addEventListener('click', () => { endCall(); });

      controls.get('mic')?.addEventListener('click', async () => {
        if (!callActive) return;
        if (micEnabled) {
          markMicState(false);
          await stopPcmCapture({ commit: true });
          setConnection('online');
          setVoiceState('idle', 'Microphone is muted');
        } else {
          try {
            await startMicrophone();
          } catch (error) {
            markMicState(false);
            setVoiceState('error', error.message || 'Microphone unavailable');
          }
        }
        setPromptAvailability();
      });

      controls.get('pause')?.addEventListener('click', async () => {
        if (!callActive) return;
        paused = !paused;
        if (paused && micEnabled) {
          markMicState(false);
          await stopPcmCapture({ commit: false });
        }
        togglePressed(controls.get('pause'), paused);
        setConnection(paused ? 'paused' : 'online');
        setVoiceState(paused ? 'paused' : 'idle', paused ? 'Call paused' : 'Microphone is muted');
        setPromptAvailability();
      });

      controls.get('private')?.addEventListener('click', async () => {
        if (!callActive) return;
        privateMode = !privateMode;
        if (privateMode && micEnabled) {
          markMicState(false);
          await stopPcmCapture({ commit: false });
        }
        togglePressed(controls.get('private'), privateMode);
        setConnection(privateMode ? 'private' : 'online');
        setVoiceState(privateMode ? 'private' : 'idle', privateMode ? 'Capture is off' : 'Microphone is muted');
        setPromptAvailability();
      });

      controls.get('interrupt')?.addEventListener('click', () => {
        if (!callActive) return;
        if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: 'response.cancel', reason: 'client_interrupt' }));
        clearPlayback();
        assistantText = null;
        setPlaybackStatus('Interrupted');
        setVoiceState(micEnabled ? 'listening' : 'idle', micEnabled ? 'Listening...' : 'Response stopped');
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
      refreshCallClass();
      setPromptAvailability();
    }

    root.querySelectorAll('[data-voice-view]').forEach((button) => {
      button.addEventListener('click', () => {
        const view = button.dataset.voiceView || 'live';
        root.dataset.activeView = view;
        root.querySelectorAll('[data-voice-view]').forEach((item) => {
          item.setAttribute('aria-selected', item === button ? 'true' : 'false');
        });
      });
    });

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
