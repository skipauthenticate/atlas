(() => {
  function appendMessage(transcript, role, text) {
    const item = document.createElement('div');
    item.className = `voice-message ${role === 'Atlas' ? 'assistant' : 'user'}`;

    const label = document.createElement('span');
    label.textContent = role;

    const body = document.createElement('p');
    body.textContent = text;

    item.append(label, body);
    transcript.append(item);
    item.scrollIntoView({ block: 'nearest' });
    return body;
  }

  function setConnection(label, state, className) {
    if (!label) return;
    label.textContent = state;
    label.className = `status ${className}`;
    label.setAttribute('data-voice-connection', '');
  }

  function websocketUrl(path) {
    const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${scheme}//${window.location.host}${path}`;
  }

  function audioUrlFromDelta(delta, mediaType) {
    const binary = atob(delta || '');
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    const blob = new Blob([bytes], { type: mediaType || 'audio/wav' });
    return URL.createObjectURL(blob);
  }

  function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.addEventListener('load', () => {
        const value = String(reader.result || '');
        resolve(value.includes(',') ? value.split(',', 2)[1] : value);
      });
      reader.addEventListener('error', () => reject(reader.error || new Error('audio read failed')));
      reader.readAsDataURL(blob);
    });
  }

  function browserRecorderMimeType() {
    const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/ogg'];
    return candidates.find((type) => MediaRecorder.isTypeSupported?.(type)) || '';
  }

  window.addEventListener('DOMContentLoaded', () => {
    const root = document.querySelector('[data-voice-console="true"]');
    if (!root) return;

    const enabled = root.dataset.assistantEnabled === 'true';
    const socketPath = root.dataset.websocketPath || '/v1/realtime';
    const form = root.querySelector('[data-voice-prompt]');
    const input = document.getElementById('voice-prompt-input');
    const submit = document.getElementById('voice-prompt-submit');
    const transcript = root.querySelector('[data-transcript-stream]');
    const connection = root.querySelector('[data-voice-connection]');
    const audio = root.querySelector('[data-response-audio]');
    const playbackStatus = root.querySelector('[data-playback-status]');
    const pending = [];
    const playbackQueue = [];
    let socket = null;
    let assistantText = null;
    let playbackEnabled = false;
    let currentAudioUrl = null;

    if (!form || !input || !submit || !transcript) return;

    function setPlaybackStatus(state) {
      if (playbackStatus) playbackStatus.textContent = state;
    }

    function revokeCurrentAudioUrl() {
      if (!currentAudioUrl) return;
      URL.revokeObjectURL(currentAudioUrl);
      currentAudioUrl = null;
    }

    function playNextAudio() {
      if (!audio || !playbackEnabled || currentAudioUrl || playbackQueue.length === 0) return;
      currentAudioUrl = playbackQueue.shift();
      audio.src = currentAudioUrl;
      setPlaybackStatus('playing assistant audio');
      audio.play().catch(() => {
        setPlaybackStatus('playback blocked');
        revokeCurrentAudioUrl();
      });
    }

    audio?.addEventListener('ended', () => {
      revokeCurrentAudioUrl();
      setPlaybackStatus(playbackQueue.length ? 'audio queued' : 'playback idle');
      playNextAudio();
    });

    audio?.addEventListener('error', () => {
      revokeCurrentAudioUrl();
      setPlaybackStatus('playback error');
      playNextAudio();
    });

    function connect() {
      if (!enabled) return null;
      if (socket && socket.readyState <= WebSocket.OPEN) return socket;

      socket = new WebSocket(websocketUrl(socketPath));
      setConnection(connection, 'connecting', 'status-queued');

      socket.addEventListener('open', () => {
        setConnection(connection, 'online', 'status-done');
        while (pending.length) socket.send(JSON.stringify(pending.shift()));
      });

      socket.addEventListener('message', (event) => {
        let payload;
        try {
          payload = JSON.parse(event.data);
        } catch (_error) {
          return;
        }

        if (payload.type === 'response.created') {
          assistantText = appendMessage(transcript, 'Atlas', '');
          return;
        }
        if (payload.type === 'response.text.delta' && assistantText) {
          assistantText.textContent += payload.delta || '';
          return;
        }
        if (payload.type === 'response.text.done' && assistantText) {
          assistantText.textContent = payload.text || assistantText.textContent;
          return;
        }
        if (payload.type === 'response.tool_call.created') {
          const name = payload.tool_call?.name || 'local tool';
          appendMessage(transcript, 'Atlas', `Tool call requested: ${name}`);
          return;
        }
        if (payload.type === 'response.tool_call.requires_confirmation') {
          const name = payload.tool_call?.name || 'local tool';
          appendMessage(transcript, 'Atlas', `Tool confirmation required: ${name}`);
          return;
        }
        if (payload.type === 'response.audio.delta' && payload.delta) {
          playbackQueue.push(audioUrlFromDelta(payload.delta, payload.media_type));
          setPlaybackStatus('audio queued');
          playNextAudio();
          return;
        }
        if (payload.type === 'response.audio.done') {
          if (payload.status === 'skipped') setPlaybackStatus('no assistant audio');
          else if (!currentAudioUrl && playbackQueue.length === 0) setPlaybackStatus('audio ready');
          return;
        }
        if (payload.type === 'response.interrupted' || payload.type === 'response.cancelled') {
          playbackQueue.length = 0;
          audio?.pause();
          revokeCurrentAudioUrl();
          setPlaybackStatus(payload.type === 'response.cancelled' ? 'playback cancelled' : 'playback interrupted');
          return;
        }
        if (payload.type === 'response.done') {
          assistantText = null;
          return;
        }
        if (payload.type === 'error') {
          appendMessage(transcript, 'Atlas', payload.error?.message || 'Realtime error');
        }
      });

      socket.addEventListener('close', () => {
        setConnection(connection, 'offline', 'status-queued');
        socket = null;
      });

      socket.addEventListener('error', () => {
        setConnection(connection, 'error', 'status-failed');
      });

      return socket;
    }

    function sendRealtimeEvent(message) {
      const activeSocket = connect();
      if (activeSocket && activeSocket.readyState === WebSocket.OPEN) {
        activeSocket.send(JSON.stringify(message));
      } else {
        pending.push(message);
      }
    }

    form.addEventListener('submit', (event) => {
      event.preventDefault();
      const text = input.value.trim();
      if (!text || !enabled) return;

      appendMessage(transcript, 'User', text);
      input.value = '';
      sendRealtimeEvent({ type: 'input_text', text });
    });

    const transport = root.querySelector('[data-voice-transport]');
    if (transport) {
      const controls = new Map(
        Array.from(transport.querySelectorAll('[data-transport-action]')).map((button) => [
          button.dataset.transportAction,
          button,
        ]),
      );
      const timer = transport.querySelector('[data-voice-timer]');
      const volume = transport.querySelector('[data-transport-volume]');
      const volumeOutput = transport.querySelector('[data-volume-output]');
      let sessionStartedAt = null;
      let timerId = null;
      let paused = false;
      let privateMode = false;
      let micEnabled = false;
      let micStream = null;
      let mediaRecorder = null;
      let micChunkReads = 0;
      let micStopPending = false;

      const formatElapsed = () => {
        if (!sessionStartedAt) return '00:00';
        const seconds = Math.max(Math.floor((Date.now() - sessionStartedAt) / 1000), 0);
        const minutes = String(Math.floor(seconds / 60)).padStart(2, '0');
        const remainder = String(seconds % 60).padStart(2, '0');
        return `${minutes}:${remainder}`;
      };

      const startTimer = () => {
        if (!sessionStartedAt) sessionStartedAt = Date.now();
        if (timerId) return;
        timerId = window.setInterval(() => {
          if (timer) timer.textContent = formatElapsed();
        }, 1000);
      };

      const stopTimer = () => {
        if (timerId) window.clearInterval(timerId);
        timerId = null;
      };

      const togglePressed = (button, pressed) => {
        button?.setAttribute('aria-pressed', pressed ? 'true' : 'false');
        button?.classList.toggle('active', pressed);
      };

      const setPromptAvailability = () => {
        const blocked = !enabled || paused || privateMode;
        input.disabled = blocked;
        submit.disabled = blocked;
      };

      const stopMicTracks = () => {
        micStream?.getTracks().forEach((track) => track.stop());
        micStream = null;
      };

      const commitMicAudio = () => {
        if (micChunkReads > 0) {
          micStopPending = true;
          return;
        }
        micStopPending = false;
        sendRealtimeEvent({ type: 'input_audio_buffer.commit' });
      };

      const sendMicChunk = async (blob) => {
        if (!blob || blob.size === 0) return;
        micChunkReads += 1;
        try {
          const audio = await blobToBase64(blob);
          sendRealtimeEvent({
            type: 'input_audio_buffer.append',
            audio,
            media_type: blob.type || mediaRecorder?.mimeType || 'audio/webm',
          });
        } catch (error) {
          appendMessage(transcript, 'Atlas', error.message || 'Microphone audio read failed');
        } finally {
          micChunkReads -= 1;
          if (micStopPending && micChunkReads === 0) commitMicAudio();
        }
      };

      const startMicStreaming = async () => {
        if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
          throw new Error('Browser microphone recording is unavailable');
        }
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mimeType = browserRecorderMimeType();
        mediaRecorder = new MediaRecorder(micStream, mimeType ? { mimeType } : undefined);
        mediaRecorder.addEventListener('dataavailable', (event) => {
          sendMicChunk(event.data);
        });
        mediaRecorder.addEventListener('stop', () => {
          stopMicTracks();
          commitMicAudio();
        });
        mediaRecorder.start(750);
        sendRealtimeEvent({ type: 'session.update', media_type: mediaRecorder.mimeType || 'audio/webm' });
      };

      const stopMicStreaming = () => {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
          mediaRecorder.stop();
        } else {
          stopMicTracks();
        }
        mediaRecorder = null;
      };

      controls.get('mic')?.addEventListener('click', async () => {
        if (!micEnabled) {
          try {
            connect();
            await startMicStreaming();
            micEnabled = true;
            togglePressed(controls.get('mic'), true);
            setConnection(connection, 'listening', 'status-done');
            startTimer();
          } catch (error) {
            micEnabled = false;
            togglePressed(controls.get('mic'), false);
            stopMicTracks();
            setConnection(connection, 'mic error', 'status-failed');
            appendMessage(transcript, 'Atlas', error.message || 'Microphone unavailable');
          }
        } else {
          micEnabled = false;
          togglePressed(controls.get('mic'), false);
          stopMicStreaming();
          stopTimer();
        }
      });

      controls.get('pause')?.addEventListener('click', () => {
        paused = !paused;
        togglePressed(controls.get('pause'), paused);
        setPromptAvailability();
        setConnection(connection, paused ? 'paused' : 'ready', paused ? 'status-queued' : 'status-done');
      });

      controls.get('private')?.addEventListener('click', () => {
        privateMode = !privateMode;
        togglePressed(controls.get('private'), privateMode);
        setPromptAvailability();
        setConnection(connection, privateMode ? 'private' : 'ready', privateMode ? 'status-queued' : 'status-done');
      });

      controls.get('interrupt')?.addEventListener('click', () => {
        pending.length = 0;
        if (socket && socket.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ type: 'response.cancel' }));
        }
        assistantText = null;
        togglePressed(controls.get('interrupt'), true);
        window.setTimeout(() => togglePressed(controls.get('interrupt'), false), 250);
      });

      controls.get('play')?.addEventListener('click', () => {
        const button = controls.get('play');
        playbackEnabled = button?.getAttribute('aria-pressed') !== 'true';
        togglePressed(button, playbackEnabled);
        setPlaybackStatus(playbackEnabled ? 'playback enabled' : 'playback paused');
        if (!playbackEnabled) {
          audio?.pause();
          return;
        }
        if (audio && currentAudioUrl && audio.paused) {
          audio.play().catch(() => setPlaybackStatus('playback blocked'));
          return;
        }
        playNextAudio();
      });

      volume?.addEventListener('input', () => {
        if (volumeOutput) volumeOutput.textContent = volume.value;
        if (audio) audio.volume = Number(volume.value) / 100;
      });

      if (audio && volume) audio.volume = Number(volume.value) / 100;
    }

    const playground = root.querySelector('[data-tts-playground]');
    if (playground) {
      const playgroundForm = playground.querySelector('form');
      const playgroundInput = playground.querySelector('input[name="tts_text"]');
      const playgroundResult = playground.querySelector('[data-tts-playground-result]');
      const endpoint = playground.dataset.endpoint || '/api/voice/playground/tts';

      playgroundForm?.addEventListener('submit', async (event) => {
        event.preventDefault();
        const text = playgroundInput?.value.trim();
        if (!text || !playgroundResult) return;
        playgroundResult.textContent = 'running';
        try {
          const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ text }),
          });
          const payload = await response.json();
          if (!response.ok) throw new Error(payload.detail || 'TTS failed');
          const latency = payload.latency_ms == null ? 'latency unknown' : `${payload.latency_ms} ms`;
          playgroundResult.textContent = `${payload.provider} ${latency}`;
        } catch (error) {
          playgroundResult.textContent = error.message || 'TTS failed';
        }
      });
    }

    const sttPlayground = root.querySelector('[data-stt-playground]');
    if (sttPlayground) {
      const sttForm = sttPlayground.querySelector('form');
      const sttInput = sttPlayground.querySelector('input[name="stt_audio"]');
      const sttResult = sttPlayground.querySelector('[data-stt-playground-result]');
      const endpoint = sttPlayground.dataset.endpoint || '/api/voice/playground/stt';

      sttForm?.addEventListener('submit', async (event) => {
        event.preventDefault();
        const file = sttInput?.files?.[0];
        if (!file || !sttResult) return;
        sttResult.textContent = 'transcribing';
        const body = new FormData();
        body.append('file', file);
        try {
          const response = await fetch(endpoint, { method: 'POST', body });
          const payload = await response.json();
          if (!response.ok) throw new Error(payload.detail || 'STT failed');
          const latency = payload.latency_ms == null ? 'latency unknown' : `${payload.latency_ms} ms`;
          sttResult.textContent = `${payload.text || '(empty transcript)'} · ${latency}`;
        } catch (error) {
          sttResult.textContent = error.message || 'STT failed';
        }
      });
    }

    const modelPlayground = root.querySelector('[data-model-playground]');
    if (modelPlayground) {
      const modelForm = modelPlayground.querySelector('form');
      const modelInput = modelPlayground.querySelector('input[name="model_text"]');
      const modelResult = modelPlayground.querySelector('[data-model-playground-result]');
      const endpoint = modelPlayground.dataset.endpoint || '/api/voice/playground/model';

      modelForm?.addEventListener('submit', async (event) => {
        event.preventDefault();
        const text = modelInput?.value.trim();
        if (!text || !modelResult) return;
        modelResult.textContent = 'running';
        try {
          const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ text }),
          });
          const payload = await response.json();
          if (!response.ok) throw new Error(payload.detail || 'Model failed');
          const latency = payload.latency_ms == null ? 'latency unknown' : `${payload.latency_ms} ms`;
          modelResult.textContent = `${payload.text || '(empty response)'} · ${latency}`;
        } catch (error) {
          modelResult.textContent = error.message || 'Model failed';
        }
      });
    }
  });
})();
