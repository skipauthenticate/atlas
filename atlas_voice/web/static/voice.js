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
    const pending = [];
    let socket = null;
    let assistantText = null;

    if (!form || !input || !submit || !transcript) return;

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

    form.addEventListener('submit', (event) => {
      event.preventDefault();
      const text = input.value.trim();
      if (!text || !enabled) return;

      appendMessage(transcript, 'User', text);
      input.value = '';
      const message = { type: 'input_text', text };
      const activeSocket = connect();
      if (activeSocket && activeSocket.readyState === WebSocket.OPEN) {
        activeSocket.send(JSON.stringify(message));
      } else {
        pending.push(message);
      }
    });

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
  });
})();
