(() => {
  'use strict';

  const timecode = (value) => {
    const total = Math.max(Math.floor(Number(value) || 0), 0);
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const seconds = total % 60;
    if (hours) {
      return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  };

  const node = (tag, className, text) => {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text !== undefined) item.textContent = text;
    return item;
  };

  const setInlineStatus = (element, text, state = "") => {
    if (!element) return;
    element.textContent = text;
    if (state) element.dataset.state = state;
    else delete element.dataset.state;
  };

  const renderTranscript = (details, segments) => {
    const container = details.querySelector('[data-transcript-list]');
    if (!container) return;
    if (!segments.length) {
      container.replaceChildren(node('p', 'transcript-load-state', 'No transcript is available yet.'));
      return;
    }

    const contextLayout = details.dataset.transcriptLayout === 'context';
    const audio = document.getElementById(details.dataset.audioTarget || '');
    const fragment = document.createDocumentFragment();
    segments.forEach((segment) => {
      const article = node(
        'article',
        contextLayout ? 'context-transcript-row' : 'recording-transcript-row',
      );
      const meta = node('div');
      meta.append(node('strong', '', segment.speaker || 'Speaker'));
      if (contextLayout) {
        const timestamp = node('time', '', timecode(segment.start));
        timestamp.dateTime = `PT${Math.max(Number(segment.start) || 0, 0)}S`;
        meta.append(timestamp);
      } else {
        const seek = node('button', 'transcript-time', timecode(segment.start));
        seek.type = 'button';
        seek.addEventListener('click', () => {
          if (!audio) return;
          audio.currentTime = Math.max(Number(segment.start) || 0, 0);
          audio.play().catch(() => {});
        });
        meta.append(seek);
      }
      article.append(meta, node('p', '', segment.text || ''));
      fragment.append(article);
    });
    container.replaceChildren(fragment);
  };

  const loadTranscript = async (details) => {
    if (details.dataset.transcriptLoaded === 'true' || details.dataset.transcriptLoading === 'true') return;
    const endpoint = details.dataset.transcriptEndpoint;
    const container = details.querySelector('[data-transcript-list]');
    if (!endpoint || !container) return;
    details.dataset.transcriptLoading = 'true';
    details.setAttribute('aria-busy', 'true');
    container.replaceChildren(node('p', 'transcript-load-state', 'Loading transcript...'));
    try {
      const response = await fetch(endpoint, { headers: { accept: 'application/json' } });
      if (!response.ok) throw new Error('Transcript request failed');
      const payload = await response.json();
      renderTranscript(details, Array.isArray(payload.segments) ? payload.segments : []);
      details.dataset.transcriptLoaded = 'true';
    } catch (_error) {
      container.replaceChildren(
        node('p', 'transcript-load-state transcript-load-error', 'Transcript could not be loaded.'),
      );
    } finally {
      details.dataset.transcriptLoading = 'false';
      details.removeAttribute('aria-busy');
    }
  };

  const startStatusPoll = (statusNote) => {
    const endpoint = statusNote.dataset.statusEndpoint;
    if (!endpoint) return;
    let stopped = false;

    const canReloadWithoutInterrupting = () => {
      const active = document.activeElement;
      const userHasFocus = active && active !== document.body && active !== document.documentElement;
      const audio = document.getElementById('recording-audio');
      const audioIsPlaying = audio && !audio.paused;
      const expandedSection = document.querySelector('.recording-document details[open]');
      return !userHasFocus && !audioIsPlaying && !expandedSection;
    };

    const offerRefresh = (payload) => {
      stopped = true;
      const spinner = statusNote.querySelector('.processing-spinner');
      if (spinner) spinner.remove();
      const title = statusNote.querySelector('strong');
      const description = statusNote.querySelector('p');
      if (title) title.textContent = payload.status === 'failed' ? 'Processing needs attention' : 'Processing complete';
      if (description) description.textContent = 'Refresh when you are ready to view the latest recording.';
      if (canReloadWithoutInterrupting()) {
        window.location.reload();
        return;
      }
      if (!statusNote.querySelector('[data-refresh-recording]')) {
        const refresh = node('button', 'quiet-button', 'Refresh notes');
        refresh.type = 'button';
        refresh.dataset.refreshRecording = 'true';
        refresh.addEventListener('click', () => window.location.reload());
        statusNote.append(refresh);
      }
    };

    const poll = async () => {
      if (stopped) return;
      try {
        const response = await fetch(endpoint, {
          headers: { accept: 'application/json' },
          cache: 'no-store',
        });
        if (!response.ok) throw new Error('Status request failed');
        const payload = await response.json();
        if (payload.terminal) {
          offerRefresh(payload);
          return;
        }
      } catch (_error) {
        // A transient polling error should not replace the useful processing state.
      }
      window.setTimeout(poll, 5000);
    };

    window.setTimeout(poll, 5000);
  };

  const setupUploadDialog = () => {
    const dialog = document.querySelector('[data-upload-dialog]');
    const opener = document.querySelector('[data-upload-dialog-open]');
    const file = dialog?.querySelector('[data-upload-file]');
    const fileName = dialog?.querySelector('[data-upload-file-name]');
    if (!dialog || !opener) return;

    opener.addEventListener('click', () => {
      if (typeof dialog.showModal === 'function') dialog.showModal();
      else dialog.setAttribute('open', '');
    });
    dialog.querySelectorAll('[data-upload-dialog-close]').forEach((button) => {
      button.addEventListener('click', () => {
        if (typeof dialog.close === 'function') dialog.close();
        else dialog.removeAttribute('open');
      });
    });
    dialog.addEventListener('click', (event) => {
      if (event.target === dialog && typeof dialog.close === 'function') dialog.close();
    });
    file?.addEventListener('change', () => {
      if (fileName) fileName.textContent = file.files?.[0]?.name || 'Choose an audio or video file.';
    });
  };

  const setupTemplateConfirmation = () => {
    const select = document.querySelector('[data-template-select]');
    const panel = document.querySelector('[data-template-confirmation]');
    const title = panel?.querySelector('[data-template-confirm-title]');
    const copy = panel?.querySelector('[data-template-confirm-copy]');
    const status = panel?.querySelector('[data-template-status]');
    const apply = panel?.querySelector('[data-template-apply]');
    const cancel = panel?.querySelector('[data-template-cancel]');
    const description = document.querySelector('[data-template-description]');
    if (!select || !panel || !apply || !cancel) return;

    let current = select.dataset.currentTemplate || select.value;
    const selectedOption = () => select.options[select.selectedIndex];
    const refreshDescription = () => {
      if (description) description.textContent = selectedOption()?.dataset.description || '';
    };
    const close = ({ restore = false } = {}) => {
      if (restore) select.value = current;
      panel.hidden = true;
      setInlineStatus(status, "");
      refreshDescription();
    };

    select.addEventListener('change', () => {
      refreshDescription();
      if (select.value === current) {
        close();
        return;
      }
      const option = selectedOption();
      if (title) title.textContent = 'Use ' + (option?.textContent.trim() || 'this style') + '?';
      if (copy) {
        copy.textContent = 'Your transcript stays unchanged. Atlas will only update the notes.';
      }
      panel.hidden = false;
      apply.focus();
    });

    cancel.addEventListener('click', () => {
      close({ restore: true });
      select.focus();
    });

    apply.addEventListener('click', async () => {
      const templateId = select.value;
      if (!templateId || templateId === current) {
        close();
        return;
      }
      apply.disabled = true;
      cancel.disabled = true;
      select.disabled = true;
      setInlineStatus(status, "Updating notes...", "loading");
      try {
        const response = await fetch(select.dataset.endpoint, {
          method: 'POST',
          headers: { 'content-type': 'application/json', accept: 'application/json' },
          body: JSON.stringify({ template_id: templateId, confirmed: true }),
        });
        const payload = await response.json();
        if (!response.ok || payload.error) throw new Error(payload.error || 'Template update failed');
        current = templateId;
        select.dataset.currentTemplate = current;
        setInlineStatus(status, payload.message || "Note style saved.", "success");
        window.setTimeout(() => window.location.reload(), 500);
      } catch (error) {
        setInlineStatus(status, error.message || "Note style could not be changed.", "error");
        apply.disabled = false;
        cancel.disabled = false;
        select.disabled = false;
      }
    });

    refreshDescription();
  };

  const setupSpeakerEditor = () => {
    const editor = document.querySelector('[data-speaker-editor]');
    const form = editor?.querySelector('[data-speaker-name-form]');
    const status = editor?.querySelector('[data-speaker-status]');
    const audio = document.getElementById('recording-audio');
    if (!editor) return;

    editor.querySelectorAll('[data-speaker-seek]').forEach((button) => {
      button.addEventListener('click', () => {
        if (!audio) return;
        audio.currentTime = Math.max(Number(button.dataset.speakerSeek) || 0, 0);
        audio.play().catch(() => {});
      });
    });

    form?.addEventListener('submit', async (event) => {
      event.preventDefault();
      const submit = form.querySelector('button[type="submit"]');
      const fields = Array.from(form.querySelectorAll('[data-speaker-name]'));
      const names = Object.fromEntries(
        fields.map((field) => [field.dataset.speakerLabel || '', field.value.trim()]),
      );
      if (submit) submit.disabled = true;
      fields.forEach((field) => { field.disabled = true; });
      setInlineStatus(status, "Saving names...", "loading");
      try {
        const response = await fetch(editor.dataset.endpoint, {
          method: 'POST',
          headers: { 'content-type': 'application/json', accept: 'application/json' },
          body: JSON.stringify({ names }),
        });
        const payload = await response.json();
        if (!response.ok || payload.error) throw new Error(payload.error || 'Names could not be saved');
        setInlineStatus(status, payload.message || "Names saved.", "success");
        window.setTimeout(() => window.location.reload(), payload.queued ? 700 : 350);
      } catch (error) {
        setInlineStatus(status, error.message || "Names could not be saved.", "error");
        if (submit) submit.disabled = false;
        fields.forEach((field) => { field.disabled = false; });
      }
    });
  };

  window.addEventListener('DOMContentLoaded', () => {
    setupUploadDialog();
    setupTemplateConfirmation();
    setupSpeakerEditor();
    document.querySelectorAll('[data-lazy-transcript]').forEach((details) => {
      details.addEventListener('toggle', () => {
        if (details.open) loadTranscript(details);
      });
      if (details.open) loadTranscript(details);
    });
    document.querySelectorAll('[data-recording-status-poll]').forEach(startStatusPoll);
  });
})();
