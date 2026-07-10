# Atlas Voice Design System

Status: foundation for all current and new Atlas Voice interfaces.

Atlas Voice uses a native CSS design system. It preserves the existing monochrome interface, coral voice accent, system sans type, local icon family, and 6px or 8px geometry. It is intentionally quiet so the product can explain complex audio work in ordinary language.

## Design read

This is a preservation-focused product interface for people who should not need AI or model knowledge.

- Design variance: 3
- Motion intensity: 2
- Visual density: 6
- Theme: light
- Foundation: semantic CSS tokens plus existing local SVG icons
- Redesign mode: targeted evolution

The interface should feel dependable, compact, and easy to scan. Decoration never competes with state or content.

## Source files and load order

1. `atlas_voice/web/static/app.css` loads the existing application layout and feature-specific composition.
2. `atlas_voice/web/static/design-system.css` loads after it and defines semantic tokens plus shared component contracts.
3. Page-specific styles may consume the tokens but must not redefine the palette or geometry.

During the legacy migration, the design-system stylesheet is the final shared layer. Its compatibility aliases keep older selectors working while new work adopts the `--atlas-*` tokens directly.

## Principles

### Explain the outcome

Primary surfaces say what Atlas is doing and what the person can do next. Model names, providers, quantization, and hardware detail belong in an advanced disclosure or diagnostics view.

Prefer:

- "Atlas heard 3 voices."
- "Names update the transcript and notes."
- "Processing your recording."
- "Needs attention."

Avoid:

- "Diarization returned 3 clusters."
- "Running an inference pass."
- "Provider exception."
- "Pipeline terminal state."

### One accent, used with purpose

Coral is reserved for live voice activity, current attention, and a small number of active-state accents. Primary actions remain near-black. Coral is not used for long text or as a white-text button background.

### Geometry communicates family

- Controls and compact labels use 6px corners.
- Panels, dialogs, fields, and grouped choices use 8px corners.
- Fully round geometry is limited to avatars, real status indicators, and native circular controls.
- Borders and spacing usually establish grouping. Elevation is reserved for overlays and true layers.

### State is never color alone

Every state includes plain text. Border color, fill, or an icon may reinforce the state, but never replace its label.

## Tokens

### Color

| Token | Role |
| --- | --- |
| `--atlas-color-canvas` | Main quiet page background |
| `--atlas-color-canvas-muted` | Secondary application background |
| `--atlas-color-surface` | Primary surface and input background |
| `--atlas-color-surface-muted` | Selected, grouped, or subdued surface |
| `--atlas-color-surface-pressed` | Pressed and compact hover surface |
| `--atlas-color-text` | Primary text |
| `--atlas-color-text-secondary` | Supporting text with readable contrast |
| `--atlas-color-text-muted` | Metadata and tertiary labels |
| `--atlas-color-border` | Default separator |
| `--atlas-color-border-strong` | Selected or structural separator |
| `--atlas-color-action` | Primary action background |
| `--atlas-color-action-hover` | Primary action hover background |
| `--atlas-color-action-text` | Text on a primary action |
| `--atlas-color-accent` | Coral voice and attention accent |
| `--atlas-color-focus` | Keyboard focus outline |
| `--atlas-color-success` | Confirmed success |
| `--atlas-color-warning` | Caution that can be resolved |
| `--atlas-color-danger` | Failure or destructive warning |
| `--atlas-color-backdrop` | Dialog backdrop |

Use semantic state colors only when the state also has a visible label.

### Type

The default stack is `--atlas-font-sans`:

`ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`

Use `--atlas-font-mono` only for machine identifiers, timestamps, and technical values. Do not use it to make ordinary copy look technical.

Available sizes:

- `--atlas-type-2xs`: compact metadata
- `--atlas-type-xs`: helper and status copy
- `--atlas-type-sm`: field labels and secondary copy
- `--atlas-type-control`: buttons and compact titles
- `--atlas-type-body`: standard reading copy
- `--atlas-type-title-sm`: small section title
- `--atlas-type-title`: panel and dialog title
- `--atlas-type-display`: page title

Use `--atlas-leading-body` for interface copy and `--atlas-leading-reading` for transcripts or notes.

### Space

The spacing scale is:

`0, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64px`

Use the corresponding `--atlas-space-*` token. Most control gaps use 6px or 8px. Most panel padding uses 16px, 20px, or 24px.

### Radius

| Token | Value | Use |
| --- | --- | --- |
| `--atlas-radius-control` | 6px | Buttons, inputs, compact labels |
| `--atlas-radius-panel` | 8px | Panels, dialogs, grouped choices |
| `--atlas-radius-round` | 999px | Avatars and genuinely round controls |

### Elevation

| Token | Use |
| --- | --- |
| `--atlas-elevation-0` | Default flat layout |
| `--atlas-elevation-1` | Raised popover or contextual panel |
| `--atlas-elevation-2` | Strong popover |
| `--atlas-elevation-dialog` | Modal dialog |

Do not add shadows to every section. A border or spacing break is the default.

### Controls

| Token | Value | Use |
| --- | --- | --- |
| `--atlas-control-compact` | 32px | Dense secondary control |
| `--atlas-control-default` | 38px | Standard button |
| `--atlas-control-field` | 40px | Input or select |
| `--atlas-control-touch` | 44px | Choice row or spacious touch target |
| `--atlas-control-icon` | 36px | Icon-only button |

Button labels stay on one line at desktop. Shorten the label or allow more width instead of wrapping it.

### Motion

Atlas uses motion only for feedback and state transition.

- `--atlas-motion-fast`: 120ms
- `--atlas-motion-standard`: 180ms
- `--atlas-motion-slow`: 240ms
- `--atlas-ease-standard`: routine interaction
- `--atlas-ease-enter`: a panel entering view

Do not add perpetual decoration. All motion collapses to effectively instant when reduced motion is requested.

### Layering

| Token | Intended layer |
| --- | --- |
| `--atlas-z-base` | Document content |
| `--atlas-z-raised` | Raised content within a component |
| `--atlas-z-sticky` | Sticky toolbar or footer |
| `--atlas-z-popover` | Menu or popover |
| `--atlas-z-overlay` | Page overlay |
| `--atlas-z-dialog` | Dialog content |
| `--atlas-z-toast` | Short-lived status above overlays |

Native dialogs use the browser top layer. The scale still applies to content inside them.

## Compatibility aliases

Existing application CSS continues to use aliases including:

- `--bg`, `--bg-soft`
- `--surface`, `--surface-raised`, `--surface-muted`
- `--ink`, `--muted`, `--subtle`
- `--line`, `--line-strong`
- `--accent`, `--accent-strong`, `--accent-soft`, `--accent-wash`
- `--voice-accent`
- `--warn`, `--fail`, `--done`, `--info`
- `--shadow-sm`, `--shadow-md`
- `--radius`, `--radius-sm`

New components should use semantic `--atlas-*` tokens. Compatibility aliases exist for gradual migration, not as the preferred API.

## Processing levels

Always present processing levels in this order:

1. Light
2. Torch
3. Fire

Accuracy defines the ranking. Speed does not.

### Light

Label: "Good accuracy"

Description: "Finishes sooner and uses the least device memory."

Use Light for a quick, dependable pass. Never describe it as bad, weak, cheap, or low quality.

### Torch

Label: "More accurate"

Description: "The recommended balance for everyday recordings."

Torch is the default recommendation when one profile must be preselected.

### Fire

Label: "Most accurate"

Description: "Runs the strongest checks and may take considerably longer."

Fire must be the highest measured accuracy option. A larger or newer model does not qualify by itself. Do not promise speed.

### Presentation rules

- Use the exact names Light, Torch, and Fire.
- Keep technical model detail out of the choice row.
- Show one short promise, then one short practical description.
- Do not use traffic-light colors to imply good or bad.
- The selected level uses the same neutral selection treatment for all three choices.
- If local benchmarks cannot maintain the accuracy order, do not expose the tier labels until the order is corrected.

## Plain-language states

### Background capture

| Internal state | Primary label | Supporting copy |
| --- | --- | --- |
| ambient | Listening now | Speech is transcribed locally. |
| paused | Paused | Capture is stopped. You can resume quickly. |
| private | Private. Microphone off. | Nothing new is captured or transcribed. |
| inactive | Not running | Your saved choice applies when the local listener starts. |

"Listening" means capture is active. "Paused" means capture stops but the listener remains ready. "Private" means the microphone is off and nothing new is accepted.

### Recording work

| Internal state | Display label |
| --- | --- |
| queued | Waiting to start |
| processing | Processing |
| done | Ready |
| failed | Needs attention |
| resummarizing | Updating notes |
| saving | Saving |
| saved | Saved |
| connecting | Connecting |
| resuming | Resuming microphone |
| offline | Not running |

A failure message states what could not happen and, when possible, the next action. Avoid raw exception text on a primary surface.

### Product vocabulary

| Technical term | Primary interface term |
| --- | --- |
| diarization | Separate voices or speaker analysis |
| cluster | Voice |
| inference | Processing |
| model tier | Processing level |
| template | Note style |
| terminal state | Ready or Needs attention |
| speaker-count constraint | Main speaker hint |
| unknown speaker count | Atlas decides or Not sure |

Advanced settings may show technical terms after the plain-language label.

## Component patterns

### Processing level choice

Class: `.quality-choice`

Use native radio inputs inside full-row labels. The entire row is clickable. The selected state has a strong border, a subtle inset marker, and the native checked control.

A settings version may also use `.settings-quality-choice` for tighter padding.

Use `.document-quality` when the selected processing level needs a compact read-only summary.

### Voice mode bar

Class: `.voice-profile-bar`

Keep the Voice mode picker in its own row outside the Live call and Transcript tab list. Each choice shows the shared mode icon, exact mode name, and a short promise. Every choice keeps a minimum 44px target at all viewport widths.

Use the same neutral border, surface, and inset marker for Light, Torch, and Fire. The icons identify the modes, but color never ranks them.

The adjacent live status uses one of these plain-language states:

- "[Mode] selected for next call"
- "Switching to [Mode]"
- "[Mode] active"

Disable additional mode requests while a switch is pending. Keep raw model names and paths out of the bar; diagnostics belong in Voice Settings.

### Capture mode explanation

Class: `.capture-mode-explanation`

Pair it with one of:

- `.capture-ambient`
- `.capture-paused`
- `.capture-private`

The block always contains a short strong label and one sentence of explanation. Selection color reinforces the mode, while the copy communicates whether the listener is actually running.

### Recording upload dialog

Class: `.recording-upload-dialog`

The dialog contains one task:

1. Choose a recording.
2. Give a main-speaker hint or choose Not sure.
3. Choose a processing level.
4. Confirm with "Add recording."

The speaker count is explicitly a hint. Atlas may retain additional coherent voices.

Use a native `dialog`, an accessible title through `aria-labelledby`, a visible Cancel action, and an icon-only close button with an accessible name.

### Note-style confirmation

Class: `.template-confirmation`

Changing a note style never starts work immediately. Show the pending style, explain that the transcript stays unchanged, and require an explicit "Use style" action. "Keep current" restores the prior selection.

The block uses `hidden` when inactive and an `aria-live` status for progress or failure.

### People editor

Class: `.recording-people`

Use one `.speaker-name-row` per detected voice. A numbered `.speaker-swatch` is a functional identity marker. Each row contains:

- Main speaker or Additional voice label
- Editable display name
- Speaking duration
- Optional Listen action

Saving names updates the transcript immediately and refreshes notes. Internal speaker identifiers remain hidden from primary copy.

### Inline status

Class: `.inline-status`

Empty status elements are hidden. When text is present, the status is visually grouped with its component.

Optional state hooks:

- `data-state="loading"`
- `data-state="success"`
- `data-state="warning"`
- `data-state="error"`

Use `aria-live="polite"` for asynchronous progress that does not require immediate interruption. Use assertive announcements only for urgent safety or data-loss conditions.

### Grouped search

Classes: `.search-main-form`, `.search-scope`, `.search-group`, and `.search-match`

Search starts with one prominent input and a short Search action. Scope choices use ordinary product terms: Everything, Notes, Transcript, and People.

Results are grouped by recording. Each group shows the recording title and status once, then lists matching notes, transcript text, or people. This removes repeated metadata and makes the result set easier to scan.

- Highlight only the matching words with `mark`.
- Keep the entire match row as one link to its recording.
- Preserve visible keyboard focus on scope and result links.
- Let snippets wrap naturally and keep source labels compact.
- Empty states explain that exact titles are not required.
- On small screens, match source and snippet stack into one column.

## Accessibility

### Contrast

- Body and control text meet WCAG AA contrast.
- Coral is not used for small body text on white.
- Supporting gray text stays readable against white and muted surfaces.
- Disabled controls remain legible and are also programmatically disabled.

### Keyboard

- Every action uses a native button, link, input, select, summary, or dialog control.
- Visible focus is a 2px near-black outline with separation from the component edge.
- Choice rows keep the native radio in the tab order.
- Dialog focus returns to the opener when the current behavior supports it.
- Hidden panels use the `hidden` attribute or equivalent programmatic state.

### Forms

- Labels appear above or beside their inputs. Placeholder text is never the only label.
- Helper text explains consequences before submission.
- Error text appears near the relevant field or action.
- Disabled controls cannot be the only explanation of why an action is unavailable.

### Touch and responsive layout

- Prefer a 44px target where space permits.
- Compact 32px controls are reserved for dense secondary actions.
- Multi-column forms collapse below 680px.
- Primary actions remain visible without horizontal scrolling.

### Motion

- Motion communicates feedback or a state change.
- Reduced-motion preferences are honored.
- Avoid auto-playing, perpetual, and decorative animation.

### Status and sound

- Never rely on sound, color, or a moving waveform alone.
- Live capture state has visible text.
- Status changes use an appropriate live region.
- Audio preview controls have visible or accessible names.

## Icons

Atlas uses the existing local Lucide SVG family in `atlas_voice/web/static/icons`.

- Default size: 16px
- Compact size: 14px
- Prominent utility size: 20px to 24px
- Decorative icons use empty alt text.
- Icon-only controls require `aria-label` and usually `title`.
- Do not mix icon families or draw new SVG paths for routine actions.

## New feature checklist

Before shipping a new interface:

1. Start with an existing component pattern or document why a new one is needed.
2. Use semantic `--atlas-*` tokens.
3. Keep the monochrome palette and one coral accent.
4. Use 6px controls and 8px containers.
5. Write primary copy without model or AI knowledge assumptions.
6. Provide loading, empty, success, failure, and disabled states where relevant.
7. Verify keyboard order, visible focus, labels, live regions, and contrast.
8. Collapse multi-column layout for small screens.
9. Honor reduced motion.
10. Keep Light, Torch, and Fire in strict measured-accuracy order.
11. Use the existing icon family.
12. Run a source diff check and focused UI tests.
