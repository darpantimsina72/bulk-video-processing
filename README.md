# Bulk Video Processing — Translation & Syncing App

Desktop app (Python/Tkinter) for the end-to-end audio dubbing pipeline:
transcription → translation → review → punctuation → TTS → syncing.
Works on **Windows** and **macOS**.

---

## Get the app

Click the green **Code** button on this GitHub page → **Download ZIP**,
then unzip it somewhere simple:

- Windows: `C:\Apps\BulkVideoProcessing` (short path — avoids "path too long" errors)
- macOS: `~/Applications/BulkVideoProcessing` or anywhere you like

*(If you have git: `git clone` works too.)*

## Windows setup (one time)

1. Double-click **`setup_windows.bat`** and follow the messages.
   It finds/checks Python, builds the environment, installs everything,
   and even installs FFmpeg automatically via winget.
2. If Python isn't installed yet, the script opens the download page for you —
   during install, **tick "Add python.exe to PATH"**, then run the script again.
3. Done. Launch the app with **`run_windows.bat`**.

> If SmartScreen warns about the .bat file: click *More info → Run anyway*.

## macOS setup (one time)

1. Double-click **`setup_mac.command`**.
   It finds/installs Python, Tkinter, all dependencies, and FFmpeg (via Homebrew).
2. Done. Launch the app with **`run_app.command`**.

> If macOS blocks the file ("unidentified developer"): right-click → Open → Open.
> If it says the file isn't executable, run once in Terminal:
> `chmod +x setup_mac.command run_app.command`

## First-run configuration

1. **LLM provider** — click **⚙ LLM Settings** in the app. Choose one:
   - Vertex AI (service-account JSON file)
   - Gemini API key
   - OpenAI-compatible base URL (LiteLLM proxy, OpenRouter, …)

   Use *Test Connection* to verify, then Save.
2. **ElevenLabs key** — paste it in the API Key box in the TTS Settings panel.
3. **Language** — pick it in the *Language* dropdown (Bengali, Hindi, Kannada,
   Malayalam, Tamil, Telugu, Gujarati, Marathi, Assamese, Odia, Nepali).
4. **Prompts** — click **📝 Edit Prompts** to adapt the five prompt stages
   per language (Translation, Review, Punctuation, Emotion tags, Syncing).

Keys and settings are stored **only on your computer**
(`api.txt`, `llm_settings.json`, `TTS_Key.json`, `vertex_key.json`) —
they are never uploaded to GitHub.

## Using the app — project workflow

The app opens to the **Projects screen** (like DaVinci Resolve):

- **➕ New Project** → name it, pick the English audio + target language →
  the workspace opens with everything pre-loaded.
- **Double-click an old project** → continue exactly where you left off.
  All old runs appear here automatically.

Inside the workspace, the **stage bar at the bottom** tracks the process:

```
①  Setup & Run   →   ②  Transcript   →   ③  Translation   →   ④  Result
```

- Press **▶ Run Pipeline** once (stage ①) — stages turn ✓ as they finish.
- **② Transcript** — the English SRT, side by side with
  **③ Translation** — editable; *💾 Save Translation* keeps versions,
  *🔁 Re-Dub with this text* re-voices without paying for re-translation.
- **④ Result** — English waveform on top, dubbed output below;
  **▶ English / ▶ Dubbed** to listen and compare.
- **⌂ Projects** (bottom-right) returns to the project screen.
  Everything is saved automatically — close the app any time.

All options live in one **⚙ Settings** window (no more Simple/Advanced
modes): TTS voice & key, region tuning, the LLM provider row and
prompt-chain options. Untick **Auto-open result** to disable the automatic
jump to ④ Result when a run finishes.

### Reaper-style mixer (v1.8.0)

The Compare view is now a small multitrack editor, modelled on REAPER:

- **▶ PLAY ALL** (or **space**) plays every track together as one mix, with
  a single shared playhead across all tracks. Click anywhere to seek — the
  mix restarts from there while playing.
- **Track strip** above the waveforms: per-track **M**ute, **S**olo
  (Reaper rules: any solo silences the rest), **volume** (0–150 %) and
  **pan** sliders — all react live during playback.
- **＋ Track** imports extra tracks — audio *or video* files (music beds,
  reference videos; audio is extracted via ffmpeg). Drag the item left/right
  to position it on the timeline, ✕ in the strip removes it.
- **⤓ Mix** renders the current mix (mute/solo/volume/pan honoured) to
  `<project>_mix.wav`.
- Mixer state and extra tracks persist per project in
  `<project>_tracks.json` and are restored on reopen.

### Compare view (v1.7.0)

- **Scroll left/right** with the scrollbar under the waveforms, the ◀ ▶
  buttons, or the mouse wheel; **ctrl+wheel** zooms.
- **Drag a dub chunk** left/right (Reaper-style) to nudge its timing — on
  release the synced audio and SRT are rebuilt at the new position. A
  one-time `_backup.wav` of the original synced audio is kept next to it.
- Reopening an old project now restores the waveform, stage ticks and the
  Compare view automatically.

### 🎭 Multi-speaker dubbing (v1.7.0)

Click **🎭 Speakers** (next to ▶ Run Pipeline) after the translation exists.
Each paragraph of the translated script gets a voice dropdown — leave it on
*default voice* or pick any ElevenLabs voice per paragraph (e.g. narrator vs
interviewee). Saved per project (`<base>_speakers.json`) and honoured by
▶ Run Pipeline and 🔁 Re-Dub.

### Better syncing (v1.7.0)

Sections that don't fit their English slot are no longer force-fit/pushed to
odd positions: they start exactly at their English start and bleed into the
following gap at natural pace, and an order-preserving sweep guarantees
chunks never overlap. Fine-tune by dragging chunks in Compare.

## Updating the app

Click **⟳ Check for Updates** (top row in the Translation tab).

- The app compares its version with this GitHub repository.
- If a newer version exists, it **asks you first** — nothing updates automatically.
- Your API keys and settings are never touched.
- Every file that gets replaced is backed up to `_update_backup/<timestamp>/`.
- Restart the app after updating.

Maintainers: to publish an update, push your changes **and bump the number in
the `VERSION` file** (and `APP_VERSION` in `Translation_and_Syncing_App.py`) —
the update button triggers only when `VERSION` on GitHub is higher.

## Sending feedback from the app

Click **💬 Send Feedback** (top row in the Translation tab). Pick a type
(Feedback / Improvement / Bug), write your message, optionally attach
screenshot files (PNG/JPG, up to 20 MB each), and hit **Send**. The report
lands with the developer as a GitHub issue, with screenshots linked.

If GitHub can't be reached (no internet, or no `github_token.txt`), the
feedback is saved to `feedback_outbox/<timestamp>/` next to the app — send
that folder to the developer by e-mail or chat instead.

Maintainers — one-time setup:

1. Create a **private** GitHub repo for feedback (default:
   `darpantimsina72/app-feedback` — change `FEEDBACK_REPO` in
   `Translation_and_Syncing_App.py` if you use another name). Initialize it
   with a README so it has a default branch.
2. Create a **fine-grained personal access token** scoped to *only that
   repo*, with **Issues: Read & write** and **Contents: Read & write**.
   This token cannot touch the app repo, so installs can never alter the
   code the updater pulls.
3. Put the token in a `github_token.txt` file next to the app on each
   install (same file the updater uses for private repos).
4. Watch the feedback repo (GitHub → Watch → All activity) to get an e-mail
   for every new report. Screenshots are committed to its `feedback`
   branch under `feedback_attachments/`.

## Translation memory (feedback loop)

The app remembers translations a human has reviewed and reuses them:

- When you click **✔ Continue to Dubbing** in the review window (normal run
  or re-dub), the reviewed script is saved to a local memory
  (`data/translation_memory.db`).
- Next time the **same English content** is processed (single or batch), the
  proofed script is reused directly — **no LLM call, zero cost** — and the
  status bar shows a 🧠 memory hit.
- When only **parts** of the content were seen before, the approved
  translations are injected into the translation prompt so wording stays
  consistent with what reviewers approved.

Controlled by the **Translation memory** checkbox in the Translation tab
(on by default). The memory is per-computer and never uploaded.
To temporarily bypass reuse without losing capture, launch with the
environment variable `TM_REUSE_DISABLE=1`.

## Where output files go

Next to your input audio file, in a new subfolder named after the file.
The final synced audio ends with `_synced`.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Python not found" in setup | Install Python 3.11+ from python.org, tick **Add to PATH**, re-run setup |
| MP3 won't load / export fails | FFmpeg missing — Windows: `winget install ffmpeg` then restart PC; macOS: `brew install ffmpeg` |
| "FFmpeg Not Found" warning at start | Same as above — the app also finds ffmpeg placed in an `ffmpeg/bin/` folder next to the app |
| App crashes / odd error dialog | Details are appended to `error_log.txt` next to the app — send that file when reporting problems |
| "Path too long" on Windows | Move the folder to a short path like `C:\Apps\BulkVideoProcessing` and re-run setup |
| Setup fails midway | All setup scripts are safe to re-run |
| Update check fails | Needs internet access to github.com; corporate proxies with SSL interception are handled automatically |
