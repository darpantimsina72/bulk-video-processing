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
