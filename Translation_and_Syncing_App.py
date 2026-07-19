"""
End-to-End Multilingual Audio Dubbing Pipeline
==============================================
Integrates Translation, TTS, and Audio Sync into a single automated flow.
The target language is selected from the Language dropdown — see
TTS_LANGUAGES for the full list (Bengali, Hindi, Kannada, Malayalam,
Assamese, Odia, Nepali, Tamil, Telugu, Gujarati, Marathi).

Stage 1 – Translation
  • Detect regions → ElevenLabs transcription → SRT → Vertex AI (translate / review / punctuate)
  • Saves:  <base>.srt   and   <base>_FinalScript.txt

Stage 2 – TTS
  • Converts the target-language translation text to audio via ElevenLabs
    (default; eleven_v3 auto-detects the script) or Google Cloud TTS
    (per-language locale, e.g. bn-IN, hi-IN, ta-IN, …)
  • Saves:  <base>_tts.mp3  (same folder as the source audio)

Stage 3 – Sync
  • Transcribes original English audio → English SRT
  • Transcribes TTS target-language audio → target SRT
  • Calls Gemini to map the two SRTs (using the translation as script)
  • Syncs target SRT to English timing
  • Cuts and reassembles TTS audio → Saves:  synced_<base>_tts.mp3

ElevenLabs API key is entered in the UI (top of the TTS Settings panel). Once
a valid key is pasted, voices that advertise support for the current language
are fetched and surfaced in the dropdown. The key is also persisted to
api.txt so it is reused across sessions. The most recently used language is
saved to last_language.txt.

Works in Single File and Batch modes.
All steps after file selection are FULLY AUTOMATIC — no further button clicks required.

Required files (same folder as this script):
    api.txt                       — Optional cached ElevenLabs API key (auto-managed)
    last_language.txt             — Last-used target language (auto-managed)
    vertex_key.json               — Vertex AI / Google Cloud service account JSON
    TTS_Key.json                  — Google Cloud TTS service account JSON
                                    (can be the same file as vertex_key.json)
    prompts/SyncingPrompt_<Language>.txt
    prompts/Step1_Translation_Prompt_<Language>.txt
    prompts/Step2_Review_Prompt_<Language>.txt
    prompts/Step3_Punctuation_Prompt_<Language>.txt

pip dependencies:
    librosa matplotlib numpy sounddevice google-genai pydub
    google-cloud-texttospeech google-auth
"""

import io
import json
import math
import mimetypes
import os
import re
import shutil
import ssl
import sys
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
import tkinter as tk
from copy import deepcopy
from tkinter import filedialog, ttk, messagebox, scrolledtext
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Translation memory — proofed-translation feedback loop (optional module;
# the app must keep working if translation_memory.py wasn't copied along).
try:
    import translation_memory
except Exception:
    translation_memory = None

def _fatal_import_error(lib_name: str, detail: str):
    """A required library failed to import. Print + log + dialog, then exit
    with a NON-ZERO code so the launcher scripts keep the console open."""
    msg = (f"Required library '{lib_name}' failed to load:\n\n{detail}\n\n"
           f"Fix: run the setup script again "
           f"(setup_windows.bat on Windows, setup_mac.command on macOS).")
    print(msg, file=sys.stderr)
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "error_log.txt"), "a", encoding="utf-8") as f:
            f.write(f"\n[startup] {msg}\n")
    except OSError:
        pass
    try:
        import tkinter.messagebox as _mb
        _mb.showerror("Missing Library", msg)
    except Exception:
        pass
    raise SystemExit(1)


try:
    import librosa
except Exception as _e:          # numba/llvmlite breakage isn't ImportError
    _fatal_import_error("librosa", repr(_e))

try:
    import sounddevice as sd
except Exception as _e:
    _fatal_import_error("sounddevice", repr(_e))

try:
    from pydub import AudioSegment as _AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    from google.cloud import texttospeech as _tts_module
    from google.oauth2 import service_account as _sa_module
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    import pyphen as _pyphen_module
    PYPHEN_AVAILABLE = True
except ImportError:
    PYPHEN_AVAILABLE = False

# Optional SpaCy for English meaningful-chunk SRT splitting
_SPACY_NLP = None
_SPACY_TRIED = False

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── App version + update source ─────────────────────────────────────────────
APP_VERSION  = "1.7.0"
GITHUB_REPO  = "darpantimsina72/bulk-video-processing"   # owner/repo on GitHub

# ─── Cross-platform setup ────────────────────────────────────────────────────
IS_WINDOWS = sys.platform.startswith("win")
IS_MAC     = sys.platform == "darwin"

# Font families differ per OS; Tk silently falls back to an ugly default when
# a family is missing, so pick explicitly.
if IS_WINDOWS:
    MONO_FONT, UI_FONT = "Consolas", "Segoe UI"
elif IS_MAC:
    MONO_FONT, UI_FONT = "Menlo", "Helvetica Neue"
else:
    MONO_FONT, UI_FONT = "DejaVu Sans Mono", "DejaVu Sans"


def _find_ffmpeg() -> Optional[str]:
    """Locate ffmpeg. Checks PATH, then a bundled ffmpeg/bin folder next to
    the app, then common Windows install locations. Returns the executable
    path or None."""
    exe = "ffmpeg.exe" if IS_WINDOWS else "ffmpeg"
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidates = [os.path.join(SCRIPT_DIR, "ffmpeg", "bin", exe),
                  os.path.join(SCRIPT_DIR, "ffmpeg", exe)]
    if IS_WINDOWS:
        candidates += [
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"),
            os.path.expandvars(r"%ProgramFiles%\ffmpeg\bin\ffmpeg.exe"),
            r"C:\ffmpeg\bin\ffmpeg.exe",
        ]
    elif IS_MAC:
        candidates += ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


FFMPEG_PATH = _find_ffmpeg()
if FFMPEG_PATH and not shutil.which("ffmpeg"):
    # Make ffmpeg visible to pydub/librosa even when it isn't on PATH.
    os.environ["PATH"] = os.path.dirname(FFMPEG_PATH) + os.pathsep + os.environ.get("PATH", "")
    if PYDUB_AVAILABLE:
        _AudioSegment.converter = FFMPEG_PATH


# ─── Crash logging ───────────────────────────────────────────────────────────
ERROR_LOG = os.path.join(SCRIPT_DIR, "error_log.txt")


def _log_exception(exc_type, exc_value, exc_tb) -> str:
    """Append a traceback to error_log.txt; return the formatted text."""
    import traceback, datetime
    text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.datetime.now().isoformat(timespec='seconds')}] "
                    f"v{APP_VERSION} on {sys.platform}\n{text}")
    except OSError:
        pass
    return text


def _tk_exception_handler(exc_type, exc_value, exc_tb):
    """Replacement for Tk's default callback-exception printer: log the
    traceback and show it in a dialog instead of dying silently (on Windows
    the console window closes and the error is never seen)."""
    text = _log_exception(exc_type, exc_value, exc_tb)
    short = str(exc_value) or exc_type.__name__
    try:
        # Never attach a modal error box to a minimized window — on Windows
        # the box is invisible but still grabs input, so the app looks
        # frozen and cannot be restored from the taskbar.
        _root = tk._default_root
        if _root is not None and _root.state() == "iconic":
            _root.deiconify()
    except Exception:
        pass
    try:
        messagebox.showerror(
            "Unexpected Error",
            f"{short}\n\nFull details were saved to error_log.txt "
            "next to the app.")
    except Exception:
        print(text, file=sys.stderr)


# ─── In-app updater (manual, opt-in — never updates without asking) ──────────
# Files/folders never touched by an update (machine-specific state + secrets).
_UPDATE_PROTECTED = {
    ".git", ".venv", "__pycache__", "_update_backup",
    "api.txt", "llm_settings.json", "last_language.txt",
    "TTS_Key.json", "vertex_key.json", "error_log.txt",
    "github_token.txt", "launch_log.txt",
    "data",  # local translation-memory DB — never overwrite
    "feedback_outbox",  # locally saved feedback awaiting manual delivery
}


def _github_token() -> str:
    """Optional: a GitHub personal-access token in github_token.txt next to
    the app enables updates from a PRIVATE repository. Not needed when the
    repo is public. The file is gitignored and never uploaded."""
    p = os.path.join(SCRIPT_DIR, "github_token.txt")
    try:
        with open(p, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _fetch_url(url: str, timeout: int = 30, headers: Optional[dict] = None) -> bytes:
    h = {"User-Agent": "TranslationSyncApp-Updater"}
    if headers:
        h.update(headers)
    tok = _github_token()
    if tok and "github" in url:
        h["Authorization"] = f"token {tok}"
    req = urllib.request.Request(url, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.URLError as e:
        # Corporate proxies / broken cert stores: retry without verification.
        if isinstance(getattr(e, "reason", None), ssl.SSLError):
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
                return r.read()
        raise


def _version_tuple(v: str) -> tuple:
    nums = re.findall(r"\d+", v or "")
    return tuple(int(n) for n in nums[:4]) if nums else (0,)


def fetch_remote_version() -> str:
    """Read the VERSION file from the GitHub repo's main branch."""
    # The API endpoint works for both public and (with token) private repos.
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/VERSION?ref=main"
    return _fetch_url(url, headers={"Accept": "application/vnd.github.raw"}
                      ).decode("utf-8").strip()


def download_and_apply_update() -> str:
    """Download the main branch as a zip and copy its files over the app
    folder. Protected files (keys, settings) are never touched; every file
    that gets replaced is first copied into _update_backup/<timestamp>/.
    Returns the backup folder path."""
    import zipfile, tempfile, datetime
    # API zipball works for both public and (with token) private repos.
    blob = _fetch_url(f"https://api.github.com/repos/{GITHUB_REPO}/zipball/main",
                      timeout=300)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(SCRIPT_DIR, "_update_backup", stamp)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = os.path.join(tmp, "update.zip")
        with open(zpath, "wb") as f:
            f.write(blob)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        roots = [d for d in os.listdir(tmp)
                 if os.path.isdir(os.path.join(tmp, d))]
        if not roots:
            raise RuntimeError("Downloaded update archive was empty.")
        src_root = os.path.join(tmp, roots[0])
        for dirpath, dirnames, filenames in os.walk(src_root):
            rel = os.path.relpath(dirpath, src_root)
            rel = "" if rel == "." else rel
            top = rel.split(os.sep)[0] if rel else ""
            if top in _UPDATE_PROTECTED:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in _UPDATE_PROTECTED]
            for fn in filenames:
                if not rel and fn in _UPDATE_PROTECTED:
                    continue
                src = os.path.join(dirpath, fn)
                dst = os.path.join(SCRIPT_DIR, rel, fn)
                if os.path.exists(dst):
                    with open(src, "rb") as a, open(dst, "rb") as b:
                        if a.read() == b.read():
                            continue          # unchanged — skip
                    bak = os.path.join(backup_dir, rel, fn)
                    os.makedirs(os.path.dirname(bak), exist_ok=True)
                    shutil.copy2(dst, bak)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
    return backup_dir


# ─── In-app feedback (message + optional screenshots → GitHub issue) ─────────
# Feedback goes to a DEDICATED repo, not the app repo: the github_token.txt
# distributed with installs only ever needs write access to this repo, so it
# can never be used to alter the app code the updater pulls from main.
# One-time setup: create the (private) repo below with a README, then issue a
# fine-grained PAT with Issues+Contents read/write on it → github_token.txt.
FEEDBACK_REPO = "darpantimsina72/app-feedback"   # owner/repo on GitHub
FEEDBACK_BRANCH = "feedback"   # attachments branch, keeps default branch lean
FEEDBACK_MAX_ATTACHMENT_MB = 20


def _github_api_json(url: str, payload: Optional[dict] = None,
                     method: str = "GET") -> dict:
    """Call the GitHub REST API using the token from github_token.txt.
    Returns the decoded JSON response; raises urllib.error.HTTPError on 4xx/5xx."""
    h = {"User-Agent": "TranslationSyncApp-Feedback",
         "Accept": "application/vnd.github+json"}
    tok = _github_token()
    if tok:
        h["Authorization"] = f"token {tok}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        if isinstance(getattr(e, "reason", None), ssl.SSLError):
            with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as r:
                return json.loads(r.read().decode("utf-8"))
        raise


def _ensure_feedback_branch():
    """Create the attachments branch off the default branch when missing."""
    base = f"https://api.github.com/repos/{FEEDBACK_REPO}"
    try:
        _github_api_json(f"{base}/git/ref/heads/{FEEDBACK_BRANCH}")
        return
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    default = _github_api_json(base).get("default_branch") or "main"
    head = _github_api_json(f"{base}/git/ref/heads/{default}")
    _github_api_json(f"{base}/git/refs", method="POST", payload={
        "ref": f"refs/heads/{FEEDBACK_BRANCH}",
        "sha": head["object"]["sha"],
    })


def upload_feedback_attachment(path: str, stamp: str) -> str:
    """Commit one screenshot to the feedback branch; return its GitHub URL."""
    import base64
    fn = re.sub(r"[^\w.\-]+", "_", os.path.basename(path)) or "attachment"
    with open(path, "rb") as f:
        blob = f.read()
    dest = f"feedback_attachments/{stamp}/{fn}"
    url = (f"https://api.github.com/repos/{FEEDBACK_REPO}/contents/"
           + urllib.parse.quote(dest))
    resp = _github_api_json(url, method="PUT", payload={
        "message": f"Feedback attachment ({stamp})",
        "content": base64.b64encode(blob).decode("ascii"),
        "branch": FEEDBACK_BRANCH,
    })
    return (resp.get("content") or {}).get("html_url") or (
        f"https://github.com/{FEEDBACK_REPO}/blob/{FEEDBACK_BRANCH}/{dest}")


def create_feedback_issue(title: str, body: str, label: str) -> str:
    """Open a GitHub issue on the feedback repo; return its URL."""
    resp = _github_api_json(
        f"https://api.github.com/repos/{FEEDBACK_REPO}/issues",
        method="POST",
        payload={"title": title, "body": body,
                 "labels": [label, "in-app-feedback"]})
    return resp.get("html_url", "")


def save_feedback_locally(kind: str, sender: str, message: str,
                          attachments: List[str]) -> str:
    """Offline / no-token fallback: bundle the feedback into feedback_outbox/
    so the user can send the folder to the developer manually."""
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(SCRIPT_DIR, "feedback_outbox", stamp)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "message.txt"), "w", encoding="utf-8") as f:
        f.write(f"Type: {kind}\n"
                f"From: {sender or '(not given)'}\n"
                f"App:  v{APP_VERSION} · {sys.platform} · "
                f"Python {sys.version.split()[0]}\n"
                f"Date: {stamp}\n\n{message}\n")
    for p in attachments:
        try:
            shutil.copy2(p, os.path.join(folder, os.path.basename(p)))
        except OSError:
            pass
    return folder


# ─── Per-file output folder helper ──────────────────────────────────────────
def _prepare_output_dir(audio_path: str) -> str:
    """
    Given the path to an input audio file, create (if needed) a sibling
    folder named after the audio file (without extension) and copy the
    original audio into that folder. Returns the path to the new folder.

    All pipeline outputs (SRT, FinalScript, TTS, sync logs, synced audio,
    etc.) for a given input file should be written inside this folder so
    that each input gets its own self-contained results directory.
    """
    src_dir   = os.path.dirname(os.path.abspath(audio_path))
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    # Already inside its own output folder (e.g. a reopened project whose
    # audio is the copy that lives in the results dir) — reuse it. Creating
    # a sibling again would nest folder/folder/folder… one level per re-run.
    if os.path.basename(src_dir) == base_name:
        return src_dir
    out_dir   = os.path.join(src_dir, base_name)
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        # If the directory cannot be created, fall back to source folder
        return src_dir

    # Copy the original audio file into the new folder (if not already there)
    dst_audio = os.path.join(out_dir, os.path.basename(audio_path))
    try:
        if (os.path.abspath(audio_path) != os.path.abspath(dst_audio)
                and not os.path.exists(dst_audio)):
            shutil.copy2(audio_path, dst_audio)
    except Exception:
        # Non-fatal — the rest of the pipeline can still run
        pass

    return out_dir


# ─── UI settings (Simple mode, auto-open result) ─────────────────────────────
UI_SETTINGS_FILE = os.path.join(SCRIPT_DIR, "ui_settings.json")


def _ui_settings_load() -> dict:
    try:
        with open(UI_SETTINGS_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _ui_settings_save(d: dict) -> None:
    try:
        with open(UI_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass


# ─── Projects (launch-screen project manager) ─────────────────────────────────
PROJECTS_FILE  = os.path.join(SCRIPT_DIR, "projects.json")
_PROJECTS_LOCK = threading.Lock()


def _find_latest_output_file(outdir: str) -> str:
    """Newest synced/dubbed audio file in *outdir* ('' if none)."""
    cands = []
    try:
        for f in os.listdir(outdir):
            fl = f.lower()
            if ("_synced" in fl or "_redub" in fl) \
                    and fl.endswith((".wav", ".mp3")):
                p = os.path.join(outdir, f)
                try:
                    cands.append((os.path.getmtime(p), p))
                except Exception:
                    pass
    except Exception:
        pass
    return max(cands)[1] if cands else ""


def _projects_load() -> List[dict]:
    """Project registry entries, most recently updated first. Never raises."""
    try:
        with open(PROJECTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("projects", []) if isinstance(data, dict) else []
        entries = [e for e in entries if isinstance(e, dict) and e.get("base")]
        entries.sort(key=lambda e: e.get("updated", e.get("created", "")),
                     reverse=True)
        return entries
    except Exception:
        return []


def _projects_save(entries: List[dict]) -> None:
    with open(PROJECTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"projects": entries}, f, ensure_ascii=False, indent=2)


def _project_key(e: dict):
    return (os.path.abspath(e.get("base", "")), e.get("language", ""))


def _project_upsert(entry: dict) -> None:
    """Insert/refresh a project (keyed by base+language). Never raises."""
    try:
        with _PROJECTS_LOCK:
            entries = _projects_load()
            key = _project_key(entry)
            old = next((e for e in entries if _project_key(e) == key), None)
            if old:
                old.update({k: v for k, v in entry.items() if v})
                old["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                merged = old
            else:
                entry.setdefault("created",
                                 time.strftime("%Y-%m-%d %H:%M:%S"))
                entry["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                entries.insert(0, entry)
                merged = entry
            _projects_save(entries)
            # Portable copy inside the project folder
            try:
                with open(os.path.join(merged.get("outdir", ""),
                                       "project.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(merged, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
    except Exception:
        pass


def _project_remove(base: str, language: str) -> None:
    try:
        with _PROJECTS_LOCK:
            key = (os.path.abspath(base), language)
            entries = [e for e in _projects_load()
                       if _project_key(e) != key]
            _projects_save(entries)
    except Exception:
        pass


def _project_stage_status(e: dict) -> Dict[str, bool]:
    """Which stages have outputs on disk (stateless — survives crashes)."""
    base   = e.get("base", "")
    outdir = e.get("outdir", "")
    return {
        "transcript":  os.path.exists(base + "_sync_en.srt")
                       or os.path.exists(base + ".srt"),
        "translation": os.path.exists(base + "_FinalScript.txt"),
        "result":      bool(_find_latest_output_file(outdir)),
    }


def _projects_sync_from_history() -> None:
    """Every run-history entry becomes a project card (old runs included)."""
    try:
        known = {_project_key(e) for e in _projects_load()}
        for h in _history_load():
            key = (os.path.abspath(h.get("base", "")), h.get("language", ""))
            if key in known:
                continue
            _project_upsert({
                "name":       os.path.basename(h.get("base", "")) or "run",
                "language":   h.get("language", ""),
                "audio_path": h.get("audio_path", ""),
                "outdir":     h.get("outdir", ""),
                "base":       h.get("base", ""),
                "created":    h.get("ts", ""),
            })
    except Exception:
        pass


# ─── Run history (History tab / re-dub) ──────────────────────────────────────
RUN_HISTORY_FILE  = os.path.join(SCRIPT_DIR, "run_history.json")
_RUN_HISTORY_LOCK = threading.Lock()


def _history_load() -> List[dict]:
    """Return run-history entries, newest first. Never raises."""
    try:
        with open(RUN_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("runs", []) if isinstance(data, dict) else []
        entries = [e for e in entries if isinstance(e, dict) and e.get("base")]
        entries.sort(key=lambda e: e.get("ts", ""), reverse=True)
        return entries
    except Exception:
        return []


def _history_save(entries: List[dict]) -> None:
    with open(RUN_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump({"runs": entries}, f, ensure_ascii=False, indent=2)


def _history_record(audio_path: str, outdir: str, base: str,
                    language: str, source: str) -> None:
    """Insert/refresh a run entry, keyed by (base, language). Never raises —
    history is best-effort and must not break a pipeline run."""
    try:
        with _RUN_HISTORY_LOCK:
            entries = _history_load()
            key = (os.path.abspath(base), language)
            entries = [e for e in entries
                       if (os.path.abspath(e.get("base", "")),
                           e.get("language")) != key]
            entries.insert(0, {
                "ts":         time.strftime("%Y-%m-%d %H:%M:%S"),
                "language":   language,
                "audio_path": os.path.abspath(audio_path),
                "outdir":     os.path.abspath(outdir),
                "base":       os.path.abspath(base),
                "source":     source,
            })
            _history_save(entries)
    except Exception:
        pass


def _history_remove(base: str, language: str) -> None:
    """Drop one entry from the run history. Never raises."""
    try:
        with _RUN_HISTORY_LOCK:
            key = (os.path.abspath(base), language)
            entries = [e for e in _history_load()
                       if (os.path.abspath(e.get("base", "")),
                           e.get("language")) != key]
            _history_save(entries)
    except Exception:
        pass


def _history_next_redub_rev(outdir: str) -> int:
    """Next free re-dub revision number for *outdir* (2, 3, …)."""
    n = 2
    try:
        names = os.listdir(outdir)
    except Exception:
        names = []
    while any(f"_redub{n:02d}" in name for name in names):
        n += 1
    return n


# ─── TTS Language / Voice catalogue ─────────────────────────────────────────
# Single source of truth for every per-language artefact: BCP-47 locale code,
# native autonym, ElevenLabs language-token filter, and Google Cloud TTS voice
# catalogue per engine family (Standard / WaveNet / Chirp3-HD).
#
# Languages without native Google Cloud TTS coverage (Assamese, Odia, Nepali)
# carry "google_unavailable": True — the UI disables the Google engine row
# and only ElevenLabs is offered for synthesis on those.
#
# Chirp3-HD voice characters (Algenib, Aoede, Charon, Fenrir, Kore, Leda,
# Orus, Puck, Schedar, Zephyr) are the same set across all Indic locales
# where Chirp3 is offered; only the locale prefix changes.

def _std_voices(code: str) -> List[str]:
    return [f"{code}-Standard-A", f"{code}-Standard-B",
            f"{code}-Standard-C", f"{code}-Standard-D"]

def _wn_voices(code: str) -> List[str]:
    return [f"{code}-Wavenet-A", f"{code}-Wavenet-B",
            f"{code}-Wavenet-C", f"{code}-Wavenet-D"]

def _c3_voices(code: str) -> List[str]:
    return [f"{code}-Chirp3-HD-Algenib", f"{code}-Chirp3-HD-Aoede",
            f"{code}-Chirp3-HD-Charon",  f"{code}-Chirp3-HD-Fenrir",
            f"{code}-Chirp3-HD-Kore",    f"{code}-Chirp3-HD-Leda",
            f"{code}-Chirp3-HD-Orus",    f"{code}-Chirp3-HD-Puck",
            f"{code}-Chirp3-HD-Schedar", f"{code}-Chirp3-HD-Zephyr"]

TTS_LANGUAGES = {
    "Bengali": {
        "code": "bn-IN", "autonym": "বাংলা", "tag": "BN", "display_name": "Bangla",
        "el_tokens": ("bn", "ben", "bengali", "bangla", "bn-in", "bn-bd", "বাংলা"),
        "Standard": _std_voices("bn-IN"),
        "WaveNet":  _wn_voices("bn-IN"),
        "Chirp3":   _c3_voices("bn-IN"),
    },
    "Hindi": {
        "code": "hi-IN", "autonym": "हिन्दी", "tag": "HI", "display_name": "Hindi",
        "el_tokens": ("hi", "hin", "hindi", "हिन्दी", "hi-in"),
        "Standard": _std_voices("hi-IN"),
        "WaveNet":  _wn_voices("hi-IN"),
        "Chirp3":   _c3_voices("hi-IN"),
    },
    "Kannada": {
        "code": "kn-IN", "autonym": "ಕನ್ನಡ", "tag": "KN", "display_name": "Kannada",
        "el_tokens": ("kn", "kan", "kannada", "ಕನ್ನಡ", "kn-in"),
        "Standard": _std_voices("kn-IN"),
        "WaveNet":  _wn_voices("kn-IN"),
        "Chirp3":   _c3_voices("kn-IN"),
    },
    "Malayalam": {
        "code": "ml-IN", "autonym": "മലയാളം", "tag": "ML", "display_name": "Malayalam",
        "el_tokens": ("ml", "mal", "malayalam", "മലയാളം", "ml-in"),
        "Standard": _std_voices("ml-IN"),
        "WaveNet":  _wn_voices("ml-IN"),
        "Chirp3":   _c3_voices("ml-IN"),
    },
    "Tamil": {
        "code": "ta-IN", "autonym": "தமிழ்", "tag": "TA", "display_name": "Tamil",
        "el_tokens": ("ta", "tam", "tamil", "தமிழ்", "ta-in", "ta-lk"),
        "Standard": _std_voices("ta-IN"),
        "WaveNet":  _wn_voices("ta-IN"),
        "Chirp3":   _c3_voices("ta-IN"),
    },
    "Telugu": {
        "code": "te-IN", "autonym": "తెలుగు", "tag": "TE", "display_name": "Telugu",
        "el_tokens": ("te", "tel", "telugu", "తెలుగు", "te-in"),
        "Standard": _std_voices("te-IN"),
        "WaveNet":  _wn_voices("te-IN"),
        "Chirp3":   _c3_voices("te-IN"),
    },
    "Gujarati": {
        "code": "gu-IN", "autonym": "ગુજરાતી", "tag": "GU", "display_name": "Gujarati",
        "el_tokens": ("gu", "guj", "gujarati", "ગુજરાતી", "gu-in"),
        "Standard": _std_voices("gu-IN"),
        "WaveNet":  _wn_voices("gu-IN"),
        "Chirp3":   _c3_voices("gu-IN"),
    },
    "Marathi": {
        "code": "mr-IN", "autonym": "मराठी", "tag": "MR", "display_name": "Marathi",
        "el_tokens": ("mr", "mar", "marathi", "मराठी", "mr-in"),
        "Standard": _std_voices("mr-IN"),
        "WaveNet":  _wn_voices("mr-IN"),
        "Chirp3":   _c3_voices("mr-IN"),
    },
    # Google Cloud TTS does not currently expose native voices for the
    # following languages — the UI disables the Google engine row when
    # any of these is selected and only ElevenLabs is available.
    "Assamese": {
        "code": "as-IN", "autonym": "অসমীয়া", "tag": "AS", "display_name": "Assamese",
        "el_tokens": ("as", "asm", "assamese", "অসমীয়া", "as-in"),
        "google_unavailable": True,
        "Standard": [], "WaveNet": [], "Chirp3": [],
    },
    "Odia": {
        "code": "or-IN", "autonym": "ଓଡ଼ିଆ", "tag": "OR", "display_name": "Odia",
        "el_tokens": ("or", "ori", "odia", "oriya", "ଓଡ଼ିଆ", "or-in"),
        "google_unavailable": True,
        "Standard": [], "WaveNet": [], "Chirp3": [],
    },
    "Nepali": {
        "code": "ne-NP", "autonym": "नेपाली", "tag": "NE", "display_name": "Nepali",
        "el_tokens": ("ne", "nep", "nepali", "नेपाली", "ne-np"),
        "google_unavailable": True,
        "Standard": [], "WaveNet": [], "Chirp3": [],
    },
}
TTS_LANGUAGE_NAMES   = list(TTS_LANGUAGES.keys())
TTS_DEFAULT_LANGUAGE = "Bengali"

def _lang_display_name(language: str) -> str:
    """Return the output-folder-friendly display name for a language (e.g. 'Bangla' for Bengali)."""
    return TTS_LANGUAGES.get(language, {}).get("display_name", language)

def _tts_output_name(language: str, audio_path: str, suffix: str = "_tts") -> str:
    """
    Build the output filename stem using the convention:
        {DisplayName}_({audio_base}){suffix}.wav
    e.g.  Bangla_(MyLecture)_tts.wav  or  Bangla_(MyLecture)_synced.wav
    """
    display   = _lang_display_name(language)
    audio_base = os.path.splitext(os.path.basename(audio_path))[0]
    return f"{display}_({audio_base}){suffix}.wav"

def _strip_emotion_tags(text: str) -> str:
    """Strip ElevenLabs inline emotion/accent tags like [calm],
    [bengali accent], [pause] — including closers like [/fast]."""
    return re.sub(r'\[/?[\w\s]+\]', '', text).strip()
TTS_DEFAULT_ENGINE   = "Chirp3"
TTS_DEFAULT_VOICE    = "bn-IN-Chirp3-HD-Aoede"

# ─── ElevenLabs TTS settings ─────────────────────────────────────────────────
# Default voice ID is empty — voices are auto-fetched from the ElevenLabs API
# once a valid key is supplied (see _fetch_voices_for_language).
ELEVENLABS_TTS_VOICE_ID = ""
# eleven_v3 auto-detects the language from the input text (Indic scripts
# inclusive). It does not accept a language_code parameter — sending one
# triggers HTTP 400 unsupported_language errors on multilingual models.
ELEVENLABS_TTS_MODEL    = "eleven_v3"
# Selectable ElevenLabs TTS models — all multilingual / Indic-capable.
# Only eleven_v3 understands inline audio tags ([calm], [pause], …); for the
# other models synthesize_tts_elevenlabs strips the tags before sending.
ELEVENLABS_TTS_MODELS   = {
    "eleven_v3":              "v3 — expressive (audio tags)",
    "eleven_multilingual_v2": "Multilingual v2 — stable",
    "eleven_turbo_v2_5":      "Turbo v2.5 — fast",
    "eleven_flash_v2_5":      "Flash v2.5 — fastest",
}
TTS_PLATFORMS           = ["ElevenLabs", "Google TTS"]
TTS_DEFAULT_PLATFORM    = "ElevenLabs"

# Gemini models available for translation / review / punctuation / mapping
GEMINI_MODELS         = ["gemini-2.5-pro", "gemini-3.5-flash"]
GEMINI_DEFAULT_MODEL  = "gemini-2.5-pro"

# ─── LLM provider settings ────────────────────────────────────────────────────
# The translation / review / punctuation / mapping / emotion steps can run on:
#   1. Vertex AI          — service-account JSON file (vertex_key.json)
#   2. Gemini API         — plain Google AI Studio API key
#   3. OpenAI-compatible  — any /v1/chat/completions endpoint (LiteLLM proxy,
#                           OpenRouter, vLLM, …) via a base URL + optional key
# Selection + credentials are edited in the "LLM Settings" dialog and persisted
# to llm_settings.json next to this script.
LLM_PROVIDER_VERTEX   = "Vertex AI (JSON file)"
LLM_PROVIDER_GEMINI   = "Gemini API key"
LLM_PROVIDER_OPENAI   = "OpenAI-compatible (Base URL)"
LLM_PROVIDERS         = [LLM_PROVIDER_VERTEX, LLM_PROVIDER_GEMINI, LLM_PROVIDER_OPENAI]
LLM_SETTINGS_FILE     = os.path.join(SCRIPT_DIR, "llm_settings.json")
LLM_DEFAULT_BASE_URL  = "http://172.18.1.17:14005"

# Step4: emotion / accent tag enrichment before ElevenLabs TTS.
# When enabled, runs an extra Gemini pass that injects ElevenLabs v3 inline
# audio tags ([<lang> accent], [calm], [contemplative], [slow], [pause], …)
# into the Step3 punctuated script so the final speech sounds human and
# reflective (Sadhguru-style cadence) rather than flat.
STEP4_EMOTION_ENABLED = True

def _lang_tokens(language: str) -> tuple:
    """ElevenLabs voice-metadata tokens for a given UI language name."""
    return TTS_LANGUAGES.get(language, {}).get("el_tokens", ())

def _read_last_language() -> str:
    """Restore the user's previously-selected language across sessions."""
    try:
        p = os.path.join(SCRIPT_DIR, "last_language.txt")
        if not os.path.exists(p):
            return TTS_DEFAULT_LANGUAGE
        with open(p, "r", encoding="utf-8") as f:
            v = f.read().strip()
        return v if v in TTS_LANGUAGES else TTS_DEFAULT_LANGUAGE
    except Exception:
        return TTS_DEFAULT_LANGUAGE

def _write_last_language(language: str) -> None:
    """Persist the user's language choice. Best-effort, never crashes."""
    if language not in TTS_LANGUAGES:
        return
    try:
        p = os.path.join(SCRIPT_DIR, "last_language.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(language)
    except Exception:
        pass

def _read_el_model() -> str:
    """Restore the user's ElevenLabs model choice across sessions."""
    try:
        p = os.path.join(SCRIPT_DIR, "el_model.txt")
        if not os.path.exists(p):
            return ELEVENLABS_TTS_MODEL
        with open(p, "r", encoding="utf-8") as f:
            v = f.read().strip()
        return v if v in ELEVENLABS_TTS_MODELS else ELEVENLABS_TTS_MODEL
    except Exception:
        return ELEVENLABS_TTS_MODEL

def _write_el_model(model_id: str) -> None:
    """Persist the ElevenLabs model choice. Best-effort, never crashes."""
    if model_id not in ELEVENLABS_TTS_MODELS:
        return
    try:
        p = os.path.join(SCRIPT_DIR, "el_model.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(model_id)
    except Exception:
        pass

# Characters per ElevenLabs TTS request chunk
ELEVENLABS_CHUNK_CHARS = 1000

# ─── TTS byte-chunk limit ─────────────────────────────────────────────────────
TTS_MAX_BYTES = 4800   # Safe limit below the 5000-byte API cap

# ─── Default region detection params — English audio ─────────────────────────
DEFAULT_THR_DB  = -42.0
DEFAULT_HYS_DB  = 6.0
DEFAULT_MIN_MS  = 150

# ─── Default region detection params — target-language TTS audio ────────────
DEFAULT_BN_THR_DB = -42.0
DEFAULT_BN_HYS_DB = 10.0
DEFAULT_BN_MIN_MS = 80

# ─── Colour palette ──────────────────────────────────────────────────────────
# Modern dark theme: deep slate panels, soft moonlight text, vibrant accents.
BG           = "#0b1220"   # app background (deep midnight)
PANEL        = "#111827"   # primary card/panel surface (gray-900)
PANEL2       = "#1e293b"   # tinted slate panel (slate-800)
PANEL3       = "#1f2937"   # tinted gray panel (gray-800)
PANEL_BORDER = "#334155"   # default 1-px panel border (slate-700)
ACCENT       = "#34d399"   # emerald accent
GRID         = "#1f2937"   # subtle grid lines on plots
TEXT         = "#e2e8f0"   # primary text (slate-200)
TEXT_MUTED   = "#94a3b8"   # secondary text (slate-400)
TEXT_FAINT   = "#64748b"   # tertiary text (slate-500)
# Input fields use LIGHT bg + DARK text. macOS Tk 9 (Aqua) ignores dark bg on
# Entry/Spinbox/Combobox and draws a native white field — light text becomes
# invisible. Light-field colors render correctly on both macOS and Windows.
INPUT_BG     = "#e2e8f0"   # input / entry / spinbox bg (light slate-200)
INPUT_FG     = "#0f172a"   # input text colour (dark slate-950)
BTN_BG       = "#1e293b"   # default button bg (slate-800)
BTN_FG       = "#e2e8f0"   # default button fg
BTN_ACT      = "#334155"   # button hover/active
WAVEFORM     = "#f59e0b"   # waveform peaks (amber)
REG_FILL     = "#1e3a8a"   # region rectangle fill (deep blue-900)
REG_EDGE     = "#60a5fa"   # region edge (blue-400)
REG_LABEL    = "#93c5fd"   # region labels (blue-300)
THR_LINE     = "#f87171"   # threshold line (red-400)
CURSOR_C     = "#4ade80"   # playback cursor (green-400)
CURSOR_W     = 1.5
TR_ACCENT    = "#22c55e"   # success/accent green
PLAY_COL_W   = 40          # px — ▶ button column in the review window

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a"}

IS_MAC = sys.platform == "darwin"


def _btn_fg(color):
    """macOS Aqua draws native LIGHT buttons and ignores tk.Button bg=.
    Pale fg colours become invisible there — map them to a dark equivalent.
    On Windows/Linux (bg honoured, dark buttons) colours pass through."""
    if not IS_MAC:
        return color
    try:
        c = str(color)
        if c.lower() in ("white", "snow", "ivory", "ghostwhite"):
            return "#0f172a"
        if c.startswith("#") and len(c) == 7:
            r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
            if (0.299 * r + 0.587 * g + 0.114 * b) / 255.0 > 0.55:
                return "#0f172a"
    except Exception:
        pass
    return color


# ═════════════════════════════════════════════════════════════════════════════
#  Shared helpers — ElevenLabs
# ═════════════════════════════════════════════════════════════════════════════

# In-memory ElevenLabs key set by the UI when the user pastes one. Falls back
# to api.txt on disk so the same key is reused across sessions.
_API_KEY_RUNTIME: Optional[str] = None

# ElevenLabs voice cache keyed by (api-key fingerprint, language name) so we
# don't re-fetch the voice list on every TTS run. Reset whenever a new key
# is set, or per-language when the user clicks "Refresh Voices".
_EL_VOICE_CACHE: Dict[tuple, List[Dict[str, str]]] = {}


_VOICE_ID_RE = re.compile(r"^[A-Za-z0-9]{12,40}$")


def _sanitize_voice_id(raw) -> str:
    """
    Return *raw* unchanged if it is a clean ElevenLabs voice_id, else "".

    ElevenLabs voice IDs are 20-char alphanumeric tokens. The UI dropdown
    shows formatted labels like "✦ Aria — abc12345…  [bn · premade]" — we
    must NOT mangle those into fake IDs by stripping punctuation, because
    the truncated 8-char fragment in the label cannot reconstruct the real
    voice_id (and ElevenLabs returns HTTP 404 voice_not_found).

    Strict policy: input must already match the voice_id regex, otherwise
    callers must look it up in the options map by display label.
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    return s if _VOICE_ID_RE.match(s) else ""


def _api_key_fingerprint(api_key: str) -> str:
    """Stable, non-sensitive cache key derived from the API key."""
    if not api_key:
        return ""
    return api_key[-8:] if len(api_key) >= 8 else "x" * len(api_key)


def _read_api_key_file() -> str:
    api_file = os.path.join(SCRIPT_DIR, "api.txt")
    if not os.path.exists(api_file):
        return ""
    try:
        with open(api_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _write_api_key_file(api_key: str) -> None:
    """Persist a validated API key to api.txt so the next launch picks it up."""
    if not api_key:
        return
    try:
        api_file = os.path.join(SCRIPT_DIR, "api.txt")
        with open(api_file, "w", encoding="utf-8") as f:
            f.write(api_key.strip())
    except Exception:
        # Persistence is best-effort — a write failure must not crash the app.
        pass


def set_runtime_api_key(api_key: Optional[str]) -> None:
    """Update the in-memory ElevenLabs key (called by the UI on paste)."""
    global _API_KEY_RUNTIME
    _API_KEY_RUNTIME = (api_key or "").strip() or None


def _get_api_key():
    """
    Resolve the ElevenLabs API key.

    Order of precedence:
      1. In-memory key set by the UI (latest paste).
      2. api.txt cached on disk from a previous successful validation.
    Raises FileNotFoundError / ValueError with a friendly message if missing.
    """
    if _API_KEY_RUNTIME:
        return _API_KEY_RUNTIME
    key = _read_api_key_file()
    if not key:
        raise ValueError(
            "No ElevenLabs API key configured.\n"
            "Paste your key into the API Key box at the top of the TTS Settings panel.")
    return key


def _redact_api_key(api_key: str) -> str:
    """Safe representation for logs / status messages."""
    if not api_key:
        return "<empty>"
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:3]}…{api_key[-4:]}"


def _validate_api_key(api_key: str, timeout: float = 15.0) -> Dict[str, str]:
    """
    Verify an ElevenLabs API key by hitting /v1/user.

    Returns a dict like {"ok": True, "tier": "...", "name": "..."} on success.
    Raises ValueError with a user-friendly message on failure (invalid /
    expired / network / quota).
    """
    if not api_key or not api_key.strip():
        raise ValueError("API key is empty.")
    api_key = api_key.strip()
    req = urllib.request.Request(
        "https://api.elevenlabs.io/v1/user",
        method="GET",
        headers={"xi-api-key": api_key, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise ValueError("Invalid or expired ElevenLabs API key (401).") from None
        if e.code == 429:
            raise ValueError("ElevenLabs API rate limit hit (429). Try again shortly.") from None
        raise ValueError(f"ElevenLabs API error: HTTP {e.code}.") from None
    except urllib.error.URLError as e:
        raise ValueError(f"Network error reaching ElevenLabs: {e.reason}") from None
    except Exception as e:
        raise ValueError(f"Could not validate ElevenLabs key: {e}") from None

    sub = payload.get("subscription") or {}
    return {
        "ok": True,
        "tier": str(sub.get("tier", "")),
        "name": str(payload.get("first_name") or payload.get("xi_api_key") or "user"),
    }


def _voice_supports_language(voice: dict, lang_tokens: tuple) -> bool:
    """
    Heuristic: does this ElevenLabs voice metadata indicate support for the
    language identified by *lang_tokens* (from TTS_LANGUAGES[...]["el_tokens"])?

    Looks at:
      • labels.language / labels.languages
      • verified_languages (newer schema, list of {language, ...})
      • language / language_code top-level fields
      • name / description text mentions
    """
    if not isinstance(voice, dict):
        return False
    haystacks: List[str] = []

    labels = voice.get("labels") or {}
    if isinstance(labels, dict):
        for key in ("language", "languages", "accent"):
            v = labels.get(key)
            if isinstance(v, str):
                haystacks.append(v)
            elif isinstance(v, list):
                haystacks.extend(str(x) for x in v)

    verified = voice.get("verified_languages") or []
    if isinstance(verified, list):
        for entry in verified:
            if isinstance(entry, dict):
                for key in ("language", "code", "name", "locale"):
                    val = entry.get(key)
                    if isinstance(val, str):
                        haystacks.append(val)
            elif isinstance(entry, str):
                haystacks.append(entry)

    for key in ("language", "language_code", "locale"):
        v = voice.get(key)
        if isinstance(v, str):
            haystacks.append(v)

    for key in ("name", "description"):
        v = voice.get(key)
        if isinstance(v, str):
            haystacks.append(v)

    blob = " ".join(haystacks).lower()
    if not blob:
        return False
    for token in lang_tokens:
        t = str(token).lower()
        # Word-boundary match for short codes, substring for full names.
        if len(t) <= 3:
            if re.search(rf"\b{re.escape(t)}\b", blob):
                return True
        else:
            if t in blob:
                return True
    return False


def _fetch_voices_for_language(api_key: str,
                               language: str = TTS_DEFAULT_LANGUAGE,
                               force_refresh: bool = False,
                               timeout: float = 30.0) -> List[Dict[str, str]]:
    """
    Pull the user's full voice catalogue from ElevenLabs and return EVERY
    voice on the account.

    Voices that advertise support for *language* (per TTS_LANGUAGES tokens)
    are sorted to the top and marked with a ✦ prefix so they're easy to
    find, but the dropdown shows everything — eleven_v3 auto-detects the
    target language from the input text and works on any voice.

    Each entry is a small dict: {"voice_id", "name", "label"}.
    Result is cached per (API-key fingerprint, language) to avoid repeated
    API calls.
    """
    api_key = (api_key or "").strip()
    if not api_key:
        raise ValueError("API key is empty.")

    lang_tokens = _lang_tokens(language)
    cache_key   = (_api_key_fingerprint(api_key), language)
    if not force_refresh and cache_key in _EL_VOICE_CACHE:
        return _EL_VOICE_CACHE[cache_key]

    req = urllib.request.Request(
        "https://api.elevenlabs.io/v1/voices",
        method="GET",
        headers={"xi-api-key": api_key, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise ValueError("Invalid or expired ElevenLabs API key (401).") from None
        if e.code == 429:
            raise ValueError("ElevenLabs API rate limit hit (429). Try again shortly.") from None
        raise ValueError(f"Could not fetch voices (HTTP {e.code}).") from None
    except urllib.error.URLError as e:
        raise ValueError(f"Network error fetching voices: {e.reason}") from None
    except Exception as e:
        raise ValueError(f"Could not fetch voices: {e}") from None

    voices = payload.get("voices") or []
    if not isinstance(voices, list):
        voices = []

    matched_voices: List[Dict[str, str]] = []
    other_voices:   List[Dict[str, str]] = []
    for v in voices:
        if not isinstance(v, dict):
            continue
        vid = v.get("voice_id") or v.get("voiceId") or ""
        if not vid:
            continue
        name = v.get("name") or "Unnamed voice"
        labels = v.get("labels") or {}
        accent = ""
        category = str(v.get("category", "")).strip()
        if isinstance(labels, dict):
            accent = str(labels.get("accent") or labels.get("language") or "")

        meta_bits: List[str] = []
        if accent:
            meta_bits.append(accent)
        if category and category.lower() not in ("premade",):
            meta_bits.append(category)
        meta = f"  [{' · '.join(meta_bits)}]" if meta_bits else ""

        is_match = _voice_supports_language(v, lang_tokens)
        prefix = "✦ " if is_match else "  "
        entry = {
            "voice_id": vid,
            "name": str(name),
            "label": f"{prefix}{name} — {vid[:8]}…{meta}",
        }
        (matched_voices if is_match else other_voices).append(entry)

    # Sort each bucket alphabetically for stable display order.
    matched_voices.sort(key=lambda e: e["name"].lower())
    other_voices.sort(key=lambda e: e["name"].lower())

    all_voices = matched_voices + other_voices

    _EL_VOICE_CACHE[cache_key] = all_voices
    return all_voices


def _clear_el_voice_cache(language: Optional[str] = None,
                          api_key: Optional[str] = None) -> None:
    """Clear ElevenLabs voice cache. If *language* is given, clear only that
    language's entries; otherwise clear everything. If *api_key* is also
    given, scope the clear to that key's fingerprint."""
    if language is None and api_key is None:
        _EL_VOICE_CACHE.clear()
        return
    fp = _api_key_fingerprint(api_key) if api_key else None
    for key in list(_EL_VOICE_CACHE.keys()):
        key_fp, key_lang = key
        if (language is None or key_lang == language) and \
           (fp is None or key_fp == fp):
            _EL_VOICE_CACHE.pop(key, None)


def _multipart_body(fields, files):
    boundary = "----ElevenLabsBoundary7MA4YWxkTrZu0gW"
    body = b""
    for name, value in fields:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f"name=\"{name}\"\r\n\r\n{value}\r\n").encode()
    for name, filename, mime, data in files:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f"name=\"{name}\"; filename=\"{filename}\"\r\n"
                 f"Content-Type: {mime}\r\n\r\n").encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, boundary


def _transcribe_audio(audio_path, api_key):
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    mime, _ = mimetypes.guess_type(audio_path)
    mime = mime or "audio/mpeg"
    body, boundary = _multipart_body(
        fields=[("model_id", "scribe_v2")],
        files=[("file", os.path.basename(audio_path), mime, audio_data)],
    )
    req = urllib.request.Request(
        "https://api.elevenlabs.io/v1/speech-to-text",
        data=body, method="POST",
        headers={"xi-api-key": api_key,
                 "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=180, context=_SSL_CTX) as resp:
        return json.loads(resp.read().decode())



# ═════════════════════════════════════════════════════════════════════════════
#  Shared helpers — SRT / region detection
# ═════════════════════════════════════════════════════════════════════════════

def _srt_ts(seconds: float) -> str:
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _build_subtitle_srt(regions, words) -> str:
    if not regions:
        return ""
    word_list  = [w for w in words if w.get("type", "word") == "word"]
    buckets    = [[] for _ in regions]
    word_idx   = 0
    region_idx = 0
    while word_idx < len(word_list) and region_idx < len(regions):
        w          = word_list[word_idx]
        word_start = w.get("start", 0.0)
        reg_end    = regions[region_idx][1]
        if word_start <= reg_end:
            buckets[region_idx].append(w.get("text", "").strip())
            word_idx += 1
        else:
            region_idx += 1
    while word_idx < len(word_list):
        buckets[-1].append(word_list[word_idx].get("text", "").strip())
        word_idx += 1
    lines = []
    idx   = 1
    for (rs, re), bucket in zip(regions, buckets):
        text = " ".join(t for t in bucket if t)
        if not text:
            # Region had no words assigned — skip rather than emitting a blank
            # subtitle entry that can confuse downstream SRT parsers.
            continue
        lines += [str(idx), f"{_srt_ts(rs)} --> {_srt_ts(re)}", text, ""]
        idx += 1
    return "\n".join(lines)


def _build_target_subtitle_srt(regions: list, words: list) -> str:
    """
    Build a target-language TTS SRT from (regions, words).

    ElevenLabs Scribe often returns all word timestamps as 0.0 when the
    source audio is TTS-generated (very uniform amplitude / pitch).  When
    that happens the normal timestamp-bucketing approach puts every word into
    the very first region, producing gibberish subtitles.

    Strategy
    --------
    1. Filter to actual word tokens (non-empty text, type == "word").
    2. Reliability check: if ≥ half the words have start > 0.05 s → the
       timestamps are good → fall back to the standard bucketing used by
       _build_subtitle_srt.
    3. Otherwise distribute words proportionally across regions weighted by
       each region's duration (same approach as Step-1 English SRT that is
       used for translation — no SpaCy, just regions → pour words in).
    """
    if not regions:
        return ""

    word_list = [
        w for w in words
        if w.get("type", "word") == "word" and w.get("text", "").strip()
    ]
    if not word_list:
        return ""

    # ── Reliability check ────────────────────────────────────────────────────
    nonzero = sum(1 for w in word_list if w.get("start", 0.0) > 0.05)
    timestamps_reliable = nonzero >= len(word_list) / 2

    if timestamps_reliable:
        # Use exact timestamp bucketing (same as _build_subtitle_srt)
        buckets    = [[] for _ in regions]
        word_idx   = 0
        region_idx = 0
        while word_idx < len(word_list) and region_idx < len(regions):
            w          = word_list[word_idx]
            word_start = w.get("start", 0.0)
            reg_end    = regions[region_idx][1]
            if word_start <= reg_end:
                buckets[region_idx].append(w.get("text", "").strip())
                word_idx += 1
            else:
                region_idx += 1
        while word_idx < len(word_list):
            buckets[-1].append(word_list[word_idx].get("text", "").strip())
            word_idx += 1
    else:
        # Proportional distribution by region duration
        total_dur = sum(re - rs for rs, re in regions)
        if total_dur <= 0:
            # Edge case: zero-length regions — spread words evenly
            n_reg = len(regions)
            per_reg = max(1, len(word_list) // n_reg)
            buckets = []
            for i, _ in enumerate(regions):
                start_i = i * per_reg
                end_i   = start_i + per_reg if i < n_reg - 1 else len(word_list)
                buckets.append([w.get("text", "").strip()
                                 for w in word_list[start_i:end_i]])
        else:
            # Calculate how many words each region should receive,
            # weighted by its fraction of the total audio duration.
            n_words  = len(word_list)
            counts   = []
            assigned = 0
            for i, (rs, re) in enumerate(regions):
                if i < len(regions) - 1:
                    c = max(1, round(n_words * (re - rs) / total_dur))
                    c = min(c, n_words - assigned - (len(regions) - 1 - i))
                    c = max(c, 1)
                else:
                    c = n_words - assigned   # last region gets the remainder
                counts.append(c)
                assigned += c

            buckets = []
            wi = 0
            for c in counts:
                buckets.append([w.get("text", "").strip()
                                 for w in word_list[wi: wi + c]])
                wi += c

    # ── Build SRT lines ──────────────────────────────────────────────────────
    lines = []
    idx   = 1
    for (rs, re), bucket in zip(regions, buckets):
        text = " ".join(t for t in bucket if t)
        if not text:
            continue
        lines += [str(idx), f"{_srt_ts(rs)} --> {_srt_ts(re)}", text, ""]
        idx += 1
    return "\n".join(lines)


# Backwards-compat alias for older callers/scripts.
_build_bn_subtitle_srt = _build_target_subtitle_srt


def _extract_translation_from_finalscript(combined: str,
                                          language: str = TTS_DEFAULT_LANGUAGE) -> str:
    """
    Pull the translation text out of a FinalScript file. Accepts the
    current "=== <LANGUAGE> TRANSLATION ===" marker plus legacy markers
    ("BENGALI" / "TELUGU") so older FinalScript files keep working.
    """
    markers = [f"=== {language.upper()} TRANSLATION ==="]
    for lang_name in TTS_LANGUAGES:
        m = f"=== {lang_name.upper()} TRANSLATION ==="
        if m not in markers:
            markers.append(m)
    markers.append("=== TELUGU TRANSLATION ===")  # legacy
    for marker in markers:
        if marker in combined:
            return combined.split(marker, 1)[1].strip()
    return combined.strip()


def _load_spacy_model():
    """Load and cache a SpaCy English model.  Returns None if unavailable."""
    global _SPACY_NLP, _SPACY_TRIED
    if _SPACY_TRIED:
        return _SPACY_NLP
    _SPACY_TRIED = True
    try:
        import spacy  # noqa: F401
        for name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
            try:
                _SPACY_NLP = spacy.load(name)
                return _SPACY_NLP
            except OSError:
                continue
        _SPACY_NLP = None
    except ImportError:
        _SPACY_NLP = None
    return _SPACY_NLP


def _get_spacy_chunk_boundaries(word_texts: List[str]) -> List[int]:
    """
    Given a list of plain word strings, return a sorted list of word indices
    at which new subtitle chunks should begin.  The first entry is always 0.

    Split criteria (SpaCy dependency parse):
      • Sentence boundaries
      • Coordinating conjunction (dep_==cc) whose head is a VERB / AUX / ROOT
        → the conjunction starts a new clause, not just a noun phrase
      • Subordinating conjunction (SCONJ) or adverbial/relative clause token
        (dep_ in advcl / relcl / acl) that immediately follows a comma

    Constraints:
      • MIN_WORDS_TO_SPLIT  — text shorter than this is never split
      • MIN_CHUNK_WORDS     — resulting chunks shorter than this are merged
        back into the preceding chunk

    Falls back to [0] (= no split) when SpaCy is unavailable.
    """
    MIN_WORDS_TO_SPLIT = 6
    MIN_CHUNK_WORDS    = 3

    if len(word_texts) < MIN_WORDS_TO_SPLIT:
        return [0]

    nlp = _load_spacy_model()
    if nlp is None:
        return [0]

    full_text = " ".join(word_texts)

    # Build char-offset → word-index table
    char_starts: List[int] = []
    pos = 0
    for wt in word_texts:
        char_starts.append(pos)
        pos += len(wt) + 1          # +1 for the space separator

    def _char_to_word(char_off: int) -> int:
        """Return the word index whose span contains char_off."""
        for i in range(len(char_starts) - 1, -1, -1):
            if char_off >= char_starts[i]:
                return i
        return 0

    doc    = nlp(full_text)
    tokens = list(doc)

    # SpaCy 3.x-safe sentence boundary detection: use doc.sents rather than
    # tok.is_sent_start, which can return None (not just False) when the model
    # hasn't explicitly assigned a boundary, causing silent misses.
    sent_starts: Set[int] = {sent.start for sent in doc.sents}

    split_tok_set: Set[int] = {0}

    for i, tok in enumerate(tokens):
        if i == 0:
            continue

        # 1. SpaCy sentence boundary
        if i in sent_starts:
            split_tok_set.add(i)
            continue

        # 2. Coordinating conjunction joining clauses (not bare NP coordination)
        if tok.dep_ == "cc":
            head = tok.head
            if head.pos_ in ("VERB", "AUX") or head.dep_ == "ROOT":
                split_tok_set.add(i)
            continue

        # 3. Clause token that immediately follows a comma
        if i > 0 and tokens[i - 1].text == ",":
            if tok.pos_ == "SCONJ" or tok.dep_ in ("advcl", "relcl", "acl"):
                split_tok_set.add(i)

    # Map token-level split positions → word-level split positions
    split_word_set: Set[int] = set()
    for tok_i in split_tok_set:
        split_word_set.add(_char_to_word(tokens[tok_i].idx))
    split_word_set.add(0)

    # Enforce minimum chunk size: merge tiny chunks into the previous one
    sorted_splits = sorted(split_word_set)
    n_words       = len(word_texts)
    merged: List[int] = []
    for sp in sorted_splits:
        next_sp   = next((s for s in sorted_splits if s > sp), n_words)
        chunk_len = next_sp - sp
        if chunk_len < MIN_CHUNK_WORDS and merged:
            continue            # too short → discard this split boundary
        merged.append(sp)

    # If the last chunk ended up too short, drop its boundary
    while len(merged) > 1:
        last_start = merged[-1]
        if (n_words - last_start) < MIN_CHUNK_WORDS:
            merged.pop()
        else:
            break

    return merged if merged else [0]


def _split_words_into_chunks(word_objs: List[dict]) -> List[List[dict]]:
    """
    Split a list of ElevenLabs word objects into subtitle-sized chunks.

    Each chunk is a contiguous sub-list of word_objs.
    Returns [word_objs] unchanged when SpaCy is unavailable or the region
    is too short to warrant splitting.
    """
    if not word_objs:
        return []

    # Strip empty-text entries before building the SpaCy input.  Empty strings
    # (punctuation artefacts, spacing tokens) create phantom words that shift
    # every character offset in the mapping table and cause split boundaries to
    # land on the wrong word index.
    word_objs = [w for w in word_objs if w.get("text", "").strip()]
    if not word_objs:
        return []

    texts      = [w.get("text", "").strip() for w in word_objs]
    boundaries = _get_spacy_chunk_boundaries(texts)

    if len(boundaries) <= 1:
        return [word_objs]

    n      = len(word_objs)
    chunks = []
    for k, start in enumerate(boundaries):
        end   = boundaries[k + 1] if k + 1 < len(boundaries) else n
        chunk = word_objs[start:end]
        if chunk:
            chunks.append(chunk)

    return chunks if chunks else [word_objs]


def _build_english_subtitle_srt(regions, words) -> str:
    """
    Build the English SRT with intra-segment meaningful-chunk splitting.

    For each waveform region the function:
      1. Collects the ElevenLabs word objects that fall inside that region
         (same bucketing logic as _build_subtitle_srt).
      2. Uses SpaCy to split the region's words into meaningful sub-chunks
         at clause / phrase / grammatical boundaries.
      3. Assigns timestamps per chunk:
           • First chunk   → start = segment start  (always)
           • Middle chunks → start/end = actual word-level timestamps
           • Last chunk    → end = last-word-end  if  last-word-end < seg-end,
                                   seg-end         otherwise
    """
    if not regions:
        return ""

    word_list = [w for w in words if w.get("type", "word") == "word"]

    # ── bucket words into regions (identical to original logic) ─────────────
    buckets: List[List[dict]] = [[] for _ in regions]
    word_idx   = 0
    region_idx = 0

    while word_idx < len(word_list) and region_idx < len(regions):
        w          = word_list[word_idx]
        word_start = w.get("start", 0.0)
        reg_end    = regions[region_idx][1]
        if word_start <= reg_end:
            buckets[region_idx].append(w)
            word_idx += 1
        else:
            region_idx += 1

    while word_idx < len(word_list):
        buckets[-1].append(word_list[word_idx])
        word_idx += 1

    # ── build SRT lines ──────────────────────────────────────────────────────
    sub_counter = 1
    lines: List[str] = []

    for (reg_start, reg_end), bucket in zip(regions, buckets):
        if not bucket:
            # Empty region → keep a placeholder subtitle
            lines += [str(sub_counter),
                      f"{_srt_ts(reg_start)} --> {_srt_ts(reg_end)}", "", ""]
            sub_counter += 1
            continue

        chunks   = _split_words_into_chunks(bucket)
        n_chunks = len(chunks)

        for c_idx, chunk_words in enumerate(chunks):
            is_first = (c_idx == 0)
            is_last  = (c_idx == n_chunks - 1)

            text = " ".join(
                w.get("text", "").strip()
                for w in chunk_words
                if w.get("text", "").strip()
            )

            # ── start time ───────────────────────────────────────────────────
            if is_first:
                t_start = reg_start
            else:
                t_start = chunk_words[0].get("start", reg_start)

            # ── end time ─────────────────────────────────────────────────────
            if is_last:
                last_word_end = chunk_words[-1].get("end", reg_end)
                # Use the earlier of (last-word-end, segment-end)
                t_end = last_word_end if last_word_end < reg_end else reg_end
            else:
                # End at the actual end timestamp of the last word in this chunk
                last_w = chunk_words[-1]
                t_end  = last_w.get("end",
                         last_w.get("start", t_start) + 1.0)

            lines += [str(sub_counter),
                      f"{_srt_ts(t_start)} --> {_srt_ts(t_end)}",
                      text, ""]
            sub_counter += 1

    return "\n".join(lines)


def _parse_srt_to_duration_format(final_srt: str) -> str:
    """
    Convert an SRT string into the duration-annotated format sent to Gemini.

    Each output line looks like:
        [4.139s] A little pause... and we are back. [-0.033s]

    Prefix [Xs]  — how long this subtitle is displayed on screen (seconds).
    Suffix [Xs]  — gap between the END of this subtitle and the START of the
                   next one.  Zero or negative means no gap: the next subtitle
                   starts immediately (or even overlaps slightly).

    A legend explaining these conventions is prepended so the model can use
    the timing information when deciding translation length and phrasing.
    """
    if not final_srt:
        return ""
    pattern = re.compile(
        r'\d+\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|\n$|$)',
        re.MULTILINE)
    matches = list(pattern.finditer(final_srt))

    def toSec(ts):
        h, m, s_ms = ts.split(':')
        s, ms = s_ms.split(',')
        return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

    lines = []
    for i, m in enumerate(matches):
        start    = toSec(m.group(1))
        end      = toSec(m.group(2))
        duration = end - start
        gap      = toSec(matches[i+1].group(1)) - end if i < len(matches)-1 else 0.0
        cleanText = m.group(3).replace('\n', ' ').strip()
        lines.append(f"[{duration:.3f}s] {cleanText} [{gap:.3f}s]")

    legend = (
        "=== FORMAT LEGEND ===\n"
        "Each subtitle line is written as:\n"
        "    [DURATION] subtitle text [GAP]\n"
        "\n"
        "  • DURATION (prefix, e.g. [4.139s]) — how long this subtitle is\n"
        "    displayed on screen.  Use this to judge how long the translated\n"
        "    audio should be: a short duration means a short, punchy translation.\n"
        "\n"
        "  • GAP (suffix, e.g. [0.033s] or [-0.033s]) — the silence between\n"
        "    the end of this subtitle and the start of the next one.\n"
        "    Zero or negative gap means there is NO pause — the next subtitle\n"
        "    starts immediately (or slightly overlaps).  A positive gap means\n"
        "    there is a natural breath / pause between the two lines.\n"
        "=== END LEGEND ===\n"
    )
    return legend + "\n" + "\n".join(lines)


def _parse_srt_to_analysis_format(final_srt: str) -> str:
    """
    Convert an SRT string into the detailed analysis format (mirrors Format_srt_V2.py).

    Outputs a header row followed by one data line per segment:
        [Segment duration][Gap after segment][Gap %][Total available] [Syllables] [Syl/s] [Rel syl/s] [Words] text

    Uses pyphen for syllable counting when available; falls back to a vowel-group
    estimate otherwise.
    """
    if not final_srt:
        return ""

    # ── Parse SRT ─────────────────────────────────────────────────────────────
    def _to_sec(ts):
        h, m, s_ms = ts.split(':')
        s, ms = s_ms.split(',')
        return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

    pattern = re.compile(
        r'\d+\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|\n$|$)',
        re.MULTILINE)
    matches = list(pattern.finditer(final_srt))
    if not matches:
        return ""

    segments = []
    for m in matches:
        start = _to_sec(m.group(1))
        end   = _to_sec(m.group(2))
        text  = m.group(3).replace('\n', ' ').strip()
        segments.append({'start': start, 'end': end, 'text': text})

    # ── Syllable counting ─────────────────────────────────────────────────────
    if PYPHEN_AVAILABLE:
        dic = _pyphen_module.Pyphen(lang='en')
        def _count_syllables(text):
            words = re.findall(r'\b[a-zA-Z]+\b', text)
            count = 0
            for w in words:
                count += len(dic.inserted(w).split('-'))
            return count
    else:
        def _count_syllables(text):
            # Rough fallback: count vowel groups
            return max(1, len(re.findall(r'[aeiouAEIOU]+', text)))

    # ── Pass 1: base calculations & average syllables/sec ─────────────────────
    total_syl_per_sec = 0.0
    valid_count = 0

    for seg in segments:
        seg['duration']    = max(seg['end'] - seg['start'], 0.001)
        seg['words']       = len(re.findall(r'\b\w+\b', seg['text']))
        seg['syllables']   = _count_syllables(seg['text'])
        seg['syl_per_sec'] = seg['syllables'] / seg['duration']
        if seg['words'] > 0:
            total_syl_per_sec += seg['syl_per_sec']
            valid_count += 1

    avg_syl_per_sec = total_syl_per_sec / valid_count if valid_count > 0 else 1.0
    if avg_syl_per_sec == 0:
        avg_syl_per_sec = 1.0

    # ── Pass 2: gaps, relative values & format output ─────────────────────────
    header = (
        "[Segment duration][Gap after segment][Gap after segment in percentage wrt segment length]"
        "[Total duration available for dubbing segment] [Number of syllables in the segment] "
        "[Number of syllables per second] [Relative number of syllables per second] "
        "[Number of words] Actual text of segment"
    )
    output_lines = [header]

    for i, seg in enumerate(segments):
        gap             = segments[i+1]['start'] - seg['end'] if i < len(segments)-1 else 0.0
        seg['gap']      = gap
        seg['gap_pct']  = (gap / seg['duration']) * 100
        seg['total_avail']     = seg['duration'] + gap
        seg['rel_syl_per_sec'] = seg['syl_per_sec'] / avg_syl_per_sec

        line = (
            f"[{seg['duration']:.2f}s]"
            f"[{seg['gap']:.2f}s]"
            f"[{seg['gap_pct']:.0f}%]"
            f"[{seg['total_avail']:.2f}s] "
            f"[{seg['syllables']}] "
            f"[{seg['syl_per_sec']:.2f}] "
            f"[{seg['rel_syl_per_sec']:.2f}] "
            f"[{seg['words']}] "
            f"{seg['text']}"
        )
        output_lines.append(line)

    return '\n'.join(output_lines)


def _format_timestamps_as_text(timestamps: list) -> str:
    """
    Format a list of sync timestamp dicts as a human-readable text file.

    Each entry comes from _build_timestamps() and contains:
        index, orig_start_ms, orig_end_ms, synced_start_ms
    """
    header = "[Index] [Orig Start] [Orig End] [Orig Duration] [Synced Start]"
    lines  = [header]
    for entry in timestamps:
        orig_dur = entry['orig_end_ms'] - entry['orig_start_ms']
        lines.append(
            f"[{entry['index']}] "
            f"[{entry['orig_start_ms']}ms] "
            f"[{entry['orig_end_ms']}ms] "
            f"[{orig_dur}ms] "
            f"[{entry['synced_start_ms']}ms]"
        )
    return '\n'.join(lines)


def _parse_timestamps_text(path: str) -> list:
    """Inverse of _format_timestamps_as_text — read a *_sync_timestamps.txt
    file back into the list-of-dicts shape sync_audio_with_timestamps()
    expects. Returns [] when the file is missing or unparsable."""
    entries = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                m = re.match(r"\[(\d+)\]\s+\[(\d+)ms\]\s+\[(\d+)ms\]\s+"
                             r"\[\d+ms\]\s+\[(\d+)ms\]", line.strip())
                if m:
                    entries.append({
                        "index":           int(m.group(1)),
                        "orig_start_ms":   int(m.group(2)),
                        "orig_end_ms":     int(m.group(3)),
                        "synced_start_ms": int(m.group(4)),
                    })
    except Exception:
        return []
    return entries


def _detect_regions_from_audio(y, sr, threshold_db=-42.0, hysteresis_db=6.0, min_sil_ms=150):
    hop   = max(1, int(sr * 0.010))
    win   = hop * 2
    n_fr  = len(y) // hop
    frames = np.array([
        np.sqrt(np.mean(y[i*hop: i*hop+win]**2)) for i in range(n_fr)
    ], dtype=np.float32)
    thr_open  = 10 ** (threshold_db / 20.0)
    thr_close = 10 ** ((threshold_db - abs(hysteresis_db)) / 20.0)
    min_sil_f = max(1, int((min_sil_ms/1000.0) * sr / hop))
    active, raw, seg_start = False, [], 0
    for i, rms in enumerate(frames):
        if not active:
            if rms >= thr_open:
                active, seg_start = True, i
        else:
            if rms < thr_close:
                active = False
                raw.append([seg_start, i])
    if active:
        raw.append([seg_start, n_fr-1])
    merged = []
    for seg in raw:
        if merged and (seg[0] - merged[-1][1]) < min_sil_f:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg[:])
    hop_s = hop / sr
    return [(r[0]*hop_s, r[1]*hop_s) for r in merged]


# ═════════════════════════════════════════════════════════════════════════════
#  LLM provider layer (Vertex AI / Gemini API / OpenAI-compatible)
# ═════════════════════════════════════════════════════════════════════════════

_LLM_SETTINGS_DEFAULTS: Dict[str, str] = {
    "provider":        LLM_PROVIDER_VERTEX,
    "vertex_json":     "",                    # blank → SCRIPT_DIR/vertex_key.json
    "gemini_api_key":  "",
    "openai_base_url": LLM_DEFAULT_BASE_URL,
    "openai_api_key":  "",
    "openai_model":    "",
    "prompt_caching":  "1",
}
_LLM_SETTINGS: Dict[str, str] = dict(_LLM_SETTINGS_DEFAULTS)


def _load_llm_settings() -> None:
    """Load llm_settings.json (if present) over the defaults."""
    global _LLM_SETTINGS
    _LLM_SETTINGS = dict(_LLM_SETTINGS_DEFAULTS)
    try:
        with open(LLM_SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k in _LLM_SETTINGS_DEFAULTS:
                if k in data and isinstance(data[k], str):
                    _LLM_SETTINGS[k] = data[k]
        if _LLM_SETTINGS["provider"] not in LLM_PROVIDERS:
            _LLM_SETTINGS["provider"] = LLM_PROVIDER_VERTEX
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[LLM] Could not read {LLM_SETTINGS_FILE}: {e} — using defaults.")


def _save_llm_settings() -> None:
    with open(LLM_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(_LLM_SETTINGS, f, indent=2)


def _get_llm_settings() -> Dict[str, str]:
    return _LLM_SETTINGS


def _llm_provider_label() -> str:
    """Short human-readable description of the active provider, for the UI."""
    s = _get_llm_settings()
    p = s.get("provider", LLM_PROVIDER_VERTEX)
    if p == LLM_PROVIDER_OPENAI:
        model = s.get("openai_model") or "(model not set)"
        return f"{model} via {s.get('openai_base_url') or '(base URL not set)'}"
    return f"{GEMINI_DEFAULT_MODEL} via {p}"


def _get_vertex_context():
    key_file = (_get_llm_settings().get("vertex_json") or "").strip() \
               or os.path.join(SCRIPT_DIR, "vertex_key.json")
    if not os.path.exists(key_file):
        raise FileNotFoundError(f"Vertex service-account JSON not found at {key_file}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_file
    with open(key_file, "r", encoding="utf-8") as f:
        key_data = json.load(f)
    project_id = key_data.get("project_id")
    if not project_id:
        raise ValueError(f"{os.path.basename(key_file)} is missing 'project_id'.")
    return project_id


def _make_genai_client():
    """google-genai Client for the Vertex or Gemini-API-key providers."""
    if not GENAI_AVAILABLE:
        raise ImportError("google-genai not installed. Run: pip install google-genai")
    s = _get_llm_settings()
    if s.get("provider") == LLM_PROVIDER_GEMINI:
        api_key = (s.get("gemini_api_key") or "").strip()
        if not api_key:
            raise ValueError("Gemini API key is empty — open LLM Settings and paste it.")
        return genai.Client(api_key=api_key)
    project_id = _get_vertex_context()
    return genai.Client(vertexai=True, project=project_id, location="us-central1")


def _openai_chat(prompt: str, model: str, timeout: float = 900.0) -> str:
    """Single-turn /v1/chat/completions call against the configured base URL."""
    s = _get_llm_settings()
    base = (s.get("openai_base_url") or "").strip().rstrip("/")
    if not base:
        raise ValueError("Base URL is empty — open LLM Settings and set it.")
    url = (base + "/chat/completions") if base.endswith("/v1") \
          else (base + "/v1/chat/completions")
    model = (model or "").strip()
    if not model:
        raise ValueError("Model name is empty — open LLM Settings and set it.")
    headers = {"Content-Type": "application/json"}
    api_key = (s.get("openai_api_key") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:400]
        except Exception:
            pass
        raise RuntimeError(f"LLM endpoint returned HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach LLM endpoint {url}: {e.reason}") from e
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise ValueError(f"Unexpected response from {url}: {str(data)[:400]}")


# Prompt-cache registry: (model, sha1-of-prefix) → Gemini cache name.
# None marks a prefix as uncacheable (too small / API rejected) so we stop
# retrying. Caches live server-side with a 1-hour TTL and are recreated
# transparently when they expire.
_GENAI_CACHE_REGISTRY: Dict[Tuple[str, str], Optional[str]] = {}


def _genai_cached_generate(client, model: str, static_prefix: Optional[str],
                           dynamic: str, use_cache: bool) -> str:
    """generate_content with explicit Gemini prompt caching for the static
    prompt prefix. Any cache failure falls back to a plain inline call."""
    inline = (static_prefix or "") + dynamic
    if not (use_cache and static_prefix):
        return client.models.generate_content(model=model, contents=inline).text

    import hashlib
    key = (model, hashlib.sha1(static_prefix.encode("utf-8")).hexdigest())
    cache_name = _GENAI_CACHE_REGISTRY.get(key, "")
    if cache_name is None:                       # known-uncacheable prefix
        return client.models.generate_content(model=model, contents=inline).text
    if not cache_name:
        try:
            cache = client.caches.create(
                model=model,
                config=genai_types.CreateCachedContentConfig(
                    contents=[static_prefix], ttl="3600s"))
            cache_name = cache.name
            _GENAI_CACHE_REGISTRY[key] = cache_name
        except Exception:
            # Prefix below the model's cache minimum, or API doesn't support
            # caching — remember that and never retry for this prefix.
            _GENAI_CACHE_REGISTRY[key] = None
            return client.models.generate_content(model=model, contents=inline).text
    try:
        return client.models.generate_content(
            model=model, contents=dynamic,
            config=genai_types.GenerateContentConfig(cached_content=cache_name)
        ).text
    except Exception:
        # Cache likely expired — forget it so the next call recreates it.
        _GENAI_CACHE_REGISTRY.pop(key, None)
        return client.models.generate_content(model=model, contents=inline).text


def _llm_generate(prompt: str, model: str = GEMINI_DEFAULT_MODEL,
                  static_prefix: Optional[str] = None) -> str:
    """Provider-agnostic text generation. All pipeline LLM calls go through here.

    *static_prefix* is the reusable part (the per-language prompt file); *prompt*
    is the per-request part. Splitting them enables prompt caching: explicit
    Gemini context caching on the Vertex / Gemini-key providers, and implicit
    (automatic server-side) prefix caching on OpenAI-compatible endpoints —
    which also relies on the static prefix coming first in the request."""
    s = _get_llm_settings()
    if s.get("provider") == LLM_PROVIDER_OPENAI:
        return _openai_chat((static_prefix or "") + prompt,
                            (s.get("openai_model") or "").strip() or model)
    client = _make_genai_client()
    use_cache = s.get("prompt_caching", "1") == "1"
    return _genai_cached_generate(client, model, static_prefix, prompt, use_cache)


def _validate_llm_config() -> None:
    """Raise with a user-readable message if the active provider is unusable."""
    s = _get_llm_settings()
    p = s.get("provider", LLM_PROVIDER_VERTEX)
    if p == LLM_PROVIDER_OPENAI:
        if not (s.get("openai_base_url") or "").strip():
            raise ValueError("OpenAI-compatible base URL is empty — open LLM Settings.")
        if not (s.get("openai_model") or "").strip():
            raise ValueError("Model name is empty — open LLM Settings.")
        return
    if not GENAI_AVAILABLE:
        raise ImportError("google-genai not installed. Run: pip install google-genai")
    if p == LLM_PROVIDER_GEMINI:
        if not (s.get("gemini_api_key") or "").strip():
            raise ValueError("Gemini API key is empty — open LLM Settings.")
        return
    _get_vertex_context()


_load_llm_settings()


def _load_lang_prompt(stage: str, language: str) -> str:
    """
    Load a per-language prompt file from the prompts/ directory.

    Layout:
        prompts/Step1_Translation_Prompt_<Language>.txt
        prompts/Step2_Review_Prompt_<Language>.txt
        prompts/Step3_Punctuation_Prompt_<Language>.txt
        prompts/SyncingPrompt_<Language>.txt

    Falls back to the legacy top-level <stage>.txt (Bengali-only) so older
    installs keep working until prompts/ is populated.
    """
    fname = f"{stage}_{language}.txt"
    p = os.path.join(SCRIPT_DIR, "prompts", fname)
    if not os.path.exists(p):
        legacy = os.path.join(SCRIPT_DIR, f"{stage}.txt")
        if os.path.exists(legacy):
            return open(legacy, "r", encoding="utf-8").read()
        raise FileNotFoundError(
            f"Prompt file not found: prompts/{fname}\n"
            f"Create it (adapt from prompts/{stage}_Bengali.txt) so the "
            f"pipeline can target {language}.")
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def _run_gemini_pipeline(formatted_srt: str, model: str = GEMINI_DEFAULT_MODEL,
                         language: str = TTS_DEFAULT_LANGUAGE,
                         steps: int = 3, tm_glossary: str = ""):
    """Runs translation (→ review → punctuation) on the configured LLM provider.

    *steps* selects the prompt chain depth:
        1 — translation only (Step1 prompt)
        2 — translation + review (Step1, Step2)
        3 — translation + review + punctuation (Step1, Step2, Step3)
    Skipped steps pass the previous step's text through unchanged.

    Prompt files are sent as the static prefix of each request so they get
    prompt-cached across audios (see _llm_generate).
    *tm_glossary* (optional) is a translation-memory block of previously
    approved translations, appended to the Step-1 dynamic input so proofed
    phrasing is reused for consistency.
    Returns (translation, review, punctuation, tr_input, rev_input, punc_input)."""
    steps = max(1, min(3, int(steps or 3)))

    p1 = _load_lang_prompt("Step1_Translation_Prompt", language)
    tr_dyn     = f"\n\n=== Formatted SRT Content ===\n{formatted_srt}"
    if tm_glossary:
        tr_dyn += tm_glossary
    tr_input   = p1 + tr_dyn
    tr_result  = _llm_generate(tr_dyn, model, static_prefix=p1)

    rev_result, rev_input = tr_result, ""
    if steps >= 2:
        p2 = _load_lang_prompt("Step2_Review_Prompt", language)
        rev_dyn    = (f"\n\nEnglish text\n{formatted_srt}\n\n"
                      f"{language} Script for Tuning\n{tr_result}")
        rev_input  = p2 + rev_dyn
        rev_result = _llm_generate(rev_dyn, model, static_prefix=p2)

    punc_result, punc_input = rev_result, ""
    if steps >= 3:
        p3 = _load_lang_prompt("Step3_Punctuation_Prompt", language)
        punc_dyn    = f"\n\n{rev_result}"
        punc_input  = p3 + punc_dyn
        punc_result = _llm_generate(punc_dyn, model, static_prefix=p3)

    return tr_result, rev_result, punc_result, tr_input, rev_input, punc_input


def _strip_code_fence(text: str) -> str:
    """Strip ```lang ... ``` fences Gemini sometimes wraps its output in."""
    if not text:
        return text
    m = re.match(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n```\s*$", text, re.DOTALL)
    return m.group(1) if m else text


def _extract_srt_entries(srt_text: str) -> List[Tuple[float, float, str]]:
    """Return (start_sec, end_sec, text) for every SRT block, in order."""
    if not srt_text:
        return []
    pattern = re.compile(
        r'\d+\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|\n$|$)',
        re.MULTILINE)

    def _to_sec(ts):
        h, m, s_ms = ts.split(':')
        s, ms = s_ms.split(',')
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    return [(_to_sec(m.group(1)), _to_sec(m.group(2)),
             m.group(3).replace('\n', ' ').strip())
            for m in pattern.finditer(srt_text)]


def _split_translation_paragraphs(text: str) -> List[str]:
    """Split a dubbing script into its blank-line-separated paragraphs.

    The translation prompts make the LLM emit roughly one paragraph per
    English subtitle segment, so these rows pair up with
    _extract_srt_texts() in the manual-review window."""
    text = _strip_code_fence(text or "").strip()
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]


def _pair_review_rows(en_entries: List[Tuple[float, float, str]],
                      tr_paragraphs: List[str]
                      ) -> List[Tuple[str, str, Optional[float], Optional[float]]]:
    """Pair English SRT segments with translation paragraphs for review.

    Counts rarely match 1:1 — the punctuation pass merges several spoken
    pulses into one continuous paragraph — so when there are more English
    segments than paragraphs, consecutive segments are grouped onto each
    paragraph proportionally by character share. Returns ordered
    (english_text, translation_text, start_sec, end_sec) rows; the times
    span the grouped English segments (None when a row has no English)."""
    en = [(s0, s1, t.strip()) for (s0, s1, t) in en_entries if t and t.strip()]
    tr = [p.strip() for p in tr_paragraphs if p and p.strip()]
    if not tr:
        return [(t, "", s0, s1) for (s0, s1, t) in en]
    if not en:
        return [("", p, None, None) for p in tr]
    if len(en) <= len(tr):
        return [(en[i][2], p, en[i][0], en[i][1]) if i < len(en)
                else ("", p, None, None)
                for i, p in enumerate(tr)]

    tr_total = float(sum(len(p) for p in tr)) or 1.0
    en_total = float(sum(len(t) for (_, _, t) in en)) or 1.0
    prefix = [0.0]                            # prefix[j] = chars in en[:j]
    for (_, _, t) in en:
        prefix.append(prefix[-1] + len(t))

    rows, i, acc = [], 0, 0.0
    for k, p in enumerate(tr):
        if k == len(tr) - 1:
            j = len(en)                       # last paragraph takes the rest
        else:
            acc += len(p)
            target = acc / tr_total * en_total
            # ≥1 segment per row, and leave ≥1 for every remaining paragraph
            j_min = i + 1
            j_max = len(en) - (len(tr) - 1 - k)
            j = j_min
            while j < j_max and prefix[j] < target:
                j += 1
            # step back if the previous boundary is closer to the target
            if j > j_min and (prefix[j] - target) > (target - prefix[j - 1]):
                j -= 1
        rows.append((" ".join(t for (_, _, t) in en[i:j]), p,
                     en[i][0], en[j - 1][1]))
        i = j
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  Translation memory (feedback loop) helpers
#
#  Capture: when the user clicks "Continue to Dubbing" in the review window,
#  the reviewed script is human-proofed — store it (full + row pairs).
#  Reuse: before calling the LLM, an exact English match returns the proofed
#  script for zero tokens; partial matches are injected into the translation
#  prompt as approved reference translations.
#  All helpers are no-ops when translation_memory.py is missing, and they
#  never raise into the pipeline.
# ─────────────────────────────────────────────────────────────────────────────

def _tm_lang(language: str) -> str:
    """Normalized language key used across store/lookup."""
    return (language or "").strip().lower()


def _tm_source_text(en_entries: List[Tuple[float, float, str]]) -> str:
    """Timing-independent English key: the segment texts joined in order.
    (Hashing the formatted SRT would bake in timings/syllable metrics that
    drift between transcription runs of the same audio.)"""
    return " ".join(t.strip() for (_s0, _s1, t) in en_entries if t and t.strip())


def _tm_lookup_full(language: str, en_entries) -> Optional[str]:
    """Proofed full script for this exact English content, or None."""
    if translation_memory is None:
        return None
    src = _tm_source_text(en_entries)
    if not src:
        return None
    try:
        return translation_memory.lookup_full(_tm_lang(language), src)
    except Exception:
        return None


def _tm_glossary_block(language: str, en_entries, cap: int = 40) -> str:
    """Prompt block of previously approved translations found in this source.
    Empty string when there are none (or memory is unavailable)."""
    if translation_memory is None:
        return ""
    src = _tm_source_text(en_entries)
    if not src:
        return ""
    try:
        pairs = translation_memory.lookup_pairs_in_source(
            _tm_lang(language), src, cap=cap)
    except Exception:
        return ""
    if not pairs:
        return ""
    lines = [f"English: {en}\n{language}: {tr}" for en, tr in pairs]
    return ("\n\n=== Previously approved translations "
            "(reuse these verbatim wherever the same English appears) ===\n"
            + "\n\n".join(lines))


def _tm_capture(language: str, en_entries, proofed_text: str,
                source: str = "") -> int:
    """Store a human-reviewed script: full doc + review-row pairs.
    Returns the number of pairs stored (0 when memory is unavailable)."""
    if translation_memory is None:
        return 0
    src = _tm_source_text(en_entries)
    proofed_text = (proofed_text or "").strip()
    if not proofed_text:
        return 0
    lang = _tm_lang(language)
    try:
        if src:
            translation_memory.store_full(lang, src, proofed_text, source)
        rows = _pair_review_rows(
            en_entries, _split_translation_paragraphs(proofed_text))
        pairs = [(en, tr) for (en, tr, _s0, _s1) in rows if en and tr]
        return translation_memory.store_pairs(lang, pairs, source) or 0
    except Exception:
        return 0


def _run_emotion_enrichment(text: str,
                            language: str = TTS_DEFAULT_LANGUAGE,
                            model: str = GEMINI_DEFAULT_MODEL,
                            status_cb=None) -> str:
    """
    Step4: inject ElevenLabs v3 emotion / accent tags into a punctuated script.

    Runs a Gemini pass with prompts/Step4_Emotion_Prompt_<Language>.txt to
    prepend a `[<language> accent]` tag and sprinkle calm/contemplative/slow/
    pause tags through the script in a Sadhguru-style cadence. Words and
    punctuation of the input are preserved verbatim — only tags are added.

    Best-effort: on ANY failure (missing prompt, network, Gemini error) the
    original text is returned so the TTS step is never blocked.
    """
    if not STEP4_EMOTION_ENABLED or not text or not text.strip():
        return text
    try:
        if status_cb:
            status_cb(f"Step4: Emotion enrichment ({language})…")
        prompt = _load_lang_prompt("Step4_Emotion_Prompt", language)
        enriched = _llm_generate(f"\n\n{text}", model, static_prefix=prompt) or ""
        enriched = _strip_code_fence(enriched).strip()
        if not enriched:
            if status_cb:
                status_cb("Step4: Emotion enrichment returned empty — using original text.")
            return text
        if status_cb:
            status_cb("Step4: Emotion enrichment ✓")
        return enriched
    except Exception as e:
        if status_cb:
            status_cb(f"Step4: Emotion enrichment skipped ({e}). Using original text.")
        return text


def _read_syncing_prompt(language: str = TTS_DEFAULT_LANGUAGE) -> str:
    return _load_lang_prompt("SyncingPrompt", language)


def _call_gemini_mapping(en_srt: str, te_srt: str, script_text: str,
                         model: str = GEMINI_DEFAULT_MODEL,
                         language: str = TTS_DEFAULT_LANGUAGE) -> str:
    """Calls Gemini to map English and target-language SRTs. Returns detailed
    mapping text tagged with the language's 2-letter code (HI, BN, TA, …)."""
    base_prompt = _read_syncing_prompt(language)
    dynamic = (
        f"\n\n=== English SRT Content ===\n{en_srt}\n\n"
        f"=== {language} SRT Content ===\n{te_srt}\n\n"
        f"=== Video Script ===\n{script_text}"
    )
    raw = _llm_generate(dynamic, model, static_prefix=base_prompt)

    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in Gemini mapping response.")
        json_str = json_match.group(0)

    data     = json.loads(json_str)
    detailed = data.get("detailed", [])
    tag      = TTS_LANGUAGES.get(language, {}).get("tag", "BN")
    lang_key = language.lower()
    lines    = ["=== DETAILED SUBTITLE MAPPING ===", ""]
    for i, m in enumerate(detailed, 1):
        en_s = ", ".join(str(x) for x in m.get("english", []))
        # Prefer the language-specific key, falling back to legacy
        # "bengali" / "telugu" keys for older prompts.
        tgt_list = (m.get(lang_key) or m.get("bengali")
                    or m.get("telugu") or m.get("translation") or [])
        tgt_s = ", ".join(str(x) for x in tgt_list)
        lines.append(f"[{i}] EN [{en_s}] -> {tag} [{tgt_s}]")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  TTS helpers
# ═════════════════════════════════════════════════════════════════════════════

def _split_text_into_chunks(text: str, max_bytes: int = TTS_MAX_BYTES) -> list:
    """
    Split *text* into a list of chunks where every chunk's UTF-8 encoded size
    is ≤ max_bytes.  Splits preferentially at:
      1. Paragraph breaks  (\\n)
      2. Sentence endings  (. ! ? ।)
      3. Word boundaries   (space)
      4. Hard character split (last resort)
    """
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]

    def _fits(s):
        return len(s.encode("utf-8")) <= max_bytes

    def _split_para(para: str) -> list:
        pieces  = []
        current = ""
        sentences = re.split(r'(?<=[.!?।])\s+', para)
        for sent in sentences:
            if not sent:
                continue
            candidate = (current + " " + sent).strip() if current else sent
            if _fits(candidate):
                current = candidate
            else:
                if current:
                    pieces.append(current)
                if _fits(sent):
                    current = sent
                else:
                    words   = sent.split()
                    current = ""
                    for word in words:
                        candidate = (current + " " + word).strip() if current else word
                        if _fits(candidate):
                            current = candidate
                        else:
                            if current:
                                pieces.append(current)
                            if not _fits(word):
                                buf = ""
                                for ch in word:
                                    if _fits(buf + ch):
                                        buf += ch
                                    else:
                                        pieces.append(buf)
                                        buf = ch
                                current = buf
                            else:
                                current = word
        if current:
            pieces.append(current)
        return pieces

    chunks     = []
    current    = ""
    paragraphs = text.split("\n")

    for idx, para in enumerate(paragraphs):
        sep       = "\n" if idx < len(paragraphs) - 1 else ""
        candidate = (current + "\n" + para).lstrip("\n") if current else para

        if _fits(candidate + sep):
            current = candidate + sep
        else:
            if current:
                chunks.append(current.strip())
                current = ""
            if _fits(para):
                current = para + sep
            else:
                sub = _split_para(para)
                if sub:
                    chunks.extend(sub[:-1])
                    current = sub[-1] + sep if sub else ""

    if current.strip():
        chunks.append(current.strip())

    return chunks or [text]


def _build_tts_client():
    if not TTS_AVAILABLE:
        raise ImportError(
            "google-cloud-texttospeech not installed.\n"
            "Run: pip install google-cloud-texttospeech google-auth")
    key_file = os.path.join(SCRIPT_DIR, "TTS_Key.json")
    if not os.path.exists(key_file):
        raise FileNotFoundError(f"TTS_Key.json not found at {key_file}")
    creds = _sa_module.Credentials.from_service_account_file(
        key_file,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return _tts_module.TextToSpeechClient(credentials=creds)


def synthesize_tts(text: str, output_path: str, status_cb=None,
                   lang_code: str = None,
                   voice_name: str = TTS_DEFAULT_VOICE) -> str:
    """
    Convert text to speech and save WAV to output_path.
    Splits text into byte-safe chunks (≤ 4800 UTF-8 bytes) and joins the audio.
    lang_code is derived from voice_name when not provided.
    Returns output_path.
    """
    import wave as _wave_mod

    # Derive lang_code from voice_name (e.g. "hi-IN-Chirp3-HD-Aoede" → "hi-IN")
    if not lang_code:
        parts     = voice_name.split("-")
        lang_code = "-".join(parts[:2]) if len(parts) >= 2 else \
            TTS_LANGUAGES.get(TTS_DEFAULT_LANGUAGE, {}).get("code", "bn-IN")

    if status_cb:
        status_cb("TTS: Connecting to Google Cloud TTS…")
    client = _build_tts_client()

    # Split into byte-safe chunks
    chunks = _split_text_into_chunks(text)
    total  = len(chunks)

    _SAMPLE_RATE = 24000
    voice_params = _tts_module.VoiceSelectionParams(
        language_code=lang_code, name=voice_name)
    audio_config = _tts_module.AudioConfig(
        audio_encoding=_tts_module.AudioEncoding.LINEAR16,
        sample_rate_hertz=_SAMPLE_RATE)

    out_base         = os.path.splitext(output_path)[0]
    chunk_log_path   = out_base + "_chunks.txt"
    chunk_log_lines  = [
        f"TTS Chunk Log — {os.path.basename(output_path)}",
        f"Platform : Google Cloud TTS",
        f"Voice    : {voice_name}  ({lang_code})",
        f"Total chunks: {total}",
        "",
    ]

    chunk_pcm_list = []
    for i, chunk in enumerate(chunks, 1):
        if status_cb:
            if total > 1:
                status_cb(f"TTS: Generating audio… chunk {i} of {total}")
            else:
                status_cb("TTS: Generating audio…")

        synthesis_input = _tts_module.SynthesisInput(text=chunk)
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice_params, audio_config=audio_config)
        pcm_bytes = response.audio_content
        chunk_pcm_list.append(pcm_bytes)

        # Save individual chunk as WAV
        chunk_audio_path = f"{out_base}_chunk_{i:02d}.wav"
        with _wave_mod.open(chunk_audio_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(_SAMPLE_RATE)
            wf.writeframes(pcm_bytes)

        chunk_log_lines += [
            f"=== CHUNK {i} of {total} ===",
            f"Characters : {len(chunk)}",
            f"Bytes (UTF-8): {len(chunk.encode('utf-8'))}",
            f"Audio saved : {os.path.basename(chunk_audio_path)}",
            "--- Text ---",
            chunk,
            "",
        ]

    # Write chunk manifest
    with open(chunk_log_path, "w", encoding="utf-8") as lf:
        lf.write("\n".join(chunk_log_lines))

    if status_cb:
        chunk_note = f" ({total} chunks joined)" if total > 1 else ""
        status_cb(f"TTS: Saving → {os.path.basename(output_path)}…{chunk_note}")

    # Concatenate all PCM chunks and write single WAV
    all_pcm = b"".join(chunk_pcm_list)
    with _wave_mod.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(all_pcm)

    return output_path


def _split_text_for_elevenlabs(text: str,
                               max_chars: int = ELEVENLABS_CHUNK_CHARS) -> list:
    """
    Split text into chunks of at most max_chars characters.

    Strategy (in priority order):
      1. Accumulate consecutive paragraphs into one chunk as long as the total
         stays under max_chars.  Blank lines between paragraphs are ignored —
         they are NOT flush points.  This prevents tiny single-paragraph chunks.
      2. If a single paragraph exceeds max_chars, split it at sentence boundaries.
      3. If a sentence exceeds max_chars, split it at word boundaries.
    """
    if len(text) <= max_chars:
        return [text]

    def _fits(s: str) -> bool:
        return len(s) <= max_chars

    def _split_para(para: str) -> list:
        pieces, current = [], ""
        for sent in re.split(r'(?<=[.!?।])\s+', para):
            if not sent:
                continue
            candidate = (current + " " + sent).strip() if current else sent
            if _fits(candidate):
                current = candidate
            else:
                if current:
                    pieces.append(current)
                if _fits(sent):
                    current = sent
                else:
                    # Split at word boundaries
                    current = ""
                    for word in sent.split():
                        candidate = (current + " " + word).strip() if current else word
                        if _fits(candidate):
                            current = candidate
                        else:
                            if current:
                                pieces.append(current)
                            current = word
        if current:
            pieces.append(current)
        return pieces

    # Walk paragraph by paragraph.  Blank lines are skipped — they are NOT
    # treated as chunk boundaries.  Paragraphs keep accumulating into `current`
    # until adding the next one would exceed max_chars.  This ensures that
    # several short paragraphs are joined into a single chunk rather than each
    # being sent as a separate (tiny) API call.
    chunks, current = [], ""
    for para in text.split("\n"):
        if not para.strip():
            continue                    # skip blank lines — don't flush here
        candidate = (current + "\n" + para).strip() if current else para
        if _fits(candidate):
            current = candidate         # still under limit — keep accumulating
        else:
            if current:
                chunks.append(current)  # flush what we have
            if _fits(para):
                current = para          # start fresh with this paragraph
            else:
                sub = _split_para(para) # paragraph itself is too long — split it
                chunks.extend(sub[:-1])
                current = sub[-1] if sub else ""
    if current:
        chunks.append(current)
    chunks = chunks if chunks else [text]

    # Safety: never split inside an ElevenLabs v3 tag like "[bengali accent]"
    # or "[calm]". If a chunk ends with an unclosed "[" (more "[" than "]"),
    # peel the trailing fragment off and prepend it to the next chunk so the
    # tag stays intact when sent to ElevenLabs.
    if len(chunks) > 1:
        fixed = []
        for idx, ch in enumerate(chunks):
            if idx < len(chunks) - 1 and ch.count("[") > ch.count("]"):
                cut = ch.rfind("[")
                if cut > 0:
                    head, tail = ch[:cut].rstrip(), ch[cut:]
                    fixed.append(head)
                    chunks[idx + 1] = tail + " " + chunks[idx + 1].lstrip()
                    continue
            fixed.append(ch)
        chunks = fixed

    return chunks


_INDIC_FONT_CACHE = "unset"


def _indic_matplotlib_font():
    """First installed font family that can render Indic scripts, for
    matplotlib text labels (DejaVu, the default, draws boxes). Cached.
    Returns a family-name list (matplotlib accepts a fallback list) or None."""
    global _INDIC_FONT_CACHE
    if _INDIC_FONT_CACHE != "unset":
        return _INDIC_FONT_CACHE
    fams = None
    try:
        from matplotlib import font_manager
        avail = {f.name for f in font_manager.fontManager.ttflist}
        cands = ["Nirmala UI",                # Windows — most Indic scripts
                 "Kohinoor Devanagari",       # macOS
                 "Devanagari Sangam MN",      # macOS
                 "Bangla Sangam MN",          # macOS — Bengali
                 "Kohinoor Bangla",
                 "Noto Sans Devanagari",
                 "Noto Sans Bengali",
                 "Arial Unicode MS"]
        found = [c for c in cands if c in avail]
        fams = found or None
    except Exception:
        fams = None
    _INDIC_FONT_CACHE = fams
    return fams


# Sentence terminators: ASCII . ! ? …  plus Indic danda । and double-danda ॥
_SENTENCE_END_RE = re.compile(r'(?<=[.!?…।॥])\s+')
_ONLY_TAGS_RE    = re.compile(r'(?:\[[^\]]+\]\s*)+$')


def _split_script_into_sentences(text: str) -> List[str]:
    """
    Split a script into sentence-level segments for per-sentence TTS.
    Blank lines are hard paragraph breaks; single newlines inside a paragraph
    are treated as spaces. Inline emotion tags ([calm], [pause]…) that stand
    alone stay attached to the sentence that follows them.
    """
    sentences = []
    for para in re.split(r'\n\s*\n', text):
        para = re.sub(r'\s*\n\s*', ' ', para.strip())
        if not para:
            continue
        parts = [p.strip() for p in _SENTENCE_END_RE.split(para) if p.strip()]
        carry = ""
        for p in parts:
            if carry:
                p = carry + " " + p
                carry = ""
            if _ONLY_TAGS_RE.fullmatch(p):
                carry = p          # tag-only fragment → prefix of next sentence
                continue
            sentences.append(p)
        if carry:                  # trailing tag-only fragment
            if sentences:
                sentences[-1] += " " + carry
            else:
                sentences.append(carry)
    return sentences


def _elevenlabs_tts_post(chunk: str, api_key: str, voice_id: str, model_id: str,
                         previous_text: str = None, next_text: str = None) -> bytes:
    """
    Single ElevenLabs text-to-speech request → raw MP3 bytes.

    previous_text / next_text enable request-stitching: when synthesizing one
    sentence in isolation the surrounding script is sent as context so prosody
    matches the neighbouring segments. Models that reject those fields get one
    automatic retry without them.
    """
    body = {
        "text": chunk,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.35,
            "similarity_boost": 0.80,
            "style": 0.40,
            "use_speaker_boost": True,
        },
    }
    # ElevenLabs caps stitching context; trailing/leading 600 chars is plenty
    # for prosody continuity without bloating the request.
    if previous_text:
        body["previous_text"] = previous_text[-600:]
    if next_text:
        body["next_text"] = next_text[:600]

    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "audio/mpeg",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=180, context=_SSL_CTX) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        if e.code in (400, 422) and (previous_text or next_text):
            return _elevenlabs_tts_post(chunk, api_key, voice_id, model_id)
        if e.code == 401:
            raise ValueError("ElevenLabs rejected the API key (401). Re-paste a valid key.") from None
        if e.code == 404:
            raise ValueError(
                f"ElevenLabs voice not found (404). voice_id={voice_id!r} "
                "is not on this account. Click 'Refresh Voices' and pick "
                "a voice from the dropdown again.") from None
        if e.code == 422:
            raise ValueError(
                "ElevenLabs rejected the request (422). "
                f"Voice may not support the target language. Details: {err_body}") from None
        if e.code == 429:
            raise ValueError("ElevenLabs rate limit hit (429). Try again shortly.") from None
        raise ValueError(f"ElevenLabs TTS error (HTTP {e.code}): {err_body}") from None
    except urllib.error.URLError as e:
        raise ValueError(f"Network error during ElevenLabs TTS: {e.reason}") from None


def synthesize_tts_elevenlabs(text: str, output_path: str, api_key: str,
                               voice_id: str = ELEVENLABS_TTS_VOICE_ID,
                               model_id: str = ELEVENLABS_TTS_MODEL,
                               status_cb=None) -> str:
    """
    Convert target-language text to speech using ElevenLabs TTS (eleven_v3
    auto-detects the script) and save MP3 to output_path. Sends text in
    chunks of ~ELEVENLABS_CHUNK_CHARS characters and concatenates the
    resulting audio. Returns output_path.

    Validates inputs up-front so we never POST a broken request:
      • API key non-empty
      • voice_id non-empty (raises a clear error if no voice is loaded)
      • text non-empty (after stripping)
    """
    if not api_key or not api_key.strip():
        raise ValueError("ElevenLabs API key is missing — paste it in the TTS Settings panel.")
    if not voice_id or not str(voice_id).strip():
        raise ValueError(
            "No ElevenLabs voice loaded yet. Paste a valid ElevenLabs API key in the "
            "TTS Settings panel — the voice list will populate automatically.")
    if not text or not text.strip():
        raise ValueError("TTS text is empty — nothing to synthesize.")

    api_key  = api_key.strip()
    # Sanitize voice_id BEFORE it ever touches the URL. If a display label
    # (e.g. "✦ Aria — abc12345…") leaks through here, urllib raises
    # "URL can't contain control characters (found at least ' ')".
    raw_voice_id = str(voice_id).strip()
    voice_id = _sanitize_voice_id(raw_voice_id)
    if not voice_id:
        raise ValueError(
            "Invalid ElevenLabs voice_id "
            f"(received: {raw_voice_id!r}). Re-select a voice from the "
            "dropdown — the value must be the raw voice ID, not a display label.")
    model_id = (model_id or ELEVENLABS_TTS_MODEL).strip() or ELEVENLABS_TTS_MODEL

    # Inline audio tags ([calm], [pause], [fast]…) are an eleven_v3 feature.
    # Older models (multilingual v2, turbo/flash v2.5) would read them aloud,
    # so strip them from the script for anything that isn't v3.
    if not model_id.startswith("eleven_v3"):
        stripped = _strip_emotion_tags(text)
        if stripped.strip():
            text = stripped

    if status_cb:
        status_cb("TTS: Connecting to ElevenLabs…")

    chunks = _split_text_for_elevenlabs(text)
    total  = len(chunks)

    out_base        = os.path.splitext(output_path)[0]
    chunk_log_path  = out_base + "_chunks.txt"
    chunk_log_lines = [
        f"TTS Chunk Log — {os.path.basename(output_path)}",
        f"Platform : ElevenLabs",
        f"Voice ID : {voice_id}",
        f"Model    : {model_id}",
        f"Total chunks: {total}",
        "",
    ]

    chunk_bytes_list = []

    for i, chunk in enumerate(chunks, 1):
        if status_cb:
            if total > 1:
                status_cb(f"TTS: ElevenLabs generating audio… chunk {i} of {total}")
            else:
                status_cb("TTS: ElevenLabs generating audio…")

        # Indic-tuned voice settings: slightly higher stability + style 0
        # produce cleaner pronunciation of conjunct consonants and matras
        # across Devanagari / Bengali / Tamil / Telugu / Kannada / Malayalam /
        # Gujarati / Odia / Assamese scripts.
        # NOTE: do NOT send `language_code` — eleven_v3 auto-detects the
        # target language from the input text. Passing language_code triggers
        # HTTP 400 `unsupported_language` on multilingual models.
        # Lower stability + raised style give eleven_v3 room to act on the
        # inline emotion / accent tags injected by Step4 (e.g. [bengali accent],
        # [calm], [slow], [pause]) so delivery feels human and reflective —
        # closer to a wise teacher (Sadhguru-style cadence) than a flat read.
        audio_bytes = _elevenlabs_tts_post(chunk, api_key, voice_id, model_id)
        chunk_bytes_list.append(audio_bytes)

        # Save individual chunk as MP3 (ElevenLabs returns MP3 bytes)
        chunk_audio_path = f"{out_base}_chunk_{i:02d}.mp3"
        with open(chunk_audio_path, "wb") as cf:
            cf.write(audio_bytes)

        # Add entry to chunk log
        chunk_log_lines += [
            f"=== CHUNK {i} of {total} ===",
            f"Characters : {len(chunk)}",
            f"Bytes (UTF-8): {len(chunk.encode('utf-8'))}",
            f"Audio saved : {os.path.basename(chunk_audio_path)}",
            "--- Text ---",
            chunk,
            "",
        ]

    # Write chunk manifest
    with open(chunk_log_path, "w", encoding="utf-8") as lf:
        lf.write("\n".join(chunk_log_lines))

    if status_cb:
        chunk_note = f" ({total} chunks joined)" if total > 1 else ""
        status_cb(f"TTS: Saving → {os.path.basename(output_path)}…{chunk_note}")

    # ElevenLabs returns MP3 — decode with pydub and export as WAV
    try:
        from pydub import AudioSegment
        combined = AudioSegment.empty()
        for raw in chunk_bytes_list:
            seg = AudioSegment.from_file(io.BytesIO(raw), format="mp3")
            combined += seg
        combined.export(output_path, format="wav")
    except ImportError:
        if status_cb:
            status_cb("TTS: Warning — pydub not found; saving raw MP3 bytes (install pydub for WAV output).")
        with open(output_path, "wb") as f:
            for raw in chunk_bytes_list:
                f.write(raw)

    return output_path


# ─── Multi-speaker dubbing ────────────────────────────────────────────────────
# Per-project map of paragraph-index → ElevenLabs voice. Saved next to the
# other pipeline outputs as <base>_speakers.json so it travels with the
# project and is reused on every re-run / re-dub.

def _speakers_file_path(base: str) -> str:
    return base + "_speakers.json"


def _speakers_load(base: str) -> dict:
    try:
        with open(_speakers_file_path(base), "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _speakers_save(base: str, d: dict) -> None:
    with open(_speakers_file_path(base), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def _speakers_voice_map(base: str) -> Dict[int, str]:
    """{paragraph_index: voice_id} for every valid saved assignment."""
    out: Dict[int, str] = {}
    for k, v in (_speakers_load(base).get("assignments") or {}).items():
        try:
            idx = int(k)
        except Exception:
            continue
        vid = _sanitize_voice_id((v or {}).get("voice_id", ""))
        if vid:
            out[idx] = vid
    return out


def synthesize_tts_elevenlabs_multi(paragraphs: List[str],
                                    voice_map: Dict[int, str],
                                    output_path: str, api_key: str,
                                    default_voice_id: str,
                                    model_id: str = ELEVENLABS_TTS_MODEL,
                                    status_cb=None, enrich_cb=None) -> str:
    """Multi-speaker ElevenLabs TTS: each paragraph is voiced by
    voice_map.get(index, default_voice_id); consecutive paragraphs that share
    a voice are synthesized together (fewer API calls, natural flow), and all
    pieces are concatenated in order into one WAV at output_path.

    enrich_cb (optional): called once per same-voice group with the group's
    text — used for emotion enrichment so paragraph indices stay stable."""
    if not api_key or not api_key.strip():
        raise ValueError("ElevenLabs API key is missing — paste it in the TTS Settings panel.")
    default_voice_id = _sanitize_voice_id(default_voice_id)
    if not default_voice_id:
        raise ValueError("No valid default ElevenLabs voice selected.")
    paragraphs = [p for p in (paragraphs or []) if p and p.strip()]
    if not paragraphs:
        raise ValueError("TTS text is empty — nothing to synthesize.")
    model_id = (model_id or ELEVENLABS_TTS_MODEL).strip() or ELEVENLABS_TTS_MODEL

    # Group consecutive paragraphs by their assigned voice.
    groups: List[Tuple[str, List[str]]] = []
    for i, para in enumerate(paragraphs):
        vid = _sanitize_voice_id(voice_map.get(i, "")) or default_voice_id
        if groups and groups[-1][0] == vid:
            groups[-1][1].append(para)
        else:
            groups.append((vid, [para]))

    out_base        = os.path.splitext(output_path)[0]
    chunk_log_lines = [
        f"TTS Chunk Log — {os.path.basename(output_path)} (multi-speaker)",
        f"Platform : ElevenLabs",
        f"Model    : {model_id}",
        f"Speakers : {len({g[0] for g in groups})} distinct voice(s), "
        f"{len(groups)} group(s)",
        "",
    ]
    chunk_bytes_list = []
    n_chunk = 0
    for g_no, (vid, paras) in enumerate(groups, 1):
        text = "\n\n".join(paras)
        if enrich_cb:
            try:
                enriched = enrich_cb(text)
                if enriched and enriched.strip():
                    text = enriched
            except Exception:
                pass
        if not model_id.startswith("eleven_v3"):
            stripped = _strip_emotion_tags(text)
            if stripped.strip():
                text = stripped
        chunks = _split_text_for_elevenlabs(text)
        for chunk in chunks:
            n_chunk += 1
            if status_cb:
                status_cb(f"TTS: speaker group {g_no}/{len(groups)} "
                          f"(voice …{vid[-6:]}) — chunk {n_chunk}…")
            audio_bytes = _elevenlabs_tts_post(chunk, api_key, vid, model_id)
            chunk_bytes_list.append(audio_bytes)
            chunk_audio_path = f"{out_base}_chunk_{n_chunk:02d}.mp3"
            with open(chunk_audio_path, "wb") as cf:
                cf.write(audio_bytes)
            chunk_log_lines += [
                f"=== CHUNK {n_chunk} · group {g_no} · voice {vid} ===",
                f"Characters : {len(chunk)}",
                f"Audio saved : {os.path.basename(chunk_audio_path)}",
                "--- Text ---",
                chunk,
                "",
            ]

    with open(out_base + "_chunks.txt", "w", encoding="utf-8") as lf:
        lf.write("\n".join(chunk_log_lines))

    if status_cb:
        status_cb(f"TTS: Saving → {os.path.basename(output_path)}… "
                  f"({n_chunk} chunks, {len(groups)} speaker groups joined)")
    try:
        from pydub import AudioSegment
        combined = AudioSegment.empty()
        for raw in chunk_bytes_list:
            combined += AudioSegment.from_file(io.BytesIO(raw), format="mp3")
        combined.export(output_path, format="wav")
    except ImportError:
        with open(output_path, "wb") as f:
            for raw in chunk_bytes_list:
                f.write(raw)
    return output_path


# ═════════════════════════════════════════════════════════════════════════════
#  Sync algorithm helpers (from Audio_File_Sync_New.py)
# ═════════════════════════════════════════════════════════════════════════════

def _sync_srt_ts(ms: float) -> str:
    ms = max(0, int(round(ms)))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def _parse_srt_time(t: str) -> float:
    t = t.strip().replace(",", ".")
    hms, ms_part = t.split(".")
    h, m, s = hms.split(":")
    return (int(h)*3600 + int(m)*60 + int(s)) * 1000 + int(ms_part)


class Subtitle:
    def __init__(self, index, start, end, text):
        self.index = index
        self.start = start
        self._dur  = end - start
        self.text  = text

    @property
    def length(self): return self._dur
    @property
    def end(self): return self.start + self._dur
    def shift(self, delta): self.start += delta


class MappingGroup:
    def __init__(self, no, en, te):
        self.no = no
        self.en = en
        self.te = te

    @property
    def mtype(self):
        e, t = len(self.en), len(self.te)
        if e == 1 and t == 1: return "1to1"
        if t == 1 and e >  1: return "Mto1"
        if e == 1 and t >  1: return "1toM"
        return "MtoM"


class Section:
    def __init__(self, no, start, end, gap_before=0.0, gap_after=0.0):
        self.no = no; self.start = start; self.end = end
        self.gap_before = gap_before; self.gap_after = gap_after

    @property
    def length(self):         return self.end - self.start
    @property
    def len_gap_after(self):  return self.length + self.gap_after
    @property
    def len_gap_before(self): return self.length + self.gap_before
    @property
    def len_both_gaps(self):  return self.gap_before + self.length + self.gap_after
    @property
    def start_pad(self):      return self.start - self.gap_before
    @property
    def end_pad(self):        return self.end   + self.gap_after


def _parse_srt_from_string(content: str) -> Dict[int, Subtitle]:
    subs: Dict[int, Subtitle] = {}
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        m = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            lines[1])
        if not m:
            continue
        subs[idx] = Subtitle(idx, _parse_srt_time(m.group(1)),
                             _parse_srt_time(m.group(2)),
                             "\n".join(lines[2:]))
    return subs


def _write_srt_from_dict(subs: Dict[int, Subtitle]) -> str:
    lines = []
    for i, s in enumerate(sorted(subs.values(), key=lambda x: x.start), 1):
        lines += [str(i), f"{_sync_srt_ts(s.start)} --> {_sync_srt_ts(s.end)}", s.text, ""]
    return "\n".join(lines)


# ─── Caption-style re-chunking ───────────────────────────────────────────────
# Re-flow a synced SRT into short, single-line caption cues — useful for
# burned-in subtitles, social media reels, and karaoke-style overlays.
def _caption_chunks(text: str, max_chars: int) -> List[str]:
    """
    Split a Bengali subtitle line into chunks each ≤ max_chars characters,
    preserving Unicode and breaking at whitespace whenever possible. If a
    single token is longer than max_chars (rare for Bengali words but happens
    with conjunct-heavy compounds), it is hard-split at the character limit
    so no chunk ever exceeds the cap.
    """
    text = (text or "").replace("\n", " ").strip()
    if not text:
        return []
    if max_chars <= 0:
        return [text]
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    current = ""
    # Tokenise on whitespace — Bengali, like English, uses spaces between words.
    for word in text.split():
        if len(word) > max_chars:
            # Flush whatever we accumulated before the oversized word.
            if current:
                chunks.append(current)
                current = ""
            # Hard-split the long word.
            for k in range(0, len(word), max_chars):
                piece = word[k:k + max_chars]
                if len(piece) == max_chars:
                    chunks.append(piece)
                else:
                    current = piece
            continue
        candidate = (current + " " + word).strip() if current else word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks


def build_caption_srt(synced_srt_text: str,
                      max_chars: int = 10,
                      max_secs:  float = 1.0) -> str:
    """
    Re-chunk a synced SRT into single-line caption cues that satisfy:
        • each cue contains AT MOST `max_chars` characters
        • each cue spans AT MOST `max_secs` seconds

    Time within each source cue is split proportionally by character count so
    shorter chunks get shorter on-screen durations and the audio still lines
    up. Cues that are already short enough pass through unchanged.

    Returns a fresh SRT text (UTF-8 safe). Empty cues are dropped.
    """
    subs = _parse_srt_from_string(synced_srt_text or "")
    if not subs:
        return ""

    out_lines: List[str] = []
    out_idx = 1
    for s in sorted(subs.values(), key=lambda x: x.start):
        text = (s.text or "").replace("\n", " ").strip()
        if not text:
            continue

        chunks = _caption_chunks(text, max_chars)
        if not chunks:
            continue

        cue_dur_ms = max(1, int(round(s.end - s.start)))   # ms (Subtitle uses ms)
        max_chunk_ms = max(50, int(round(max_secs * 1000.0)))

        # Distribute cue duration proportional to chunk length, but cap each
        # chunk at max_secs so captions never linger past the limit.
        total_chars = sum(len(c) for c in chunks) or 1
        cursor_ms = float(s.start)
        for ci, chunk in enumerate(chunks):
            share = (len(chunk) / total_chars) * cue_dur_ms
            chunk_ms = min(max_chunk_ms, max(120, int(round(share))))
            start_ms = cursor_ms
            end_ms = min(float(s.end), start_ms + chunk_ms)
            # On the last piece, never exceed the original cue end.
            if ci == len(chunks) - 1:
                end_ms = float(s.end)
                # But still respect the per-chunk cap.
                if end_ms - start_ms > max_chunk_ms:
                    end_ms = start_ms + max_chunk_ms
            if end_ms <= start_ms:
                end_ms = start_ms + 120

            out_lines += [
                str(out_idx),
                f"{_sync_srt_ts(start_ms)} --> {_sync_srt_ts(end_ms)}",
                chunk,
                "",
            ]
            out_idx += 1
            cursor_ms = end_ms

    return "\n".join(out_lines)


def _parse_mapping_from_string(content: str) -> List[MappingGroup]:
    """
    Parse Gemini mapping output. Accepts every per-language 2-letter tag
    registered in TTS_LANGUAGES (BN, HI, KN, ML, AS, OR, NE, TA, TE, GU, MR)
    plus the legacy full-word `BENGALI` / `BANGLA` forms, so prompts can
    migrate incrementally without breaking the syncing pipeline.
    """
    groups = []
    _tags = {info.get("tag", "") for info in TTS_LANGUAGES.values()}
    _tags.update({"BN", "TE", "BENGALI", "BANGLA"})
    tag_alt = "|".join(sorted(t for t in _tags if t))
    pattern = re.compile(
        r"\[(\d+)\]\s*EN\s*\[([^\]]+)\]\s*->\s*"
        rf"(?:{tag_alt})\s*\[([^\]]+)\]",
        re.IGNORECASE)
    for line in content.splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        groups.append(MappingGroup(
            no=int(m.group(1)),
            en=[int(x.strip()) for x in m.group(2).split(",")],
            te=[int(x.strip()) for x in m.group(3).split(",")]))
    groups.sort(key=lambda g: g.no)
    return groups


def _build_section_table(mappings, en_subs, te_subs, processed):
    raw = {}
    for mg in mappings:
        if mg.no in processed:
            valid = [i for i in mg.te if i in te_subs]
            if not valid: continue
            raw[mg.no] = (min(te_subs[i].start for i in valid),
                          max(te_subs[i].end   for i in valid))
        else:
            valid = [i for i in mg.en if i in en_subs]
            if not valid: continue
            raw[mg.no] = (min(en_subs[i].start for i in valid),
                          max(en_subs[i].end   for i in valid))
    if not raw:
        return {}
    ordered = sorted(raw.keys(), key=lambda n: raw[n][0])
    table   = {}
    for i, no in enumerate(ordered):
        start, end = raw[no]
        gb = max(0.0, start - raw[ordered[i-1]][1]) if i > 0 else 0.0
        ga = max(0.0, raw[ordered[i+1]][0] - end) if i < len(ordered)-1 else 0.0
        table[no] = Section(no, start, end, gb, ga)
    return table


MIN_SPRING = 10.0

def _te_valid(mg, te): return [i for i in mg.te if i in te]
def _te_start(mg, te):
    v = _te_valid(mg, te); return min(te[i].start for i in v) if v else 0.0
def _te_end(mg, te):
    v = _te_valid(mg, te); return max(te[i].end   for i in v) if v else 0.0
def _te_len(mg, te): return _te_end(mg, te) - _te_start(mg, te)
def _te_len_np(mg, te): return sum(te[i].length for i in mg.te if i in te)
def _shift(mg, delta, te):
    for i in mg.te:
        if i in te: te[i].shift(delta)

def _align_center(mg, te, target):
    cur = (_te_start(mg, te) + _te_end(mg, te)) / 2
    _shift(mg, target - cur, te)
def _align_start(mg, te, target): _shift(mg, target - _te_start(mg, te), te)
def _align_end(mg, te, target):   _shift(mg, target - _te_end(mg, te), te)


def _compress_springs(mg, te, anchor_start, target_end):
    valid = _te_valid(mg, te)
    if not valid: return
    if len(valid) == 1:
        te[valid[0]].start = anchor_start; return
    gaps = [max(0.0, te[valid[k+1]].start - te[valid[k]].end) for k in range(len(valid)-1)]
    total_content = sum(te[i].length for i in valid)
    available     = target_end - anchor_start
    space_gaps    = available - total_content
    new_gaps      = list(gaps)
    free          = list(range(len(gaps)))
    for _ in range(len(gaps)):
        ft = sum(gaps[k] for k in free)
        if ft <= 0: break
        tgt = space_gaps - sum(new_gaps[k] for k in range(len(gaps)) if k not in free)
        if tgt < 0: tgt = 0.0
        ratio  = tgt / ft if ft > 0 else 0.0
        frozen = []
        for k in free:
            p = gaps[k] * ratio
            if p <= MIN_SPRING: new_gaps[k] = MIN_SPRING; frozen.append(k)
            else:               new_gaps[k] = p
        for k in frozen: free.remove(k)
        if not frozen: break
    cursor = anchor_start
    for k, idx in enumerate(valid):
        te[idx].start = cursor
        cursor += te[idx].length
        if k < len(gaps): cursor += new_gaps[k]


def _attach_fixed(mg, te, anchor_start, spring=MIN_SPRING):
    cursor = anchor_start
    for idx in _te_valid(mg, te):
        te[idx].start = cursor
        cursor += te[idx].length + spring


def _fix_min_spring_forward(te, ordered_ids, min_gap=MIN_SPRING):
    for k in range(1, len(ordered_ids)):
        needed = te[ordered_ids[k-1]].end + min_gap
        if te[ordered_ids[k]].start < needed:
            te[ordered_ids[k]].start = needed


def _process_round(rnd, mappings, en_subs, te, processed, log_lines=None):
    count = 0
    for mg in mappings:
        if mg.no in processed:
            continue
        # Rebuild the section table after each section is processed so every
        # subsequent section sees the updated neighbour positions.
        sec_tbl = _build_section_table(mappings, en_subs, te, processed)
        if mg.no not in sec_tbl:
            continue
        tl      = sec_tbl[mg.no]
        te_len  = _te_len(mg, te)
        np_len  = _te_len_np(mg, te)
        mt      = mg.mtype
        te_v    = _te_valid(mg, te)
        en_v    = [i for i in mg.en if i in en_subs]

        eligible = False
        if   rnd == 1: eligible = te_len <= tl.length
        elif rnd == 2: eligible = (mt in ("1toM","MtoM")) and (np_len <= tl.length)
        elif rnd == 3: eligible = te_len <= tl.len_gap_after
        elif rnd == 4: eligible = (mt in ("1toM","MtoM")) and (np_len <= tl.len_gap_after)
        elif rnd == 5:
            bound = tl.len_both_gaps + MIN_SPRING
            eligible = (te_len <= bound) or (np_len <= bound)
        if not eligible:
            continue

        # Capture TE positions before placement for the log
        te_before = {i: (te[i].start, te[i].end) for i in te_v} if log_lines is not None else {}

        strategy = ""
        if rnd == 1:
            if mt == "MtoM":
                ne, nt = len(en_v), len(te_v)
                if ne == nt:
                    for eid, tid in zip(en_v, te_v):
                        te[tid].start = en_subs[eid].start
                    _fix_min_spring_forward(te, te_v)
                    strategy = "MtoM_equal"
                else:
                    for tid in te_v:
                        if en_v: te[tid].start = en_subs[en_v.pop(0)].start
                    _fix_min_spring_forward(te, te_v)
                    strategy = "MtoM_unequal"
            else:
                _align_center(mg, te, (tl.start + tl.end) / 2)
                strategy = "align_center"
        elif rnd == 2: _compress_springs(mg, te, tl.start, tl.end); strategy = "compress_springs"
        elif rnd == 3:
            if mt in ("1to1","Mto1"): _align_start(mg, te, tl.start); strategy = "align_start"
            else:                      _attach_fixed(mg, te, tl.start, MIN_SPRING); strategy = "attach_fixed"
        elif rnd == 4: _attach_fixed(mg, te, tl.start, MIN_SPRING); strategy = "attach_fixed"
        elif rnd == 5:
            target_end = tl.end_pad - MIN_SPRING
            if mt in ("1to1","Mto1"): _align_end(mg, te, target_end); strategy = "align_end"
            else:
                _attach_fixed(mg, te, tl.start, MIN_SPRING)
                _align_end(mg, te, target_end)
                strategy = "attach_fixed+align_end"

        processed.add(mg.no)
        count += 1

        if log_lines is not None:
            en_info = "  ".join(
                f"EN{i}:[{en_subs[i].start:.2f}s-{en_subs[i].end:.2f}s]"
                for i in mg.en if i in en_subs)
            te_info = "  ".join(
                f"TE{i}:{te_before[i][0]:.2f}s→{te[i].start:.2f}s"
                for i in te_v if i in te_before)
            log_lines.append(
                f"  Sec {mg.no:>3} [{mt:<5}]  slot:[{tl.start:.2f}s-{tl.end:.2f}s]"
                f"  {en_info}  {te_info}  [{strategy}]")

    return count


def _process_overflow(mappings, en_subs, te, processed, log_lines=None,
                      en_audio_duration: float = None):
    """
    Place sections that remain unprocessed after iterations 1 & 2.

    Each unprocessed section is moved to its corresponding English section's
    start time PLUS an extra offset. The extra offset is the length of the
    English audio file (en_audio_duration, in SECONDS) when supplied;
    otherwise we fall back to the max end of the English subtitle file.

    NOTE: Subtitle.start/.end are in MILLISECONDS (see _parse_srt_time),
    so en_audio_duration must be converted from seconds → ms here to keep
    the units consistent. Without this conversion the offset is ~1000×
    too small and overflow positioning effectively becomes zero.
    """
    srt_end_ms = max((s.end for s in en_subs.values()), default=0.0)
    if en_audio_duration is not None and en_audio_duration > 0:
        total_dur = float(en_audio_duration) * 1000.0  # seconds → ms
        offset_label = "EN-audio-len"
    elif srt_end_ms > 0:
        total_dur = srt_end_ms
        offset_label = "EN-srt-end"
    else:
        total_dur = 0.0
        offset_label = "ZERO-fallback (no audio length, no SRT end)"

    en_starts  = {}
    for mg in mappings:
        if mg.no in processed: continue
        valid = [i for i in mg.en if i in en_subs]
        if valid: en_starts[mg.no] = min(en_subs[i].start for i in valid)

    # Sections WITH a valid English anchor are placed directly at their
    # English start and allowed to BLEED OVER into the next slot (ported
    # from fast-syncs). No compression, no shoving past placed audio — the
    # order-preserving sweep that runs after placement resolves overlaps by
    # nudging later clips forward, which keeps speech at natural pace and
    # as close to its English timing as physically possible.
    anchored = sorted(
        [mg for mg in mappings if mg.no not in processed and mg.no in en_starts],
        key=lambda mg: en_starts[mg.no])
    # Sections with NO usable English reference still go after the end.
    orphans = [mg for mg in mappings
               if mg.no not in processed and mg.no not in en_starts]

    for mg in anchored:
        te_v = _te_valid(mg, te)
        if not te_v:
            processed.add(mg.no)
            continue
        want = en_starts[mg.no]
        te_before = {i: te[i].start for i in te_v} if log_lines is not None else {}
        _align_start(mg, te, want)
        processed.add(mg.no)
        if log_lines is not None:
            te_info = "  ".join(
                f"TE{i}:{te_before[i]:.2f}s→{te[i].start:.2f}s" for i in te_v)
            log_lines.append(
                f"  Sec {mg.no:>3} [{mg.mtype:<5}]  anchored at EN start "
                f"{want/1000.0:.2f}s (may bleed into the next slot)  "
                f"{te_info}  [bleed-over]")

    if orphans and log_lines is not None:
        log_lines.append(
            f"  Overflow offset for {len(orphans)} EN-less section(s): "
            f"{total_dur/1000.0:.2f}s ({total_dur:.0f} ms) [{offset_label}]")
    placed_ends = []
    for mg in orphans:
        te_v = _te_valid(mg, te)
        if not te_v:
            processed.add(mg.no)
            continue
        ds = total_dur
        if placed_ends and ds < placed_ends[-1]: ds = placed_ends[-1]
        te_before = {i: te[i].start for i in te_v} if log_lines is not None else {}
        _align_start(mg, te, ds)
        placed_ends.append(_te_end(mg, te))
        processed.add(mg.no)
        if log_lines is not None:
            te_info = "  ".join(
                f"TE{i}:{te_before[i]:.2f}s→{te[i].start:.2f}s" for i in te_v)
            log_lines.append(
                f"  Sec {mg.no:>3} [{mg.mtype:<5}]  overflow at {ds:.2f}s  {te_info}  [overflow]")
    return len(anchored) + len(orphans)


def run_sync_from_strings(en_srt_text, te_srt_text, mapping_text,
                          en_audio_duration: float = None):
    """
    Full sync algorithm. Returns (synced_subs, original_te_subs, sync_log).

    en_audio_duration (optional): length of the original English audio in
    SECONDS. When supplied (and > 0) it is used as the overflow offset for
    sections that remain unprocessed after iteration 2 (Stage 3d). If
    omitted/zero, we fall back to the end of the English subtitle file.
    Both values are converted to milliseconds inside _process_overflow
    because Subtitle.start/.end are stored in ms.
    """
    en_subs  = _parse_srt_from_string(en_srt_text)
    te       = {k: deepcopy(v) for k, v in _parse_srt_from_string(te_srt_text).items()}
    orig_te  = {k: deepcopy(v) for k, v in _parse_srt_from_string(te_srt_text).items()}
    mappings = _parse_mapping_from_string(mapping_text)
    if not mappings:
        raise ValueError("No mapping groups found.")
    processed: Set[int] = set()
    log_lines = [
        f"=== Sync Log — {len(mappings)} sections | {len(en_subs)} EN subs | {len(te)} TE subs ==="]
    # Log which overflow offset will be used so debugging is obvious.
    _srt_end_ms = max((s.end for s in en_subs.values()), default=0.0)
    if en_audio_duration is not None and en_audio_duration > 0:
        log_lines.append(
            f"  English audio length: {en_audio_duration:.2f}s "
            f"({en_audio_duration*1000.0:.0f} ms) [used for overflow]")
    elif _srt_end_ms > 0:
        log_lines.append(
            f"  English audio length: not supplied — falling back to "
            f"EN-SRT end = {_srt_end_ms/1000.0:.2f}s ({_srt_end_ms:.0f} ms)")
    else:
        log_lines.append(
            "  WARNING: no English audio length AND no EN-SRT end — "
            "overflow offset will be 0.")
    for iteration in (1, 2):
        for rnd in range(1, 6):
            log_lines.append(f"\n--- Iteration {iteration}, Round {rnd} ---")
            c = _process_round(rnd, mappings, en_subs, te, processed, log_lines=log_lines)
            if c == 0:
                log_lines.append("  (none processed)")
    remaining = len(mappings) - len(processed)
    if remaining:
        log_lines.append(f"\n--- Overflow ({remaining} unprocessed sections) ---")
        _process_overflow(mappings, en_subs, te, processed,
                          log_lines=log_lines,
                          en_audio_duration=en_audio_duration)
    # Order-preserving sweep (ported from fast-syncs): dubbed clips must stay
    # in recording order and must never overlap — overlapping segments get
    # mixed on top of each other in the final audio and sound garbled. Walk
    # the clips in recording order and nudge any offender forward.
    ordered_ids = sorted(te.keys())
    n_pushed, prev_end = 0, None
    for i in ordered_ids:
        if prev_end is not None and te[i].start < prev_end + MIN_SPRING:
            delta = (prev_end + MIN_SPRING) - te[i].start
            te[i].shift(delta)
            if delta > 1.0:
                n_pushed += 1
                log_lines.append(
                    f"  sweep: TE{i} +{delta/1000.0:.2f}s "
                    "(keep order / avoid overlap)")
        prev_end = te[i].end
    if n_pushed:
        log_lines.append(
            f"\n--- Order sweep: {n_pushed} subtitle(s) nudged forward ---")
    log_lines.append(f"\n=== Complete: {len(processed)}/{len(mappings)} sections synced ===")
    all_te_idx = {i for mg in mappings for i in mg.te}
    synced = {i: te[i] for i in sorted(all_te_idx) if i in te}
    # Safety net: dubbed subtitles the LLM mapping never mentioned used to be
    # DROPPED from the output entirely (missing audio in the dub). Keep them
    # at their original TTS timing instead so no speech disappears.
    missing = [i for i in sorted(te) if i not in all_te_idx]
    if missing:
        for i in missing:
            synced[i] = te[i]
        log_lines.append(
            f"⚠ {len(missing)} dubbed subtitle(s) were missing from the "
            f"mapping ({missing[:10]}{'…' if len(missing) > 10 else ''}) — "
            "kept at their original timing instead of being dropped.")
    return synced, orig_te, "\n".join(log_lines)


def _build_timestamps(original_te_subs, synced_subs):
    entries = []
    for idx in sorted(synced_subs.keys()):
        synced   = synced_subs[idx]
        original = original_te_subs.get(idx)
        if original is None: continue
        entries.append({
            "index":           idx,
            "orig_start_ms":   int(round(original.start)),
            "orig_end_ms":     int(round(original.end)),
            "synced_start_ms": int(round(synced.start)),
        })
    return entries


def sync_audio_with_timestamps(audio_path: str, timestamps: list, out_path: str, status_cb=None):
    """Cut & overlay TTS audio according to syncing timestamps. Saves as WAV to out_path."""
    if not PYDUB_AVAILABLE:
        raise ImportError("pydub not installed. Run: pip install pydub")
    if status_cb: status_cb("Sync Audio: Loading audio…")
    ext    = os.path.splitext(audio_path)[1].lower().lstrip(".")
    fmt    = {"m4a": "mp4", "aac": "adts"}.get(ext, ext)
    source = _AudioSegment.from_file(audio_path, format=fmt)

    sorted_ts = sorted(timestamps, key=lambda e: e["synced_start_ms"])
    last_entry = sorted_ts[-1]

    last_end = max(e["synced_start_ms"] + (e["orig_end_ms"] - e["orig_start_ms"])
                   for e in sorted_ts)
    canvas   = _AudioSegment.silent(duration=last_end * 2,
                                    frame_rate=source.frame_rate)

    if status_cb: status_cb(f"Sync Audio: Mixing {len(sorted_ts)} segments…")
    for entry in sorted_ts:
        if entry is last_entry:
            # Last segment: extend to end of TTS source so no word gets clipped
            seg = source[entry["orig_start_ms"]:]
        else:
            seg = source[entry["orig_start_ms"]: entry["orig_end_ms"]]
        canvas = canvas.overlay(seg, position=entry["synced_start_ms"])

    # Trim with generous tail, then smooth fade-out so ending never sounds abrupt
    tail_end = last_end + 1500
    canvas   = canvas[:tail_end].fade_out(1200)
    if status_cb: status_cb(f"Sync Audio: Exporting → {os.path.basename(out_path)}…")
    canvas.export(out_path, format="wav")
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
#  Integrated pipeline backend
# ═════════════════════════════════════════════════════════════════════════════

def run_full_pipeline_single(
        audio_path: str,
        y_data,
        sr: int,
        regions: list,
        status_cb: Callable,
        gemini_model: str = GEMINI_DEFAULT_MODEL,
        language: str = TTS_DEFAULT_LANGUAGE,
) -> dict:
    """
    Runs Stages 2 & 3 after Stage 1 (translation) has already been done.
    Expects:
      audio_path  — path to the original English audio
      y_data      — already-loaded numpy audio array
      sr          — sample rate
      regions     — already-detected regions from Stage 1
    Returns dict with keys: tts_path, synced_path (or error key)
    """
    # Each input audio file gets its own per-file output folder (named after
    # the file, without extension). All outputs (and a copy of the original
    # audio) are placed inside that folder.
    outdir  = _prepare_output_dir(audio_path)
    base    = os.path.join(outdir, os.path.splitext(os.path.basename(audio_path))[0])
    results = {}

    # ── Stage 1 outputs (passed in) ──────────────────────────────────────────
    # Read the FinalScript to recover the target-language translation.
    final_script_path = base + "_FinalScript.txt"
    if not os.path.exists(final_script_path):
        return {"error": f"FinalScript not found: {final_script_path}"}

    with open(final_script_path, "r", encoding="utf-8") as f:
        combined = f.read()

    translation_text = _extract_translation_from_finalscript(combined, language)

    # ── Stage 2: TTS ─────────────────────────────────────────────────────────
    status_cb(f"Stage 2/TTS: Emotion detection ({language})…")
    tts_path = os.path.join(outdir, _tts_output_name(language, audio_path, "_tts"))
    try:
        enriched = _run_emotion_enrichment(translation_text, language=language,
                                           model=gemini_model, status_cb=status_cb)
        status_cb(f"Stage 2/TTS: Converting {language} text to speech…")
        synthesize_tts(_strip_emotion_tags(enriched), tts_path, status_cb=status_cb)
        results["tts_path"] = tts_path
        status_cb(f"Stage 2/TTS: Done → {os.path.basename(tts_path)}")
    except Exception as e:
        return {"error": f"TTS failed: {e}", **results}

    # ── Stage 3a: Transcribe English audio → English SRT ─────────────────────
    status_cb("Stage 3a/Sync: Transcribing English audio for SRT…")
    try:
        api_key     = _get_api_key()
        en_result   = _transcribe_audio(audio_path, api_key)
        en_words    = en_result.get("words", [])
        if not en_words:
            return {"error": "No word data from ElevenLabs for English audio", **results}
        en_srt = _build_english_subtitle_srt(regions, en_words)
        status_cb("Stage 3a/Sync: English SRT generated.")
    except Exception as e:
        return {"error": f"English SRT generation failed: {e}", **results}

    # ── Stage 3b: Load TTS audio, detect regions, transcribe → target SRT ────
    status_cb(f"Stage 3b/Sync: Loading TTS {language} audio…")
    try:
        te_y, te_sr = librosa.load(tts_path, sr=None, mono=True)
        te_regions  = _detect_regions_from_audio(te_y, te_sr, DEFAULT_THR_DB, DEFAULT_HYS_DB, DEFAULT_MIN_MS)
        if not te_regions:
            return {"error": "No regions detected in TTS audio", **results}
        status_cb(f"Stage 3b/Sync: {len(te_regions)} regions in TTS audio — transcribing…")
        te_result = _transcribe_audio(tts_path, api_key)
        te_words  = te_result.get("words", [])
        if not te_words:
            return {"error": f"No word data from ElevenLabs for {language} TTS audio", **results}
        te_srt = _build_target_subtitle_srt(te_regions, te_words)
        status_cb(f"Stage 3b/Sync: {language} SRT generated.")
    except Exception as e:
        return {"error": f"{language} SRT generation failed: {e}", **results}

    # ── Stage 3c: Gemini SRT mapping ─────────────────────────────────────────
    status_cb("Stage 3c/Sync: Calling Gemini for SRT mapping…")
    try:
        mapping_text = _call_gemini_mapping(en_srt, te_srt, translation_text,
                                            gemini_model, language=language)
        status_cb("Stage 3c/Sync: Mapping received from Gemini.")
    except Exception as e:
        return {"error": f"Gemini mapping failed: {e}", **results}

    # ── Stage 3d: Sync SRTs ──────────────────────────────────────────────────
    status_cb("Stage 3d/Sync: Syncing SRTs…")
    try:
        try:
            _en_audio_dur = float(len(y_data)) / float(sr) if sr else 0.0
        except Exception:
            _en_audio_dur = 0.0
        synced_subs, orig_te_subs, sync_log = run_sync_from_strings(
            en_srt, te_srt, mapping_text,
            en_audio_duration=_en_audio_dur)
        status_cb(f"Stage 3d/Sync: {len(synced_subs)} subtitles synced.")
        # Save sync log
        sync_log_path = base + "_sync_log.txt"
        with open(sync_log_path, "w", encoding="utf-8") as _f:
            _f.write(sync_log)
        # Save the synced Bengali SRT so the Captions exporter can re-chunk it.
        try:
            synced_srt_text = _write_srt_from_dict(synced_subs)
            synced_srt_path = base + "_sync_synced.srt"
            with open(synced_srt_path, "w", encoding="utf-8") as _f:
                _f.write(synced_srt_text)
            results["synced_srt_path"] = synced_srt_path
        except Exception:
            pass
    except Exception as e:
        return {"error": f"SRT sync failed: {e}", **results}

    # ── Stage 3e: Create synced audio ────────────────────────────────────────
    status_cb("Stage 3e/Sync: Building audio timestamps…")
    try:
        timestamps   = _build_timestamps(orig_te_subs, synced_subs)
        synced_name  = _tts_output_name(language, audio_path, "_synced")
        synced_path  = os.path.join(outdir, synced_name)
        sync_audio_with_timestamps(tts_path, timestamps, synced_path, status_cb=status_cb)
        results["synced_path"] = synced_path
        status_cb(f"Stage 3e/Sync: Done → {os.path.basename(synced_path)}")
    except Exception as e:
        return {"error": f"Audio sync failed: {e}", **results}

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  Main Application
# ═════════════════════════════════════════════════════════════════════════════

class EndToEndApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"End-to-End Audio Dubbing Pipeline  ·  v{APP_VERSION}")
        self.root.configure(bg=BG)
        self.root.geometry("1200x900")
        self.root.minsize(900, 700)

        # audio
        self.audio_data  = None
        self.sample_rate = None
        self.duration    = 0.0
        self.filepath    = None

        # view
        self.zoom_slider_val = 50.0
        self.scroll_pos      = 0.0
        self.regions         = []

        # playback
        self.cursor_pos      = 0.0
        self.is_playing      = False
        self._play_start_t   = 0.0
        self._play_start_pos = 0.0
        self._play_lock      = threading.Lock()

        # pipeline state
        self.transcription_text = ""
        self.transcription_raw  = None
        self.final_srt          = ""
        self.formatted_srt      = ""
        self.punctuation_result = ""

        # TTS settings (shared between single and batch)
        self._tts_platform_var    = None   # tk.StringVar — "ElevenLabs" or "Google TTS"
        self._tts_language_var    = None   # tk.StringVar — set in _build_tts_settings_panel
        self._tts_engine_var      = None
        self._tts_voice_var       = None
        self._tts_rb_wavenet      = None
        self._tts_rb_chirp3       = None
        self._tts_google_frame    = None   # sub-frame shown only when Google TTS selected
        self._tts_el_frame        = None   # sub-frame shown only when ElevenLabs selected
        self._el_voice_var        = None   # StringVar — selected Bengali voice (label)
        self._el_model_var        = None   # StringVar — ElevenLabs model id (raw)
        self._el_model_cb_widgets = []     # [(combobox, display StringVar), …]
        # Legacy var kept None — Bengali voice IDs are auto-loaded; manual
        # override has been removed per the simplified Bengali workflow.
        self._el_custom_voice_var = None
        self._api_key_var         = None   # StringVar — ElevenLabs API key
        self._last_validated_key  = None   # str — last key that passed _validate_api_key

        # Bengali region detection parameters
        self.bn_thr_var       = None   # tk.DoubleVar
        self.bn_hys_var       = None   # tk.DoubleVar
        self.bn_minsilence_var = None  # tk.IntVar

        # single-file pipeline cancel
        self._pipeline_cancel = threading.Event()

        # batch state
        self._batch_running   = False
        self._batch_stop_req  = False
        self._batch_files     = []
        self._batch_folder    = ""
        self._batch_stop_step = tk.StringVar(value="Full Pipeline")

        # TTS-only tab state
        self._tts_tab_out_path        = ""
        # TTS Studio sentence-segment state
        self._tts_segments      = []      # [{text, path, start_ms, dur_ms}, …]
        self._tts_seg_audio     = None    # merged waveform (np.float32 mono)
        self._tts_seg_sr        = 0
        self._tts_seg_selected  = None    # selected segment index
        self._tts_seg_playing   = False
        self._tts_seg_play_t0   = 0.0     # perf_counter at playback start
        self._tts_seg_play_from = 0.0     # playback start position (s)
        self._tts_seg_play_end  = 0.0     # playback end position (s)
        self._tts_seg_params    = None    # TTS params of the last segment run
        self._tts_seg_out_path  = ""      # merged output file of the last run
        self._tts_el_frame_mirror     = None
        self._tts_google_frame_mirror = None
        self._tts_voice_cb_mirror     = None

        # Single-file pipeline stop step
        self._single_stop_step = tk.StringVar(value="Translation")

        # Gemini model selector
        self._gemini_model_var = tk.StringVar(value=GEMINI_DEFAULT_MODEL)

        # Translation prompt-chain depth: 1 = translate only, 2 = +review,
        # 3 = +punctuation (default, original behaviour)
        self._translation_steps_var = tk.IntVar(value=3)

        # Emotion enrichment toggle (Step 4)
        self._emotion_enabled_var = tk.BooleanVar(value=True)

        # Manual review of the translation before dubbing (single-file
        # pipeline only). When on, the pipeline pauses after translation
        # and shows a side-by-side English/translation review window with
        # Skip / Continue buttons.
        self._review_enabled_var = tk.BooleanVar(value=True)

        # Translation memory (feedback loop). When on: an exact English
        # match reuses the human-proofed script with zero LLM cost, partial
        # matches are injected into the translation prompt, and reviewed
        # scripts are captured back into memory on "Continue to Dubbing".
        self._tm_enabled_var = tk.BooleanVar(value=True)

        # ── Audio Syncing tab state ──────────────────────────────────────────
        # Two independent waveform panels (English + Bengali). Each "side"
        # holds its own audio data, sample rate, duration, file path,
        # regions, region-detection params, waveform fig/ax/canvas, and
        # related label widgets. Populated lazily in _build_audio_sync_tab.
        self._as_state = {
            "en": {
                "audio": None, "sr": None, "dur": 0.0, "regions": [],
                "filepath": None,
                "thr_var": None, "hys_var": None, "min_var": None,
                "fig": None, "ax": None, "canvas": None,
                "file_label": None, "region_count_label": None,
                # Per-side zoom / scroll state
                "zoom_val": 50.0, "scroll_pos": 0.0,
                "sb_canvas": None,
                "sb_drag_start_x": None, "sb_drag_start_pos": 0.0,
                "zoom_readout": None,
            },
            "bn": {
                "audio": None, "sr": None, "dur": 0.0, "regions": [],
                "filepath": None,
                "thr_var": None, "hys_var": None, "min_var": None,
                "fig": None, "ax": None, "canvas": None,
                "file_label": None, "region_count_label": None,
                # Per-side zoom / scroll state
                "zoom_val": 50.0, "scroll_pos": 0.0,
                "sb_canvas": None,
                "sb_drag_start_x": None, "sb_drag_start_pos": 0.0,
                "zoom_readout": None,
            },
        }
        # Stage indicator vars (S3a..S3e) for the Audio Syncing tab
        self._as_stage_vars   = {}
        self._as_stage_labels = {}
        self._as_running      = False

        # Lifecycle flags used by long-running `after` loops so they exit
        # cleanly when the window closes.
        self._closing = False
        self._playhead_after_id = None

        # Path of the most recent synced Bengali SRT — populated by the
        # pipeline workers so the Captions exporter knows what to re-chunk.
        self._last_synced_srt_path: Optional[str] = None

        self._build_ui()
        self._playhead_tick()
        self._check_api_key_badge()

    # ─────────────────────────────────────────────────────────────────────────
    #  UI Build
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.status = tk.Label(
            self.root,
            text="Open an audio file to start the end-to-end dubbing pipeline",
            bg=INPUT_BG, fg=TEXT_MUTED, font=(MONO_FONT, 9), anchor="w")
        self.status.pack(fill="x", side="bottom", ipady=3, padx=6)

        self._init_zoom_scroll_vars()

        style = ttk.Style()
        style.theme_use("clam")

        # Notebook (tabs)
        style.configure("Dark.TNotebook", background=BG, borderwidth=0)
        style.configure("Dark.TNotebook.Tab", background=PANEL, foreground=TEXT_MUTED,
                        padding=[14, 6], font=(MONO_FONT, 10, "bold"),
                        borderwidth=0)
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", PANEL2), ("active", PANEL3)],
                  foreground=[("selected", TR_ACCENT), ("active", TEXT)])

        # Combobox (dropdowns) — light fields + dark text (visible on the
        # native white field macOS Tk 9 draws regardless of fieldbackground)
        style.configure("TCombobox",
                        fieldbackground=INPUT_BG,
                        background=PANEL2,
                        foreground=INPUT_FG,
                        bordercolor=PANEL_BORDER,
                        lightcolor=PANEL_BORDER,
                        darkcolor=PANEL_BORDER,
                        arrowcolor=TEXT_MUTED,
                        selectbackground="#1e3a8a",
                        selectforeground="#f8fafc",
                        insertcolor=INPUT_FG,
                        padding=4)
        style.map("TCombobox",
                  fieldbackground=[("readonly", INPUT_BG), ("focus", INPUT_BG)],
                  foreground=[("readonly", INPUT_FG)],
                  arrowcolor=[("active", ACCENT)],
                  bordercolor=[("focus", ACCENT)])
        # Dropdown listbox (uses option database — applied at root level)
        try:
            self.root.option_add("*TCombobox*Listbox.background", PANEL)
            self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
            self.root.option_add("*TCombobox*Listbox.selectBackground", "#1e3a8a")
            self.root.option_add("*TCombobox*Listbox.selectForeground", "#f8fafc")
            self.root.option_add("*TCombobox*Listbox.font", (MONO_FONT, 9))
            self.root.option_add("*TCombobox*Listbox.borderWidth", 0)
        except Exception:
            pass

        # Scrollbars — slim dark style
        style.configure("Vertical.TScrollbar",
                        background=PANEL2, troughcolor=BG,
                        bordercolor=BG, arrowcolor=TEXT_MUTED,
                        gripcount=0)
        style.configure("Horizontal.TScrollbar",
                        background=PANEL2, troughcolor=BG,
                        bordercolor=BG, arrowcolor=TEXT_MUTED,
                        gripcount=0)
        style.map("Vertical.TScrollbar",
                  background=[("active", PANEL3), ("pressed", PANEL_BORDER)])
        style.map("Horizontal.TScrollbar",
                  background=[("active", PANEL3), ("pressed", PANEL_BORDER)])

        # ── Workspace: notebook + bottom stage bar (shown once a project is
        #    open); the Project Manager screen is shown first instead.
        self._workspace = tk.Frame(self.root, bg=BG)
        self._current_project: Optional[dict] = None

        self._build_stage_bar(self._workspace)       # packs side="bottom"

        self.notebook = ttk.Notebook(self._workspace, style="Dark.TNotebook")
        self.notebook.pack(fill="both", expand=True)

        self.main_tab    = tk.Frame(self.notebook, bg=BG)
        self.batch_tab   = tk.Frame(self.notebook, bg=BG)
        self.tts_tab     = tk.Frame(self.notebook, bg=BG)
        self.sync_tab    = tk.Frame(self.notebook, bg=BG)
        self.scripts_tab = tk.Frame(self.notebook, bg=BG)
        self.hist_tab    = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(self.main_tab,    text="  Single File  ")
        self.notebook.add(self.batch_tab,   text="  Batch Process  ")
        self.notebook.add(self.tts_tab,     text="  TTS  ")
        self.notebook.add(self.sync_tab,    text="  Audio Syncing  ")
        self.notebook.add(self.scripts_tab, text="  Scripts  ")
        self.notebook.add(self.hist_tab,    text="  History  ")

        self._build_main_tab()
        self._build_batch_tab()
        self._build_tts_tab()
        self._build_audio_sync_tab()
        self._build_scripts_tab()
        self._build_history_tab()

        # ── Project Manager (launch screen) ──────────────────────────────────
        self._pm_frame = tk.Frame(self.root, bg=BG)
        self._build_project_manager()
        self._show_project_manager()

    # ── Single File Tab ───────────────────────────────────────────────────────
    def _build_main_tab(self):
        """
        Studio-style single-file tab. The main window is only: toolbar,
        run bar, clickable pipeline stages, and one big swappable view
        (Waveform / Log / Compare). Every setting lives in the ⚙ Settings
        dialog, built once at startup and kept hidden.
        """
        tab = self.main_tab

        # Toolbar
        toolbar = tk.Frame(tab, bg=PANEL, height=52, bd=0,
                           highlightbackground=PANEL_BORDER,
                           highlightthickness=1)
        toolbar.pack(fill="x", side="top")
        toolbar.pack_propagate(False)
        self._build_toolbar(toolbar)

        _ui = _ui_settings_load()
        self._auto_open_var = tk.BooleanVar(
            value=bool(_ui.get("auto_open_result", True)))

        self._btn(toolbar, " ⚙ Settings ", self._open_sf_settings_dialog,
                  bg="#1e1b3a", fg="#a78bfa", abg="#312e81").pack(
            side="right", padx=(4, 10), pady=8)

        # ── Run bar: run/cancel + view switcher + status ─────────────────────
        run_bar = tk.Frame(tab, bg=PANEL3, height=44, bd=0,
                           highlightbackground=PANEL_BORDER, highlightthickness=1)
        run_bar.pack(fill="x", side="top")
        run_bar.pack_propagate(False)
        self._sf_run_bar = run_bar

        self.btn_run_pipeline = self._btn(
            run_bar, "▶  Run Pipeline", self._run_full_pipeline,
            bg="#0f1d14", fg=TR_ACCENT, abg="#1f4d2e")
        self.btn_run_pipeline.pack(side="left", padx=(14, 6), pady=8)

        self.btn_cancel_pipeline = self._btn(
            run_bar, "✕  Cancel", self._cancel_pipeline,
            bg="#1d0f0f", fg="#f87171", abg="#4d1f1f")
        self.btn_cancel_pipeline.pack(side="left", padx=(0, 10), pady=8)
        self.btn_cancel_pipeline.config(state="disabled")

        self._btn(run_bar, "🎭 Speakers", self._open_speakers_dialog,
                  bg="#1e1b3a", fg="#a78bfa", abg="#312e81").pack(
            side="left", padx=(0, 10), pady=8)

        self.api_badge = tk.Label(run_bar, text="api.txt: checking…",
                                  bg=PANEL3, fg=TEXT_FAINT, font=(MONO_FONT, 9))
        self.api_badge.pack(side="left", padx=(0, 12))

        tk.Frame(run_bar, bg=PANEL_BORDER, width=2, height=26).pack(
            side="left", padx=(0, 10), pady=9)
        self._sf_view_btns = {}
        for key, label in (("wave", " Waveform "), ("log", " Log "),
                           ("compare", " Compare ")):
            b = self._btn(run_bar, label,
                          lambda k=key: self._sf_show_view(k),
                          bg=BTN_BG, fg=TEXT_FAINT, abg=BTN_ACT)
            b.pack(side="left", padx=2, pady=8)
            self._sf_view_btns[key] = b

        self.tr_status = tk.Label(run_bar, text="", bg=PANEL3, fg=TEXT_FAINT,
                                  font=(MONO_FONT, 9))
        self.tr_status.pack(side="right", padx=14)

        # Pipeline progress panel (rows are clickable → preview output files)
        self._build_pipeline_panel(tab)

        # Settings dialog — holds TTS settings, region tuning and pipeline
        # options. Built now (hidden) so all shared vars/widgets exist.
        self._build_sf_settings_dialog()

        # ── Main swappable views ─────────────────────────────────────────────
        self._sf_main = tk.Frame(tab, bg=BG)
        self._sf_main.pack(fill="both", expand=True, padx=10, pady=(4, 0))

        # Waveform view
        self._sf_wave_view = tk.Frame(self._sf_main, bg=BG)
        canvas_frame = tk.Frame(self._sf_wave_view, bg=BG)
        canvas_frame.pack(fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(12, 3))
        self.fig.patch.set_facecolor(BG)
        # Fixed margins once — tight_layout() per frame is far too slow.
        self.fig.subplots_adjust(left=0.05, right=0.995, top=0.88, bottom=0.22)
        self._style_axes()
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.get_tk_widget().configure(bg=BG, cursor="crosshair")
        self._draw_placeholder()

        self._sb_drag_start_x   = None
        self._sb_drag_start_pos = 0.0
        self.sb_canvas = tk.Canvas(self._sf_wave_view, height=14, bg="#334155",
                                   highlightthickness=0, cursor="hand2")
        self.sb_canvas.pack(fill="x", pady=(1, 2))
        self.sb_canvas.bind("<Configure>",       self._sb_on_configure)
        self.sb_canvas.bind("<ButtonPress-1>",   self._sb_on_press)
        self.sb_canvas.bind("<B1-Motion>",       self._sb_on_drag)
        self.sb_canvas.bind("<ButtonRelease-1>", self._sb_on_release)
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>", "<Shift-MouseWheel>", "<Shift-Button-4>", "<Shift-Button-5>", "<Button-6>", "<Button-7>"):
            self.sb_canvas.bind(seq, self._on_waveform_scroll)

        self.canvas.mpl_connect("button_press_event",  self._on_canvas_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_hover)

        # Log view — full-window pipeline log
        self._sf_log_view = tk.Frame(self._sf_main, bg=BG)
        log_hdr = tk.Frame(self._sf_log_view, bg=PANEL2, height=32)
        log_hdr.pack(fill="x")
        log_hdr.pack_propagate(False)
        tk.Label(log_hdr, text="PIPELINE LOG", bg=PANEL2, fg="#60a5fa",
                 font=(MONO_FONT, 9, "bold")).pack(side="left", padx=12, pady=6)
        self._btn(log_hdr, " Clear ", self._sf_log_clear,
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT).pack(
            side="right", padx=8, pady=3)
        self._sf_log = scrolledtext.ScrolledText(
            self._sf_log_view, wrap="word", bg="#0b1220", fg="#93c5fd",
            insertbackground=INPUT_FG, font=(MONO_FONT, 9),
            relief="flat", bd=0, state="disabled")
        self._sf_log.pack(fill="both", expand=True, pady=(2, 2))

        # Compare view — English on top, synced dub below, SRT text on chunks
        self._sf_cmp_view = tk.Frame(self._sf_main, bg=BG)
        self._sf_cmp_build(self._sf_cmp_view)

        self._sf_show_view("wave")

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>", "<Shift-MouseWheel>", "<Shift-Button-4>", "<Shift-Button-5>", "<Button-6>", "<Button-7>"):
            self.canvas.get_tk_widget().bind(seq, self._on_waveform_scroll)

        self.root.bind("<space>", lambda e: self._toggle_play())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _save_ui_settings(self):
        d = _ui_settings_load()
        d["auto_open_result"] = bool(self._auto_open_var.get())
        _ui_settings_save(d)

    def _build_toolbar(self, tb):
        self._btn(tb, "Open Audio", self._pick_file).pack(side="left", padx=(14, 6), pady=10)
        self._btn(tb, "Reset View", self._reset_view).pack(side="left", padx=6, pady=10)

        tk.Frame(tb, bg="#334155", width=2, height=28).pack(side="left", padx=10, pady=12)

        self.play_btn = self._btn(tb, "  Play  ", self._toggle_play,
                                  bg="#0f1d14", fg="#4ade80", abg="#1f4d2e")
        self.play_btn.pack(side="left", padx=4, pady=10)
        self._btn(tb, "  Stop  ", self._stop_playback,
                  bg="#1f1213", fg="#f87171", abg="#3a1414").pack(side="left", padx=4, pady=10)

        self.cursor_label = tk.Label(tb, text="Cursor: 0.00s", bg=PANEL,
                                     fg=CURSOR_C, font=(MONO_FONT, 10))
        self.cursor_label.pack(side="left", padx=16)
        self.file_label = tk.Label(tb, text="No file loaded",
                                   bg=PANEL, fg=TEXT_FAINT, font=(MONO_FONT, 10))
        self.file_label.pack(side="left", padx=14)
        self.info_label = tk.Label(tb, text="", bg=PANEL, fg=ACCENT, font=(MONO_FONT, 10))
        self.info_label.pack(side="right", padx=18)

    def _build_tts_settings_panel(self, parent):
        """TTS Settings panel redesigned as two side-by-side cards.

        Left card  — ElevenLabs API key entry + validation status.
        Right card — Bengali voice picker + Language / Platform / Google engine controls.
        """
        # Caches shared with the mirror panel so widgets stay in sync.
        self._el_voice_options: List[Dict[str, str]] = []
        self._el_voice_cb_widgets: List[ttk.Combobox] = []
        self._el_voice_labels: List[tk.Label] = []
        self._el_status_labels: List[tk.Label] = []
        self._api_key_entries: List[tk.Entry] = []

        # Primary StringVars — mirror panels bind to these same vars.
        self._api_key_var        = tk.StringVar()
        self._el_voice_var       = tk.StringVar()
        self._el_selected_voice_id: str = ""
        self._el_label_to_id: Dict[str, str] = {}
        self._el_search_entries: List[tk.Entry] = []
        self._el_custom_voice_var = tk.StringVar()

        cached = _read_api_key_file()
        if cached:
            self._api_key_var.set(cached)
            set_runtime_api_key(cached)

        # ── Outer container ──────────────────────────────────────────────────
        outer = tk.Frame(parent, bg="#1e1b3a", bd=0,
                         highlightbackground="#5b4fbf", highlightthickness=1)
        outer.pack(fill="x", side="top")
        self._tts_panel_frame = outer   # anchor for Simple-mode hint bar

        # Thin header strip
        hdr = tk.Frame(outer, bg="#211e40", height=26)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="TTS SETTINGS", bg="#211e40", fg="#a78bfa",
                 font=(MONO_FONT, 8, "bold")).pack(side="left", padx=14, pady=5)
        tk.Frame(hdr, bg="#5b4fbf", width=1, height=14).pack(side="left", pady=6)
        tk.Label(hdr, text="Configure API authentication and voice selection",
                 bg="#211e40", fg="#64748b", font=(MONO_FONT, 8)).pack(side="left", padx=10)

        # Two-card row
        cards = tk.Frame(outer, bg="#1e1b3a")
        cards.pack(fill="x", padx=10, pady=(6, 10))

        _card_bg  = "#0d0b1f"
        _card_bdr = "#3d3580"
        _ttl_fg   = "#a78bfa"
        _ttl_font = (MONO_FONT, 8, "bold")

        # ── LEFT CARD: API Key ───────────────────────────────────────────────
        lcard = tk.Frame(cards, bg=_card_bg,
                         highlightbackground=_card_bdr, highlightthickness=1)
        lcard.pack(side="left", fill="y", padx=(0, 8), ipadx=12, ipady=2)

        # Card title
        lc_hdr = tk.Frame(lcard, bg="#100e26", height=24)
        lc_hdr.pack(fill="x")
        lc_hdr.pack_propagate(False)
        tk.Label(lc_hdr, text="⚡  ElevenLabs API Key",
                 bg="#100e26", fg=_ttl_fg, font=_ttl_font
                 ).pack(side="left", padx=10, pady=4)

        # Key entry row
        key_row = tk.Frame(lcard, bg=_card_bg)
        key_row.pack(anchor="w", padx=10, pady=(8, 4))

        tk.Label(key_row, text="Key:", bg=_card_bg, fg=TEXT_MUTED,
                 font=(UI_FONT, 9)).pack(side="left", padx=(0, 6))
        api_entry = tk.Entry(key_row, textvariable=self._api_key_var,
                             width=32, bg=INPUT_BG, fg=INPUT_FG,
                             insertbackground=INPUT_FG, relief="flat",
                             font=(UI_FONT, 9), show="•")
        api_entry.pack(side="left", padx=(0, 6))
        api_entry.bind("<<Paste>>",
                       lambda e: self.root.after(50, self._on_api_key_changed))
        api_entry.bind("<FocusOut>", lambda e: self._on_api_key_changed())
        api_entry.bind("<Return>",   lambda e: self._on_api_key_changed())
        self._api_key_var.trace_add("write", lambda *_: self._schedule_api_key_changed())
        self._api_key_entries.append(api_entry)

        self._btn(key_row, "Validate", self._on_api_key_changed,
                  bg="#172554", fg=REG_LABEL, abg="#1e3a8a").pack(side="left")

        # Status label (wrappable, below key row)
        status = tk.Label(lcard, text="No API key.",
                          bg=_card_bg, fg=TEXT_MUTED,
                          font=(UI_FONT, 8), wraplength=310, justify="left")
        status.pack(anchor="w", padx=10, pady=(2, 10))
        self._el_status_labels.append(status)

        # ── RIGHT CARD: Voice Configuration ─────────────────────────────────
        rcard = tk.Frame(cards, bg=_card_bg,
                         highlightbackground=_card_bdr, highlightthickness=1)
        rcard.pack(side="left", fill="both", expand=True, ipadx=12, ipady=2)

        # Card title
        rc_hdr = tk.Frame(rcard, bg="#100e26", height=24)
        rc_hdr.pack(fill="x")
        rc_hdr.pack_propagate(False)
        tk.Label(rc_hdr, text="🎤  Voice Configuration",
                 bg="#100e26", fg=_ttl_fg, font=_ttl_font
                 ).pack(side="left", padx=10, pady=4)

        # Voice picker + search row
        vrow = tk.Frame(rcard, bg=_card_bg)
        vrow.pack(anchor="w", padx=10, pady=(8, 4))

        _vlbl = tk.Label(vrow, text=f"{self._current_language()} Voice:",
                         bg=_card_bg, fg=TEXT_MUTED, font=(UI_FONT, 9))
        _vlbl.pack(side="left", padx=(0, 6))
        self._el_voice_labels.append(_vlbl)
        voice_cb = ttk.Combobox(vrow, textvariable=self._el_voice_var,
                                state="readonly", width=30, font=(UI_FONT, 9))
        voice_cb.pack(side="left", padx=(0, 8))
        voice_cb.bind("<<ComboboxSelected>>", self._on_el_voice_pick)
        self._el_voice_cb_widgets.append(voice_cb)

        tk.Frame(vrow, bg="#5b4fbf", width=1, height=20).pack(side="left", padx=(0, 8), pady=2)

        # Search field — bordered container that glows on focus
        _vsf = tk.Frame(vrow, bg=INPUT_BG, highlightbackground="#334155",
                        highlightthickness=1)
        _vsf.pack(side="left", padx=(0, 6))
        tk.Label(_vsf, text="🔍", bg=INPUT_BG, fg=TEXT_MUTED,
                 font=(UI_FONT, 9)).pack(side="left", padx=(5, 0))
        search_entry = tk.Entry(_vsf, width=11, bg=INPUT_BG, fg=INPUT_FG,
                                insertbackground=INPUT_FG, relief="flat",
                                font=(UI_FONT, 9), bd=0)
        search_entry.pack(side="left", padx=(2, 5), pady=3)
        self._install_placeholder(search_entry, "search voices…")
        search_entry.bind("<FocusIn>",
                          lambda e, f=_vsf: f.config(highlightbackground=ACCENT), "+")
        search_entry.bind("<FocusOut>",
                          lambda e, f=_vsf: f.config(highlightbackground="#334155"), "+")
        search_entry.bind("<KeyRelease>", lambda e: self._on_voice_search())
        search_entry.bind("<Return>",     lambda e: self._cycle_voice_match(+1))
        search_entry.bind("<Escape>",     lambda e: self._clear_voice_search())
        self._el_search_entries.append(search_entry)

        self._btn(vrow, "Search", lambda: self._cycle_voice_match(+1),
                  bg="#1e3a8a", fg="#bfdbfe", abg="#1d4ed8").pack(side="left", padx=(0, 6))
        self._btn(vrow, "↻  Refresh", self._refresh_bengali_voices,
                  bg=TR_ACCENT, fg="#052e16", abg="#16a34a").pack(side="left")

        # Language / Platform / Google TTS row
        prow = tk.Frame(rcard, bg=_card_bg)
        prow.pack(anchor="w", padx=10, pady=(2, 10))

        tk.Label(prow, text="Language:", bg=_card_bg, fg=TEXT_MUTED,
                 font=(UI_FONT, 9)).pack(side="left", padx=(0, 4))
        self._tts_language_var = tk.StringVar(value=_read_last_language())
        lang_cb = ttk.Combobox(prow, textvariable=self._tts_language_var,
                               values=TTS_LANGUAGE_NAMES, state="readonly", width=14,
                               font=(UI_FONT, 9))
        lang_cb.pack(side="left", padx=(0, 12))
        lang_cb.bind("<<ComboboxSelected>>", self._on_tts_language_change)

        tk.Frame(prow, bg="#5b4fbf", width=1, height=20).pack(side="left", padx=(0, 10), pady=2)

        tk.Label(prow, text="Platform:", bg=_card_bg, fg=TEXT_MUTED,
                 font=(UI_FONT, 9)).pack(side="left", padx=(0, 4))
        self._tts_platform_var = tk.StringVar(value=TTS_DEFAULT_PLATFORM)
        platform_cb = ttk.Combobox(prow, textvariable=self._tts_platform_var,
                                   values=TTS_PLATFORMS, state="readonly", width=12,
                                   font=(UI_FONT, 9))
        platform_cb.pack(side="left", padx=(0, 8))
        platform_cb.bind("<<ComboboxSelected>>", self._on_tts_platform_change)

        # ElevenLabs sub-frame — holds the ElevenLabs model picker
        ef = tk.Frame(prow, bg=_card_bg)
        self._tts_el_frame = ef
        self._build_el_model_picker(ef, _card_bg)

        # Google-TTS-only sub-frame (Engine + Voice)
        gf = tk.Frame(prow, bg=_card_bg)
        self._tts_google_frame = gf

        tk.Frame(gf, bg=ACCENT, width=1, height=20).pack(side="left", padx=(0, 10), pady=2)
        tk.Label(gf, text="Engine:", bg=_card_bg, fg=TEXT,
                 font=(MONO_FONT, 9)).pack(side="left", padx=(0, 4))
        self._tts_engine_var = tk.StringVar(value=TTS_DEFAULT_ENGINE)
        for engine in ("Standard", "WaveNet", "Chirp3"):
            rb = tk.Radiobutton(
                gf, text=engine, variable=self._tts_engine_var, value=engine,
                bg=_card_bg, fg=TEXT, selectcolor=_card_bg,
                activebackground=_card_bg, activeforeground="#a78bfa",
                font=(MONO_FONT, 9), command=self._on_tts_engine_change)
            rb.pack(side="left", padx=4)
            if engine == "WaveNet":
                self._tts_rb_wavenet = rb
            elif engine == "Chirp3":
                self._tts_rb_chirp3 = rb

        tk.Frame(gf, bg="#5b4fbf", width=1, height=20).pack(side="left", padx=(8, 8), pady=2)
        tk.Label(gf, text="Voice:", bg=_card_bg, fg=TEXT,
                 font=(MONO_FONT, 9)).pack(side="left", padx=(0, 4))
        self._tts_voice_var = tk.StringVar(value=TTS_DEFAULT_VOICE)
        self._tts_voice_cb = ttk.Combobox(gf, textvariable=self._tts_voice_var,
                                          state="readonly", width=28,
                                          font=(MONO_FONT, 9))
        self._tts_voice_cb.pack(side="left", padx=(0, 8))

        # Initialise voice list and platform visibility
        self._refresh_tts_voice_list()
        self._apply_tts_platform_visibility()

        # Auto-validate cached key, or show hint
        if self._api_key_var.get().strip():
            self.root.after(150, self._on_api_key_changed)
        else:
            self._set_el_status("Paste your ElevenLabs API key to load voices.",
                                TEXT_MUTED)

    # ─────────────────────────────────────────────────────────────────────────
    #  ElevenLabs API key + Bengali voice picker
    # ─────────────────────────────────────────────────────────────────────────
    def _build_elevenlabs_keybar(self, parent, panel_bg="#1e1b3a", primary: bool = False):
        """
        Render the ElevenLabs API key entry + Bengali voice picker into *parent*.

        Called once for the Single File tab (primary=True) and again (primary
        False) for each mirror surface (TTS tab, Audio Syncing tab). All
        widgets share the same StringVars so editing one updates the rest.

        Layout:  [API Key:][entry][Validate]   [Bengali Voice:][combobox][Refresh]   status_badge
        """
        if primary:
            # Bound to all panels — central source of truth
            self._api_key_var = tk.StringVar()
            self._el_voice_var = tk.StringVar()
            # Authoritative voice_id captured the moment the user picks a
            # voice. Never derived from the label string at TTS time — that
            # round-trip broke when labels contained whitespace / Unicode and
            # produced HTTP 404 voice_not_found errors.
            self._el_selected_voice_id: str = ""
            # display-label -> clean voice_id (rebuilt on every voice load).
            self._el_label_to_id: Dict[str, str] = {}
            # Each search entry gets its own private StringVar (sharing one
            # across mirror panels broke the placeholder logic and caused
            # the dropdown to filter on the literal hint string).
            self._el_search_entries: List[tk.Entry] = []
            # Custom voice ID was a Telugu-era manual override; Bengali voices
            # are auto-loaded so we keep the StringVar empty for compatibility
            # but never expose it to the user.
            self._el_custom_voice_var = tk.StringVar()

            cached = _read_api_key_file()
            if cached:
                self._api_key_var.set(cached)
                set_runtime_api_key(cached)

        tk.Label(parent, text="API Key:", bg=panel_bg, fg=TEXT_MUTED,
                 font=(UI_FONT, 9)).pack(side="left", padx=(0, 4))
        api_entry = tk.Entry(parent, textvariable=self._api_key_var,
                             width=22, bg=INPUT_BG, fg=INPUT_FG,
                             insertbackground=INPUT_FG, relief="flat",
                             font=(UI_FONT, 9), show="•")
        api_entry.pack(side="left", padx=(0, 4))
        # Trigger auto-fetch on paste, on focus-out, and on Enter.
        api_entry.bind("<<Paste>>",
                       lambda e: self.root.after(50, self._on_api_key_changed))
        api_entry.bind("<FocusOut>", lambda e: self._on_api_key_changed())
        api_entry.bind("<Return>",   lambda e: self._on_api_key_changed())
        # Catch Ctrl-V / right-click paste consistently — on some Tk builds the
        # <<Paste>> virtual event lags; trace the StringVar as a safety net.
        if primary:
            self._api_key_var.trace_add("write", lambda *_: self._schedule_api_key_changed())
        self._api_key_entries.append(api_entry)

        self._btn(parent, "Validate", self._on_api_key_changed,
                  bg="#172554", fg=REG_LABEL, abg="#1e3a8a").pack(
            side="left", padx=(0, 10), pady=5)

        tk.Frame(parent, bg="#5b4fbf", width=2, height=24).pack(side="left", padx=(0, 10), pady=10)

        _vlbl = tk.Label(parent, text=f"{self._current_language()} Voice:",
                         bg=panel_bg, fg=TEXT_MUTED, font=(UI_FONT, 9))
        _vlbl.pack(side="left", padx=(0, 4))
        self._el_voice_labels.append(_vlbl)
        voice_cb = ttk.Combobox(parent, textvariable=self._el_voice_var,
                                state="readonly", width=34,
                                font=(UI_FONT, 9))
        voice_cb.pack(side="left", padx=(0, 6))
        voice_cb.bind("<<ComboboxSelected>>", self._on_el_voice_pick)
        self._el_voice_cb_widgets.append(voice_cb)

        # ── Voice search field + button ──────────────────────────────────────
        # Filters the Bengali Voice dropdown live as the user types. The
        # button cycles through matching voices; Enter / Esc work too.
        # Note: each entry has its own internal text — sharing a StringVar
        # across mirror panels broke the placeholder logic.
        _ksf = tk.Frame(parent, bg=INPUT_BG, highlightbackground="#334155",
                        highlightthickness=1)
        _ksf.pack(side="left", padx=(0, 6), pady=5)
        tk.Label(_ksf, text="🔍", bg=INPUT_BG, fg=TEXT_MUTED,
                 font=(UI_FONT, 9)).pack(side="left", padx=(5, 0))
        search_entry = tk.Entry(_ksf, width=11, bg=INPUT_BG, fg=INPUT_FG,
                                insertbackground=INPUT_FG, relief="flat",
                                font=(UI_FONT, 9), bd=0)
        search_entry.pack(side="left", padx=(2, 5), pady=3)
        self._install_placeholder(search_entry, "search voices…")
        search_entry.bind("<FocusIn>",
                          lambda e, f=_ksf: f.config(highlightbackground=ACCENT), "+")
        search_entry.bind("<FocusOut>",
                          lambda e, f=_ksf: f.config(highlightbackground="#334155"), "+")
        search_entry.bind("<KeyRelease>", lambda e: self._on_voice_search())
        search_entry.bind("<Return>",     lambda e: self._cycle_voice_match(+1))
        search_entry.bind("<Escape>",     lambda e: self._clear_voice_search())
        self._el_search_entries.append(search_entry)

        self._btn(parent, "Search", lambda: self._cycle_voice_match(+1),
                  bg="#1e3a8a", fg="#bfdbfe", abg="#1d4ed8").pack(
            side="left", padx=(0, 6), pady=5)

        self._btn(parent, "↻  Refresh", self._refresh_bengali_voices,
                  bg=TR_ACCENT, fg="#052e16", abg="#16a34a").pack(
            side="left", padx=(0, 10), pady=5)

        status = tk.Label(parent, text="No API key.", bg=panel_bg, fg=TEXT_MUTED,
                          font=(UI_FONT, 9))
        status.pack(side="left", padx=(8, 0))
        self._el_status_labels.append(status)

        if primary:
            # Already had a cached key when widgets were built — kick off an
            # auto-validate + voice fetch so the dropdown is populated before
            # the user does anything.
            if self._api_key_var.get().strip():
                self.root.after(150, self._on_api_key_changed)
            else:
                self._set_el_status("Paste your ElevenLabs API key to load voices.",
                                    TEXT_MUTED)

    def _schedule_api_key_changed(self):
        """Debounce StringVar trace so we don't fire mid-typing on every keystroke."""
        if getattr(self, "_api_key_debounce", None):
            try:
                self.root.after_cancel(self._api_key_debounce)
            except Exception:
                pass
        self._api_key_debounce = self.root.after(700, self._on_api_key_changed)

    def _set_el_status(self, msg: str, colour: str = TEXT_FAINT):
        for lbl in self._el_status_labels:
            try:
                lbl.config(text=msg, fg=colour)
            except Exception:
                pass

    def _install_placeholder(self, entry: tk.Entry, hint: str):
        """Native Tk Entry has no placeholder — fake it with focus events."""
        entry._placeholder = hint
        entry._placeholder_active = False

        def _show():
            if not entry.get():
                entry._placeholder_active = True
                entry.config(fg=TEXT_FAINT)
                entry.insert(0, hint)

        def _hide(_e=None):
            if entry._placeholder_active:
                entry._placeholder_active = False
                entry.delete(0, "end")
                entry.config(fg=INPUT_FG)

        entry.bind("<FocusIn>", _hide)
        entry.bind("<FocusOut>", lambda _e: _show())
        _show()

    def _voice_search_query(self) -> str:
        """Current search filter text, ignoring the placeholder."""
        for ent in getattr(self, "_el_search_entries", []):
            try:
                if getattr(ent, "_placeholder_active", False):
                    continue
                txt = ent.get().strip()
                # Belt-and-braces: never treat the placeholder hint as a query.
                if not txt or txt == getattr(ent, "_placeholder", ""):
                    continue
                return txt.lower()
            except Exception:
                continue
        return ""

    def _filtered_voice_options(self) -> List[Dict[str, str]]:
        """Apply the current search filter to the loaded voice list."""
        q = self._voice_search_query()
        if not q:
            return list(self._el_voice_options)
        return [o for o in self._el_voice_options
                if q in o["label"].lower() or q in o["voice_id"].lower()]

    def _on_voice_search(self):
        """Re-render the dropdown contents using the current filter."""
        filtered = self._filtered_voice_options()
        labels = [o["label"] for o in filtered]
        for cb in self._el_voice_cb_widgets:
            try:
                cb["values"] = labels
            except Exception:
                continue
        # Auto-select first match so the user can hear/use it immediately.
        if labels:
            current = self._el_voice_var.get() if self._el_voice_var else ""
            if current not in labels:
                self._el_voice_var.set(labels[0])
                self._on_el_voice_pick()

    def _cycle_voice_match(self, direction: int = 1):
        """Step the selection through the filtered list (Search button / Enter)."""
        filtered = self._filtered_voice_options()
        labels = [o["label"] for o in filtered]
        if not labels:
            return
        current = self._el_voice_var.get() if self._el_voice_var else ""
        if current in labels:
            idx = (labels.index(current) + direction) % len(labels)
        else:
            idx = 0
        self._el_voice_var.set(labels[idx])
        self._on_el_voice_pick()
        # Make sure every mirror combobox shows the filtered list too.
        for cb in self._el_voice_cb_widgets:
            try:
                cb["values"] = labels
            except Exception:
                continue

    def _clear_voice_search(self):
        """Reset the search box and re-show the full voice list."""
        for ent in getattr(self, "_el_search_entries", []):
            try:
                ent.delete(0, "end")
                ent._placeholder_active = False
                ent.config(fg=TEXT)
            except Exception:
                continue
        self._on_voice_search()

    def _rebuild_label_id_map(self) -> None:
        """Rebuild the display-label → voice_id map from `_el_voice_options`.

        Only options whose voice_id passes strict sanitization are indexed,
        so a corrupt cache entry can never poison the map.
        """
        mapping: Dict[str, str] = {}
        for o in self._el_voice_options:
            vid = _sanitize_voice_id(o.get("voice_id"))
            if not vid:
                continue
            label = o.get("label") or ""
            if label:
                mapping[label] = vid
            # Also key by normalised label and by raw voice_id so any of them
            # resolves, no matter how the combobox round-trips whitespace.
            mapping[re.sub(r"\s+", " ", label).strip()] = vid
            mapping[vid] = vid
        self._el_label_to_id = mapping

    def _set_el_voice_options(self, options: List[Dict[str, str]],
                              keep_selection: bool = True):
        """Update the ElevenLabs voice combobox(es) with a fresh list of options.

        Sanitizes every incoming option, rebuilds the label→id map, refreshes
        the dropdown, and re-captures the selected voice_id atomically so a
        stale label can never reach the API.
        """
        # Sanitize every voice_id up-front; drop anything malformed.
        clean_options: List[Dict[str, str]] = []
        for o in (options or []):
            if not isinstance(o, dict):
                continue
            vid = _sanitize_voice_id(o.get("voice_id"))
            if not vid:
                continue
            clean_options.append({
                "voice_id": vid,
                "name": str(o.get("name") or "Unnamed voice"),
                "label": str(o.get("label") or vid),
            })
        self._el_voice_options = clean_options
        self._rebuild_label_id_map()

        # Apply the current search filter to what the dropdown actually shows.
        filtered = self._filtered_voice_options()
        labels = [o["label"] for o in filtered]
        for cb in self._el_voice_cb_widgets:
            try:
                cb["values"] = labels
            except Exception:
                continue

        prev_id = getattr(self, "_el_selected_voice_id", "") or ""

        # 1. Try to keep the previous selection (by id) if still in the list.
        if keep_selection and prev_id:
            for o in self._el_voice_options:
                if o["voice_id"] == prev_id:
                    self._el_voice_var.set(o["label"])
                    self._el_selected_voice_id = o["voice_id"]
                    return

        # 2. Otherwise prefer the first Bengali-tagged voice (label starts ✦).
        if labels:
            chosen = next(
                (o for o in self._el_voice_options
                 if o["label"] in labels and o["label"].startswith("✦")),
                None,
            )
            if chosen is None:
                chosen = next(
                    (o for o in self._el_voice_options
                     if o["label"] == labels[0]),
                    None,
                )
            if chosen is not None:
                self._el_voice_var.set(chosen["label"])
                self._el_selected_voice_id = chosen["voice_id"]
                return

        # 3. Empty list — clear everything.
        self._el_voice_var.set("")
        self._el_selected_voice_id = ""

    def _on_el_voice_pick(self, _event=None):
        """User picked a voice in the combobox — capture its voice_id NOW.

        We resolve the label to a clean voice_id at selection time (UI thread)
        and stash it in `_el_selected_voice_id`. TTS code reads that field
        directly, so the formatted display label can never reach the URL.
        """
        if not self._el_voice_var:
            self._el_selected_voice_id = ""
            return
        sel = (self._el_voice_var.get() or "").strip()
        if not sel:
            self._el_selected_voice_id = ""
            return
        # Direct map lookup (label, normalised label, or raw voice_id).
        vid = self._el_label_to_id.get(sel)
        if not vid:
            vid = self._el_label_to_id.get(re.sub(r"\s+", " ", sel).strip())
        if not vid:
            # Last-ditch: user typed/pasted a clean voice_id. Strict only.
            vid = _sanitize_voice_id(sel)
        self._el_selected_voice_id = vid or ""

    def _resolve_el_voice_id(self) -> str:
        """Return the captured voice_id for the currently selected voice.

        Always returns a sanitized, URL-safe alphanumeric voice_id, or "" if
        no valid voice is currently selected. Never parses the display label
        at request time — the label round-trip is the source of every random
        voice_id corruption we've seen.
        """
        # Fast path: the id captured at selection time.
        cached = _sanitize_voice_id(getattr(self, "_el_selected_voice_id", ""))
        if cached:
            return cached

        # Selection happened before the pick handler ran (e.g. programmatic
        # set during _set_el_voice_options on a thread boundary). Re-resolve
        # via the label→id map without any string parsing of the label body.
        if not self._el_voice_var:
            return _sanitize_voice_id(ELEVENLABS_TTS_VOICE_ID)
        sel = (self._el_voice_var.get() or "").strip()
        if not sel:
            return _sanitize_voice_id(ELEVENLABS_TTS_VOICE_ID)

        vid = self._el_label_to_id.get(sel)
        if not vid:
            vid = self._el_label_to_id.get(re.sub(r"\s+", " ", sel).strip())
        if vid:
            self._el_selected_voice_id = vid
            return vid

        # Strict raw-id acceptance only (no label salvage).
        clean = _sanitize_voice_id(sel)
        if clean:
            self._el_selected_voice_id = clean
            return clean
        return _sanitize_voice_id(ELEVENLABS_TTS_VOICE_ID)

    def _current_language(self) -> str:
        """Resolve the currently-selected pipeline language from the UI."""
        try:
            v = self._tts_language_var.get() if self._tts_language_var else ""
        except Exception:
            v = ""
        return v if v in TTS_LANGUAGES else TTS_DEFAULT_LANGUAGE

    def _on_api_key_changed(self):
        """
        User just pasted / edited an API key. Validate it on a worker thread,
        and on success fetch voices for the currently-selected language and
        populate the dropdown.
        """
        key = (self._api_key_var.get() if self._api_key_var else "").strip()
        if not key:
            set_runtime_api_key(None)
            self._set_el_status("No API key.", TEXT_MUTED)
            self._set_el_voice_options([])
            return

        lang = self._current_language()

        # If this is the same key we already validated successfully, no need
        # to re-hit the API — just re-show the cached voice list.
        cache_key = (_api_key_fingerprint(key), lang)
        if (getattr(self, "_last_validated_key", None) == key
                and cache_key in _EL_VOICE_CACHE):
            cached = _EL_VOICE_CACHE.get(cache_key) or []
            self._set_el_voice_options(cached)
            tagged = sum(1 for v in cached if v["label"].startswith("✦"))
            suffix = (f" (✦ = {tagged} {lang}-tagged)" if tagged
                      else f" (eleven_v3 auto-detects {lang})")
            self._set_el_status(
                f"✔ API key OK · {len(cached)} voice(s) loaded{suffix}",
                TR_ACCENT)
            return

        set_runtime_api_key(key)
        self._set_el_status("Validating API key…", "#d97706")

        def _worker():
            try:
                _validate_api_key(key)
            except Exception as e:
                self.root.after(0, lambda err=str(e):
                    self._set_el_status(f"✗ {err}", "#f87171"))
                return

            self.root.after(0, lambda:
                self._set_el_status(f"Loading {lang} voices…", "#d97706"))
            try:
                voices = _fetch_voices_for_language(key, lang, force_refresh=True)
            except Exception as e:
                self.root.after(0, lambda err=str(e):
                    self._set_el_status(f"✗ {err}", "#f87171"))
                return

            # Persist the validated key so the next launch picks it up.
            _write_api_key_file(key)

            def _apply():
                self._last_validated_key = key
                self._set_el_voice_options(voices)
                if voices:
                    tagged = sum(1 for v in voices if v["label"].startswith("✦"))
                    suffix = (f" (✦ = {tagged} {lang}-tagged)" if tagged
                              else f" (eleven_v3 auto-detects {lang})")
                    self._set_el_status(
                        f"✔ API key OK · {len(voices)} voice(s) loaded{suffix}",
                        TR_ACCENT)
                else:
                    self._set_el_status(
                        "✗ No voices found on this account.",
                        "#f87171")
                # Refresh the api.txt OK / MISSING badge
                try:
                    self._check_api_key_badge()
                except Exception:
                    pass

            self.root.after(0, _apply)

        threading.Thread(target=_worker, daemon=True).start()

    def _refresh_el_voices(self):
        """Force-refresh the ElevenLabs voice list for the current language."""
        key = (self._api_key_var.get() if self._api_key_var else "").strip()
        if not key:
            self._set_el_status("Paste an API key first.", "#f87171")
            return
        _clear_el_voice_cache(language=self._current_language(), api_key=key)
        self._last_validated_key = None
        self._on_api_key_changed()

    # Backwards-compat alias — older UI bindings may still reference this.
    _refresh_bengali_voices = _refresh_el_voices

    def _on_tts_platform_change(self, _event=None):
        self._apply_tts_platform_visibility()

    def _build_el_model_picker(self, parent, panel_bg):
        """ElevenLabs model selector (v3 / multilingual v2 / turbo / flash).

        Rendered inside the ElevenLabs platform sub-frame so it shows and
        hides with the platform toggle. Every surface shares
        self._el_model_var (raw model id); each combobox keeps its own
        display var, kept in sync across mirrors on selection."""
        if self._el_model_var is None:
            self._el_model_var = tk.StringVar(value=_read_el_model())

        tk.Frame(parent, bg="#5b4fbf", width=1, height=20).pack(
            side="left", padx=(0, 10), pady=2)
        tk.Label(parent, text="Model:", bg=panel_bg, fg=TEXT_MUTED,
                 font=(UI_FONT, 9)).pack(side="left", padx=(0, 4))

        disp = tk.StringVar(value=ELEVENLABS_TTS_MODELS.get(
            self._el_model_var.get(), self._el_model_var.get()))
        cb = ttk.Combobox(parent, textvariable=disp, state="readonly",
                          width=26, font=(UI_FONT, 9),
                          values=list(ELEVENLABS_TTS_MODELS.values()))
        cb.pack(side="left", padx=(0, 8))

        def _picked(_event=None):
            label = disp.get()
            for mid, lbl in ELEVENLABS_TTS_MODELS.items():
                if lbl == label:
                    self._el_model_var.set(mid)
                    _write_el_model(mid)
                    break
            for _cb, ovar in self._el_model_cb_widgets:
                if ovar is not disp:
                    ovar.set(label)

        cb.bind("<<ComboboxSelected>>", _picked)
        self._el_model_cb_widgets.append((cb, disp))

    def _get_el_model(self) -> str:
        """Raw ElevenLabs model id currently selected (falls back to default)."""
        try:
            v = (self._el_model_var.get() if self._el_model_var else "").strip()
        except Exception:
            v = ""
        return v if v in ELEVENLABS_TTS_MODELS else ELEVENLABS_TTS_MODEL

    def _apply_tts_platform_visibility(self):
        """Show the correct platform sub-frame; hide the other (both panels)."""
        platform = self._tts_platform_var.get() if self._tts_platform_var else TTS_DEFAULT_PLATFORM
        # Keep the S2 pipeline-stage description in sync with the platform.
        _s2 = getattr(self, "_stage_desc_labels", {}).get("S2")
        if _s2:
            try:
                _name = "Google Cloud TTS" if platform == "Google TTS" else "ElevenLabs"
                _s2.config(text=f"TTS — Dubbed Audio ({_name})")
            except Exception:
                pass
        if platform == "Google TTS":
            for ef in (self._tts_el_frame, self._tts_el_frame_mirror):
                if ef:
                    ef.pack_forget()
            for gf in (self._tts_google_frame, self._tts_google_frame_mirror):
                if gf:
                    gf.pack(side="left", fill="y")
        else:  # ElevenLabs
            for gf in (self._tts_google_frame, self._tts_google_frame_mirror):
                if gf:
                    gf.pack_forget()
            for ef in (self._tts_el_frame, self._tts_el_frame_mirror):
                if ef:
                    ef.pack(side="left", fill="y")

    def _register_lang_label(self, widget, template):
        """Register a widget whose text follows the selected language.
        *template* contains {LANG}, replaced with the language in caps."""
        if not hasattr(self, "_lang_dyn_labels"):
            self._lang_dyn_labels = []
        self._lang_dyn_labels.append((widget, template))
        try:
            widget.config(text=template.format(LANG=self._current_language().upper()))
        except Exception:
            pass

    def _on_tts_language_change(self, _event=None):
        lang = self._current_language()
        # Update every "<X> Voice:" label to the new language.
        for lbl in getattr(self, "_el_voice_labels", []):
            try:
                lbl.config(text=f"{lang} Voice:")
            except Exception:
                pass
        # Update every registered language-dependent label (regions headers,
        # audio-sync section titles, captions header, …).
        for lbl, template in getattr(self, "_lang_dyn_labels", []):
            try:
                lbl.config(text=template.format(LANG=lang.upper()))
            except Exception:
                pass
        # Persist choice across launches.
        _write_last_language(lang)
        # Refresh Google TTS voice list / engine availability.
        self._refresh_tts_voice_list()
        # If Google TTS is unavailable for this language, force-switch to
        # ElevenLabs and surface a hint.
        lang_data = TTS_LANGUAGES.get(lang, {})
        if lang_data.get("google_unavailable") and self._tts_platform_var \
                and self._tts_platform_var.get() == "Google TTS":
            self._tts_platform_var.set("ElevenLabs")
            self._apply_tts_platform_visibility()
            self._set_el_status(
                f"Google TTS has no native {lang} voices — using ElevenLabs.",
                "#d97706")
        # Refresh ElevenLabs voice list with the new-language filter.
        try:
            key = (self._api_key_var.get() if self._api_key_var else "").strip()
            if key:
                self._last_validated_key = None  # force re-evaluate per-lang cache
                self._on_api_key_changed()
        except Exception:
            pass

    def _on_tts_engine_change(self):
        self._refresh_tts_voice_list()

    def _refresh_tts_voice_list(self):
        lang   = self._tts_language_var.get()
        engine = self._tts_engine_var.get()
        lang_data = TTS_LANGUAGES.get(lang, {})

        # If Google has no voices for this language, leave the engine row
        # populated but empty — the platform-visibility logic hides it.
        std_avail = bool(lang_data.get("Standard"))
        wn_avail = bool(lang_data.get("WaveNet"))
        c3_avail = bool(lang_data.get("Chirp3"))

        if self._tts_rb_wavenet:
            self._tts_rb_wavenet.config(state="normal" if wn_avail else "disabled")
        if not wn_avail and engine == "WaveNet":
            self._tts_engine_var.set("Standard")
            engine = "Standard"

        if self._tts_rb_chirp3:
            self._tts_rb_chirp3.config(state="normal" if c3_avail else "disabled")
        if not c3_avail and engine == "Chirp3":
            self._tts_engine_var.set("Standard")
            engine = "Standard"

        voices = lang_data.get(engine, [])
        self._tts_voice_cb["values"] = voices
        if self._tts_voice_cb_mirror:
            self._tts_voice_cb_mirror["values"] = voices

        # Pick a sensible default
        if voices:
            preferred = f"{lang_data.get('code','')}-{engine.capitalize()}-HD-Algenib"
            if preferred in voices:
                self._tts_voice_var.set(preferred)
            elif self._tts_voice_var.get() not in voices:
                self._tts_voice_var.set(voices[0])

    def _get_tts_params(self):
        """Return (platform, lang_code, voice_name, el_voice_id, language,
        el_model) from the TTS settings panel. el_voice_id is resolved from
        the auto-loaded ElevenLabs voice dropdown; el_model is the raw
        ElevenLabs model id from the Model picker."""
        platform  = self._tts_platform_var.get() if self._tts_platform_var else TTS_DEFAULT_PLATFORM
        lang      = self._tts_language_var.get() if self._tts_language_var else TTS_DEFAULT_LANGUAGE
        if lang not in TTS_LANGUAGES:
            lang = TTS_DEFAULT_LANGUAGE
        voice     = self._tts_voice_var.get()    if self._tts_voice_var    else TTS_DEFAULT_VOICE
        lang_code = TTS_LANGUAGES.get(lang, {}).get("code", "bn-IN")
        el_voice_id = self._resolve_el_voice_id()
        return platform, lang_code, voice, el_voice_id, lang, self._get_el_model()

    def _get_en_region_params(self):
        """Return (thr_db, hys_db, min_ms) for English audio region detection."""
        try:
            thr = float(self.thr_var.get())
            hys = float(self.hys_var.get())
            ms  = int(self.minsilence_var.get())
        except Exception:
            thr, hys, ms = DEFAULT_THR_DB, DEFAULT_HYS_DB, DEFAULT_MIN_MS
        return thr, hys, ms

    def _get_bn_region_params(self):
        """Return (thr_db, hys_db, min_ms) for Bengali TTS audio region detection."""
        try:
            thr = float(self.bn_thr_var.get())
            hys = float(self.bn_hys_var.get())
            ms  = int(self.bn_minsilence_var.get())
        except Exception:
            thr, hys, ms = DEFAULT_BN_THR_DB, DEFAULT_BN_HYS_DB, DEFAULT_BN_MIN_MS
        return thr, hys, ms

    def _build_regions_panel(self, parent):
        rp = tk.Frame(parent, bg=PANEL2, height=44, bd=0,
                      highlightbackground=TEXT_MUTED, highlightthickness=1)
        rp.pack(fill="x", side="top")
        rp.pack_propagate(False)
        self._regions_frame_en = rp     # hidden in Simple mode

        tk.Label(rp, text="ENGLISH REGIONS", bg=PANEL2, fg=REG_EDGE,
                 font=(MONO_FONT, 9, "bold")).pack(side="left", padx=(14, 10), pady=10)
        tk.Frame(rp, bg=TEXT_MUTED, width=2, height=26).pack(side="left", padx=(0, 12), pady=9)

        tk.Label(rp, text="Threshold (dBFS)", bg=PANEL2, fg=TEXT,
                 font=(MONO_FONT, 9)).pack(side="left", padx=(0, 4))
        self.thr_var = tk.DoubleVar(value=-42.0)
        self._thr_spinbox = tk.Spinbox(rp, from_=-80.0, to=0.0, increment=1.0,
                   textvariable=self.thr_var, width=6, bg=INPUT_BG, fg=INPUT_FG,
                   insertbackground=INPUT_FG, relief="flat", font=(MONO_FONT, 10),
                   buttonbackground="#334155")
        self._thr_spinbox.pack(side="left", padx=(0, 16))

        tk.Label(rp, text="Hysteresis (dB)", bg=PANEL2, fg=TEXT,
                 font=(MONO_FONT, 9)).pack(side="left", padx=(0, 4))
        self.hys_var = tk.DoubleVar(value=6.0)
        self._hys_spinbox = tk.Spinbox(rp, from_=0.0, to=40.0, increment=0.5,
                   textvariable=self.hys_var, width=6, bg=INPUT_BG, fg=INPUT_FG,
                   insertbackground=INPUT_FG, relief="flat", font=(MONO_FONT, 10),
                   buttonbackground="#334155")
        self._hys_spinbox.pack(side="left", padx=(0, 16))

        tk.Label(rp, text="Min silence (ms)", bg=PANEL2, fg=TEXT,
                 font=(MONO_FONT, 9)).pack(side="left", padx=(0, 4))
        self.minsilence_var = tk.IntVar(value=150)
        self._minsilence_spinbox = tk.Spinbox(rp, from_=10, to=5000, increment=10,
                   textvariable=self.minsilence_var, width=7, bg=INPUT_BG, fg=INPUT_FG,
                   insertbackground=INPUT_FG, relief="flat", font=(MONO_FONT, 10),
                   buttonbackground="#334155")
        self._minsilence_spinbox.pack(side="left", padx=(0, 16))

        self._region_debounce_id = None
        self._btn(rp, "Re-Apply", self._apply_regions,
                  bg="#172554", fg=REG_LABEL, abg="#1e3a8a").pack(side="left", padx=(0, 8), pady=7)
        self._btn(rp, "Clear", self._clear_regions,
                  bg="#1f1213", fg="#f87171", abg="#3a1414").pack(side="left", padx=(0, 12), pady=7)
        self.region_count_label = tk.Label(rp, text="", bg=PANEL2, fg=REG_LABEL,
                                           font=(MONO_FONT, 9))
        self.region_count_label.pack(side="left", padx=4)

    def _build_bn_regions_panel(self, parent):
        """Bengali TTS audio region detection panel (separate defaults from English)."""
        rp = tk.Frame(parent, bg="#0c1f2c", height=44, bd=0,
                      highlightbackground="#1e3a52", highlightthickness=1)
        rp.pack(fill="x", side="top")
        rp.pack_propagate(False)
        self._regions_frame_bn = rp     # hidden in Simple mode

        _bn_regions_lbl = tk.Label(rp, text="BENGALI REGIONS", bg="#0c1f2c", fg="#38bdf8",
                 font=(MONO_FONT, 9, "bold"))
        _bn_regions_lbl.pack(side="left", padx=(14, 10), pady=10)
        self._register_lang_label(_bn_regions_lbl, "{LANG} REGIONS")
        tk.Frame(rp, bg=TEXT_MUTED, width=2, height=26).pack(side="left", padx=(0, 12), pady=9)

        tk.Label(rp, text="Threshold (dBFS)", bg="#0c1f2c", fg=TEXT,
                 font=(MONO_FONT, 9)).pack(side="left", padx=(0, 4))
        self.bn_thr_var = tk.DoubleVar(value=DEFAULT_BN_THR_DB)
        tk.Spinbox(rp, from_=-80.0, to=0.0, increment=1.0,
                   textvariable=self.bn_thr_var, width=6, bg=INPUT_BG, fg=INPUT_FG,
                   insertbackground=INPUT_FG, relief="flat", font=(MONO_FONT, 10),
                   buttonbackground="#334155").pack(side="left", padx=(0, 16))

        tk.Label(rp, text="Hysteresis (dB)", bg="#0c1f2c", fg=TEXT,
                 font=(MONO_FONT, 9)).pack(side="left", padx=(0, 4))
        self.bn_hys_var = tk.DoubleVar(value=DEFAULT_BN_HYS_DB)
        tk.Spinbox(rp, from_=0.0, to=40.0, increment=0.5,
                   textvariable=self.bn_hys_var, width=6, bg=INPUT_BG, fg=INPUT_FG,
                   insertbackground=INPUT_FG, relief="flat", font=(MONO_FONT, 10),
                   buttonbackground="#334155").pack(side="left", padx=(0, 16))

        tk.Label(rp, text="Min silence (ms)", bg="#0c1f2c", fg=TEXT,
                 font=(MONO_FONT, 9)).pack(side="left", padx=(0, 4))
        self.bn_minsilence_var = tk.IntVar(value=DEFAULT_BN_MIN_MS)
        tk.Spinbox(rp, from_=10, to=5000, increment=10,
                   textvariable=self.bn_minsilence_var, width=7, bg=INPUT_BG, fg=INPUT_FG,
                   insertbackground=INPUT_FG, relief="flat", font=(MONO_FONT, 10),
                   buttonbackground="#334155").pack(side="left", padx=(0, 16))

        tk.Label(rp, text="(used for Stage 3b TTS audio)", bg="#0c1f2c", fg=TEXT_FAINT,
                 font=(MONO_FONT, 8, "italic")).pack(side="left", padx=(4, 0))

    def _build_translation_panel(self, parent):
        tp = tk.Frame(parent, bg=PANEL3, bd=0,
                      highlightbackground=PANEL_BORDER, highlightthickness=1)
        tp.pack(fill="x", side="top")
        self._tp_frame = tp             # anchor for restoring hidden panels

        # ── Header row (Run/Cancel live on the main tab's run bar) ───────────
        row = tk.Frame(tp, bg=PANEL3, height=40)
        row.pack(fill="x")
        row.pack_propagate(False)

        tk.Label(row, text="PIPELINE OPTIONS", bg=PANEL3, fg=TR_ACCENT,
                 font=(MONO_FONT, 9, "bold")).pack(side="left", padx=(14, 8), pady=10)
        tk.Frame(row, bg="#1f4d2e", width=2, height=26).pack(side="left", padx=(0, 10), pady=7)
        tk.Label(row, text="These options apply to the next ▶ Run Pipeline.",
                 bg=PANEL3, fg=TEXT_FAINT, font=(MONO_FONT, 9)).pack(side="left")

        # ── "Run pipeline up to" selector ─────────────────────────────────────
        step_row = tk.Frame(tp, bg=PANEL3, height=34)
        step_row.pack(fill="x")
        step_row.pack_propagate(False)

        tk.Label(step_row, text="Run pipeline up to:",
                 bg=PANEL3, fg=TEXT_FAINT, font=(MONO_FONT, 9, "bold")
                 ).pack(side="left", padx=(14, 10), pady=6)

        _single_step_opts = [
            ("SRT only",      "English SRT"),
            ("+ Translation", "Translation"),
            ("+ TTS Audio",   "TTS Audio"),
            ("Full Pipeline", "Full Pipeline"),
        ]
        for label, value in _single_step_opts:
            is_default = (value == "Translation")
            tk.Radiobutton(
                step_row, text=label,
                variable=self._single_stop_step, value=value,
                bg=PANEL3, fg=ACCENT if is_default else TEXT,
                selectcolor=PANEL,
                activebackground=PANEL3, activeforeground=ACCENT,
                font=(MONO_FONT, 9, "bold" if is_default else "normal"),
                cursor="hand2",
            ).pack(side="left", padx=(0, 18), pady=4)

        # ── LLM provider row ───────────────────────────────────────────────────
        model_row = tk.Frame(tp, bg=PANEL3, height=34)
        model_row.pack(fill="x")
        model_row.pack_propagate(False)
        self._model_row = model_row     # hidden in Simple mode

        self._llm_powered_label = tk.Label(
            model_row, text=f"Powered by {_llm_provider_label()}",
            bg=PANEL3, fg=TEXT_FAINT, font=(MONO_FONT, 9, "bold"))
        self._llm_powered_label.pack(side="left", padx=(14, 10), pady=6)

        tk.Button(model_row, text="⚙ LLM Settings",
                  command=self._open_llm_settings_dialog,
                  bg=PANEL, fg=_btn_fg(TEXT), activebackground=PANEL3,
                  activeforeground=_btn_fg(ACCENT), font=(MONO_FONT, 9),
                  cursor="hand2", relief="flat", padx=8,
                  ).pack(side="left", padx=(6, 0), pady=4)

        tk.Button(model_row, text="📝 Edit Prompts",
                  command=self._open_prompt_editor_dialog,
                  bg=PANEL, fg=_btn_fg(TEXT), activebackground=PANEL3,
                  activeforeground=_btn_fg(ACCENT), font=(MONO_FONT, 9),
                  cursor="hand2", relief="flat", padx=8,
                  ).pack(side="left", padx=(6, 0), pady=4)

        tk.Button(model_row, text=f"⟳ Check for Updates (v{APP_VERSION})",
                  command=self._check_for_updates,
                  bg=PANEL, fg=_btn_fg(TEXT), activebackground=PANEL3,
                  activeforeground=_btn_fg(ACCENT), font=(MONO_FONT, 9),
                  cursor="hand2", relief="flat", padx=8,
                  ).pack(side="left", padx=(6, 0), pady=4)

        tk.Button(model_row, text="💬 Send Feedback",
                  command=self._open_feedback_dialog,
                  bg=PANEL, fg=_btn_fg(TEXT), activebackground=PANEL3,
                  activeforeground=_btn_fg(ACCENT), font=(MONO_FONT, 9),
                  cursor="hand2", relief="flat", padx=8,
                  ).pack(side="left", padx=(6, 0), pady=4)

        # ── Options row (own row so nothing clips off the right edge) ─────────
        opts_row = tk.Frame(tp, bg=PANEL3, height=34)
        opts_row.pack(fill="x")
        opts_row.pack_propagate(False)

        # Prompt-chain depth: how many LLM passes the translation makes
        tk.Label(opts_row, text="Prompt steps:", bg=PANEL3, fg=TEXT_FAINT,
                 font=(MONO_FONT, 9, "bold")).pack(side="left", padx=(14, 4), pady=6)
        for n, tip in ((1, "1 (translate)"), (2, "2 (+review)"), (3, "3 (+punctuation)")):
            tk.Radiobutton(
                opts_row, text=tip, variable=self._translation_steps_var, value=n,
                bg=PANEL3, fg=TEXT, selectcolor=PANEL,
                activebackground=PANEL3, activeforeground=ACCENT,
                font=(MONO_FONT, 9), cursor="hand2",
            ).pack(side="left", padx=(0, 6), pady=6)

        tk.Frame(opts_row, bg=PANEL_BORDER, width=2, height=20).pack(
            side="left", padx=(8, 12), pady=7)

        tk.Checkbutton(
            opts_row, text="Emotion Enhancement",
            variable=self._emotion_enabled_var,
            bg=PANEL3, fg=TEXT, selectcolor=PANEL,
            activebackground=PANEL3, activeforeground=ACCENT,
            font=(MONO_FONT, 9),
        ).pack(side="left", padx=(0, 12), pady=6)

        tk.Checkbutton(
            opts_row, text="Review before dubbing",
            variable=self._review_enabled_var,
            bg=PANEL3, fg=TEXT, selectcolor=PANEL,
            activebackground=PANEL3, activeforeground=ACCENT,
            font=(MONO_FONT, 9), cursor="hand2",
        ).pack(side="left", padx=(0, 6), pady=6)

        if translation_memory is not None:
            tk.Checkbutton(
                opts_row, text="Translation memory",
                variable=self._tm_enabled_var,
                bg=PANEL3, fg=TEXT, selectcolor=PANEL,
                activebackground=PANEL3, activeforeground=ACCENT,
                font=(MONO_FONT, 9), cursor="hand2",
            ).pack(side="left", padx=(0, 6), pady=6)

    # ── LLM provider settings dialog ─────────────────────────────────────────

    # ── In-app updater (opt-in) ──────────────────────────────────────────
    def _check_for_updates(self):
        """Contact GitHub in a background thread; never blocks the UI and
        never updates without explicit user confirmation."""
        def worker():
            try:
                remote = fetch_remote_version()
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: messagebox.showerror(
                    "Update Check Failed",
                    "Could not reach GitHub to check for updates.\n\n"
                    f"{err}\n\nCheck your internet connection and try again."))
                return
            if _version_tuple(remote) <= _version_tuple(APP_VERSION):
                self.root.after(0, lambda: messagebox.showinfo(
                    "Up to Date",
                    f"You already have the latest version (v{APP_VERSION})."))
                return
            self.root.after(0, lambda: self._prompt_and_apply_update(remote))
        threading.Thread(target=worker, daemon=True).start()

    def _prompt_and_apply_update(self, remote_version: str):
        proceed = messagebox.askyesno(
            "Update Available",
            "A newer version is available on GitHub.\n\n"
            f"   Installed:  v{APP_VERSION}\n"
            f"   Latest:     v{remote_version}\n\n"
            "Update now?\n\n"
            "• Your API keys and settings are never touched.\n"
            "• Prompt files may be refreshed from GitHub — any file that\n"
            "  gets replaced is first backed up to the _update_backup folder.\n"
            "• Restart the app after the update finishes.")
        if not proceed:
            return

        def worker():
            try:
                backup = download_and_apply_update()
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: messagebox.showerror(
                    "Update Failed",
                    f"The update could not be applied:\n\n{err}\n\n"
                    "Nothing was changed, or changed files were backed up.\n"
                    "You can retry, or re-download the app from GitHub."))
                return
            self.root.after(0, lambda: messagebox.showinfo(
                "Update Complete",
                f"Updated to v{remote_version}.\n\n"
                f"Backups of replaced files: {backup}\n\n"
                "Please close and reopen the app to use the new version."))
        threading.Thread(target=worker, daemon=True).start()

    # ── Feedback dialog (message + screenshots → GitHub issue) ───────────
    def _open_feedback_dialog(self):
        """Collect a message + optional screenshots and file them as a
        GitHub issue on the app repo. Falls back to saving the feedback
        into feedback_outbox/ when GitHub can't be reached."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Send Feedback")
        dlg.configure(bg=PANEL)
        dlg.resizable(False, False)
        dlg.transient(self.root)

        tk.Label(dlg, text="Share feedback, ideas or bug reports",
                 bg=PANEL, fg=TEXT, font=(MONO_FONT, 11, "bold")
                 ).pack(anchor="w", padx=14, pady=(12, 2))
        tk.Label(dlg, text="Sent straight to the developer — screenshots welcome.",
                 bg=PANEL, fg=TEXT_FAINT, font=(MONO_FONT, 9)
                 ).pack(anchor="w", padx=14, pady=(0, 8))

        kind_var = tk.StringVar(value="Feedback")
        kind_row = tk.Frame(dlg, bg=PANEL)
        kind_row.pack(anchor="w", padx=14)
        for label in ("💬 Feedback", "💡 Improvement", "🐞 Bug"):
            tk.Radiobutton(
                kind_row, text=label, variable=kind_var,
                value=label.split(" ", 1)[1],
                bg=PANEL, fg=TEXT, selectcolor=PANEL3,
                activebackground=PANEL, activeforeground=ACCENT,
                font=(MONO_FONT, 9), cursor="hand2",
            ).pack(side="left", padx=(0, 12))

        name_row = tk.Frame(dlg, bg=PANEL)
        name_row.pack(fill="x", padx=14, pady=(8, 0))
        tk.Label(name_row, text="Your name (optional):", bg=PANEL,
                 fg=TEXT_FAINT, font=(MONO_FONT, 9)).pack(side="left")
        name_entry = tk.Entry(name_row, bg=INPUT_BG, fg=INPUT_FG,
                              font=(MONO_FONT, 9), width=28, relief="flat")
        name_entry.pack(side="left", padx=(6, 0), ipady=2)

        msg_box = scrolledtext.ScrolledText(
            dlg, bg=PANEL3, fg=TEXT, insertbackground=TEXT,
            font=(MONO_FONT, 10), wrap="word", width=64, height=9)
        msg_box.pack(fill="both", expand=True, padx=14, pady=(8, 4))
        msg_box.focus_set()

        attachments: List[str] = []
        attach_lbl = tk.Label(dlg, text="No screenshots attached.", bg=PANEL,
                              fg=TEXT_FAINT, font=(MONO_FONT, 8),
                              anchor="w", justify="left")

        def _refresh_attach_label():
            if attachments:
                names = ", ".join(os.path.basename(p) for p in attachments)
                attach_lbl.config(text=f"Attached: {names}")
            else:
                attach_lbl.config(text="No screenshots attached.")

        def _add_attachments():
            paths = filedialog.askopenfilenames(
                parent=dlg, title="Attach screenshot(s)",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.gif *.bmp"),
                           ("All files", "*.*")])
            for p in paths:
                try:
                    mb = os.path.getsize(p) / (1024 * 1024)
                except OSError:
                    continue
                if mb > FEEDBACK_MAX_ATTACHMENT_MB:
                    messagebox.showwarning(
                        "File Too Large",
                        f"{os.path.basename(p)} is {mb:.0f} MB — the limit "
                        f"is {FEEDBACK_MAX_ATTACHMENT_MB} MB per file.",
                        parent=dlg)
                    continue
                if p not in attachments:
                    attachments.append(p)
            _refresh_attach_label()

        def _clear_attachments():
            attachments.clear()
            _refresh_attach_label()

        attach_row = tk.Frame(dlg, bg=PANEL)
        attach_row.pack(fill="x", padx=14)
        tk.Button(attach_row, text="📎 Attach Screenshot(s)",
                  command=_add_attachments,
                  bg=PANEL3, fg=_btn_fg(TEXT), activebackground=PANEL3,
                  activeforeground=_btn_fg(ACCENT), font=(MONO_FONT, 9),
                  cursor="hand2", relief="flat", padx=8,
                  ).pack(side="left", pady=2)
        tk.Button(attach_row, text="Clear", command=_clear_attachments,
                  bg=PANEL3, fg=_btn_fg(TEXT), activebackground=PANEL3,
                  activeforeground=_btn_fg(ACCENT), font=(MONO_FONT, 9),
                  cursor="hand2", relief="flat", padx=8,
                  ).pack(side="left", padx=(6, 0), pady=2)
        attach_lbl.pack(fill="x", padx=14, pady=(2, 0))

        status_lbl = tk.Label(dlg, text="", bg=PANEL, fg=TEXT_FAINT,
                              font=(MONO_FONT, 9), anchor="w")
        status_lbl.pack(fill="x", padx=14, pady=(4, 0))

        btn_row = tk.Frame(dlg, bg=PANEL)
        btn_row.pack(fill="x", padx=14, pady=(6, 12))
        send_btn = tk.Button(
            btn_row, text="Send  ➤",
            bg="#2563eb", fg=_btn_fg("white"), activebackground=PANEL3,
            activeforeground=_btn_fg(ACCENT), font=(MONO_FONT, 9, "bold"),
            cursor="hand2", relief="flat", padx=14)
        send_btn.pack(side="right")
        tk.Button(btn_row, text="Cancel", command=dlg.destroy,
                  bg=PANEL3, fg=_btn_fg(TEXT), activebackground=PANEL3,
                  activeforeground=_btn_fg(ACCENT), font=(MONO_FONT, 9),
                  cursor="hand2", relief="flat", padx=10,
                  ).pack(side="right", padx=(0, 8))

        def _send():
            message = msg_box.get("1.0", "end").strip()
            if not message:
                messagebox.showwarning("Empty Message",
                                       "Please write a message first.",
                                       parent=dlg)
                return
            send_btn.config(state="disabled")
            status_lbl.config(text="Sending…")
            kind = kind_var.get()
            sender = name_entry.get().strip()
            files = list(attachments)
            threading.Thread(
                target=self._submit_feedback,
                args=(dlg, send_btn, status_lbl, kind, sender, message, files),
                daemon=True).start()

        send_btn.config(command=_send)

    def _submit_feedback(self, dlg, send_btn, status_lbl,
                         kind, sender, message, files):
        """Background worker: upload screenshots, open the GitHub issue.
        All UI updates go through root.after()."""
        import datetime
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            links = []
            if files:
                self.root.after(0, lambda: status_lbl.config(
                    text="Uploading screenshots…"))
                _ensure_feedback_branch()
                for p in files:
                    links.append((os.path.basename(p),
                                  upload_feedback_attachment(p, stamp)))
            self.root.after(0, lambda: status_lbl.config(
                text="Creating report…"))
            first_line = message.strip().splitlines()[0][:60]
            title = f"[{kind}] {first_line}"
            body = (f"**Type:** {kind}\n"
                    f"**From:** {sender or '(not given)'}\n"
                    f"**App:** v{APP_VERSION} · {sys.platform} · "
                    f"Python {sys.version.split()[0]}\n\n---\n\n{message}\n")
            if links:
                body += "\n---\n\n**Screenshots:**\n" + "\n".join(
                    f"- [{name}]({url})" for name, url in links)
            issue_url = create_feedback_issue(title, body,
                                              label=kind.lower())

            def _done():
                dlg.destroy()
                messagebox.showinfo(
                    "Feedback Sent",
                    "Thank you! Your feedback was delivered to the "
                    "developer.\n\n" + (f"Reference: {issue_url}"
                                        if issue_url else ""))
            self.root.after(0, _done)
        except Exception as e:
            err = str(e)
            folder = ""
            try:
                folder = save_feedback_locally(kind, sender, message, files)
            except OSError:
                pass

            def _failed():
                send_btn.config(state="normal")
                status_lbl.config(text="")
                hint = ("Sending needs the github_token.txt file next to the "
                        "app (the same one used for updates) and an internet "
                        "connection.")
                saved = (f"\n\nYour feedback was saved to:\n{folder}\n"
                         "You can send that folder to the developer manually."
                         if folder else "")
                messagebox.showerror(
                    "Could Not Send Feedback",
                    f"{hint}\n\nDetails: {err}{saved}", parent=dlg)
            self.root.after(0, _failed)

    def _open_llm_settings_dialog(self):
        """Modal dialog: pick provider (Vertex JSON / Gemini key / OpenAI base
        URL) and its credentials. Saved to llm_settings.json."""
        s = _get_llm_settings()

        dlg = tk.Toplevel(self.root)
        dlg.title("LLM Settings")
        dlg.configure(bg=PANEL)
        dlg.resizable(False, False)
        dlg.transient(self.root)
        try:
            dlg.grab_set()
        except tk.TclError:
            pass   # not viewable yet (minimized parent) — run non-modal

        provider_var    = tk.StringVar(value=s.get("provider", LLM_PROVIDER_VERTEX))
        vertex_json_var = tk.StringVar(value=s.get("vertex_json", ""))
        gemini_key_var  = tk.StringVar(value=s.get("gemini_api_key", ""))
        base_url_var    = tk.StringVar(value=s.get("openai_base_url", LLM_DEFAULT_BASE_URL))
        openai_key_var  = tk.StringVar(value=s.get("openai_api_key", ""))
        openai_model_var = tk.StringVar(value=s.get("openai_model", ""))

        body = tk.Frame(dlg, bg=PANEL, padx=16, pady=12)
        body.pack(fill="both", expand=True)

        tk.Label(body, text="TRANSLATION LLM PROVIDER", bg=PANEL, fg=ACCENT,
                 font=(MONO_FONT, 10, "bold")).pack(anchor="w", pady=(0, 8))

        def _mk_section(title):
            frame = tk.Frame(body, bg=PANEL3, padx=10, pady=8,
                             highlightbackground="#3b82f6", highlightthickness=1)
            frame.pack(fill="x", pady=(0, 8))
            tk.Radiobutton(frame, text=title, variable=provider_var, value=title,
                           bg=PANEL3, fg=TEXT, selectcolor=PANEL,
                           activebackground=PANEL3, activeforeground=ACCENT,
                           font=(MONO_FONT, 9, "bold"), cursor="hand2",
                           ).pack(anchor="w")
            return frame

        def _mk_entry(parent, label, var, show=None, width=44):
            row = tk.Frame(parent, bg=PANEL3)
            row.pack(fill="x", pady=(4, 0))
            tk.Label(row, text=label, bg=PANEL3, fg=TEXT_FAINT,
                     font=(MONO_FONT, 9), width=12, anchor="w").pack(side="left")
            ent = tk.Entry(row, textvariable=var, width=width, show=show or "",
                           bg=INPUT_BG, fg=INPUT_FG, insertbackground=INPUT_FG,
                           font=(MONO_FONT, 9), relief="flat")
            ent.pack(side="left", fill="x", expand=True, ipady=3)
            return row

        # 1 — Vertex AI (service-account JSON)
        f_vertex = _mk_section(LLM_PROVIDER_VERTEX)
        row = _mk_entry(f_vertex, "JSON file:", vertex_json_var)
        def _browse_json():
            p = filedialog.askopenfilename(
                parent=dlg, title="Select service-account JSON",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
            if p:
                vertex_json_var.set(p)
        tk.Button(row, text="Browse…", command=_browse_json,
                  bg=PANEL, fg=_btn_fg(TEXT), font=(MONO_FONT, 8), relief="flat",
                  cursor="hand2").pack(side="left", padx=(6, 0))
        tk.Label(f_vertex, text="Blank = vertex_key.json next to this script",
                 bg=PANEL3, fg=TEXT_FAINT, font=(MONO_FONT, 8)).pack(anchor="w", pady=(2, 0))

        # 2 — Gemini API key
        f_gemini = _mk_section(LLM_PROVIDER_GEMINI)
        _mk_entry(f_gemini, "API key:", gemini_key_var, show="•")

        # 3 — OpenAI-compatible endpoint (LiteLLM proxy etc.)
        f_openai = _mk_section(LLM_PROVIDER_OPENAI)
        _mk_entry(f_openai, "Base URL:", base_url_var)
        _mk_entry(f_openai, "API key:", openai_key_var, show="•")
        _mk_entry(f_openai, "Model:", openai_model_var)
        tk.Label(f_openai,
                 text="Works with LiteLLM / OpenRouter / any /v1/chat/completions "
                      "endpoint.\nAPI key optional for local proxies.",
                 bg=PANEL3, fg=TEXT_FAINT, font=(MONO_FONT, 8),
                 justify="left").pack(anchor="w", pady=(2, 0))

        status_lbl = tk.Label(body, text="", bg=PANEL, fg=TEXT_FAINT,
                              font=(MONO_FONT, 9), wraplength=420, justify="left")
        status_lbl.pack(anchor="w", pady=(0, 6))

        cache_var = tk.BooleanVar(value=s.get("prompt_caching", "1") == "1")
        cache_row = tk.Frame(body, bg=PANEL)
        cache_row.pack(fill="x", pady=(0, 6))
        tk.Checkbutton(
            cache_row, text="Prompt caching (reuse the big prompt across audios)",
            variable=cache_var,
            bg=PANEL, fg=TEXT, selectcolor=PANEL3,
            activebackground=PANEL, activeforeground=ACCENT,
            font=(MONO_FONT, 9), cursor="hand2",
        ).pack(side="left")
        tk.Label(cache_row,
                 text="Vertex/Gemini: explicit 1-hour server cache.\n"
                      "OpenAI-compatible: automatic (implicit) prefix caching.",
                 bg=PANEL, fg=TEXT_FAINT, font=(MONO_FONT, 8),
                 justify="left").pack(side="left", padx=(12, 0))

        def _collect():
            _LLM_SETTINGS.update({
                "provider":        provider_var.get(),
                "vertex_json":     vertex_json_var.get().strip(),
                "gemini_api_key":  gemini_key_var.get().strip(),
                "openai_base_url": base_url_var.get().strip(),
                "openai_api_key":  openai_key_var.get().strip(),
                "openai_model":    openai_model_var.get().strip(),
                "prompt_caching":  "1" if cache_var.get() else "0",
            })

        def _test():
            _collect()
            status_lbl.config(text="Testing connection…", fg=TEXT_FAINT)
            def worker():
                try:
                    _validate_llm_config()
                    reply = _llm_generate("Reply with exactly: OK",
                                          self._gemini_model_var.get())
                    msg, color = f"✓ Connection OK — reply: {reply.strip()[:80]}", "#4ade80"
                except Exception as e:
                    msg, color = f"✗ {e}", "#f87171"
                if dlg.winfo_exists():
                    dlg.after(0, lambda: status_lbl.config(text=msg, fg=color))
            threading.Thread(target=worker, daemon=True).start()

        def _save():
            _collect()
            try:
                _validate_llm_config()
            except Exception as e:
                status_lbl.config(text=f"✗ {e}", fg="#f87171")
                return
            try:
                _save_llm_settings()
            except Exception as e:
                status_lbl.config(text=f"✗ Could not save settings: {e}", fg="#f87171")
                return
            if getattr(self, "_llm_powered_label", None):
                self._llm_powered_label.config(text=f"Powered by {_llm_provider_label()}")
            dlg.destroy()

        btn_row = tk.Frame(body, bg=PANEL)
        btn_row.pack(fill="x")
        tk.Button(btn_row, text="Test Connection", command=_test,
                  bg=PANEL3, fg=_btn_fg(TEXT), font=(MONO_FONT, 9), relief="flat",
                  cursor="hand2", padx=10).pack(side="left")
        tk.Button(btn_row, text="Save", command=_save,
                  bg="#2563eb", fg=_btn_fg("white"), font=(MONO_FONT, 9, "bold"),
                  relief="flat", cursor="hand2", padx=16).pack(side="right")
        tk.Button(btn_row, text="Cancel", command=dlg.destroy,
                  bg=PANEL3, fg=_btn_fg(TEXT), font=(MONO_FONT, 9), relief="flat",
                  cursor="hand2", padx=10).pack(side="right", padx=(0, 8))

    # ── Prompt editor dialog ─────────────────────────────────────────────────

    _PROMPT_STAGES = [
        ("Step 1 — Translation",  "Step1_Translation_Prompt"),
        ("Step 2 — Review",       "Step2_Review_Prompt"),
        ("Step 3 — Punctuation",  "Step3_Punctuation_Prompt"),
        ("Step 4 — Emotion tags", "Step4_Emotion_Prompt"),
        ("Syncing / Mapping",     "SyncingPrompt"),
    ]

    def _open_prompt_editor_dialog(self):
        """View / edit / create the per-language prompt files in prompts/.
        Lets a new language be seeded by copying an existing language's
        prompt, so the app is easy to hand to other language teams."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Prompt Editor")
        dlg.configure(bg=PANEL)
        dlg.geometry("860x640")
        dlg.transient(self.root)
        try:
            dlg.grab_set()
        except tk.TclError:
            pass   # not viewable yet (minimized parent) — run non-modal

        languages = list(TTS_LANGUAGES.keys())
        cur_lang  = self._tts_language_var.get() if self._tts_language_var else languages[0]
        if cur_lang not in languages:
            cur_lang = languages[0]

        lang_var  = tk.StringVar(value=cur_lang)
        stage_var = tk.StringVar(value=self._PROMPT_STAGES[0][0])
        copy_var  = tk.StringVar(value="")
        _stage_by_label = dict(self._PROMPT_STAGES)

        top = tk.Frame(dlg, bg=PANEL, padx=12, pady=10)
        top.pack(fill="x")

        tk.Label(top, text="Language:", bg=PANEL, fg=TEXT_FAINT,
                 font=(MONO_FONT, 9)).pack(side="left")
        lang_cb = ttk.Combobox(top, textvariable=lang_var, values=languages,
                               state="readonly", width=12)
        lang_cb.pack(side="left", padx=(4, 16))

        tk.Label(top, text="Prompt:", bg=PANEL, fg=TEXT_FAINT,
                 font=(MONO_FONT, 9)).pack(side="left")
        stage_cb = ttk.Combobox(top, textvariable=stage_var,
                                values=[l for l, _ in self._PROMPT_STAGES],
                                state="readonly", width=22)
        stage_cb.pack(side="left", padx=(4, 0))

        path_lbl = tk.Label(dlg, text="", bg=PANEL, fg=TEXT_FAINT,
                            font=(MONO_FONT, 8), anchor="w", padx=12)
        path_lbl.pack(fill="x")

        editor = scrolledtext.ScrolledText(
            dlg, bg=PANEL3, fg=TEXT, insertbackground=TEXT,
            font=(MONO_FONT, 10), wrap="word", undo=True)
        editor.pack(fill="both", expand=True, padx=12, pady=(4, 6))

        status_lbl = tk.Label(dlg, text="", bg=PANEL, fg=TEXT_FAINT,
                              font=(MONO_FONT, 9), anchor="w", padx=12)
        status_lbl.pack(fill="x")

        def _prompt_path():
            stage = _stage_by_label[stage_var.get()]
            return os.path.join(SCRIPT_DIR, "prompts",
                                f"{stage}_{lang_var.get()}.txt")

        def _load(_event=None):
            p = _prompt_path()
            path_lbl.config(text=p)
            editor.delete("1.0", "end")
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    editor.insert("1.0", f.read())
                status_lbl.config(text="Loaded.", fg=TEXT_FAINT)
            else:
                status_lbl.config(
                    text="File does not exist yet — use “Copy from” to seed it "
                         "from another language, edit, then Save.",
                    fg="#fbbf24")
            editor.edit_reset()

        def _copy_from(_event=None):
            src_lang = copy_var.get()
            if not src_lang or src_lang == lang_var.get():
                return
            stage = _stage_by_label[stage_var.get()]
            src = os.path.join(SCRIPT_DIR, "prompts", f"{stage}_{src_lang}.txt")
            if not os.path.exists(src):
                status_lbl.config(text=f"No {stage} prompt for {src_lang}.",
                                  fg="#f87171")
                return
            with open(src, "r", encoding="utf-8") as f:
                editor.delete("1.0", "end")
                editor.insert("1.0", f.read())
            status_lbl.config(
                text=f"Copied from {src_lang}. Adapt language references, then Save.",
                fg="#fbbf24")

        def _save():
            p = _prompt_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(editor.get("1.0", "end-1c"))
            status_lbl.config(text=f"✓ Saved {os.path.basename(p)}", fg="#4ade80")

        btm = tk.Frame(dlg, bg=PANEL, padx=12, pady=8)
        btm.pack(fill="x")
        tk.Label(btm, text="Copy from:", bg=PANEL, fg=TEXT_FAINT,
                 font=(MONO_FONT, 9)).pack(side="left")
        copy_cb = ttk.Combobox(btm, textvariable=copy_var, values=languages,
                               state="readonly", width=12)
        copy_cb.pack(side="left", padx=(4, 12))
        copy_cb.bind("<<ComboboxSelected>>", _copy_from)
        tk.Button(btm, text="Save", command=_save,
                  bg="#2563eb", fg=_btn_fg("white"), font=(MONO_FONT, 9, "bold"),
                  relief="flat", cursor="hand2", padx=16).pack(side="right")
        tk.Button(btm, text="Close", command=dlg.destroy,
                  bg=PANEL3, fg=_btn_fg(TEXT), font=(MONO_FONT, 9), relief="flat",
                  cursor="hand2", padx=10).pack(side="right", padx=(0, 8))

        lang_cb.bind("<<ComboboxSelected>>", _load)
        stage_cb.bind("<<ComboboxSelected>>", _load)
        _load()

    def _build_pipeline_panel(self, parent):
        pp = tk.Frame(parent, bg="#162032", bd=0,
                      highlightbackground="#3b82f6", highlightthickness=1)
        pp.pack(fill="x", side="top")

        self._sf_pipe_frame = pp

        hdr = tk.Frame(pp, bg="#162032", height=28)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  PIPELINE PROGRESS", bg="#162032", fg="#60a5fa",
                 font=(MONO_FONT, 9, "bold")).pack(side="left", padx=14, pady=4)
        tk.Label(hdr, text="click a stage to preview its output file",
                 bg="#162032", fg=TEXT_FAINT, font=(MONO_FONT, 8)).pack(
            side="right", padx=14, pady=4)

        body = tk.Frame(pp, bg="#162032")
        body.pack(fill="x", padx=14, pady=(0, 8))

        stage_labels = [
            ("S1a", "Transcription (ElevenLabs)", "#cbd5e1"),
            ("S1b", "Translation / Review / Punctuation (LLM)", "#cbd5e1"),
            ("S2 ", "TTS — Dubbed Audio", "#d97706"),
            ("S3a", "Sync — English SRT", "#22c55e"),
            ("S3b", "Sync — Dubbed SRT from TTS Audio", "#22c55e"),
            ("S3c", "Sync — SRT Mapping (Gemini)", "#22c55e"),
            ("S3d", "Sync — SRT Timing Sync", "#22c55e"),
            ("S3e", "Sync — Synced Audio File", "#22c55e"),
        ]

        self._stage_vars  = {}
        self._stage_labels = {}
        self._stage_desc_labels = {}
        for tag, desc, colour in stage_labels:
            row = tk.Frame(body, bg="#162032", cursor="hand2")
            row.pack(fill="x", pady=1)
            _tag_lbl = tk.Label(row, text=f" [{tag}]", bg="#162032", fg=colour,
                     font=(MONO_FONT, 9, "bold"), width=6, anchor="w",
                     cursor="hand2")
            _tag_lbl.pack(side="left")
            _desc_lbl = tk.Label(row, text=desc, bg="#162032", fg=TEXT_FAINT,
                     font=(MONO_FONT, 9), width=50, anchor="w", cursor="hand2")
            _desc_lbl.pack(side="left")
            self._stage_desc_labels[tag.strip()] = _desc_lbl
            var = tk.StringVar(value="—")
            lbl = tk.Label(row, textvariable=var, bg="#162032", fg=TEXT_MUTED,
                           font=(MONO_FONT, 9), anchor="w", cursor="hand2")
            lbl.pack(side="left", fill="x", expand=True, padx=(8, 0))
            self._stage_vars[tag.strip()] = var
            self._stage_labels[tag.strip()] = lbl
            for w in (row, _tag_lbl, _desc_lbl, lbl):
                w.bind("<Button-1>",
                       lambda _e, t=tag.strip(): self._sf_stage_preview(t))

    # ─────────────────────────────────────────────────────────────────────────
    #  Single File tab — settings dialog / views / log / stage preview / compare
    # ─────────────────────────────────────────────────────────────────────────

    def _build_sf_settings_dialog(self):
        """One ⚙ Settings window for the Single File tab: TTS settings,
        region tuning, and pipeline options. Built once at startup (hidden)
        so all shared vars/widgets exist before the mirror panels bind."""
        win = tk.Toplevel(self.root)
        self._sf_settings_win = win
        win.title("Pipeline Settings")
        win.configure(bg=BG)
        win.geometry("1100x560")
        win.transient(self.root)
        win.protocol("WM_DELETE_WINDOW", win.withdraw)
        win.withdraw()

        top = tk.Frame(win, bg=PANEL, height=44)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(top, text="PIPELINE SETTINGS", bg=PANEL, fg="#a78bfa",
                 font=(MONO_FONT, 10, "bold")).pack(side="left",
                                                    padx=(14, 10), pady=10)
        tk.Checkbutton(
            top, text="Auto-open result", variable=self._auto_open_var,
            command=self._save_ui_settings,
            bg=PANEL, fg=TEXT, selectcolor=PANEL3,
            activebackground=PANEL, activeforeground=ACCENT,
            font=(MONO_FONT, 9), cursor="hand2",
        ).pack(side="left", padx=10, pady=8)
        self._btn(top, "  Close  ", win.withdraw,
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT).pack(
            side="right", padx=12, pady=6)

        body = tk.Frame(win, bg=BG)
        body.pack(fill="both", expand=True)
        self._sf_settings_body = body

        self._build_tts_settings_panel(body)
        self._build_regions_panel(body)
        self._build_bn_regions_panel(body)
        self._build_translation_panel(body)

    def _open_sf_settings_dialog(self):
        win = self._sf_settings_win
        try:
            win.deiconify()
            win.lift()
        except Exception:
            pass

    # ── 🎭 Multi-speaker dialog ───────────────────────────────────────────────

    def _open_speakers_dialog(self):
        """Assign a different ElevenLabs voice to individual paragraphs of
        the translated script. Saved per-project; applied on the next
        ▶ Run Pipeline / Re-Dub."""
        p = self._sf_run_paths()
        if not p:
            messagebox.showinfo("Speakers", "Open an audio file first.")
            return
        base = p["base"]
        lang = self._current_language()
        text = (self.punctuation_result or "").strip()
        if not text:
            fs_path = base + "_FinalScript.txt"
            if os.path.isfile(fs_path):
                try:
                    with open(fs_path, "r", encoding="utf-8",
                              errors="replace") as f:
                        text = _extract_translation_from_finalscript(
                            f.read(), lang).strip()
                except Exception:
                    text = ""
        paras = _split_translation_paragraphs(text) if text else []
        if not paras:
            messagebox.showinfo(
                "Speakers",
                "No translated script yet — run the pipeline at least up to "
                "Translation first, then assign voices per paragraph here.")
            return

        voice_opts = list(getattr(self, "_el_voice_options", []) or [])
        id_to_label = {o["voice_id"]: o["label"] for o in voice_opts}
        DEFAULT_LABEL = "— default voice —"
        values = [DEFAULT_LABEL] + [o["label"] for o in voice_opts]

        saved = _speakers_load(base).get("assignments") or {}

        win = tk.Toplevel(self.root)
        win.title("Speakers — voice per paragraph")
        win.configure(bg=BG)
        win.geometry("980x600")
        win.transient(self.root)

        hdr = tk.Frame(win, bg=PANEL, height=48)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="🎭 SPEAKERS", bg=PANEL, fg="#a78bfa",
                 font=(MONO_FONT, 11, "bold")).pack(side="left",
                                                    padx=(14, 10), pady=12)
        tk.Label(hdr,
                 text="Pick a voice per paragraph — paragraphs left on "
                      "“default voice” use the voice from ⚙ Settings. "
                      "Applied on the next ▶ Run Pipeline.",
                 bg=PANEL, fg=TEXT_FAINT, font=(MONO_FONT, 9)).pack(
            side="left", padx=6)

        status = tk.Label(win, text="", bg=BG, fg=TR_ACCENT,
                          font=(MONO_FONT, 9), anchor="w")

        wrap = tk.Frame(win, bg=BG)
        wrap.pack(fill="both", expand=True, padx=12, pady=(8, 4))
        cv = tk.Canvas(wrap, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(cv, bg=BG)
        cv_win = cv.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda _e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>",
                lambda ev: cv.itemconfigure(cv_win, width=ev.width))

        def _wheel(ev):
            n = getattr(ev, "num", 0)
            d = getattr(ev, "delta", 0)
            cv.yview_scroll(-1 if (d > 0 or n == 4) else 1, "units")
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            cv.bind(seq, _wheel)
            inner.bind(seq, _wheel)

        row_vars: List[tk.StringVar] = []
        for i, para in enumerate(paras):
            row = tk.Frame(inner, bg=PANEL if i % 2 == 0 else PANEL2)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"¶{i + 1:>3}", bg=row["bg"], fg="#a78bfa",
                     font=(MONO_FONT, 9, "bold"), width=5).pack(
                side="left", padx=(8, 4), pady=6)
            var = tk.StringVar(value=DEFAULT_LABEL)
            asg = saved.get(str(i)) or {}
            vid = _sanitize_voice_id(asg.get("voice_id", ""))
            if vid:
                var.set(id_to_label.get(vid, vid))
            cb = ttk.Combobox(row, textvariable=var, values=values, width=34)
            cb.pack(side="left", padx=(0, 10), pady=6)
            preview = " ".join(para.split())
            if len(preview) > 110:
                preview = preview[:109] + "…"
            tk.Label(row, text=preview, bg=row["bg"], fg=TEXT,
                     font=(MONO_FONT, 9), anchor="w").pack(
                side="left", fill="x", expand=True, padx=(0, 8))
            row_vars.append(var)

        def _save():
            assignments = {}
            n = 0
            for i, var in enumerate(row_vars):
                sel = (var.get() or "").strip()
                if not sel or sel == DEFAULT_LABEL:
                    continue
                vid = self._el_label_to_id.get(sel) \
                    or self._el_label_to_id.get(
                        re.sub(r"\s+", " ", sel).strip()) \
                    or _sanitize_voice_id(sel)
                if not vid:
                    continue
                assignments[str(i)] = {"voice_id": vid, "label": sel}
                n += 1
            try:
                _speakers_save(base, {
                    "assignments": assignments,
                    "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
            except Exception as ex:
                status.config(text=f"Could not save: {ex}", fg="#f87171")
                return
            status.config(
                text=(f"Saved — {n} paragraph(s) with a custom voice. "
                      "Run the pipeline to dub with these speakers.")
                if n else
                "Saved — all paragraphs use the default voice.",
                fg=TR_ACCENT)
            self._sf_log_append(
                f"[Speakers] {n} paragraph voice assignment(s) saved → "
                f"{os.path.basename(_speakers_file_path(base))}")

        def _clear():
            for var in row_vars:
                var.set(DEFAULT_LABEL)

        btns = tk.Frame(win, bg=BG)
        btns.pack(fill="x", padx=12, pady=(2, 10))
        status.pack(in_=btns, side="left", fill="x", expand=True)
        self._btn(btns, "  Close  ", win.destroy,
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT).pack(
            side="right", padx=(6, 0))
        self._btn(btns, " Clear All ", _clear,
                  bg="#1f1213", fg="#f87171", abg="#3a1414").pack(
            side="right", padx=6)
        self._btn(btns, " 💾 Save Speakers ", _save,
                  bg="#0f1d14", fg=TR_ACCENT, abg="#1f4d2e").pack(
            side="right", padx=6)

    def _sf_show_view(self, which):
        """Swap the main area between Waveform / Log / Compare."""
        self._sf_active_view = which
        for f in (self._sf_wave_view, self._sf_log_view, self._sf_cmp_view):
            f.pack_forget()
        target = {"wave": self._sf_wave_view,
                  "log": self._sf_log_view,
                  "compare": self._sf_cmp_view}.get(which, self._sf_wave_view)
        target.pack(fill="both", expand=True)
        for key, b in self._sf_view_btns.items():
            try:
                b.config(fg=_btn_fg("#a78bfa") if key == which
                         else _btn_fg(TEXT_FAINT))
            except Exception:
                pass
        if which == "compare" and not self._sf_cmp_data:
            self._sf_cmp_load()

    # ── log ──────────────────────────────────────────────────────────────────

    def _sf_log_append(self, msg):
        ts_ui = time.strftime("%H:%M:%S")

        def _do():
            try:
                self._sf_log.config(state="normal")
                self._sf_log.insert("end", f"[{ts_ui}] {msg}\n")
                self._sf_log.see("end")
                self._sf_log.config(state="disabled")
            except Exception:
                pass
        try:
            self.root.after(0, _do)
        except Exception:
            pass
        # Also append to logs/pipeline.log with an ISO timestamp.
        try:
            logdir = os.path.join(SCRIPT_DIR, "logs")
            os.makedirs(logdir, exist_ok=True)
            with open(os.path.join(logdir, "pipeline.log"), "a",
                      encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
        except Exception:
            pass

    def _sf_log_clear(self):
        try:
            self._sf_log.config(state="normal")
            self._sf_log.delete("1.0", "end")
            self._sf_log.config(state="disabled")
        except Exception:
            pass

    # ── stage → output-file preview ──────────────────────────────────────────

    def _sf_run_paths(self):
        """Expected output paths for the currently loaded audio file."""
        fp = self.filepath
        if not fp:
            return None
        stem   = os.path.splitext(os.path.basename(fp))[0]
        base   = getattr(self, "_last_run_base", "") or ""
        outdir = getattr(self, "_last_run_outdir", "") or ""
        if not base or os.path.basename(base) != stem:
            outdir = os.path.join(os.path.dirname(fp), stem)
            base   = os.path.join(outdir, stem)
        try:
            lang = self._get_tts_params()[4]
        except Exception:
            lang = getattr(self, "_last_run_language", "") or ""
        try:
            tts_audio    = os.path.join(outdir, _tts_output_name(lang, fp, "_tts"))
            synced_audio = os.path.join(outdir, _tts_output_name(lang, fp, "_synced"))
        except Exception:
            tts_audio = synced_audio = ""
        return {"outdir": outdir, "base": base,
                "tts_audio": tts_audio, "synced_audio": synced_audio}

    def _sf_stage_preview(self, tag):
        p = self._sf_run_paths()
        if not p:
            messagebox.showinfo("No audio", "Open an audio file first.")
            return
        base = p["base"]
        MAP = {
            "S1a": [("English SRT", base + ".srt"),
                    ("Analyzed transcript", base + "_analyzed.txt")],
            "S1b": [("Final script", base + "_FinalScript.txt"),
                    ("Translation step", base + "_TranslationStep.txt"),
                    ("Review step", base + "_ReviewStep.txt")],
            "S2":  [("TTS audio", p["tts_audio"]),
                    ("TTS chunk log",
                     os.path.splitext(p["tts_audio"])[0] + "_chunks.txt"
                     if p["tts_audio"] else "")],
            "S3a": [("Sync — English SRT", base + "_sync_en.srt")],
            "S3b": [("Sync — dubbed SRT", base + "_sync_te.srt")],
            "S3c": [("SRT mapping (Gemini)", base + "_sync_mapping.txt")],
            "S3d": [("Timing sync log", base + "_sync_log.txt"),
                    ("Synced SRT", base + "_sync_synced.srt")],
            "S3e": [("Synced audio", p["synced_audio"]),
                    ("Synced SRT", base + "_sync_synced.srt"),
                    ("Synced timestamps", base + "_sync_timestamps.txt")],
        }
        cands = [(lbl, pth) for lbl, pth in MAP.get(tag, [])
                 if pth and os.path.isfile(pth)]
        if not cands:
            self.status.config(
                text=f"[{tag}] No output file yet — run the pipeline first.")
            return
        if len(cands) == 1:
            self._sf_preview_file(cands[0][1])
            return
        menu = tk.Menu(self.root, tearoff=0)
        for lbl, pth in cands:
            menu.add_command(
                label=f"{lbl}  —  {os.path.basename(pth)}",
                command=lambda q=pth: self._sf_preview_file(q))
        try:
            menu.tk_popup(self.root.winfo_pointerx(),
                          self.root.winfo_pointery())
        finally:
            menu.grab_release()

    def _sf_preview_file(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg"):
            self._sf_preview_audio(path)
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            messagebox.showerror("Preview", f"Could not read file:\n{e}")
            return
        win = tk.Toplevel(self.root)
        win.title(os.path.basename(path))
        win.configure(bg=BG)
        win.geometry("900x620")
        win.transient(self.root)
        bar = tk.Frame(win, bg=PANEL, height=36)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text=path, bg=PANEL, fg=TEXT_FAINT,
                 font=(MONO_FONT, 8), anchor="w").pack(
            side="left", fill="x", expand=True, padx=10)
        self._btn(bar, " Open Externally ",
                  lambda: self._sf_open_external(path),
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT).pack(
            side="right", padx=8, pady=3)
        txt = scrolledtext.ScrolledText(
            win, wrap="word", bg=INPUT_BG, fg=INPUT_FG,
            font=(MONO_FONT, 10), relief="flat", bd=0)
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("1.0", content)
        txt.config(state="disabled")

    def _sf_preview_audio(self, path):
        win = tk.Toplevel(self.root)
        win.title(os.path.basename(path))
        win.configure(bg=BG)
        win.geometry("520x150")
        win.transient(self.root)
        info = tk.Label(win, text="Loading audio…", bg=BG, fg=TEXT_FAINT,
                        font=(MONO_FONT, 9), anchor="w", wraplength=490,
                        justify="left")
        info.pack(fill="x", padx=12, pady=(12, 6))
        state = {"y": None, "sr": 0}

        row = tk.Frame(win, bg=BG)
        row.pack(pady=6)

        def _play():
            if state["y"] is None:
                return
            try:
                sd.stop()
                sd.play(state["y"], samplerate=state["sr"])
            except Exception as e:
                info.config(text=f"Playback error: {e}")

        def _stop():
            try:
                sd.stop()
            except Exception:
                pass

        self._btn(row, "  ▶ Play  ", _play,
                  bg="#0f1d14", fg="#22c55e", abg="#1f4d2e").pack(
            side="left", padx=6)
        self._btn(row, "  ■ Stop  ", _stop,
                  bg="#1f1213", fg="#f87171", abg="#3a1414").pack(
            side="left", padx=6)
        self._btn(row, " Open Externally ",
                  lambda: self._sf_open_external(path),
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT).pack(side="left", padx=6)
        win.bind("<Destroy>", lambda e: _stop() if e.widget is win else None)

        def _load():
            try:
                y, sr = librosa.load(path, sr=None, mono=True)
                state["y"], state["sr"] = y.astype(np.float32), int(sr)
                self.root.after(0, lambda: info.config(
                    text=f"{os.path.basename(path)}\n"
                         f"{len(y) / max(sr, 1):.1f}s · {sr} Hz"))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: info.config(
                    text=f"Could not load audio: {err}"))
        threading.Thread(target=_load, daemon=True).start()

    def _sf_open_external(self, path):
        import subprocess
        try:
            if IS_WINDOWS:
                os.startfile(path)          # noqa — Windows only
            elif IS_MAC:
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as ex:
            messagebox.showerror("Open", str(ex))

    # ── compare view (English vs synced dub, SRT text on chunks) ─────────────

    def _sf_cmp_build(self, parent):
        bar = tk.Frame(parent, bg=PANEL2, height=40, bd=0,
                       highlightbackground=PANEL_BORDER, highlightthickness=1)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        self._btn(bar, " ▶ English ", lambda: self._sf_cmp_play("en"),
                  bg="#172554", fg="#3b82f6", abg="#1e3a8a").pack(
            side="left", padx=(10, 4), pady=6)
        self._btn(bar, " ▶ Synced ", lambda: self._sf_cmp_play("out"),
                  bg="#0f1d14", fg="#22c55e", abg="#1f4d2e").pack(
            side="left", padx=4, pady=6)
        self._btn(bar, " ■ Stop ", self._sf_cmp_stop,
                  bg="#1f1213", fg="#f87171", abg="#3a1414").pack(
            side="left", padx=4, pady=6)
        self._btn(bar, " ⟳ Reload ", self._sf_cmp_load,
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT).pack(
            side="left", padx=(10, 4), pady=6)
        tk.Frame(bar, bg=PANEL_BORDER, width=2, height=24).pack(
            side="left", padx=(8, 8), pady=8)
        self._btn(bar, " − ", lambda: self._sf_cmp_zoom_by(0.67),
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT).pack(
            side="left", padx=2, pady=6)
        self._btn(bar, " + ", lambda: self._sf_cmp_zoom_by(1.5),
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT).pack(
            side="left", padx=2, pady=6)
        self._btn(bar, " Fit ", self._sf_cmp_zoom_fit,
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT).pack(
            side="left", padx=2, pady=6)
        self._btn(bar, " ◀ ", lambda: self._sf_cmp_scroll_step(-1),
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT).pack(
            side="left", padx=(8, 2), pady=6)
        self._btn(bar, " ▶ ", lambda: self._sf_cmp_scroll_step(1),
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT).pack(
            side="left", padx=2, pady=6)
        self._sf_cmp_info = tk.Label(
            bar, text="Run the full pipeline — audio appears here. "
                      "Click = seek · double-click a chunk = play it · "
                      "drag a dub chunk = move it · scroll = pan · "
                      "ctrl+scroll = zoom",
            bg=PANEL2, fg=TEXT_FAINT, font=(MONO_FONT, 9), anchor="w")
        self._sf_cmp_info.pack(side="left", fill="x", expand=True, padx=10)
        self._sf_cmp_bar = bar

        # media-player state: per-track cursor + zoom/scroll window
        self._sf_cmp_cursor       = {"en": 0.0, "out": 0.0}
        self._sf_cmp_zoom         = 1.0
        self._sf_cmp_scroll       = 0.0
        self._sf_cmp_playing_side = None
        self._sf_cmp_play_from    = 0.0
        self._sf_cmp_play_t0      = 0.0

        # Chunk-regenerate row — appears when a dubbed chunk is selected.
        # Edit the dubbed text, then ⟳ regenerates ONLY that chunk and
        # splices it back into the synced audio (timeline preserved).
        edit_row = tk.Frame(parent, bg=PANEL3, bd=0,
                            highlightbackground=PANEL_BORDER,
                            highlightthickness=1)
        self._sf_cmp_edit_row = edit_row      # packed on demand
        tk.Label(edit_row, text="Dub chunk:", bg=PANEL3, fg="#f59e0b",
                 font=(MONO_FONT, 9, "bold")).pack(side="left",
                                                   padx=(10, 6), pady=6)
        self._sf_cmp_edit = tk.Text(
            edit_row, height=2, wrap="word", bg=INPUT_BG, fg=INPUT_FG,
            insertbackground=INPUT_FG, font=(MONO_FONT, 10),
            relief="flat", bd=0)
        self._sf_cmp_edit.pack(side="left", fill="x", expand=True, pady=4)
        self._sf_cmp_regen_btn = self._btn(
            edit_row, " ⟳ Regenerate Chunk ", self._sf_cmp_regen_chunk,
            bg="#2a1a0a", fg="#f59e0b", abg="#4d2f10")
        self._sf_cmp_regen_btn.pack(side="right", padx=(4, 8), pady=4)
        self._btn(edit_row, " ▶ Chunk ", self._sf_cmp_play_sel,
                  bg="#0f1d14", fg="#22c55e", abg="#1f4d2e").pack(
            side="right", padx=4, pady=4)
        self._sf_cmp_sel = None               # selected dub-chunk index

        self._sf_cmp_fig, (self._sf_cmp_ax_top, self._sf_cmp_ax_bot) = \
            plt.subplots(2, 1, figsize=(12, 3.6))
        self._sf_cmp_fig.patch.set_facecolor(BG)

        # Horizontal scrollbar (custom canvas thumb, like the Waveform view).
        self._sf_cmp_sb = tk.Canvas(parent, height=14, bg="#1e293b",
                                    highlightthickness=0, cursor="hand2")
        self._sf_cmp_sb.pack(fill="x", side="bottom", pady=(1, 2))
        self._sf_cmp_sb.bind("<Configure>",       self._sf_cmp_sb_redraw)
        self._sf_cmp_sb.bind("<ButtonPress-1>",   self._sf_cmp_sb_press)
        self._sf_cmp_sb.bind("<B1-Motion>",       self._sf_cmp_sb_drag)
        self._sf_cmp_sb.bind("<ButtonRelease-1>", self._sf_cmp_sb_release)
        self._sf_cmp_sb_drag_x   = None
        self._sf_cmp_sb_drag_pos = 0.0

        self._sf_cmp_canvas = FigureCanvasTkAgg(self._sf_cmp_fig,
                                                master=parent)
        w = self._sf_cmp_canvas.get_tk_widget()
        w.configure(bg=BG, cursor="hand2")
        w.pack(fill="both", expand=True)
        self._sf_cmp_canvas.mpl_connect("button_press_event",
                                        self._sf_cmp_on_press)
        self._sf_cmp_canvas.mpl_connect("motion_notify_event",
                                        self._sf_cmp_on_motion)
        self._sf_cmp_canvas.mpl_connect("button_release_event",
                                        self._sf_cmp_on_release)
        self._sf_cmp_canvas.mpl_connect("scroll_event",
                                        self._sf_cmp_on_scroll)
        self._sf_cmp_press = None      # pending click / chunk-drag state
        self._sf_cmp_data = {}
        self._sf_cmp_draw()

    def _sf_cmp_load(self, keep_sel=None):
        """Load English + synced audio and their SRTs in a worker thread.
        keep_sel: dub-chunk index to keep selected after the reload."""
        p = self._sf_run_paths()
        if not p or not self.filepath:
            self._sf_cmp_info.config(text="Open an audio file first.")
            return
        en_audio  = self.filepath
        out_audio = p["synced_audio"]
        base      = p["base"]
        en_srt_p  = next((q for q in (base + "_sync_en.srt", base + ".srt")
                          if os.path.isfile(q)), None)
        out_srt_p = next((q for q in (base + "_sync_synced.srt",
                                      base + "_sync_te.srt")
                          if os.path.isfile(q)), None)
        self._sf_cmp_info.config(text="Loading audio + SRT…")

        def _env_of(y, bins=3000):
            n = len(y)
            if n == 0:
                return np.zeros(1, dtype=np.float32)
            if n <= bins:
                env = np.abs(y)
            else:
                step   = n // bins
                frames = n // step
                env = np.abs(y[:frames * step]).reshape(frames, step).max(axis=1)
            peak = float(env.max())
            return env / peak if peak > 0 else env

        def _subs_of(srt_path):
            """→ [(start_s, end_s, text)] — Subtitle stores milliseconds."""
            if not srt_path:
                return []
            try:
                with open(srt_path, "r", encoding="utf-8",
                          errors="replace") as f:
                    d = _parse_srt_from_string(f.read())
                return [(s.start / 1000.0, s.end / 1000.0, str(s.text))
                        for s in sorted(d.values(), key=lambda s: s.start)]
            except Exception:
                return []

        def _load():
            data = {}
            for side, apath, spath in (("en", en_audio, en_srt_p),
                                       ("out", out_audio, out_srt_p)):
                if not apath or not os.path.isfile(apath):
                    continue
                try:
                    y, sr = librosa.load(apath, sr=None, mono=True)
                except Exception:
                    continue
                y = y.astype(np.float32)
                data[side] = {
                    "y": y, "sr": int(sr),
                    "dur": len(y) / max(sr, 1),
                    "env": _env_of(y),
                    "subs": _subs_of(spath),
                    "title": os.path.basename(apath),
                }

            def _done():
                self._sf_cmp_data = data
                out_subs = data.get("out", {}).get("subs", [])
                if keep_sel is not None and keep_sel < len(out_subs):
                    self._sf_cmp_sel = keep_sel
                    self._sf_cmp_fill_edit(keep_sel)
                else:
                    self._sf_cmp_sel = None
                    self._sf_cmp_edit_row.pack_forget()
                if not data:
                    self._sf_cmp_info.config(
                        text="No audio found yet — run the full pipeline first.")
                elif "out" not in data:
                    self._sf_cmp_info.config(
                        text="Synced audio not found — run the FULL pipeline "
                             "(Settings → Run pipeline up to: Full Pipeline).")
                else:
                    n_en  = len(data.get("en", {}).get("subs", []))
                    n_out = len(data["out"].get("subs", []))
                    self._sf_cmp_info.config(
                        text=f"English: {n_en} chunks · Synced: {n_out} chunks "
                             "· click any chunk to hear it")
                self._sf_cmp_draw()
            self.root.after(0, _done)
        threading.Thread(target=_load, daemon=True).start()

    def _sf_cmp_view_window(self):
        """Visible time window (x0, x1, total) from zoom + scroll state."""
        total = max([d["dur"] for d in self._sf_cmp_data.values()] or [1.0])
        zoom  = max(1.0, self._sf_cmp_zoom)
        vis   = total / zoom
        x0    = self._sf_cmp_scroll * max(0.0, total - vis)
        return x0, x0 + vis, total

    @staticmethod
    def _sf_cmp_fmt(t):
        m, s = divmod(max(t, 0.0), 60)
        return f"{int(m)}:{s:04.1f}"

    def _sf_cmp_draw(self):
        fam = _indic_matplotlib_font()
        x0, x1, total = self._sf_cmp_view_window()
        vis = max(x1 - x0, 1e-6)
        try:
            px_w = max(
                self._sf_cmp_canvas.get_tk_widget().winfo_width() - 90, 200)
        except Exception:
            px_w = 1000
        px_per_s = px_w / vis
        specs = (
            (self._sf_cmp_ax_top, "en",  "#3b82f6", "English (original)"),
            (self._sf_cmp_ax_bot, "out", TR_ACCENT, "Synced dub"),
        )
        for ax, side, color, name in specs:
            ax.clear()
            ax.set_facecolor("#0f172a")
            ax.tick_params(colors=TEXT, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID)
            d = self._sf_cmp_data.get(side)
            if not d:
                ax.text(0.5, 0.5, f"{name} — not loaded",
                        ha="center", va="center", transform=ax.transAxes,
                        color=TEXT_MUTED, fontsize=9, fontfamily="monospace")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # envelope — draw only the visible slice
            env, dur = d["env"], d["dur"]
            n = len(env)
            if dur > 0 and n > 1:
                i0 = max(0, int(x0 / dur * n))
                i1 = min(n, int(min(x1, dur) / dur * n) + 1)
                if i1 > i0:
                    t = np.linspace(i0 / n * dur, i1 / n * dur, i1 - i0)
                    ax.fill_between(t, -env[i0:i1], env[i0:i1], color=color,
                                    alpha=0.75, linewidth=0, zorder=2)

            # subtitle chunks — visible only; label only when it fits.
            # Rows staggered (odd/even) so neighbouring labels don't collide.
            for k, (s0, s1, stext) in enumerate(d["subs"]):
                if s1 < x0 or s0 > x1:
                    continue
                sel = (side == "out" and self._sf_cmp_sel == k)
                ax.axvspan(s0, s1,
                           color="#f59e0b" if sel else
                           ("#facc15" if k % 2 else "#a78bfa"),
                           alpha=0.30 if sel else 0.10, zorder=1)
                ax.axvline(s0, color="#f59e0b" if sel else "#64748b",
                           linewidth=1.2 if sel else 0.5,
                           alpha=0.9 if sel else 0.6, zorder=3)
                chunk_px = (s1 - s0) * px_per_s
                if not sel and chunk_px < 30:
                    continue          # too narrow — no readable label
                max_chars = max(int(chunk_px / 7.5), 4)
                label = " ".join(stext.split())
                if len(label) > max_chars:
                    label = label[:max(max_chars - 1, 1)] + "…"
                cx = min(max((s0 + s1) / 2.0, x0), x1)
                kw = {"fontsize": 7, "ha": "center", "va": "top",
                      "color": "#f8fafc" if sel else "#cbd5e1",
                      "clip_on": True, "zorder": 4,
                      "transform": ax.get_xaxis_transform()}
                if fam:
                    kw["fontfamily"] = fam
                ax.text(cx, 0.99 if k % 2 == 0 else 0.84, label, **kw)

            # playhead cursor
            cur = self._sf_cmp_cursor.get(side, 0.0)
            if x0 <= cur <= x1:
                ax.axvline(cur, color=CURSOR_C, linewidth=1.5,
                           alpha=0.95, zorder=6)

            ax.set_ylim(-1.05, 1.05)
            ax.set_xlim(x0, x1)
            ax.set_yticks([])
            ax.grid(True, axis="x", color=GRID, linewidth=0.4,
                    linestyle="--", alpha=0.5)
            playing = "▶ " if self._sf_cmp_playing_side == side else ""
            ax.set_title(
                f"{playing}{name} · {d['title']} · "
                f"{self._sf_cmp_fmt(cur)} / {self._sf_cmp_fmt(dur)}",
                color=color, fontsize=8, pad=3,
                fontfamily="monospace", loc="left")

        def fmt_time(x, _):
            m, s2 = divmod(x, 60)
            return f"{int(m)}:{s2:04.1f}" if m else f"{s2:.1f}s"
        self._sf_cmp_ax_bot.xaxis.set_major_formatter(
            ticker.FuncFormatter(fmt_time))
        self._sf_cmp_ax_top.set_xticklabels([])
        try:
            self._sf_cmp_fig.tight_layout(pad=0.5)
        except Exception:
            pass
        self._sf_cmp_canvas.draw()
        self._sf_cmp_sb_redraw()

    # ── zoom / scroll ────────────────────────────────────────────────────────

    def _sf_cmp_zoom_by(self, factor, center_frac=0.5):
        if not self._sf_cmp_data:
            return
        x0, x1, total = self._sf_cmp_view_window()
        cx = x0 + (x1 - x0) * center_frac
        self._sf_cmp_zoom = min(max(self._sf_cmp_zoom * factor, 1.0), 200.0)
        vis = total / self._sf_cmp_zoom
        if total > vis:
            new_x0 = min(max(cx - vis * center_frac, 0.0), total - vis)
            self._sf_cmp_scroll = new_x0 / (total - vis)
        else:
            self._sf_cmp_scroll = 0.0
        self._sf_cmp_draw()

    def _sf_cmp_zoom_fit(self):
        self._sf_cmp_zoom = 1.0
        self._sf_cmp_scroll = 0.0
        if self._sf_cmp_data:
            self._sf_cmp_draw()

    def _sf_cmp_on_scroll(self, event):
        """Scroll = pan; ctrl+scroll = zoom around the pointer."""
        if not self._sf_cmp_data:
            return
        x0, x1, total = self._sf_cmp_view_window()
        if event.key == "control":
            frac = 0.5
            if event.xdata is not None and x1 > x0:
                frac = min(max((event.xdata - x0) / (x1 - x0), 0.0), 1.0)
            self._sf_cmp_zoom_by(1.25 if event.button == "up" else 0.8, frac)
            return
        vis = x1 - x0
        if total <= vis:
            return
        step = vis * 0.15 * (-1 if event.button == "up" else 1)
        new_x0 = min(max(x0 + step, 0.0), total - vis)
        self._sf_cmp_scroll = new_x0 / (total - vis)
        self._sf_cmp_draw()

    def _sf_cmp_set_scroll(self, frac):
        self._sf_cmp_scroll = min(max(frac, 0.0), 1.0)
        self._sf_cmp_draw()

    def _sf_cmp_scroll_step(self, direction):
        """◀ / ▶ buttons: pan by half a screen."""
        if not self._sf_cmp_data:
            return
        x0, x1, total = self._sf_cmp_view_window()
        vis = x1 - x0
        if total <= vis:
            return
        new_x0 = min(max(x0 + direction * vis * 0.5, 0.0), total - vis)
        self._sf_cmp_set_scroll(new_x0 / (total - vis))

    # ── compare-view scrollbar (canvas thumb) ────────────────────────────────

    def _sf_cmp_sb_geom(self):
        w = self._sf_cmp_sb.winfo_width()
        x0, x1, total = self._sf_cmp_view_window()
        vis   = max(x1 - x0, 1e-6)
        ratio = min(vis / max(total, 1e-6), 1.0)
        thumb_w    = max(24, int(w * ratio))
        scrollable = max(1, w - thumb_w)
        tx0 = int(self._sf_cmp_scroll * scrollable)
        return tx0, tx0 + thumb_w, scrollable, thumb_w

    def _sf_cmp_sb_redraw(self, _event=None):
        c = getattr(self, "_sf_cmp_sb", None)
        if not c:
            return
        w, h = c.winfo_width(), c.winfo_height()
        if w < 2:
            return
        c.delete("all")
        c.create_rectangle(0, 0, w, h, fill="#1e293b", outline="")
        tx0, tx1, _, _ = self._sf_cmp_sb_geom()
        c.create_rectangle(tx0 + 1, 2, tx1 - 1, h - 2,
                           fill="#475569", outline="#64748b")

    def _sf_cmp_sb_press(self, event):
        if not self._sf_cmp_data:
            return
        tx0, tx1, scrollable, thumb_w = self._sf_cmp_sb_geom()
        if tx0 <= event.x <= tx1:
            self._sf_cmp_sb_drag_x   = event.x
            self._sf_cmp_sb_drag_pos = self._sf_cmp_scroll
        else:
            pos = min(max((event.x - thumb_w / 2) / scrollable, 0.0), 1.0)
            self._sf_cmp_sb_drag_x   = event.x
            self._sf_cmp_sb_drag_pos = pos
            self._sf_cmp_set_scroll(pos)

    def _sf_cmp_sb_drag(self, event):
        if self._sf_cmp_sb_drag_x is None or not self._sf_cmp_data:
            return
        _, _, scrollable, _ = self._sf_cmp_sb_geom()
        dx = event.x - self._sf_cmp_sb_drag_x
        self._sf_cmp_set_scroll(self._sf_cmp_sb_drag_pos + dx / scrollable)

    def _sf_cmp_sb_release(self, _event=None):
        self._sf_cmp_sb_drag_x = None

    # ── compare-view mouse: click = seek, drag dub chunk = move (Reaper) ─────

    def _sf_cmp_on_press(self, event):
        """Press: double-click plays the chunk; single press arms either a
        click-seek (handled on release) or a chunk drag (handled on motion)."""
        if event.xdata is None:
            return
        side = ("en" if event.inaxes is self._sf_cmp_ax_top else
                "out" if event.inaxes is self._sf_cmp_ax_bot else None)
        d = self._sf_cmp_data.get(side) if side else None
        if not d:
            return
        t = float(event.xdata)
        hit_idx = next((k for k, (a0, a1, _t) in enumerate(d["subs"])
                        if a0 <= t <= a1), None)

        if getattr(event, "dblclick", False):
            self._sf_cmp_press = None
            if hit_idx is None:
                return
            s0, s1, stext = d["subs"][hit_idx]
            try:
                sd.stop()
                self._sf_cmp_playing_side = None
                sd.play(d["y"][int(s0 * d["sr"]):int(s1 * d["sr"])],
                        samplerate=d["sr"])
            except Exception:
                pass
            label = " ".join(stext.split())
            self._sf_cmp_info.config(
                text=f"[{'EN' if side == 'en' else 'DUB'} chunk "
                     f"{s0:.1f}–{s1:.1f}s]  {label}")
            return

        self._sf_cmp_press = {
            "side": side, "x": t, "idx": hit_idx, "moved": False,
            "orig": d["subs"][hit_idx] if hit_idx is not None else None,
        }

    def _sf_cmp_on_motion(self, event):
        """Drag a dub chunk left/right (Reaper-style item move)."""
        pr = self._sf_cmp_press
        if (not pr or pr["side"] != "out" or pr["idx"] is None
                or event.xdata is None):
            return
        d = self._sf_cmp_data.get("out")
        if not d:
            return
        dx = float(event.xdata) - pr["x"]
        x0, x1, _tot = self._sf_cmp_view_window()
        vis = max(x1 - x0, 1e-6)
        try:
            px_w = max(
                self._sf_cmp_canvas.get_tk_widget().winfo_width() - 90, 200)
        except Exception:
            px_w = 1000
        if not pr["moved"] and abs(dx) * (px_w / vis) < 5:
            return                      # ignore tiny jitter — still a click
        pr["moved"] = True
        s0, s1, stext = pr["orig"]
        ns0 = max(0.0, s0 + dx)
        d["subs"][pr["idx"]] = (ns0, ns0 + (s1 - s0), stext)
        self._sf_cmp_sel = pr["idx"]
        self._sf_cmp_info.config(
            text=f"Moving chunk {pr['idx'] + 1}: {s0:.2f}s → {ns0:.2f}s "
                 f"({'+' if ns0 >= s0 else ''}{ns0 - s0:.2f}s) — release to apply")
        self._sf_cmp_draw()

    def _sf_cmp_on_release(self, event):
        pr = self._sf_cmp_press
        self._sf_cmp_press = None
        if not pr:
            return
        side = pr["side"]
        d = self._sf_cmp_data.get(side)
        if not d:
            return

        if pr["moved"]:
            # Chunk drag finished → rebuild the synced audio at the new spot.
            s0_old = pr["orig"][0]
            s0_new = d["subs"][pr["idx"]][0]
            if abs(s0_new - s0_old) < 0.005:
                d["subs"][pr["idx"]] = pr["orig"]
                self._sf_cmp_draw()
                return
            self._sf_cmp_apply_move(pr["idx"], pr["orig"], s0_new)
            return

        # Plain click → seek that track's playhead (media-player style).
        t = pr["x"]
        hit_idx = pr["idx"]
        was_playing = (self._sf_cmp_playing_side == side)
        self._sf_cmp_cursor[side] = min(max(t, 0.0), d["dur"])
        if hit_idx is not None:
            s0, s1, stext = d["subs"][hit_idx]
            label = " ".join(stext.split())
            self._sf_cmp_info.config(
                text=f"[{'EN' if side == 'en' else 'DUB'} "
                     f"{s0:.1f}–{s1:.1f}s]  {label}")
            if side == "out":
                self._sf_cmp_sel = hit_idx
                self._sf_cmp_fill_edit(hit_idx)
        else:
            self._sf_cmp_info.config(
                text=f"{'English' if side == 'en' else 'Dub'} "
                     f"cursor → {self._sf_cmp_fmt(t)}")
        if was_playing:
            self._sf_cmp_play(side)      # live seek: restart from new cursor
        else:
            self._sf_cmp_draw()

    def _sf_cmp_apply_move(self, idx, orig_sub, new_s0):
        """Persist a dragged dub chunk: shift its entry in the sync
        timestamps, rebuild the synced audio from the TTS source, and update
        the synced SRT. Falls back to slicing the synced WAV directly when
        the timestamps/TTS files are missing (old projects)."""
        p = self._sf_run_paths()
        if not p:
            return
        synced_path = p["synced_audio"]
        tts_path    = p["tts_audio"]
        ts_path     = p["base"] + "_sync_timestamps.txt"
        srt_path    = p["base"] + "_sync_synced.srt"
        if not synced_path or not os.path.isfile(synced_path):
            messagebox.showerror("Move chunk", "Synced audio file not found.")
            self._sf_cmp_load(keep_sel=idx)
            return
        if not PYDUB_AVAILABLE:
            messagebox.showerror("pydub missing",
                                 "pydub is required to move chunks.")
            self._sf_cmp_load(keep_sel=idx)
            return
        old_s0, old_s1, _txt = orig_sub
        delta_ms = int(round((new_s0 - old_s0) * 1000))
        self._sf_cmp_info.config(
            text=f"Applying move of chunk {idx + 1} "
                 f"({'+' if delta_ms >= 0 else ''}{delta_ms / 1000.0:.2f}s)…")
        self._sf_log_append(
            f"[Compare] Moving dub chunk {idx + 1}: {old_s0:.2f}s → "
            f"{new_s0:.2f}s ({'+' if delta_ms >= 0 else ''}{delta_ms} ms)")

        def _worker():
            try:
                # One-time backup of the untouched synced audio.
                bak = os.path.splitext(synced_path)[0] + "_backup.wav"
                if not os.path.exists(bak):
                    shutil.copy2(synced_path, bak)

                entries = _parse_timestamps_text(ts_path)
                if entries and tts_path and os.path.isfile(tts_path) \
                        and len(entries) > idx:
                    # Preferred path: shift the timestamp entry and rebuild
                    # the whole synced file from the clean TTS source.
                    by_start = sorted(entries,
                                      key=lambda e: e["synced_start_ms"])
                    ent = by_start[idx]
                    ent["synced_start_ms"] = max(
                        0, ent["synced_start_ms"] + delta_ms)
                    with open(ts_path, "w", encoding="utf-8") as f:
                        f.write(_format_timestamps_as_text(
                            sorted(entries, key=lambda e: e["index"])))
                    sync_audio_with_timestamps(tts_path, entries, synced_path)
                else:
                    # Fallback: splice the chunk inside the synced WAV itself.
                    synced = _AudioSegment.from_file(synced_path)
                    a0 = max(0, int(old_s0 * 1000))
                    a1 = min(len(synced), int(old_s1 * 1000))
                    seg = synced[a0:a1]
                    gap = _AudioSegment.silent(
                        duration=a1 - a0, frame_rate=synced.frame_rate)
                    canvas = synced[:a0] + gap + synced[a1:]
                    npos = max(0, a0 + delta_ms)
                    if npos + len(seg) > len(canvas):
                        canvas += _AudioSegment.silent(
                            duration=npos + len(seg) - len(canvas),
                            frame_rate=canvas.frame_rate)
                    canvas = canvas.overlay(seg, position=npos)
                    canvas.export(synced_path, format="wav")

                # Shift the matching cue in the synced SRT.
                try:
                    with open(srt_path, "r", encoding="utf-8",
                              errors="replace") as f:
                        srt_subs = _parse_srt_from_string(f.read())
                    ordered = sorted(srt_subs.values(), key=lambda s: s.start)
                    if idx < len(ordered):
                        ordered[idx].shift(delta_ms)
                        if ordered[idx].start < 0:
                            ordered[idx].start = 0
                        with open(srt_path, "w", encoding="utf-8") as f:
                            f.write(_write_srt_from_dict(srt_subs))
                except Exception:
                    pass

                self._sf_log_append(
                    f"[Compare] ✔ chunk {idx + 1} moved → "
                    f"{os.path.basename(synced_path)}")
                self.root.after(0, lambda: self._sf_cmp_load(keep_sel=idx))
            except Exception as exc:
                err = str(exc)

                def _fail(e=err):
                    self._sf_cmp_info.config(text=f"Move error: {e}")
                    messagebox.showerror("Move chunk", e)
                    self._sf_cmp_load(keep_sel=idx)
                self.root.after(0, _fail)

        threading.Thread(target=_worker, daemon=True).start()

    def _sf_cmp_fill_edit(self, idx):
        """Put chunk idx's dubbed text into the edit row and show the row."""
        d = self._sf_cmp_data.get("out")
        if not d or idx >= len(d["subs"]):
            return
        self._sf_cmp_edit.delete("1.0", "end")
        self._sf_cmp_edit.insert("1.0", d["subs"][idx][2])
        if not self._sf_cmp_edit_row.winfo_ismapped():
            self._sf_cmp_edit_row.pack(fill="x", after=self._sf_cmp_bar)

    def _sf_cmp_play_sel(self):
        i = self._sf_cmp_sel
        d = self._sf_cmp_data.get("out")
        if i is None or not d or i >= len(d["subs"]):
            return
        s0, s1, _tx = d["subs"][i]
        sr = d["sr"]
        try:
            sd.stop()
            sd.play(d["y"][int(s0 * sr):int(s1 * sr)], samplerate=sr)
        except Exception:
            pass

    def _sf_cmp_regen_chunk(self):
        """Re-synthesize ONLY the selected dubbed chunk and splice it back
        into the synced audio. The chunk's window (its start → next chunk's
        start) is replaced; everything else keeps its exact timing."""
        i = self._sf_cmp_sel
        d = self._sf_cmp_data.get("out")
        if i is None or not d or i >= len(d["subs"]):
            messagebox.showinfo(
                "No chunk", "Click a chunk on the synced dub waveform first.")
            return
        if not PYDUB_AVAILABLE:
            messagebox.showerror(
                "pydub missing", "pydub is required for chunk regeneration.")
            return
        new_text = self._sf_cmp_edit.get("1.0", "end").strip()
        if not new_text:
            messagebox.showwarning(
                "Empty text", "Chunk text is empty — type the corrected "
                              "sentence first.")
            return
        p = self._sf_run_paths()
        synced_path = p["synced_audio"] if p else ""
        if not synced_path or not os.path.isfile(synced_path):
            messagebox.showerror(
                "Not found",
                "Synced audio file not found — run the full pipeline first.")
            return
        try:
            api_key = _get_api_key()
            (tts_platform, tts_lang_code, tts_voice_name, el_voice_id,
             _lang, el_model) = self._get_tts_params()
        except Exception as e:
            messagebox.showerror("TTS Settings", str(e))
            return
        params = {"platform": tts_platform, "api_key": api_key,
                  "voice_id": el_voice_id, "model_id": el_model,
                  "lang_code": tts_lang_code, "voice_name": tts_voice_name}

        subs = list(d["subs"])
        s0, s1, _old = subs[i]
        w0_ms = int(s0 * 1000)
        w1_ms = (int(subs[i + 1][0] * 1000) if i + 1 < len(subs)
                 else int(d["dur"] * 1000))
        srt_path = p["base"] + "_sync_synced.srt"

        self._sf_cmp_regen_btn.config(state="disabled")
        self._sf_cmp_info.config(text=f"Regenerating chunk {i + 1}…")
        self._sf_log_append(
            f"[Compare] Regenerating dub chunk {i + 1} ({s0:.1f}–{s1:.1f}s)…")

        def _worker():
            try:
                prev_text = " ".join(t for _, _, t in subs[max(0, i - 2):i])
                next_text = " ".join(t for _, _, t in subs[i + 1:i + 3])
                tmp_wav = os.path.join(os.path.dirname(synced_path),
                                       "_chunk_regen_tmp.wav")
                self._tts_synth_segment_file(new_text, tmp_wav, params,
                                             prev_text, next_text)
                new_seg = _AudioSegment.from_file(tmp_wav)
                spoken_ms = len(new_seg)

                synced = _AudioSegment.from_file(synced_path)
                # One-time backup of the untouched synced audio.
                bak = os.path.splitext(synced_path)[0] + "_backup.wav"
                if not os.path.exists(bak):
                    shutil.copy2(synced_path, bak)

                win_len = max(w1_ms - w0_ms, 1)
                clipped = spoken_ms > win_len
                if clipped:
                    new_seg = new_seg[:win_len]
                else:
                    new_seg = new_seg + _AudioSegment.silent(
                        duration=win_len - spoken_ms,
                        frame_rate=new_seg.frame_rate)
                combined = synced[:w0_ms] + new_seg + synced[w1_ms:]
                combined.export(synced_path, format="wav")
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass

                # Update the synced SRT: new text + new spoken duration.
                try:
                    with open(srt_path, "r", encoding="utf-8",
                              errors="replace") as f:
                        srt_subs = _parse_srt_from_string(f.read())
                    ordered = sorted(srt_subs.values(), key=lambda s: s.start)
                    if i < len(ordered):
                        sub = ordered[i]
                        sub.text = new_text
                        sub._dur = max(min(spoken_ms, win_len), 1)
                        with open(srt_path, "w", encoding="utf-8") as f:
                            f.write(_write_srt_from_dict(srt_subs))
                except Exception:
                    pass

                if clipped:
                    self._sf_log_append(
                        f"[Compare] ⚠ chunk {i + 1}: new audio ({spoken_ms} ms) "
                        f"longer than its window ({win_len} ms) — clipped so "
                        "the timeline stays aligned")
                self._sf_log_append(
                    f"[Compare] ✔ chunk {i + 1} regenerated & merged → "
                    f"{os.path.basename(synced_path)}")

                def _done():
                    self.status.config(
                        text=f"Chunk {i + 1} regenerated → "
                             f"{os.path.basename(synced_path)}")
                    self._sf_cmp_load(keep_sel=i)
                self.root.after(0, _done)
            except Exception as exc:
                err = str(exc)

                def _fail(e=err):
                    self._sf_cmp_info.config(text=f"Regenerate error: {e}")
                    messagebox.showerror("Regenerate Chunk", e)
                self.root.after(0, _fail)
            finally:
                self.root.after(0, lambda: self._sf_cmp_regen_btn.config(
                    state="normal"))

        threading.Thread(target=_worker, daemon=True).start()

    def _sf_cmp_play(self, side):
        """Play a track from its playhead cursor (media-player style)."""
        d = self._sf_cmp_data.get(side)
        if not d:
            self._sf_cmp_info.config(text="Nothing loaded — hit ⟳ Reload.")
            return
        cur = self._sf_cmp_cursor.get(side, 0.0)
        if cur >= d["dur"] - 0.05:
            cur = 0.0
        try:
            sd.stop()
            sd.play(d["y"][int(cur * d["sr"]):], samplerate=d["sr"])
        except Exception as e:
            self._sf_cmp_info.config(text=f"Playback error: {e}")
            return
        self._sf_cmp_cursor[side]  = cur
        self._sf_cmp_playing_side  = side
        self._sf_cmp_play_from     = cur
        self._sf_cmp_play_t0       = time.perf_counter()
        self._sf_cmp_info.config(
            text=f"Playing {'English' if side == 'en' else 'synced dub'} "
                 f"from {self._sf_cmp_fmt(cur)} — click waveform to seek, "
                 "■ Stop to pause")
        self._sf_cmp_tick()

    def _sf_cmp_tick(self):
        side = self._sf_cmp_playing_side
        if not side:
            return
        d = self._sf_cmp_data.get(side)
        if not d:
            self._sf_cmp_playing_side = None
            return
        pos = self._sf_cmp_play_from + (time.perf_counter()
                                        - self._sf_cmp_play_t0)
        if pos >= d["dur"]:
            self._sf_cmp_playing_side = None
            self._sf_cmp_cursor[side] = 0.0
            self._sf_cmp_draw()
            return
        self._sf_cmp_cursor[side] = pos
        # auto-follow the playhead when zoomed in
        x0, x1, total = self._sf_cmp_view_window()
        vis = x1 - x0
        if pos > x1 and total > vis:
            new_x0 = min(max(pos - vis * 0.1, 0.0), total - vis)
            self._sf_cmp_scroll = new_x0 / (total - vis)
        self._sf_cmp_draw()
        self.root.after(150, self._sf_cmp_tick)

    def _sf_cmp_stop(self):
        """Pause: the playhead keeps its position — ▶ resumes from there."""
        side = self._sf_cmp_playing_side
        if side:
            pos = self._sf_cmp_play_from + (time.perf_counter()
                                            - self._sf_cmp_play_t0)
            d = self._sf_cmp_data.get(side)
            if d:
                self._sf_cmp_cursor[side] = min(pos, d["dur"])
        self._sf_cmp_playing_side = None
        try:
            sd.stop()
        except Exception:
            pass
        if self._sf_cmp_data:
            self._sf_cmp_draw()

    # ── Batch Tab ─────────────────────────────────────────────────────────────
    def _build_batch_tab(self):
        tab = self.batch_tab
        ctrl = tk.Frame(tab, bg=PANEL, height=54, bd=0,
                        highlightbackground=PANEL_BORDER, highlightthickness=1)
        ctrl.pack(fill="x", side="top")
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="BATCH PROCESS", bg=PANEL, fg=TR_ACCENT,
                 font=(MONO_FONT, 10, "bold")).pack(side="left", padx=(14, 10), pady=14)
        tk.Frame(ctrl, bg="#334155", width=2, height=28).pack(side="left", padx=8, pady=12)

        self._btn(ctrl, "Select Folder", self._batch_pick_folder,
                  bg="#172554", fg="#3b82f6", abg="#1e3a8a").pack(side="left", padx=(0, 8), pady=10)
        self.batch_folder_label = tk.Label(ctrl, text="No folder selected",
                                           bg=PANEL, fg=TEXT_FAINT, font=(MONO_FONT, 10))
        self.batch_folder_label.pack(side="left", padx=10)

        self.batch_stop_btn = self._btn(ctrl, "Stop Batch", self._batch_stop,
                                        bg="#1f1213", fg="#f87171", abg="#3a1414")
        self.batch_stop_btn.pack(side="right", padx=14, pady=10)
        self.batch_stop_btn.config(state="disabled")

        # ── Step selector panel ───────────────────────────────────────────────
        step_bar = tk.Frame(tab, bg=PANEL2, bd=0,
                            highlightbackground=PANEL_BORDER, highlightthickness=1)
        step_bar.pack(fill="x")

        tk.Label(step_bar, text="Run pipeline up to:",
                 bg=PANEL2, fg=TEXT_FAINT, font=(MONO_FONT, 9, "bold")
                 ).pack(side="left", padx=(14, 10), pady=8)

        _step_options = [
            ("SRT only",      "English SRT"),   # step 1a — ElevenLabs transcription → SRT
            ("+ Translation", "Translation"),   # step 1b — Gemini pipeline → FinalScript
            ("+ TTS Audio",   "TTS Audio"),     # step 2  — TTS synthesis
            ("Full Pipeline", "Full Pipeline"), # step 3  — audio sync
        ]
        for label, value in _step_options:
            is_default = (value == "Full Pipeline")
            rb = tk.Radiobutton(
                step_bar, text=label,
                variable=self._batch_stop_step, value=value,
                bg=PANEL2, fg=ACCENT if is_default else TEXT,
                selectcolor=PANEL,
                activebackground=PANEL2, activeforeground=ACCENT,
                font=(MONO_FONT, 9, "bold" if is_default else "normal"),
                cursor="hand2",
            )
            rb.pack(side="left", padx=(0, 18), pady=6)

        info_bar = tk.Frame(tab, bg=PANEL2, height=32, bd=0,
                            highlightbackground=PANEL_BORDER, highlightthickness=1)
        info_bar.pack(fill="x")
        info_bar.pack_propagate(False)
        self.batch_progress_label = tk.Label(
            info_bar, text="", bg=PANEL2, fg=ACCENT, font=(MONO_FONT, 9), anchor="w")
        self.batch_progress_label.pack(side="left", fill="x", expand=True, padx=12, pady=6)

        list_frame = tk.Frame(tab, bg=BG)
        list_frame.pack(fill="both", expand=True, padx=10, pady=8)

        cols = ("File", "Translation", "TTS", "Sync", "Output")
        self.batch_tree = ttk.Treeview(list_frame, columns=cols, show="headings",
                                       selectmode="browse")
        style = ttk.Style()
        style.configure("Batch.Treeview", background=PANEL, foreground=TEXT,
                        fieldbackground=PANEL, rowheight=26, font=(MONO_FONT, 9))
        style.configure("Batch.Treeview.Heading", background=PANEL2, foreground=REG_EDGE,
                        font=(MONO_FONT, 9, "bold"))
        style.map("Batch.Treeview", background=[("selected", "#1e3a8a")],
                  foreground=[("selected", "#f8fafc")])
        self.batch_tree.configure(style="Batch.Treeview")

        for col, width in [("File", 220), ("Translation", 160), ("TTS", 120),
                            ("Sync", 120), ("Output", 300)]:
            self.batch_tree.heading(col, text=col)
            self.batch_tree.column(col, width=width, anchor="w")

        vsb = ttk.Scrollbar(list_frame, orient="vertical",   command=self.batch_tree.yview)
        hsb = ttk.Scrollbar(list_frame, orient="horizontal", command=self.batch_tree.xview)
        self.batch_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right",  fill="y")
        hsb.pack(side="bottom", fill="x")
        self.batch_tree.pack(fill="both", expand=True)

        self.batch_tree.tag_configure("pending", foreground=TEXT_FAINT)
        self.batch_tree.tag_configure("running", foreground=ACCENT)
        self.batch_tree.tag_configure("done",    foreground=TR_ACCENT)
        self.batch_tree.tag_configure("error",   foreground="#f87171")
        self.batch_tree.tag_configure("skipped", foreground=TEXT_MUTED)

    # ── TTS Tab ───────────────────────────────────────────────────────────────
    def _build_tts_tab(self):
        """
        Studio-style layout (modelled on ElevenLabs Studio): the main view is
        only text + waveform timeline + transport. Every knob (API key,
        language, platform, voices, output path, toggles) lives in one
        ⚙ Settings dialog.
        """
        tab = self.tts_tab

        self._tts_segment_mode_var = tk.BooleanVar(
            value=bool(_ui_settings_load().get("tts_segment_mode", True)))
        self._tts_tab_out_label = None   # created inside the Settings dialog

        # ── Header: title | status | view toggle | Generate | Settings ──────
        ctrl = tk.Frame(tab, bg=PANEL, height=54, bd=0,
                        highlightbackground=PANEL_BORDER, highlightthickness=1)
        ctrl.pack(fill="x", side="top")
        ctrl.pack_propagate(False)
        tk.Label(ctrl, text="TTS STUDIO", bg=PANEL, fg="#a78bfa",
                 font=(MONO_FONT, 10, "bold")).pack(side="left", padx=(14, 10), pady=14)
        tk.Frame(ctrl, bg="#5b4fbf", width=2, height=28).pack(side="left", padx=8, pady=12)

        self._tts_view_script_btn = self._btn(
            ctrl, " Script ", lambda: self._tts_show_view("script"),
            bg=BTN_BG, fg="#a78bfa", abg=BTN_ACT)
        self._tts_view_script_btn.pack(side="left", padx=(4, 2), pady=10)
        self._tts_view_seg_btn = self._btn(
            ctrl, " Segments ", lambda: self._tts_show_view("segments"),
            bg=BTN_BG, fg=TEXT_FAINT, abg=BTN_ACT)
        self._tts_view_seg_btn.pack(side="left", padx=2, pady=10)

        self._tts_tab_status = tk.Label(
            ctrl, text="Paste a script, then ▶ Generate",
            bg=PANEL, fg=TEXT_FAINT, font=(MONO_FONT, 9), anchor="w")
        self._tts_tab_status.pack(side="left", fill="x", expand=True, padx=10)

        self._btn(ctrl, " ⚙ Settings ", self._open_tts_settings_dialog,
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT).pack(
            side="right", padx=(4, 12), pady=10)
        self._tts_tab_generate_btn = self._btn(
            ctrl, "  ▶ Generate  ", self._tts_tab_generate,
            bg="#0f1d14", fg="#22c55e", abg="#1f4d2e")
        self._tts_tab_generate_btn.pack(side="right", padx=4, pady=10)

        # ── Bottom (packed first so it claims the lowest strips):
        #    waveform timeline at the very bottom, transport bar above it ────
        wave_frame = tk.Frame(tab, bg=PANEL, bd=0,
                              highlightbackground=PANEL_BORDER, highlightthickness=1)
        wave_frame.pack(side="bottom", fill="x")
        self._tts_seg_fig, self._tts_seg_ax = plt.subplots(figsize=(11, 1.7))
        self._tts_seg_fig.patch.set_facecolor(PANEL)
        self._tts_seg_canvas = FigureCanvasTkAgg(self._tts_seg_fig, master=wave_frame)
        cw = self._tts_seg_canvas.get_tk_widget()
        cw.configure(bg=PANEL, height=150, cursor="hand2")
        cw.pack(fill="x", padx=8, pady=(2, 4))
        self._tts_seg_canvas.mpl_connect("button_press_event",
                                         self._tts_seg_on_wave_click)
        self._tts_seg_draw_placeholder()

        transport = tk.Frame(tab, bg=PANEL2, height=44, bd=0,
                             highlightbackground=PANEL_BORDER, highlightthickness=1)
        transport.pack(side="bottom", fill="x")
        transport.pack_propagate(False)
        self._btn(transport, " ▶ Play All ", self._tts_seg_play_all,
                  bg="#0f1d14", fg="#22c55e", abg="#1f4d2e").pack(
            side="left", padx=(12, 4), pady=7)
        self._btn(transport, " ▶ Segment ", self._tts_seg_play_selected,
                  bg="#0f1d14", fg="#22c55e", abg="#1f4d2e").pack(
            side="left", padx=4, pady=7)
        self._btn(transport, " ■ Stop ", self._tts_seg_stop,
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT).pack(
            side="left", padx=4, pady=7)
        self._tts_seg_regen_btn = self._btn(
            transport, " ⟳ Regenerate Selected ", self._tts_seg_regenerate,
            bg="#2a1a0a", fg="#f59e0b", abg="#4d2f10")
        self._tts_seg_regen_btn.pack(side="left", padx=(12, 4), pady=7)
        self._tts_transport_time = tk.Label(
            transport, text="0:00 / 0:00", bg=PANEL2, fg=TEXT,
            font=(MONO_FONT, 10, "bold"))
        self._tts_transport_time.pack(side="right", padx=14)

        # ── Main area: Script view ↔ Segments (card) view ───────────────────
        self._tts_main = tk.Frame(tab, bg=BG)
        self._tts_main.pack(fill="both", expand=True, padx=10, pady=(8, 4))

        # Script view
        self._tts_script_view = tk.Frame(self._tts_main, bg=BG)
        tk.Label(self._tts_script_view, text="Script  (text to synthesize):",
                 bg=BG, fg=TEXT, font=(MONO_FONT, 9, "bold")).pack(
            anchor="w", padx=2, pady=(4, 2))
        self._tts_tab_text = scrolledtext.ScrolledText(
            self._tts_script_view, wrap="word",
            bg=INPUT_BG, fg=INPUT_FG, insertbackground=INPUT_FG,
            font=(MONO_FONT, 10), relief="flat", bd=0,
            selectbackground="#1e3a8a", selectforeground="#f8fafc",
            height=16)
        self._tts_tab_text.pack(fill="both", expand=True, padx=0, pady=(0, 4))

        # Segments view — scrollable stack of sentence cards
        self._tts_seg_view = tk.Frame(self._tts_main, bg=BG)
        seg_cv = tk.Canvas(self._tts_seg_view, bg=BG, highlightthickness=0)
        seg_sb = tk.Scrollbar(self._tts_seg_view, orient="vertical",
                              command=seg_cv.yview)
        self._tts_seg_holder = tk.Frame(seg_cv, bg=BG)
        holder_id = seg_cv.create_window((0, 0), window=self._tts_seg_holder,
                                         anchor="nw")
        self._tts_seg_holder.bind(
            "<Configure>",
            lambda e: seg_cv.configure(scrollregion=seg_cv.bbox("all")))
        seg_cv.bind("<Configure>",
                    lambda e: seg_cv.itemconfigure(holder_id, width=e.width))
        seg_cv.configure(yscrollcommand=seg_sb.set)
        seg_sb.pack(side="right", fill="y")
        seg_cv.pack(side="left", fill="both", expand=True)
        self._tts_seg_cards_canvas = seg_cv
        self._tts_seg_cards = []

        self._tts_show_view("script")

    def _tts_show_view(self, which):
        """Swap the main area between the Script editor and Segment cards."""
        self._tts_active_view = which
        for f in (self._tts_script_view, self._tts_seg_view):
            f.pack_forget()
        if which == "script":
            self._tts_script_view.pack(fill="both", expand=True)
        else:
            self._tts_seg_view.pack(fill="both", expand=True)
        try:
            self._tts_view_script_btn.config(
                fg=_btn_fg("#a78bfa") if which == "script" else _btn_fg(TEXT_FAINT))
            self._tts_view_seg_btn.config(
                fg=_btn_fg("#a78bfa") if which == "segments" else _btn_fg(TEXT_FAINT))
        except Exception:
            pass

    def _open_tts_settings_dialog(self):
        """One ⚙ Settings window holding every TTS Studio option. Built once,
        then hidden/shown — the mirror widgets stay alive so the shared
        platform/voice visibility logic keeps working."""
        win = getattr(self, "_tts_settings_win", None)
        if win is not None and win.winfo_exists():
            win.deiconify()
            win.lift()
            return
        win = tk.Toplevel(self.root)
        self._tts_settings_win = win
        win.title("TTS Settings")
        win.configure(bg="#1e1b3a")
        win.geometry("920x300")
        win.transient(self.root)
        win.protocol("WM_DELETE_WINDOW", win.withdraw)

        # Rows 1-2: API key bar + language/platform/model/engine/voice —
        # bound to the SAME vars as the Single File tab.
        self._build_tts_settings_mirror(win)

        # Row 3: output path
        out_bar = tk.Frame(win, bg="#1e1b3a", height=40)
        out_bar.pack(fill="x")
        out_bar.pack_propagate(False)
        tk.Label(out_bar, text="Output:", bg="#1e1b3a", fg=TEXT_FAINT,
                 font=(MONO_FONT, 9)).pack(side="left", padx=(14, 6), pady=9)
        self._tts_tab_out_label = tk.Label(
            out_bar, text=(self._tts_tab_out_path or
                           "No output path chosen — will be prompted on Generate"),
            bg="#1e1b3a", fg=TEXT if self._tts_tab_out_path else TEXT_MUTED,
            font=(MONO_FONT, 9), anchor="w")
        self._tts_tab_out_label.pack(side="left", fill="x", expand=True, padx=4)
        self._btn(out_bar, "Choose Path", self._tts_tab_pick_output,
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT).pack(side="right", padx=10, pady=6)

        # Row 4: toggles
        tog = tk.Frame(win, bg="#1e1b3a", height=40)
        tog.pack(fill="x")
        tog.pack_propagate(False)
        for text, var, cmd in (
            ("Sentence segments (fix & regenerate one sentence)",
             self._tts_segment_mode_var, self._tts_seg_mode_changed),
            ("Emotion enrichment (eleven_v3 audio tags)",
             self._emotion_enabled_var, None),
        ):
            tk.Checkbutton(
                tog, text=text, variable=var, command=cmd,
                bg="#1e1b3a", fg=TEXT, selectcolor="#1e1b3a",
                activebackground="#1e1b3a", activeforeground="#a78bfa",
                font=(MONO_FONT, 9)).pack(side="left", padx=(14, 10), pady=8)
        self._btn(tog, "  Close  ", win.withdraw,
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT).pack(side="right", padx=12, pady=6)

    def _build_tts_settings_mirror(self, parent):
        """
        Visual copy of the TTS settings panel bound to the SAME StringVars as the
        Single File tab.  Call this AFTER _build_tts_settings_panel so that all
        vars already exist.
        """
        outer = tk.Frame(parent, bg="#1e1b3a", bd=0,
                         highlightbackground="#5b4fbf", highlightthickness=1)
        outer.pack(fill="x", side="top")

        # Mirror Row 1: API key + Bengali voice picker (always visible)
        row1 = tk.Frame(outer, bg="#1e1b3a", height=46)
        row1.pack(fill="x")
        row1.pack_propagate(False)

        tk.Label(row1, text="TTS SETTINGS", bg="#1e1b3a", fg="#a78bfa",
                 font=(MONO_FONT, 9, "bold")).pack(side="left", padx=(14, 10), pady=10)
        tk.Frame(row1, bg="#5b4fbf", width=2, height=26).pack(side="left", padx=(0, 12), pady=9)

        # primary=False so we don't recreate the StringVars — same vars,
        # same auto-fetch behaviour.
        self._build_elevenlabs_keybar(row1, panel_bg="#1e1b3a", primary=False)

        # Mirror Row 2: Language / Platform / Google TTS controls
        sp = tk.Frame(outer, bg="#1e1b3a", height=44)
        sp.pack(fill="x")
        sp.pack_propagate(False)

        # Language
        tk.Label(sp, text="Language:", bg="#1e1b3a", fg=TEXT,
                 font=(MONO_FONT, 9)).pack(side="left", padx=(14, 4), pady=10)
        lang_cb = ttk.Combobox(sp, textvariable=self._tts_language_var,
                               values=TTS_LANGUAGE_NAMES, state="readonly", width=14,
                               font=(MONO_FONT, 9))
        lang_cb.pack(side="left", padx=(0, 16))
        lang_cb.bind("<<ComboboxSelected>>", self._on_tts_language_change)

        # Platform
        tk.Label(sp, text="Platform:", bg="#1e1b3a", fg=TEXT,
                 font=(MONO_FONT, 9)).pack(side="left", padx=(0, 4))
        platform_cb = ttk.Combobox(sp, textvariable=self._tts_platform_var,
                                   values=TTS_PLATFORMS, state="readonly", width=12,
                                   font=(MONO_FONT, 9))
        platform_cb.pack(side="left", padx=(0, 8))
        platform_cb.bind("<<ComboboxSelected>>", self._on_tts_platform_change)

        # ElevenLabs sub-frame (mirror) — holds the ElevenLabs model picker
        ef = tk.Frame(sp, bg="#1e1b3a")
        self._tts_el_frame_mirror = ef
        self._build_el_model_picker(ef, "#1e1b3a")

        # Google TTS sub-frame (mirror)
        gf = tk.Frame(sp, bg="#1e1b3a")
        self._tts_google_frame_mirror = gf
        tk.Frame(gf, bg="#5b4fbf", width=2, height=26).pack(side="left", padx=(12, 12), pady=9)
        tk.Label(gf, text="Engine:", bg="#1e1b3a", fg=TEXT,
                 font=(MONO_FONT, 9)).pack(side="left", padx=(0, 4))
        for engine in ("Standard", "WaveNet", "Chirp3"):
            rb = tk.Radiobutton(
                gf, text=engine, variable=self._tts_engine_var, value=engine,
                bg="#1e1b3a", fg=TEXT, selectcolor="#1e1b3a",
                activebackground="#1e1b3a", activeforeground="#a78bfa",
                font=(MONO_FONT, 9), command=self._on_tts_engine_change)
            rb.pack(side="left", padx=4)
        tk.Frame(gf, bg="#5b4fbf", width=2, height=26).pack(side="left", padx=(8, 8), pady=9)
        tk.Label(gf, text="Voice:", bg="#1e1b3a", fg=TEXT,
                 font=(MONO_FONT, 9)).pack(side="left", padx=(0, 4))
        self._tts_voice_cb_mirror = ttk.Combobox(
            gf, textvariable=self._tts_voice_var, state="readonly", width=28,
            font=(MONO_FONT, 9))
        self._tts_voice_cb_mirror.pack(side="left", padx=(0, 8))

        # Sync mirror voice list + visibility with current state
        lang_data = TTS_LANGUAGES.get(self._tts_language_var.get(), {})
        engine    = self._tts_engine_var.get()
        voices    = lang_data.get(engine, [])
        self._tts_voice_cb_mirror["values"] = voices
        self._apply_tts_platform_visibility()

    def _tts_tab_pick_output(self):
        path = filedialog.asksaveasfilename(
            title="Save TTS audio as…",
            defaultextension=".mp3",
            filetypes=[("MP3 audio", "*.mp3"), ("All files", "*.*")])
        if path:
            self._tts_tab_out_path = path
            self._tts_set_out_label(path)

    def _tts_set_out_label(self, path):
        lbl = getattr(self, "_tts_tab_out_label", None)
        if lbl is not None:
            try:
                lbl.config(text=path, fg=TEXT)
            except tk.TclError:
                pass

    def _tts_tab_generate(self):
        script = self._tts_tab_text.get("1.0", "end").strip()
        if not script:
            messagebox.showwarning("No Script", "Paste or type a script in the text box first.")
            return

        # Resolve output path — prompt if not chosen yet
        out_path = self._tts_tab_out_path
        if not out_path:
            out_path = filedialog.asksaveasfilename(
                title="Save TTS audio as…",
                defaultextension=".mp3",
                filetypes=[("MP3 audio", "*.mp3"), ("All files", "*.*")])
            if not out_path:
                return
            self._tts_tab_out_path = out_path
            self._tts_set_out_label(out_path)

        (tts_platform, tts_lang_code, tts_voice_name, el_voice_id,
         pipeline_language, el_model) = self._get_tts_params()

        # Validate keys
        try:
            api_key = _get_api_key()
        except Exception as e:
            messagebox.showerror("ElevenLabs API Key Error", str(e)); return
        if tts_platform == "Google TTS":
            if not TTS_AVAILABLE:
                messagebox.showerror("Google TTS unavailable",
                                     "google-cloud-texttospeech not installed.\n"
                                     "Run: pip install google-cloud-texttospeech"); return

        self._tts_tab_generate_btn.config(state="disabled")
        self._tts_tab_status.config(text="Starting…", fg=TEXT_FAINT)

        segment_mode = bool(self._tts_segment_mode_var.get()) and PYDUB_AVAILABLE

        def _worker():
            try:
                def _cb(msg):
                    self.root.after(0, lambda m=msg: self._tts_tab_status.config(text=m, fg=ACCENT))

                if tts_platform == "ElevenLabs":
                    script_final = (
                        _run_emotion_enrichment(
                            script, language=pipeline_language,
                            model=GEMINI_DEFAULT_MODEL, status_cb=_cb)
                        if self._emotion_enabled_var.get() else script
                    )
                else:
                    script_final = script

                if segment_mode:
                    params = {
                        "platform": tts_platform, "api_key": api_key,
                        "voice_id": el_voice_id, "model_id": el_model,
                        "lang_code": tts_lang_code, "voice_name": tts_voice_name,
                    }
                    segs = self._tts_generate_segments_worker(
                        script_final, out_path, params, status_cb=_cb)
                    self.root.after(0, lambda: self._tts_segments_loaded(
                        segs, out_path, params))
                elif tts_platform == "ElevenLabs":
                    synthesize_tts_elevenlabs(script_final, out_path,
                                              api_key=api_key, voice_id=el_voice_id,
                                              model_id=el_model, status_cb=_cb)
                else:
                    synthesize_tts(script_final, out_path, status_cb=_cb,
                                   lang_code=tts_lang_code, voice_name=tts_voice_name)

                self.root.after(0, lambda: self._tts_tab_status.config(
                    text=f"✔ Done → {os.path.basename(out_path)}", fg=TR_ACCENT))
            except Exception as exc:
                err = str(exc)
                self.root.after(0, lambda e=err: self._tts_tab_status.config(
                    text=f"Error: {e}", fg="#f87171"))
                self.root.after(0, lambda e=err: messagebox.showerror("TTS Error", e))
            finally:
                self.root.after(0, lambda: self._tts_tab_generate_btn.config(state="normal"))

        threading.Thread(target=_worker, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────────────
    #  TTS Studio — sentence segments (per-sentence regenerate & merge)
    # ─────────────────────────────────────────────────────────────────────────

    def _tts_seg_mode_changed(self):
        d = _ui_settings_load()
        d["tts_segment_mode"] = bool(self._tts_segment_mode_var.get())
        _ui_settings_save(d)

    def _tts_seg_draw_placeholder(self):
        ax = self._tts_seg_ax
        ax.clear()
        ax.set_facecolor("#0f172a")
        ax.text(0.5, 0.5, "Generate audio — the segment timeline appears here",
                ha="center", va="center", transform=ax.transAxes,
                color=TEXT_MUTED, fontsize=10, fontfamily="monospace")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        try:
            self._tts_seg_fig.tight_layout(pad=0.4)
        except Exception:
            pass
        self._tts_seg_canvas.draw()

    # ── segment cards (ElevenLabs-Studio-style text blocks) ─────────────────

    def _tts_seg_time_str(self, i):
        s = self._tts_segments[i]
        t0 = s["start_ms"] / 1000.0
        t1 = (s["start_ms"] + s["dur_ms"]) / 1000.0

        def _f(t):
            m, sec = divmod(t, 60)
            return f"{int(m)}:{sec:04.1f}"
        return f"{_f(t0)}–{_f(t1)}"

    def _tts_seg_card_text(self, i):
        """Current (possibly user-edited) text of card i."""
        try:
            return self._tts_seg_cards[i]["text"].get("1.0", "end").strip()
        except Exception:
            return self._tts_segments[i]["text"]

    def _tts_seg_rebuild_cards(self):
        for c in self._tts_seg_cards:
            try:
                c["frame"].destroy()
            except Exception:
                pass
        self._tts_seg_cards = []

        def _wheel(event):
            cv = self._tts_seg_cards_canvas
            if event.num == 4:
                cv.yview_scroll(-2, "units")
            elif event.num == 5:
                cv.yview_scroll(2, "units")
            else:
                step = -1 if event.delta > 0 else 1
                cv.yview_scroll(step * 2, "units")
            return "break"

        for i, s in enumerate(self._tts_segments):
            f = tk.Frame(self._tts_seg_holder, bg=INPUT_BG,
                         highlightbackground=PANEL_BORDER, highlightthickness=1)
            f.pack(fill="x", pady=3, padx=2)

            left = tk.Frame(f, bg=INPUT_BG, width=76)
            left.pack(side="left", fill="y")
            left.pack_propagate(False)
            num = tk.Label(left, text=f"{i + 1:02d}", bg=INPUT_BG, fg="#a78bfa",
                           font=(MONO_FONT, 10, "bold"))
            num.pack(anchor="w", padx=8, pady=(6, 0))
            tlab = tk.Label(left, text=self._tts_seg_time_str(i), bg=INPUT_BG,
                            fg=TEXT_FAINT, font=(MONO_FONT, 7))
            tlab.pack(anchor="w", padx=8)

            lines = max(2, min(5, len(s["text"]) // 80 + 1))
            txt = tk.Text(f, height=lines, wrap="word", bg=INPUT_BG,
                          fg=INPUT_FG, insertbackground=INPUT_FG,
                          font=(MONO_FONT, 10), relief="flat", bd=0,
                          selectbackground="#1e3a8a",
                          selectforeground="#f8fafc")
            txt.insert("1.0", s["text"])
            txt.pack(side="left", fill="x", expand=True, padx=(4, 4), pady=6)

            right = tk.Frame(f, bg=INPUT_BG)
            right.pack(side="right", fill="y")
            play = tk.Label(right, text="▶", bg=INPUT_BG, fg="#22c55e",
                            font=(MONO_FONT, 11), cursor="hand2")
            play.pack(padx=(2, 4), pady=(6, 0))
            regen = tk.Label(right, text="⟳", bg=INPUT_BG, fg="#f59e0b",
                             font=(MONO_FONT, 12, "bold"), cursor="hand2")
            regen.pack(padx=(2, 4))

            def _sel(_e=None, idx=i):
                self._tts_seg_select(idx)

            def _play(_e=None, idx=i):
                self._tts_seg_select(idx)
                self._tts_seg_play_selected()

            def _regen(_e=None, idx=i):
                self._tts_seg_select(idx)
                self._tts_seg_regenerate()

            for w in (f, left, num, tlab):
                w.bind("<Button-1>", _sel)
            txt.bind("<FocusIn>", _sel)
            play.bind("<Button-1>", _play)
            regen.bind("<Button-1>", _regen)
            for w in (f, left, num, tlab, txt, play, regen, right):
                w.bind("<MouseWheel>", _wheel)
                w.bind("<Button-4>", _wheel)
                w.bind("<Button-5>", _wheel)

            self._tts_seg_cards.append(
                {"frame": f, "left": left, "num": num, "tlab": tlab,
                 "text": txt, "right": right, "play": play, "regen": regen})
        self._tts_seg_update_card_styles()

    def _tts_seg_update_card_styles(self):
        for i, c in enumerate(self._tts_seg_cards):
            sel = (i == self._tts_seg_selected)
            border = "#5b4fbf" if sel else PANEL_BORDER
            bg = "#1a2440" if sel else INPUT_BG
            try:
                c["frame"].config(highlightbackground=border,
                                  highlightcolor=border, bg=bg)
                for k in ("left", "num", "tlab", "right", "play", "regen"):
                    c[k].config(bg=bg)
                c["text"].config(bg=bg)
            except Exception:
                pass

    def _tts_seg_refresh_times(self):
        for i, c in enumerate(self._tts_seg_cards):
            try:
                c["tlab"].config(text=self._tts_seg_time_str(i))
            except Exception:
                pass

    def _tts_seg_scroll_to_card(self, idx):
        try:
            n = max(len(self._tts_seg_cards), 1)
            self._tts_seg_cards_canvas.yview_moveto(max(0.0, (idx - 1) / n))
        except Exception:
            pass

    # ── synthesis / merge (worker-thread safe: no UI access) ────────────────

    def _tts_synth_segment_file(self, text, seg_path, params,
                                prev_text="", next_text=""):
        """Synthesize one sentence to a WAV file at seg_path."""
        if params["platform"] == "ElevenLabs":
            voice_id = _sanitize_voice_id(str(params["voice_id"] or "").strip())
            if not params["api_key"] or not voice_id:
                raise ValueError(
                    "ElevenLabs API key / voice missing — check TTS Settings.")
            model_id = (params["model_id"] or ELEVENLABS_TTS_MODEL).strip() \
                or ELEVENLABS_TTS_MODEL
            snt = text
            if not model_id.startswith("eleven_v3"):
                stripped = _strip_emotion_tags(snt)
                if stripped.strip():
                    snt = stripped
            # A single sentence rarely exceeds the API limit, but guard anyway.
            pieces = _split_text_for_elevenlabs(snt) if len(snt) > 4800 else [snt]
            combined = _AudioSegment.empty()
            for p in pieces:
                raw = _elevenlabs_tts_post(
                    p, params["api_key"], voice_id, model_id,
                    previous_text=prev_text or None,
                    next_text=next_text or None)
                combined += _AudioSegment.from_file(io.BytesIO(raw), format="mp3")
            combined.export(seg_path, format="wav")
        else:
            synthesize_tts(text, seg_path,
                           lang_code=params["lang_code"],
                           voice_name=params["voice_name"])
        return seg_path

    def _tts_merge_segments(self, segs, out_path):
        """Concatenate all segment WAVs → out_path; refresh start/dur of each."""
        combined = _AudioSegment.empty()
        pos = 0
        for s in segs:
            seg = _AudioSegment.from_file(s["path"])
            s["start_ms"] = pos
            s["dur_ms"]   = len(seg)
            pos += len(seg)
            combined += seg
        combined.export(out_path, format="wav")
        return out_path

    def _tts_write_seg_manifest(self, segs, out_path):
        try:
            man = {"output": out_path,
                   "segments": [{"index": i + 1, "text": s["text"],
                                 "file": os.path.basename(s["path"]),
                                 "start_ms": s["start_ms"],
                                 "dur_ms": s["dur_ms"]}
                                for i, s in enumerate(segs)]}
            mpath = os.path.join(os.path.dirname(segs[0]["path"]), "manifest.json")
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump(man, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _tts_generate_segments_worker(self, script, out_path, params,
                                      status_cb=None):
        sentences = _split_script_into_sentences(script)
        if not sentences:
            raise ValueError("Script produced no sentences to synthesize.")
        seg_dir = os.path.splitext(out_path)[0] + "_segments"
        os.makedirs(seg_dir, exist_ok=True)
        segs, total = [], len(sentences)
        for i, snt in enumerate(sentences):
            if status_cb:
                status_cb(f"TTS: generating sentence {i + 1} of {total}…")
            seg_path = os.path.join(seg_dir, f"seg_{i + 1:03d}.wav")
            prev_text = " ".join(sentences[max(0, i - 2):i])
            next_text = " ".join(sentences[i + 1:i + 3])
            self._tts_synth_segment_file(snt, seg_path, params,
                                         prev_text, next_text)
            segs.append({"text": snt, "path": seg_path,
                         "start_ms": 0, "dur_ms": 0})
        if status_cb:
            status_cb("TTS: merging segments…")
        self._tts_merge_segments(segs, out_path)
        self._tts_write_seg_manifest(segs, out_path)
        return segs

    # ── UI state (main thread) ──────────────────────────────────────────────

    def _tts_segments_loaded(self, segs, out_path, params):
        try:
            y, sr = librosa.load(out_path, sr=None, mono=True)
        except Exception as e:
            messagebox.showerror("Load error",
                                 f"Could not load merged audio:\n{e}")
            return
        self._tts_segments     = segs
        self._tts_seg_audio    = y.astype(np.float32)
        self._tts_seg_sr       = int(sr)
        self._tts_seg_out_path = out_path
        self._tts_seg_params   = params
        self._tts_seg_selected = 0 if segs else None
        self._tts_seg_playing  = False

        self._tts_seg_rebuild_cards()
        self._tts_show_view("segments")
        dur = len(y) / max(sr, 1)
        self._tts_transport_time.config(text=f"0:00 / {self._tts_fmt_time(dur)}")
        self._tts_tab_status.config(
            text=f"{len(segs)} sentences · click a card or the timeline · "
                 "✎ edit text, ⟳ regenerate", fg=TEXT_FAINT)
        self._tts_seg_render_wave()

    @staticmethod
    def _tts_fmt_time(t):
        m, sec = divmod(max(t, 0.0), 60)
        return f"{int(m)}:{sec:04.1f}"

    def _tts_seg_select(self, idx, scroll=False):
        if idx is None or not (0 <= idx < len(self._tts_segments)):
            return
        self._tts_seg_selected = idx
        self._tts_seg_update_card_styles()
        if scroll:
            self._tts_seg_scroll_to_card(idx)
        self._tts_seg_render_wave()

    def _tts_seg_on_wave_click(self, event):
        if event.xdata is None or not self._tts_segments:
            return
        t_ms = event.xdata * 1000.0
        for i, s in enumerate(self._tts_segments):
            if s["start_ms"] <= t_ms < s["start_ms"] + s["dur_ms"]:
                self._tts_seg_select(i, scroll=True)
                return

    # ── waveform ────────────────────────────────────────────────────────────

    def _tts_seg_render_wave(self, cursor=None):
        if self._tts_seg_audio is None:
            return
        ax = self._tts_seg_ax
        ax.clear()
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors=TEXT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        y, sr = self._tts_seg_audio, self._tts_seg_sr
        dur = len(y) / max(sr, 1)

        FILLS = ("#134e4a", "#1e3a5f")
        for i, s in enumerate(self._tts_segments):
            x0 = s["start_ms"] / 1000.0
            x1 = x0 + s["dur_ms"] / 1000.0
            sel = (i == self._tts_seg_selected)
            ax.add_patch(Rectangle(
                (x0, -1.05), x1 - x0, 2.10,
                facecolor="#5b4fbf" if sel else FILLS[i % 2],
                edgecolor="none", alpha=0.50 if sel else 0.35, zorder=1))
            ax.axvline(x0, color="#a78bfa", linewidth=0.8, alpha=0.7, zorder=3)
            ax.text((x0 + x1) / 2.0, 1.02, str(i + 1),
                    color="#f8fafc" if sel else "#a78bfa", fontsize=7,
                    ha="center", va="bottom", fontfamily="monospace",
                    transform=ax.get_xaxis_transform(), clip_on=True, zorder=4)

        TARGET = 1500
        n = len(y)
        if n > TARGET:
            step = n // TARGET
            frames = n // step
            trimmed = y[:frames * step].reshape(frames, step)
            peaks_p = trimmed.max(axis=1)
            peaks_n = trimmed.min(axis=1)
            t_axis  = np.linspace(0, dur, frames)
        else:
            peaks_p = peaks_n = y
            t_axis  = np.linspace(0, dur, max(n, 1))
        ax.fill_between(t_axis, peaks_p, peaks_n, color=WAVEFORM,
                        alpha=0.85, linewidth=0, zorder=2)

        if cursor is not None and 0 <= cursor <= dur:
            ax.axvline(cursor, color=CURSOR_C, linewidth=CURSOR_W,
                       alpha=0.95, zorder=5)

        ax.set_xlim(0, max(dur, 0.01))
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, axis="x", color=GRID, linewidth=0.4,
                linestyle="--", alpha=0.5)

        def fmt_time(x, _):
            m, s2 = divmod(x, 60)
            return f"{int(m)}:{s2:04.1f}" if m else f"{s2:.1f}s"
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_time))
        try:
            self._tts_seg_fig.tight_layout(pad=0.4)
        except Exception:
            pass
        self._tts_seg_canvas.draw()

    # ── playback ────────────────────────────────────────────────────────────

    def _tts_seg_play_range(self, start_s, end_s):
        if self._tts_seg_audio is None:
            return
        self._tts_seg_stop()
        sr = self._tts_seg_sr
        chunk = self._tts_seg_audio[int(start_s * sr):int(end_s * sr)]
        if len(chunk) == 0:
            return
        self._tts_seg_playing   = True
        self._tts_seg_play_from = start_s
        self._tts_seg_play_end  = end_s
        self._tts_seg_play_t0   = time.perf_counter()

        def _run():
            try:
                sd.play(chunk, samplerate=sr)
                sd.wait()
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()
        self._tts_seg_tick()

    def _tts_seg_tick(self):
        if not self._tts_seg_playing:
            return
        pos = self._tts_seg_play_from + (time.perf_counter() - self._tts_seg_play_t0)
        dur = (len(self._tts_seg_audio) / max(self._tts_seg_sr, 1)
               if self._tts_seg_audio is not None else 0.0)
        try:
            self._tts_transport_time.config(
                text=f"{self._tts_fmt_time(pos)} / {self._tts_fmt_time(dur)}")
        except Exception:
            pass
        if pos >= self._tts_seg_play_end:
            self._tts_seg_playing = False
            self._tts_seg_render_wave()
            return
        self._tts_seg_render_wave(cursor=pos)
        self.root.after(120, self._tts_seg_tick)

    def _tts_seg_play_selected(self):
        i = self._tts_seg_selected
        if i is None or not self._tts_segments:
            return
        s = self._tts_segments[i]
        self._tts_seg_play_range(s["start_ms"] / 1000.0,
                                 (s["start_ms"] + s["dur_ms"]) / 1000.0)

    def _tts_seg_play_all(self):
        if self._tts_seg_audio is None:
            return
        self._tts_seg_play_range(
            0.0, len(self._tts_seg_audio) / max(self._tts_seg_sr, 1))

    def _tts_seg_stop(self):
        was_playing = self._tts_seg_playing
        self._tts_seg_playing = False
        try:
            sd.stop()
        except Exception:
            pass
        if was_playing and self._tts_seg_audio is not None:
            self._tts_seg_render_wave()

    # ── regenerate one sentence ─────────────────────────────────────────────

    def _tts_seg_regenerate(self):
        i = self._tts_seg_selected
        if i is None or not self._tts_segments:
            messagebox.showinfo("No segment",
                                "Generate audio first, then select a sentence.")
            return
        if self._tts_seg_params is None:
            return
        new_text = self._tts_seg_card_text(i)
        if not new_text:
            messagebox.showwarning(
                "Empty text",
                "Segment text is empty — type the corrected sentence first.")
            return
        self._tts_seg_stop()
        segs     = self._tts_segments
        params   = dict(self._tts_seg_params)
        out_path = self._tts_seg_out_path
        seg      = segs[i]
        self._tts_seg_regen_btn.config(state="disabled")
        self._tts_tab_status.config(text=f"Regenerating sentence {i + 1}…",
                                    fg=ACCENT)

        def _worker():
            try:
                prev_text = " ".join(s["text"] for s in segs[max(0, i - 2):i])
                next_text = " ".join(s["text"] for s in segs[i + 1:i + 3])
                tmp_path  = seg["path"] + ".tmp.wav"
                self._tts_synth_segment_file(new_text, tmp_path, params,
                                             prev_text, next_text)
                os.replace(tmp_path, seg["path"])
                seg["text"] = new_text
                self._tts_merge_segments(segs, out_path)
                self._tts_write_seg_manifest(segs, out_path)
                y, sr = librosa.load(out_path, sr=None, mono=True)

                def _done():
                    self._tts_seg_audio = y.astype(np.float32)
                    self._tts_seg_sr    = int(sr)
                    self._tts_seg_refresh_times()
                    dur = len(y) / max(sr, 1)
                    self._tts_transport_time.config(
                        text=f"0:00 / {self._tts_fmt_time(dur)}")
                    self._tts_seg_render_wave()
                    self._tts_tab_status.config(
                        text=f"✔ Sentence {i + 1} regenerated & merged → "
                             f"{os.path.basename(out_path)}", fg=TR_ACCENT)
                self.root.after(0, _done)
            except Exception as exc:
                err = str(exc)

                def _fail(e=err):
                    self._tts_tab_status.config(text=f"Error: {e}", fg="#f87171")
                    messagebox.showerror("Regenerate Error", e)
                self.root.after(0, _fail)
            finally:
                self.root.after(0, lambda: self._tts_seg_regen_btn.config(
                    state="normal"))

        threading.Thread(target=_worker, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────────────
    #  Audio Syncing Tab
    # ─────────────────────────────────────────────────────────────────────────
    def _build_audio_sync_tab(self):
        """
        Standalone audio syncing tab. Inputs:
          • English audio file
          • target-language (already-dubbed) audio file
          • target-language dubbing script (text)

        Runs Stage 3a → 3e (English SRT, dubbed SRT, Gemini mapping,
        SRT timing sync, synced dubbed audio). Skips Stage 1/2 entirely
        because the user is providing the dubbed audio + script directly.
        """
        tab = self.sync_tab

        # ── Top control bar ───────────────────────────────────────────────────
        ctrl = tk.Frame(tab, bg=PANEL, height=52, bd=0,
                        highlightbackground=PANEL_BORDER, highlightthickness=1)
        ctrl.pack(fill="x", side="top")
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="AUDIO SYNCING", bg=PANEL, fg=TR_ACCENT,
                 font=(MONO_FONT, 10, "bold")).pack(side="left", padx=(14, 10), pady=14)
        tk.Frame(ctrl, bg="#334155", width=2, height=28).pack(side="left", padx=8, pady=12)

        self._as_run_btn = self._btn(
            ctrl, "▶ Sync Audio", self._as_run_sync,
            bg="#0f1d20", fg="#2dd4bf", abg="#0e3a35")
        self._as_run_btn.pack(side="left", padx=(4, 14), pady=10)

        self._as_status = tk.Label(
            ctrl, text="Load English + dubbed audio and paste the script to begin.",
            bg=PANEL, fg=TEXT_FAINT, font=(MONO_FONT, 9))
        self._as_status.pack(side="left", padx=8)

        # ── Stacked layout: English (top) above dubbed audio (bottom) ──────────
        self._as_build_section(
            tab, side="en", title="ENGLISH AUDIO",
            colour=REG_EDGE, panel_bg=PANEL2,
            thr_default=DEFAULT_THR_DB, hys_default=DEFAULT_HYS_DB,
            min_default=DEFAULT_MIN_MS)

        self._as_build_section(
            tab, side="bn", title=f"{self._current_language().upper()} AUDIO",
            colour="#38bdf8", panel_bg="#0c1f2c",
            thr_default=DEFAULT_BN_THR_DB, hys_default=DEFAULT_BN_HYS_DB,
            min_default=DEFAULT_BN_MIN_MS)

        # ── Dubbing script input ───────────────────────────────────────────────
        script_hdr = tk.Frame(tab, bg="#1e1b3a", height=28, bd=0,
                              highlightbackground="#5b4fbf", highlightthickness=1)
        script_hdr.pack(fill="x", side="top", pady=(6, 0))
        script_hdr.pack_propagate(False)
        _script_lbl = tk.Label(script_hdr, text="BENGALI DUBBING SCRIPT",
                 bg="#1e1b3a", fg="#a78bfa",
                 font=(MONO_FONT, 9, "bold"))
        _script_lbl.pack(side="left", padx=14, pady=4)
        self._register_lang_label(_script_lbl, "{LANG} DUBBING SCRIPT")
        tk.Label(script_hdr,
                 text="(used as the Gemini mapping script in Stage 3c)",
                 bg="#1e1b3a", fg=TEXT_FAINT,
                 font=(MONO_FONT, 8, "italic")).pack(side="left", padx=8, pady=4)

        script_frame = tk.Frame(tab, bg=BG)
        script_frame.pack(fill="both", expand=False, padx=8, pady=(2, 6))
        self._as_script_text = scrolledtext.ScrolledText(
            script_frame, height=8, wrap="word",
            bg=INPUT_BG, fg=INPUT_FG, insertbackground=INPUT_FG,
            font=(MONO_FONT, 10), relief="flat", borderwidth=0)
        self._as_script_text.pack(fill="both", expand=True)

        # ── Stage progress indicators ─────────────────────────────────────────
        pp = tk.Frame(tab, bg="#162032", bd=0,
                      highlightbackground="#3b82f6", highlightthickness=1)
        pp.pack(fill="x", side="top")

        hdr = tk.Frame(pp, bg="#162032", height=26)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  SYNC PROGRESS", bg="#162032", fg="#60a5fa",
                 font=(MONO_FONT, 9, "bold")).pack(side="left", padx=14, pady=3)

        body = tk.Frame(pp, bg="#162032")
        body.pack(fill="x", padx=14, pady=(0, 8))

        stage_labels = [
            ("S3a", "Sync — English SRT"),
            ("S3b", "Sync — Dubbed SRT from Dubbed Audio"),
            ("S3c", "Sync — SRT Mapping (Gemini)"),
            ("S3d", "Sync — SRT Timing Sync"),
            ("S3e", "Sync — Synced Audio File"),
        ]
        for tag, desc in stage_labels:
            row = tk.Frame(body, bg="#162032")
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f" [{tag}]", bg="#162032", fg="#22c55e",
                     font=(MONO_FONT, 9, "bold"), width=6, anchor="w").pack(side="left")
            tk.Label(row, text=desc, bg="#162032", fg=TEXT_FAINT,
                     font=(MONO_FONT, 9), width=44, anchor="w").pack(side="left")
            var = tk.StringVar(value="—")
            lbl = tk.Label(row, textvariable=var, bg="#162032", fg=TEXT_MUTED,
                           font=(MONO_FONT, 9), anchor="w")
            lbl.pack(side="left", fill="x", expand=True, padx=(8, 0))
            self._as_stage_vars[tag] = var
            self._as_stage_labels[tag] = lbl

        # ── Bengali Captions panel (post-sync re-chunking) ───────────────────
        self._build_captions_panel(self.sync_tab)

    # ─────────────────────────────────────────────────────────────────────────
    #  History tab — past runs, edit saved text, re-dub
    # ─────────────────────────────────────────────────────────────────────────
    def _build_history_tab(self):
        tab = self.hist_tab

        ctrl = tk.Frame(tab, bg=PANEL, height=52, bd=0,
                        highlightbackground=PANEL_BORDER, highlightthickness=1)
        ctrl.pack(fill="x", side="top")
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="RUN HISTORY", bg=PANEL, fg=TR_ACCENT,
                 font=(MONO_FONT, 10, "bold")).pack(side="left",
                                                    padx=(14, 10), pady=14)
        tk.Frame(ctrl, bg="#334155", width=2, height=28).pack(
            side="left", padx=8, pady=12)

        self._hist_redub_btn = self._btn(
            ctrl, "✏ Edit Text & Re-Dub", self._history_redub_selected,
            bg="#0f1d14", fg=TR_ACCENT, abg="#1f4d2e")
        self._hist_redub_btn.pack(side="left", padx=(4, 6), pady=10)

        self._btn(ctrl, "📂 Open Folder", self._history_open_folder,
                  bg="#172554", fg=REG_LABEL, abg="#1e3a8a"
                  ).pack(side="left", padx=(0, 6), pady=10)
        self._btn(ctrl, "➕ Add Existing…", self._history_add_existing,
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT
                  ).pack(side="left", padx=(0, 6), pady=10)
        self._btn(ctrl, "↻ Refresh", self._history_refresh,
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT
                  ).pack(side="left", padx=(0, 6), pady=10)
        self._btn(ctrl, "🗑 Remove", self._history_remove_selected,
                  bg="#1f1213", fg="#f87171", abg="#3a1414"
                  ).pack(side="left", padx=(0, 6), pady=10)

        tk.Label(tab,
                 text="Every translated run is saved here. Select one → "
                      "“Edit Text & Re-Dub” re-opens the saved translation for "
                      "editing (even days later) and re-runs dubbing + syncing — "
                      "no re-translation cost. Nothing is overwritten: old text "
                      "is kept as _FinalScript_vNN.txt, new audio gets a "
                      "_redubNN suffix.",
                 bg=BG, fg=TEXT_FAINT, font=(MONO_FONT, 9),
                 anchor="w", justify="left", wraplength=1100
                 ).pack(fill="x", padx=14, pady=(8, 0))

        list_frame = tk.Frame(tab, bg=BG)
        list_frame.pack(fill="both", expand=True, padx=8, pady=8)

        cols = ("Date", "Language", "Audio", "Source", "Folder")
        self.hist_tree = ttk.Treeview(list_frame, columns=cols,
                                      show="headings", selectmode="browse")
        style = ttk.Style()
        style.configure("Hist.Treeview", background=PANEL, foreground=TEXT,
                        fieldbackground=PANEL, rowheight=26,
                        font=(MONO_FONT, 9))
        style.configure("Hist.Treeview.Heading", background=PANEL2,
                        foreground=REG_EDGE, font=(MONO_FONT, 9, "bold"))
        style.map("Hist.Treeview", background=[("selected", "#1e3a8a")],
                  foreground=[("selected", "#f8fafc")])
        self.hist_tree.configure(style="Hist.Treeview")
        for col, width in [("Date", 150), ("Language", 90), ("Audio", 240),
                           ("Source", 90), ("Folder", 380)]:
            self.hist_tree.heading(col, text=col)
            self.hist_tree.column(col, width=width, anchor="w")
        vsb = ttk.Scrollbar(list_frame, orient="vertical",
                            command=self.hist_tree.yview)
        self.hist_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.hist_tree.pack(fill="both", expand=True)
        self.hist_tree.bind("<Double-1>",
                            lambda _e: self._history_redub_selected())
        self.hist_tree.bind("<<TreeviewSelect>>",
                            lambda _e: self._history_load_compare())

        # ── Listen & Compare: English (top) vs dubbed output (bottom) ────────
        cmp_outer = tk.Frame(tab, bg=PANEL, bd=0,
                             highlightbackground=PANEL_BORDER,
                             highlightthickness=1)
        cmp_outer.pack(fill="x", padx=8, pady=(0, 4))

        cmp_bar = tk.Frame(cmp_outer, bg=PANEL, height=40)
        cmp_bar.pack(fill="x")
        cmp_bar.pack_propagate(False)
        tk.Label(cmp_bar, text="LISTEN & COMPARE", bg=PANEL, fg="#a78bfa",
                 font=(MONO_FONT, 9, "bold")).pack(side="left",
                                                   padx=(14, 12), pady=10)
        self._btn(cmp_bar, "▶ English", lambda: self._cmp_play("en"),
                  bg="#172554", fg=REG_LABEL, abg="#1e3a8a"
                  ).pack(side="left", padx=(0, 6), pady=6)
        self._btn(cmp_bar, "▶ Dubbed", lambda: self._cmp_play("out"),
                  bg="#0f1d14", fg=TR_ACCENT, abg="#1f4d2e"
                  ).pack(side="left", padx=(0, 6), pady=6)
        self._btn(cmp_bar, "⏹ Stop", self._cmp_stop,
                  bg="#1f1213", fg="#f87171", abg="#3a1414"
                  ).pack(side="left", padx=(0, 10), pady=6)
        self._cmp_info = tk.Label(cmp_bar, text="Select a run above to see "
                                  "both waveforms.", bg=PANEL, fg=TEXT_FAINT,
                                  font=(MONO_FONT, 9), anchor="w")
        self._cmp_info.pack(side="left", fill="x", expand=True)

        self._cmp_fig, (self._cmp_ax_en, self._cmp_ax_out) = plt.subplots(
            2, 1, figsize=(12, 3.0))
        self._cmp_fig.patch.set_facecolor(BG)
        self._cmp_fig.subplots_adjust(left=0.055, right=0.99,
                                      top=0.90, bottom=0.14, hspace=0.65)
        self._cmp_canvas = FigureCanvasTkAgg(self._cmp_fig, master=cmp_outer)
        self._cmp_canvas.get_tk_widget().configure(bg=BG)
        self._cmp_canvas.get_tk_widget().pack(fill="x")
        self._cmp_paths: Dict[str, str] = {}
        self._cmp_audio_cache: Dict[Tuple[str, float], Tuple] = {}
        self._cmp_load_token = 0
        self._cmp_draw(None, None, "", "")

        self._hist_status = tk.Label(tab, text="", bg=BG, fg=TEXT_FAINT,
                                     font=(MONO_FONT, 9), anchor="w")
        self._hist_status.pack(fill="x", padx=14, pady=(0, 8))

        self._hist_entries = []
        self._hist_redub_running = False
        self._history_refresh()

    # ── History: Listen & Compare panel ──────────────────────────────────────

    @staticmethod
    def _find_latest_output(outdir: str) -> str:
        """Newest synced/dubbed audio file in *outdir* ('' if none)."""
        return _find_latest_output_file(outdir)

    def _cmp_style_axis(self, ax, title, colour):
        ax.set_facecolor("#0b1220")
        ax.tick_params(colors=TEXT_FAINT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#1e293b")
        ax.set_title(title, color=colour, fontsize=8,
                     fontfamily="monospace", loc="left")
        ax.margins(x=0)

    def _cmp_draw(self, en_env, out_env, en_title, out_title,
                  en_sr=100, out_sr=100):
        """Draw both envelope waveforms (env arrays sampled at *sr* Hz)."""
        for ax, env, sr, title, colour in (
                (self._cmp_ax_en,  en_env,  en_sr,  en_title or
                 "ENGLISH (original)", WAVEFORM),
                (self._cmp_ax_out, out_env, out_sr, out_title or
                 "DUBBED OUTPUT", TR_ACCENT)):
            ax.clear()
            self._cmp_style_axis(ax, title, colour)
            if env is not None and len(env):
                t = np.arange(len(env)) / float(sr)
                ax.fill_between(t, -env, env, color=colour,
                                linewidth=0.4, alpha=0.9)
                ax.set_xlim(0, t[-1] if len(t) else 1)
                ax.set_ylim(-1.05, 1.05)
            else:
                ax.text(0.5, 0.5, "no audio", transform=ax.transAxes,
                        ha="center", va="center", color=TEXT_FAINT,
                        fontsize=8, fontfamily="monospace")
                ax.set_xticks([]); ax.set_yticks([])
        try:
            self._cmp_canvas.draw_idle()
        except Exception:
            pass

    @staticmethod
    def _cmp_envelope(path: str, points_per_sec: int = 100):
        """Load *path* and return (envelope, points_per_sec, duration_s)."""
        y, sr = librosa.load(path, sr=None, mono=True)
        if not len(y):
            return np.zeros(1), points_per_sec, 0.0
        hop = max(1, int(sr / points_per_sec))
        n   = len(y) // hop
        env = np.abs(y[:n * hop]).reshape(n, hop).max(axis=1)
        peak = float(env.max()) or 1.0
        return (env / peak).astype(np.float32), points_per_sec, len(y) / sr

    def _history_load_compare(self):
        """Load the selected run's English + dubbed waveforms (async)."""
        sel = self.hist_tree.selection()
        if not sel:
            return
        try:
            e = self._hist_entries[int(sel[0])]
        except Exception:
            return
        en_path  = e.get("audio_path", "")
        out_path = self._find_latest_output(e.get("outdir", ""))
        language = e.get("language", "") or "DUBBED"
        self._cmp_paths = {"en": en_path, "out": out_path}
        self._cmp_info.config(text="Loading waveforms…", fg=TEXT_FAINT)
        self._cmp_load_token += 1
        token = self._cmp_load_token

        def worker():
            def env_for(path):
                if not path or not os.path.exists(path):
                    return None, 100, 0.0
                key = (path, os.path.getmtime(path))
                if key not in self._cmp_audio_cache:
                    try:
                        self._cmp_audio_cache[key] = self._cmp_envelope(path)
                    except Exception:
                        return None, 100, 0.0
                return self._cmp_audio_cache[key]

            en_env,  en_sr,  en_dur  = env_for(en_path)
            out_env, out_sr, out_dur = env_for(out_path)

            def done():
                if token != self._cmp_load_token or getattr(self, "_closing", False):
                    return              # superseded selection or app closing
                self._cmp_draw(
                    en_env, out_env,
                    f"ENGLISH (original) — {os.path.basename(en_path) or '?'} "
                    f"· {en_dur:.1f}s",
                    f"{language.upper()} (dubbed output) — "
                    f"{os.path.basename(out_path) or 'not found'} "
                    f"· {out_dur:.1f}s",
                    en_sr, out_sr)
                if out_path:
                    self._cmp_info.config(
                        text="Ready — use ▶ English / ▶ Dubbed to listen.",
                        fg=TEXT_FAINT)
                else:
                    self._cmp_info.config(
                        text="No dubbed output file in this run's folder yet "
                             "(run the Full Pipeline or Re-Dub).",
                        fg="#fbbf24")
            try:
                self.root.after(0, done)
            except Exception:
                pass    # window closed while loading

        threading.Thread(target=worker, daemon=True).start()

    def _cmp_play(self, which: str):
        path = self._cmp_paths.get(which, "")
        if not path or not os.path.exists(path):
            self._cmp_info.config(text="No audio file for that button — "
                                  "select a run first.", fg="#fbbf24")
            return
        self._cmp_info.config(text=f"Playing {os.path.basename(path)}…",
                              fg=TR_ACCENT)

        def worker():
            try:
                y, sr = librosa.load(path, sr=None, mono=True)
                sd.stop()
                sd.play(y.astype(np.float32), samplerate=sr)
            except Exception as ex:
                self.root.after(0, lambda: self._cmp_info.config(
                    text=f"Playback failed: {ex}", fg="#f87171"))

        threading.Thread(target=worker, daemon=True).start()

    def _cmp_stop(self):
        try:
            sd.stop()
        except Exception:
            pass
        self._cmp_info.config(text="Stopped.", fg=TEXT_FAINT)

    def _show_run_result(self, base: str = None, language: str = None,
                         jump: bool = True):
        """Jump to the History tab and load the run's waveforms. Called on
        pipeline / batch / re-dub completion when Auto-open result is on.
        jump=False refreshes project state only (single-file runs now show
        their result in the in-tab Compare view instead)."""
        # Keep project state fresh regardless of the auto-open preference.
        try:
            e = self._current_project
            if e and base and os.path.abspath(base) == \
                    os.path.abspath(e.get("base", "")):
                _project_upsert(e)          # bumps 'updated'
            self._refresh_stage_bar()
            self._scripts_load_project()
        except Exception:
            pass
        if not jump:
            return
        if not getattr(self, "_auto_open_var", None) \
                or not self._auto_open_var.get():
            return
        try:
            self._history_refresh()
            target_iid = None
            if base:
                key = (os.path.abspath(base), language)
                for idx, e in enumerate(self._hist_entries):
                    if (os.path.abspath(e.get("base", "")),
                            e.get("language")) == key:
                        target_iid = str(idx)
                        break
            if target_iid is None and self.hist_tree.get_children():
                target_iid = self.hist_tree.get_children()[0]
            if target_iid is not None:
                self.hist_tree.selection_set(target_iid)
                self.hist_tree.see(target_iid)
            self.notebook.select(self.hist_tab)
            self._history_load_compare()
        except Exception:
            pass

    def _history_refresh(self):
        self._hist_entries = _history_load()
        tree = self.hist_tree
        for item in tree.get_children():
            tree.delete(item)
        for idx, e in enumerate(self._hist_entries):
            tree.insert("", "end", iid=str(idx), values=(
                e.get("ts", "?"), e.get("language", "?"),
                os.path.basename(e.get("audio_path", "")) or "?",
                e.get("source", ""),
                e.get("outdir", "")))
        n = len(self._hist_entries)
        self._hist_status.config(
            text=(f"{n} run(s) in history." if n else
                  "No runs recorded yet — run a translation, or use "
                  "➕ Add Existing… to register an old output folder."),
            fg=TEXT_FAINT)

    def _history_selected_entry(self):
        sel = self.hist_tree.selection()
        if not sel:
            self._hist_status.config(text="Select a run in the list first.",
                                     fg="#fbbf24")
            return None
        try:
            return self._hist_entries[int(sel[0])]
        except Exception:
            return None

    def _history_open_folder(self):
        e = self._history_selected_entry()
        if not e:
            return
        path = e.get("outdir", "")
        if not os.path.isdir(path):
            self._hist_status.config(text=f"Folder missing: {path}",
                                     fg="#f87171")
            return
        import subprocess
        try:
            if IS_WINDOWS:
                os.startfile(path)  # noqa — Windows only
            elif IS_MAC:
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as ex:
            self._hist_status.config(text=f"Could not open folder: {ex}",
                                     fg="#f87171")

    def _history_remove_selected(self):
        e = self._history_selected_entry()
        if not e:
            return
        if not messagebox.askyesno(
                "Remove from history",
                "Remove this entry from the history list?\n\n"
                "Files on disk are NOT deleted."):
            return
        _history_remove(e.get("base", ""), e.get("language", ""))
        self._history_refresh()

    def _history_add_existing(self):
        folder = filedialog.askdirectory(
            title="Select an output folder (contains *_FinalScript.txt)")
        if not folder:
            return
        fs = sorted(f for f in os.listdir(folder)
                    if f.endswith("_FinalScript.txt"))
        if not fs:
            self._hist_status.config(
                text="No *_FinalScript.txt found in that folder.",
                fg="#f87171")
            return
        fs_path = os.path.join(folder, fs[0])
        base = fs_path[:-len("_FinalScript.txt")]

        # Detect language from the "=== <LANG> TRANSLATION ===" marker
        language = ""
        try:
            with open(fs_path, "r", encoding="utf-8") as f:
                head = f.read(6000)
            for lang in TTS_LANGUAGES:
                if f"=== {lang.upper()} TRANSLATION ===" in head:
                    language = lang
                    break
        except Exception:
            pass
        if not language:
            language = self._current_language()

        # The pipeline copies the original audio into the output folder
        audio_path = ""
        for ext in AUDIO_EXTENSIONS:
            cand = base + ext
            if os.path.exists(cand):
                audio_path = cand
                break
        if not audio_path:
            self._hist_status.config(
                text="Added — but no matching audio file found in the folder; "
                     "re-dub will reuse the saved English SRT if present.",
                fg="#fbbf24")
        _history_record(audio_path or base + ".wav", folder, base,
                        language, "added")
        self._history_refresh()

    def _history_redub_selected(self):
        if self._hist_redub_running:
            return
        e = self._history_selected_entry()
        if not e:
            return
        base       = e.get("base", "")
        outdir     = e.get("outdir", "")
        language   = e.get("language", "")
        audio_path = e.get("audio_path", "")
        fs_path    = base + "_FinalScript.txt"
        if not os.path.exists(fs_path):
            self._hist_status.config(text=f"FinalScript missing: {fs_path}",
                                     fg="#f87171")
            return
        if language != self._current_language():
            messagebox.showwarning(
                "Language mismatch",
                f"This run is {language}, but the TTS panel is set to "
                f"{self._current_language()}.\n\nSwitch the Language dropdown "
                f"in TTS Settings to {language} (so the right voice is used), "
                "then try again.")
            return
        try:
            _get_api_key()
        except Exception as ex:
            messagebox.showerror("ElevenLabs API Key Error", str(ex))
            return
        try:
            _validate_llm_config()
        except Exception as ex:
            messagebox.showerror("LLM Provider Error", str(ex))
            return

        (tts_platform, tts_lang_code, tts_voice_name,
         el_voice_id, _lang, el_model) = self._get_tts_params()
        en_thr, en_hys, en_min = self._get_en_region_params()
        te_thr, te_hys, te_min = self._get_bn_region_params()
        gemini_model = self._gemini_model_var.get()
        emotion_on   = self._emotion_enabled_var.get()

        self._hist_redub_running = True
        self._hist_redub_btn.config(state="disabled")

        def _st(msg, colour=TEXT_FAINT):
            self.root.after(0, lambda: self._hist_status.config(
                text=msg, fg=colour))

        def _worker():
            try:
                with open(fs_path, "r", encoding="utf-8") as f:
                    combined_old = f.read()
                script_text = _extract_translation_from_finalscript(
                    combined_old, language)
                raw_eng = ""
                if combined_old.startswith("=== ENGLISH TRANSCRIPTION ==="):
                    raw_eng = combined_old.split(
                        "=== ENGLISH TRANSCRIPTION ===", 1)[1]
                    raw_eng = raw_eng.split("===", 1)[0].strip()

                # Left column of the review window: the run's segment SRT if
                # it was saved, else the whole English text as one box.
                en_entries = []
                srt_path = base + ".srt"
                if os.path.exists(srt_path):
                    with open(srt_path, "r", encoding="utf-8") as f:
                        en_entries = _extract_srt_entries(f.read())
                if not en_entries and raw_eng:
                    en_entries = [(0.0, 0.0, raw_eng)]

                y_data, sr = None, None
                if audio_path and os.path.exists(audio_path):
                    _st("Loading English audio…")
                    try:
                        y_data, sr = librosa.load(audio_path, sr=None,
                                                  mono=True)
                    except Exception:
                        y_data, sr = None, None

                # ── Review / edit window (the point of a re-dub) ────────────
                _st("Waiting for text review — edit, then Continue…",
                    "#d97706")
                res, done = {}, threading.Event()
                tr_paras = _split_translation_paragraphs(script_text)
                self.root.after(0, lambda: self._show_translation_review(
                    en_entries, tr_paras, language, res, done,
                    audio=y_data, sr=sr, audio_path=audio_path))
                done.wait()
                new_text = script_text
                if (res.get("action") == "continue"
                        and (res.get("text") or "").strip()):
                    new_text = res["text"].strip()
                    # Feedback loop: reviewed re-dub script is human-proofed —
                    # save to translation memory (same capture as the
                    # single-file review).
                    if self._tm_enabled_var.get():
                        n_pairs = _tm_capture(language, en_entries, new_text,
                                              source=base)
                        if n_pairs:
                            _st(f"🧠 Proofed translation saved to memory "
                                f"({n_pairs} pairs).")

                rev = _history_next_redub_rev(outdir)

                # Preserve the old text before replacing the canonical file.
                if new_text != script_text:
                    backup = f"{base}_FinalScript_v{rev - 1:02d}.txt"
                    if not os.path.exists(backup):
                        shutil.copy2(fs_path, backup)
                    header = (f"=== ENGLISH TRANSCRIPTION ===\n{raw_eng}\n\n"
                              if raw_eng else "")
                    with open(fs_path, "w", encoding="utf-8") as f:
                        f.write(header + f"=== {language.upper()} TRANSLATION "
                                         f"===\n{new_text}")

                api_key = _get_api_key()

                # ── TTS ─────────────────────────────────────────────────────
                def _cb(msg):
                    _st(f"Re-dub {rev:02d}: {msg}", "#d97706")

                name_src = audio_path or base + ".wav"
                tts_path = os.path.join(outdir, _tts_output_name(
                    language, name_src, f"_tts_redub{rev:02d}"))

                # Multi-speaker: honour the project's saved 🎭 voice map.
                spk_paras = _split_translation_paragraphs(new_text)
                spk_map = (_speakers_voice_map(base)
                           if tts_platform == "ElevenLabs" else {})
                spk_map = {i: v for i, v in spk_map.items()
                           if 0 <= i < len(spk_paras)}

                _st(f"Re-dub {rev:02d}: synthesizing speech…", "#d97706")
                if tts_platform == "ElevenLabs" and spk_map:
                    enrich_cb = ((lambda txt: _run_emotion_enrichment(
                                     txt, language=language,
                                     model=GEMINI_DEFAULT_MODEL,
                                     status_cb=_cb))
                                 if emotion_on else None)
                    synthesize_tts_elevenlabs_multi(
                        spk_paras, spk_map, tts_path, api_key=api_key,
                        default_voice_id=el_voice_id, model_id=el_model,
                        status_cb=_cb, enrich_cb=enrich_cb)
                elif tts_platform == "ElevenLabs":
                    _st(f"Re-dub {rev:02d}: emotion pass…", "#d97706")
                    enriched = (_run_emotion_enrichment(
                                    new_text, language=language,
                                    model=GEMINI_DEFAULT_MODEL, status_cb=_cb)
                                if emotion_on else new_text)
                    synthesize_tts_elevenlabs(enriched, tts_path,
                                              api_key=api_key,
                                              voice_id=el_voice_id,
                                              model_id=el_model,
                                              status_cb=_cb)
                else:
                    _st(f"Re-dub {rev:02d}: emotion pass…", "#d97706")
                    enriched = (_run_emotion_enrichment(
                                    new_text, language=language,
                                    model=GEMINI_DEFAULT_MODEL, status_cb=_cb)
                                if emotion_on else new_text)
                    synthesize_tts(_strip_emotion_tags(enriched), tts_path,
                                   status_cb=_cb, lang_code=tts_lang_code,
                                   voice_name=tts_voice_name)

                # ── English SRT: reuse the saved one, else re-transcribe ───
                en_srt = ""
                en_srt_path = base + "_sync_en.srt"
                if os.path.exists(en_srt_path):
                    with open(en_srt_path, "r", encoding="utf-8") as f:
                        en_srt = f.read()
                if not en_srt.strip():
                    if y_data is None:
                        raise ValueError(
                            "No saved _sync_en.srt and the English audio is "
                            f"missing ({audio_path or 'no path'}) — cannot "
                            "build the sync map.")
                    _st(f"Re-dub {rev:02d}: transcribing English audio…",
                        "#d97706")
                    en_regions = _detect_regions_from_audio(
                        y_data, sr, en_thr, en_hys, en_min)
                    en_result = _transcribe_audio(audio_path, api_key)
                    en_words  = en_result.get("words", [])
                    if not en_words:
                        raise ValueError(
                            "No word data from ElevenLabs for English audio.")
                    en_srt = _build_english_subtitle_srt(en_regions, en_words)
                    with open(en_srt_path, "w", encoding="utf-8") as f:
                        f.write(en_srt)

                # ── Dubbed SRT from the fresh TTS audio ────────────────────
                _st(f"Re-dub {rev:02d}: transcribing new {language} audio…",
                    "#d97706")
                te_y, te_sr = librosa.load(tts_path, sr=None, mono=True)
                te_regions = _detect_regions_from_audio(
                    te_y, te_sr, te_thr, te_hys, te_min)
                if not te_regions:
                    raise ValueError("No regions detected in the new TTS audio.")
                te_result = _transcribe_audio(tts_path, api_key)
                te_words  = te_result.get("words", [])
                if not te_words:
                    raise ValueError(
                        f"No word data for the new {language} audio.")
                te_srt = _build_target_subtitle_srt(te_regions, te_words)
                with open(f"{base}_sync_te_redub{rev:02d}.srt", "w",
                          encoding="utf-8") as f:
                    f.write(te_srt)

                # ── Mapping + timing sync + synced audio ───────────────────
                _st(f"Re-dub {rev:02d}: Gemini SRT mapping…", "#d97706")
                mapping_text = _call_gemini_mapping(
                    en_srt, te_srt, new_text, gemini_model, language=language)
                with open(f"{base}_sync_mapping_redub{rev:02d}.txt", "w",
                          encoding="utf-8") as f:
                    f.write(mapping_text)

                _st(f"Re-dub {rev:02d}: syncing…", "#d97706")
                en_dur = (float(len(y_data)) / float(sr)
                          if (y_data is not None and sr) else 0.0)
                synced_subs, orig_te_subs, sync_log = run_sync_from_strings(
                    en_srt, te_srt, mapping_text, en_audio_duration=en_dur)
                with open(f"{base}_sync_log_redub{rev:02d}.txt", "w",
                          encoding="utf-8") as f:
                    f.write(sync_log)
                synced_srt_path = f"{base}_sync_synced_redub{rev:02d}.srt"
                with open(synced_srt_path, "w", encoding="utf-8") as f:
                    f.write(_write_srt_from_dict(synced_subs))
                self.root.after(0, lambda p=synced_srt_path:
                                setattr(self, "_last_synced_srt_path", p))
                ts_list = _build_timestamps(orig_te_subs, synced_subs)
                with open(f"{base}_sync_timestamps_redub{rev:02d}.txt", "w",
                          encoding="utf-8") as f:
                    f.write(_format_timestamps_as_text(ts_list))

                _st(f"Re-dub {rev:02d}: building synced audio…", "#d97706")
                synced_path = os.path.join(outdir, _tts_output_name(
                    language, name_src, f"_synced_redub{rev:02d}"))
                sync_audio_with_timestamps(tts_path, ts_list, synced_path,
                                           status_cb=_cb)

                _history_record(audio_path, outdir, base, language,
                                e.get("source", "single"))
                _st(f"✔ Re-dub {rev:02d} complete → "
                    f"{os.path.basename(synced_path)}", TR_ACCENT)
                self.root.after(0, self._history_refresh)
                self.root.after(0, lambda b=base, l=language:
                    self._show_run_result(b, l))

            except Exception as exc:
                import traceback
                err, tb = str(exc), traceback.format_exc()
                _st(f"Error: {err[:90]}", "#f87171")
                self.root.after(0, self._ensure_window_visible)
                self.root.after(0, lambda e2=err, t2=tb: messagebox.showerror(
                    "Re-Dub Error", f"{e2}\n\n{t2[:600]}"))
            finally:
                def _fin():
                    self._hist_redub_running = False
                    self._hist_redub_btn.config(state="normal")
                self.root.after(0, _fin)

        threading.Thread(target=_worker, daemon=True).start()

    def _build_captions_panel(self, parent):
        """
        Captions panel: re-chunks the latest synced Bengali SRT into short
        single-line cues (default ≤10 chars / ≤1 sec) and exports them.
        """
        cp = tk.Frame(parent, bg=PANEL2, bd=0,
                      highlightbackground=PANEL_BORDER, highlightthickness=1)
        cp.pack(fill="x", side="top", pady=(6, 0))

        hdr = tk.Frame(cp, bg=PANEL2, height=28)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        _cap_lbl = tk.Label(hdr, text="  BENGALI CAPTIONS", bg=PANEL2, fg=ACCENT,
                 font=(MONO_FONT, 9, "bold"))
        _cap_lbl.pack(side="left", padx=14, pady=4)
        self._register_lang_label(_cap_lbl, "  {LANG} CAPTIONS")
        tk.Label(hdr,
                 text="(re-chunk synced SRT for burned-in captions / reels)",
                 bg=PANEL2, fg=TEXT_FAINT,
                 font=(MONO_FONT, 8, "italic")).pack(side="left", padx=8, pady=4)

        body = tk.Frame(cp, bg=PANEL2, height=44)
        body.pack(fill="x")
        body.pack_propagate(False)

        tk.Label(body, text="Max chars/cue:", bg=PANEL2, fg=TEXT,
                 font=(MONO_FONT, 9)).pack(side="left", padx=(14, 4), pady=10)
        self._cap_max_chars = tk.IntVar(value=10)
        tk.Spinbox(body, from_=1, to=80, increment=1,
                   textvariable=self._cap_max_chars, width=5,
                   bg=INPUT_BG, fg=INPUT_FG, insertbackground=INPUT_FG,
                   relief="flat", font=(MONO_FONT, 10),
                   buttonbackground=BTN_BG).pack(side="left", padx=(0, 14))

        tk.Label(body, text="Max secs/cue:", bg=PANEL2, fg=TEXT,
                 font=(MONO_FONT, 9)).pack(side="left", padx=(0, 4))
        self._cap_max_secs = tk.DoubleVar(value=1.0)
        tk.Spinbox(body, from_=0.2, to=10.0, increment=0.1, format="%.1f",
                   textvariable=self._cap_max_secs, width=5,
                   bg=INPUT_BG, fg=INPUT_FG, insertbackground=INPUT_FG,
                   relief="flat", font=(MONO_FONT, 10),
                   buttonbackground=BTN_BG).pack(side="left", padx=(0, 14))

        tk.Label(body, text="Lines:", bg=PANEL2, fg=TEXT,
                 font=(MONO_FONT, 9)).pack(side="left", padx=(0, 4))
        # Single-line is the only mode currently supported — shown read-only
        # so the constraint is visible to the user.
        tk.Label(body, text="1 (single-line)", bg=INPUT_BG, fg=INPUT_FG,
                 font=(MONO_FONT, 9), padx=8, pady=2,
                 relief="flat").pack(side="left", padx=(0, 14))

        self._btn(body, "▶ Generate Captions", self._captions_generate,
                  bg=TR_ACCENT, fg="#052e16", abg="#16a34a").pack(
            side="left", padx=(0, 6), pady=6)

        self._btn(body, "Export SRT…", self._captions_export,
                  bg="#172554", fg=REG_LABEL, abg="#1e3a8a").pack(
            side="left", padx=(0, 12), pady=6)

        self._cap_status = tk.Label(
            body, text="Run a sync first — the latest synced SRT will be re-chunked.",
            bg=PANEL2, fg=TEXT_FAINT, font=(MONO_FONT, 9), anchor="w")
        self._cap_status.pack(side="left", fill="x", expand=True, padx=8)

        # Holds the most-recent caption SRT string (in memory) so Export
        # doesn't have to regenerate.
        self._cap_last_srt: str = ""

    # ── Captions: generate + export ──────────────────────────────────────────
    def _captions_source_srt(self) -> Tuple[str, str]:
        """
        Locate the most recent synced Bengali SRT.

        Returns (srt_text, source_path). Raises ValueError with a friendly
        message if nothing is available yet.
        """
        path = getattr(self, "_last_synced_srt_path", None)
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read(), path
        raise ValueError(
            "No synced SRT found yet. Run the pipeline (Sync Audio / "
            "Run Pipeline) at least once so the captions can be re-chunked.")

    def _captions_params(self) -> Tuple[int, float]:
        try:
            mc = max(1, int(self._cap_max_chars.get()))
        except Exception:
            mc = 10
        try:
            ms = float(self._cap_max_secs.get())
        except Exception:
            ms = 1.0
        ms = max(0.2, min(ms, 30.0))
        return mc, ms

    def _captions_generate(self):
        try:
            src_text, src_path = self._captions_source_srt()
        except Exception as e:
            self._cap_status.config(text=str(e), fg="#f87171")
            return

        mc, ms = self._captions_params()
        try:
            srt_text = build_caption_srt(src_text, max_chars=mc, max_secs=ms)
        except Exception as e:
            self._cap_status.config(text=f"Error: {e}", fg="#f87171")
            return

        if not srt_text.strip():
            self._cap_last_srt = ""
            self._cap_status.config(
                text="No cues produced — check the source SRT.", fg="#f87171")
            return

        self._cap_last_srt = srt_text
        cue_count = srt_text.count(" --> ")
        self._cap_status.config(
            text=(f"✔ {cue_count} caption cue(s) generated "
                  f"(≤{mc} chars · ≤{ms:.1f}s) from "
                  f"{os.path.basename(src_path)}"),
            fg=TR_ACCENT)

    def _captions_export(self):
        # If the user clicks Export without Generate first, generate on demand
        # so the workflow is one click.
        if not getattr(self, "_cap_last_srt", "").strip():
            self._captions_generate()
            if not getattr(self, "_cap_last_srt", "").strip():
                return

        # Suggest a filename next to the source.
        src_path = getattr(self, "_last_synced_srt_path", "") or ""
        initdir = os.path.dirname(src_path) if src_path else ""
        initname = ""
        if src_path:
            base = os.path.splitext(os.path.basename(src_path))[0]
            base = base.replace("_sync_synced", "")
            mc, ms = self._captions_params()
            initname = f"{base}_captions_{mc}c_{ms:.1f}s.srt"

        out_path = filedialog.asksaveasfilename(
            title=f"Save {self._current_language()} captions SRT as…",
            defaultextension=".srt",
            initialdir=initdir or None,
            initialfile=initname or None,
            filetypes=[("SubRip subtitle", "*.srt"), ("All files", "*.*")])
        if not out_path:
            return
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(self._cap_last_srt)
        except Exception as e:
            self._cap_status.config(text=f"Save failed: {e}", fg="#f87171")
            return
        self._cap_status.config(
            text=f"✔ Saved → {os.path.basename(out_path)}", fg=TR_ACCENT)

    def _as_build_section(self, parent, side, title, colour, panel_bg,
                          thr_default, hys_default, min_default):
        """Build one half (English or dubbed audio) of the Audio Syncing tab."""
        st = self._as_state[side]

        # ── File picker row ───────────────────────────────────────────────────
        section_border = TEXT_MUTED if side == "en" else "#1e3a52"
        fp = tk.Frame(parent, bg=panel_bg, height=38, bd=0,
                      highlightbackground=section_border, highlightthickness=1)
        fp.pack(fill="x", side="top", pady=(6, 0))
        fp.pack_propagate(False)

        _title_lbl = tk.Label(fp, text=title, bg=panel_bg, fg=colour,
                 font=(MONO_FONT, 9, "bold"))
        _title_lbl.pack(side="left", padx=(10, 8), pady=8)
        if side == "bn":
            self._register_lang_label(_title_lbl, "{LANG} AUDIO")

        self._btn(fp, "Open Audio", lambda s=side: self._as_pick_file(s),
                  bg="#172554", fg=REG_LABEL, abg="#1e3a8a"
                  ).pack(side="left", padx=(0, 8), pady=5)

        self._btn(fp, "Reset View", lambda s=side: self._as_reset_view(s),
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT
                  ).pack(side="left", padx=(0, 8), pady=5)

        st["file_label"] = tk.Label(fp, text="No file loaded",
                                    bg=panel_bg, fg=TEXT_FAINT,
                                    font=(MONO_FONT, 9))
        st["file_label"].pack(side="left", padx=4)

        st["zoom_readout"] = tk.Label(fp, text="", bg=panel_bg, fg=ACCENT,
                                      font=(MONO_FONT, 9))
        st["zoom_readout"].pack(side="right", padx=14)

        # ── Regions panel ─────────────────────────────────────────────────────
        rp = tk.Frame(parent, bg=panel_bg, height=40, bd=0,
                      highlightbackground=section_border, highlightthickness=1)
        rp.pack(fill="x", side="top")
        rp.pack_propagate(False)

        tk.Label(rp, text="Thr", bg=panel_bg, fg=TEXT,
                 font=(MONO_FONT, 9)).pack(side="left", padx=(10, 2))
        st["thr_var"] = tk.DoubleVar(value=thr_default)
        tk.Spinbox(rp, from_=-80.0, to=0.0, increment=1.0,
                   textvariable=st["thr_var"], width=6,
                   bg=INPUT_BG, fg=INPUT_FG, insertbackground=INPUT_FG,
                   relief="flat", font=(MONO_FONT, 10),
                   buttonbackground="#334155").pack(side="left", padx=(0, 8))

        tk.Label(rp, text="Hys", bg=panel_bg, fg=TEXT,
                 font=(MONO_FONT, 9)).pack(side="left", padx=(0, 2))
        st["hys_var"] = tk.DoubleVar(value=hys_default)
        tk.Spinbox(rp, from_=0.0, to=40.0, increment=0.5,
                   textvariable=st["hys_var"], width=5,
                   bg=INPUT_BG, fg=INPUT_FG, insertbackground=INPUT_FG,
                   relief="flat", font=(MONO_FONT, 10),
                   buttonbackground="#334155").pack(side="left", padx=(0, 8))

        tk.Label(rp, text="MinSil", bg=panel_bg, fg=TEXT,
                 font=(MONO_FONT, 9)).pack(side="left", padx=(0, 2))
        st["min_var"] = tk.IntVar(value=min_default)
        tk.Spinbox(rp, from_=10, to=5000, increment=10,
                   textvariable=st["min_var"], width=6,
                   bg=INPUT_BG, fg=INPUT_FG, insertbackground=INPUT_FG,
                   relief="flat", font=(MONO_FONT, 10),
                   buttonbackground="#334155").pack(side="left", padx=(0, 8))

        self._btn(rp, "Re-Apply", lambda s=side: self._as_apply_regions(s),
                  bg="#172554", fg=REG_LABEL, abg="#1e3a8a"
                  ).pack(side="left", padx=(0, 4), pady=4)
        self._btn(rp, "Clear", lambda s=side: self._as_clear_regions(s),
                  bg="#1f1213", fg="#f87171", abg="#3a1414"
                  ).pack(side="left", padx=(0, 8), pady=4)

        st["region_count_label"] = tk.Label(
            rp, text="", bg=panel_bg, fg=REG_LABEL, font=(MONO_FONT, 9))
        st["region_count_label"].pack(side="left", padx=4)

        tk.Label(rp, text="(Ctrl+wheel = zoom · wheel = scroll)",
                 bg=panel_bg, fg=TEXT_FAINT,
                 font=(MONO_FONT, 8, "italic")).pack(side="right", padx=10)

        # ── Waveform ──────────────────────────────────────────────────────────
        wf_wrap = tk.Frame(parent, bg=BG, height=160)
        wf_wrap.pack(fill="x", side="top", padx=8, pady=(2, 0))
        wf_wrap.pack_propagate(False)

        fig, ax = plt.subplots(figsize=(10, 1.5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors=TEXT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(True, axis="x", color=GRID, linewidth=0.5,
                linestyle="--", alpha=0.6)
        ax.text(0.5, 0.5,
                f"Open {'English' if side == 'en' else 'dubbed'} audio — "
                "waveform loads here",
                ha="center", va="center", transform=ax.transAxes,
                color=TEXT_MUTED, fontsize=10, fontfamily="monospace")
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=wf_wrap)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.get_tk_widget().configure(bg=BG)
        canvas.draw()

        # Mouse-wheel: Ctrl held = zoom, no Ctrl = scroll
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>", "<Shift-MouseWheel>", "<Shift-Button-4>", "<Shift-Button-5>", "<Button-6>", "<Button-7>"):
            canvas.get_tk_widget().bind(
                seq, lambda e, s=side: self._as_on_waveform_scroll(e, s))

        st["fig"]    = fig
        st["ax"]     = ax
        st["canvas"] = canvas

        # ── Scrollbar canvas (drag thumb to scroll) ───────────────────────────
        sb = tk.Canvas(parent, height=14, bg="#334155",
                       highlightthickness=0, cursor="hand2")
        sb.pack(fill="x", side="top", padx=8, pady=(1, 4))
        sb.bind("<Configure>",
                lambda e, s=side: self._as_sb_redraw(s))
        sb.bind("<ButtonPress-1>",
                lambda e, s=side: self._as_sb_on_press(e, s))
        sb.bind("<B1-Motion>",
                lambda e, s=side: self._as_sb_on_drag(e, s))
        sb.bind("<ButtonRelease-1>",
                lambda e, s=side: self._as_sb_on_release(e, s))
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>", "<Shift-MouseWheel>", "<Shift-Button-4>", "<Shift-Button-5>", "<Button-6>", "<Button-7>"):
            sb.bind(seq, lambda e, s=side: self._as_on_waveform_scroll(e, s))
        st["sb_canvas"] = sb

    # ─────────────────────────────────────────────────────────────────────────
    #  Audio Syncing — file load, region detection, waveform render
    # ─────────────────────────────────────────────────────────────────────────
    def _as_pick_file(self, side):
        path = filedialog.askopenfilename(
            title=(f"Open "
                   f"{'English' if side == 'en' else self._current_language()}"
                   f" audio file"),
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg *.aiff *.aif *.m4a"),
                       ("All Files", "*.*")])
        if not path:
            return
        st = self._as_state[side]
        st["filepath"] = path
        short = os.path.basename(path)
        st["file_label"].config(text=short, fg=TEXT)
        st["region_count_label"].config(text="Loading…")
        threading.Thread(target=self._as_load_audio, args=(side,), daemon=True).start()

    def _as_load_audio(self, side):
        st = self._as_state[side]
        try:
            y, sr = librosa.load(st["filepath"], sr=None, mono=True)
            st["audio"]      = y
            st["sr"]         = sr
            st["dur"]        = len(y) / sr
            st["regions"]    = []
            st["zoom_val"]   = 50.0
            st["scroll_pos"] = 0.0
            self.root.after(0, lambda s=side: self._as_render_waveform(s))
            self.root.after(50, lambda s=side: self._as_apply_regions(s))
        except Exception as e:
            err = str(e)
            self.root.after(0, lambda m=err:
                messagebox.showerror("Load Error", m))
            self.root.after(0, lambda:
                st["region_count_label"].config(text="Load failed"))

    def _as_apply_regions(self, side):
        st = self._as_state[side]
        if st["audio"] is None:
            messagebox.showwarning("No audio",
                f"Open the {'English' if side == 'en' else 'Bengali'} audio file first.")
            return
        try:
            thr = float(st["thr_var"].get())
            hys = float(st["hys_var"].get())
            ms  = int(st["min_var"].get())
        except ValueError:
            return

        def _detect():
            regions = _detect_regions_from_audio(st["audio"], st["sr"], thr, hys, ms)
            st["regions"] = regions
            n = len(regions)
            self.root.after(0, lambda s=side: self._as_render_waveform(s))
            self.root.after(0, lambda c=n:
                st["region_count_label"].config(
                    text=f"{c} region{'s' if c != 1 else ''} found"))

        threading.Thread(target=_detect, daemon=True).start()

    def _as_clear_regions(self, side):
        st = self._as_state[side]
        st["regions"] = []
        st["region_count_label"].config(text="")
        self._as_render_waveform(side)

    def _as_render_waveform(self, side):
        st = self._as_state[side]
        if st["audio"] is None or st["fig"] is None:
            return
        ax = st["ax"]; fig = st["fig"]
        ax.clear()
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors=TEXT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(True, axis="x", color=GRID, linewidth=0.5,
                linestyle="--", alpha=0.6)

        y, sr, dur = st["audio"], st["sr"], st["dur"]

        # Compute visible window from zoom + scroll
        visible    = min(self._slider_to_seconds(st["zoom_val"]), dur)
        scrollable = max(0.0, dur - visible)
        start_t    = max(0.0, min(st["scroll_pos"] * scrollable, scrollable))
        end_t      = start_t + visible
        i0         = int(start_t * sr)
        i1         = min(int(end_t * sr), len(y))
        chunk      = y[i0:i1]
        n          = len(chunk)
        if n == 0:
            st["canvas"].draw()
            self._as_sb_redraw(side)
            return

        # Region overlays — only draw the ones intersecting the visible range
        for idx, (rs, re) in enumerate(st["regions"]):
            if re < start_t or rs > end_t:
                continue
            rx0 = max(rs, start_t); rx1 = min(re, end_t)
            ax.add_patch(Rectangle(
                (rx0, -1.05), rx1 - rx0, 2.10,
                facecolor=REG_FILL, edgecolor="none", alpha=0.55, zorder=1))
            if rs >= start_t:
                ax.axvline(rs, color=REG_EDGE, linewidth=1.0, alpha=0.9, zorder=3)
            if re <= end_t:
                ax.axvline(re, color=REG_EDGE, linewidth=1.0, alpha=0.9, zorder=3)
            lx = max(rx0, start_t) + (rx1 - rx0) * 0.02
            ax.text(lx, 1.01, f"R{idx+1}  {rs:.2f}s",
                    color=REG_LABEL, fontsize=7,
                    va="bottom", fontfamily="monospace", zorder=4,
                    transform=ax.get_xaxis_transform(), clip_on=True)

        # Down-sample the visible slice for plotting
        TARGET = 1500
        if n > TARGET:
            step    = max(1, n // TARGET)
            frames  = n // step
            trimmed = chunk[:frames*step].reshape(frames, step)
            peaks_p = trimmed.max(axis=1)
            peaks_n = trimmed.min(axis=1)
            t_axis  = np.linspace(start_t, end_t, frames)
        else:
            peaks_p = chunk; peaks_n = chunk
            t_axis  = np.linspace(start_t, end_t, n)

        ax.fill_between(t_axis, peaks_p, peaks_n,
                        color=WAVEFORM, alpha=0.85, linewidth=0, zorder=2)
        ax.axhline(0, color=TEXT_MUTED, linewidth=0.6, zorder=2)

        ax.set_xlim(start_t, end_t)
        ax.set_ylim(-1.05, 1.05)

        def fmt_time(x, _):
            m, s = divmod(x, 60)
            return f"{int(m)}:{s:05.2f}" if m else f"{s:.1f}s"
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_time))

        fname = os.path.basename(st["filepath"]) if st["filepath"] else ""
        ax.set_title(f"{fname}  ·  {dur:.2f}s  ·  {sr} Hz",
                     color=TEXT, fontsize=8, pad=4, fontfamily="monospace")

        # Update zoom readout (visible window in seconds)
        if st.get("zoom_readout") is not None:
            st["zoom_readout"].config(text=f"View: {visible:.1f}s")

        fig.tight_layout()
        st["canvas"].draw()
        self._as_sb_redraw(side)

    # ── Audio Syncing — zoom / scroll / scrollbar handlers ───────────────────
    def _as_on_waveform_scroll(self, event, side):
        st = self._as_state[side]
        if st["audio"] is None:
            return
        ctrl_held = bool(getattr(event, "state", 0) & 0x4)
        num   = getattr(event, "num", 0)
        delta = getattr(event, "delta", 0)
        up = (delta > 0) or (num in (4, 6))
        if ctrl_held:
            self._as_zoom_by_delta(5.0 if up else -5.0, side)
        else:
            self._as_scroll_by(-0.05 if up else 0.05, side)

    def _as_zoom_by_delta(self, delta, side):
        st = self._as_state[side]
        st["zoom_val"] = max(0.0, min(100.0, st["zoom_val"] + delta))
        self._as_render_waveform(side)

    def _as_scroll_by(self, delta, side):
        st = self._as_state[side]
        st["scroll_pos"] = max(0.0, min(1.0, st["scroll_pos"] + delta))
        self._as_render_waveform(side)

    def _as_reset_view(self, side):
        st = self._as_state[side]
        st["zoom_val"]   = 50.0
        st["scroll_pos"] = 0.0
        self._as_render_waveform(side)

    def _as_sb_thumb_geometry(self, side):
        st = self._as_state[side]
        sb = st.get("sb_canvas")
        if sb is None:
            return 0, 0
        w = sb.winfo_width()
        if w < 2 or st["dur"] <= 0:
            return 0, w
        visible    = min(self._slider_to_seconds(st["zoom_val"]), st["dur"])
        ratio      = visible / max(st["dur"], 0.001)
        thumb_w    = max(20, int(w * ratio))
        scrollable = max(1, w - thumb_w)
        x0         = int(st["scroll_pos"] * scrollable)
        return x0, x0 + thumb_w

    def _as_sb_redraw(self, side):
        st = self._as_state[side]
        sb = st.get("sb_canvas")
        if sb is None:
            return
        w = sb.winfo_width(); h = sb.winfo_height()
        if w < 2:
            return
        sb.delete("all")
        sb.create_rectangle(0, 0, w, h, fill="#334155", outline="")
        x0, x1 = self._as_sb_thumb_geometry(side)
        sb.create_rectangle(x0+1, 2, x1-1, h-2,
                            fill=TEXT_MUTED, outline=TEXT_MUTED, width=1)

    def _as_sb_on_press(self, event, side):
        st = self._as_state[side]
        if st["audio"] is None:
            return
        sb = st["sb_canvas"]
        x0, x1 = self._as_sb_thumb_geometry(side)
        if x0 <= event.x <= x1:
            st["sb_drag_start_x"]   = event.x
            st["sb_drag_start_pos"] = st["scroll_pos"]
        else:
            w = sb.winfo_width()
            visible  = min(self._slider_to_seconds(st["zoom_val"]),
                           max(st["dur"], 0.001))
            thumb_w  = max(20, int(w * (visible / max(st["dur"], 0.001))))
            scrollable_px = max(1, w - thumb_w)
            new_pos  = max(0.0, min(1.0, (event.x - thumb_w/2) / scrollable_px))
            st["scroll_pos"] = new_pos
            st["sb_drag_start_x"]   = event.x
            st["sb_drag_start_pos"] = new_pos
            self._as_render_waveform(side)

    def _as_sb_on_drag(self, event, side):
        st = self._as_state[side]
        if st["sb_drag_start_x"] is None or st["audio"] is None:
            return
        sb = st["sb_canvas"]
        w = sb.winfo_width()
        visible  = min(self._slider_to_seconds(st["zoom_val"]),
                       max(st["dur"], 0.001))
        thumb_w  = max(20, int(w * (visible / max(st["dur"], 0.001))))
        scrollable_px = max(1, w - thumb_w)
        dx = event.x - st["sb_drag_start_x"]
        new_pos = max(0.0, min(1.0,
                               st["sb_drag_start_pos"] + dx / scrollable_px))
        st["scroll_pos"] = new_pos
        self._as_render_waveform(side)

    def _as_sb_on_release(self, _event, side):
        self._as_state[side]["sb_drag_start_x"] = None

    # ─────────────────────────────────────────────────────────────────────────
    #  Audio Syncing — Sync worker (S3a → S3e)
    # ─────────────────────────────────────────────────────────────────────────
    def _as_set_stage(self, tag, msg, colour=None):
        var = self._as_stage_vars.get(tag)
        lbl = self._as_stage_labels.get(tag)
        if var and lbl:
            var.set(msg)
            if colour:
                lbl.config(fg=colour)

    def _as_run_sync(self):
        if self._as_running:
            return
        en = self._as_state["en"]
        te = self._as_state["bn"]

        if en["audio"] is None or not en["filepath"]:
            messagebox.showwarning("Missing English Audio",
                "Open an English audio file first.")
            return
        if te["audio"] is None or not te["filepath"]:
            messagebox.showwarning(
                f"Missing {self._current_language()} Audio",
                f"Open a {self._current_language()} (dubbed) audio file first.")
            return
        if not en["regions"]:
            messagebox.showwarning("No English Regions",
                "No regions detected in English audio. Adjust threshold / Re-Apply.")
            return
        if not te["regions"]:
            messagebox.showwarning(f"No {self._current_language()} Regions",
                "No regions detected in target-language audio. Adjust threshold / Re-Apply.")
            return

        script_text = self._as_script_text.get("1.0", "end").strip()
        if not script_text:
            messagebox.showwarning("Missing Script",
                f"Paste the {self._current_language()} dubbing script "
                f"in the text box first.")
            return

        # Validate API keys
        try:
            _get_api_key()
        except Exception as e:
            messagebox.showerror("ElevenLabs API Key Error", str(e))
            return
        try:
            _validate_llm_config()
        except Exception as e:
            messagebox.showerror("LLM Provider Error", str(e))
            return

        # Snapshot inputs for the worker thread
        en_path    = en["filepath"]
        te_path    = te["filepath"]
        en_regions = list(en["regions"])
        te_regions = list(te["regions"])
        # Capture English audio duration (in SECONDS) up-front so the worker
        # thread doesn't have to touch Tk state. Used by Stage S3d's overflow
        # logic. We prefer the cached `dur` (computed in _as_load_audio when
        # the file was opened) and fall back to recomputing from audio/sr.
        # If both are unavailable, run_sync_from_strings will fall back to
        # the EN-SRT end length — never silently zero.
        try:
            cached = float(en.get("dur") or 0.0)
            if cached > 0:
                en_audio_duration = cached
            elif en.get("audio") is not None and en.get("sr"):
                en_audio_duration = float(len(en["audio"])) / float(en["sr"])
            else:
                en_audio_duration = 0.0
        except Exception:
            en_audio_duration = 0.0

        # Reset stage indicators
        for tag in self._as_stage_vars:
            self._as_set_stage(tag, "—", TEXT_MUTED)

        self._as_running = True
        self._as_run_btn.config(state="disabled")
        self._as_status.config(text="Syncing — please wait…", fg=TEXT_FAINT)

        def _worker():
            try:
                api_key = _get_api_key()

                # Output folder = per-file folder next to the English audio
                outdir = _prepare_output_dir(en_path)
                en_base_name = os.path.splitext(os.path.basename(en_path))[0]
                base = os.path.join(outdir, en_base_name)

                pipeline_language = self._current_language()

                # Also copy the user-provided target-language audio into the same folder
                try:
                    dst_te = os.path.join(outdir, os.path.basename(te_path))
                    if (os.path.abspath(te_path) != os.path.abspath(dst_te)
                            and not os.path.exists(dst_te)):
                        shutil.copy2(te_path, dst_te)
                except Exception:
                    pass

                # Save the user-provided script so the run is reproducible
                try:
                    with open(base + "_FinalScript.txt", "w", encoding="utf-8") as f:
                        f.write(f"=== {pipeline_language.upper()} TRANSLATION ===\n"
                                + script_text)
                    _history_record(en_path, outdir, base, pipeline_language,
                                    "audio-sync")
                except Exception:
                    pass

                # ── Stage 3a: English SRT ────────────────────────────────────
                self.root.after(0, lambda:
                    self._as_set_stage("S3a", "Transcribing English…", "#d97706"))
                en_result = _transcribe_audio(en_path, api_key)
                en_words  = en_result.get("words", [])
                if not en_words:
                    raise ValueError("No word data from ElevenLabs for English audio.")
                en_srt = _build_english_subtitle_srt(en_regions, en_words)
                en_srt_path = base + "_sync_en.srt"
                with open(en_srt_path, "w", encoding="utf-8") as f:
                    f.write(en_srt)
                self.root.after(0, lambda:
                    self._as_set_stage("S3a", "✔ Done", "#22c55e"))

                # ── Stage 3b: Target-language SRT ────────────────────────────
                self.root.after(0, lambda lang=pipeline_language:
                    self._as_set_stage("S3b", f"Transcribing {lang}…", "#d97706"))
                te_result = _transcribe_audio(te_path, api_key)
                te_words  = te_result.get("words", [])
                if not te_words:
                    raise ValueError(
                        f"No word data from ElevenLabs for {pipeline_language} audio.")
                te_srt = _build_target_subtitle_srt(te_regions, te_words)
                te_srt_path = base + "_sync_te.srt"
                with open(te_srt_path, "w", encoding="utf-8") as f:
                    f.write(te_srt)
                self.root.after(0, lambda:
                    self._as_set_stage("S3b", "✔ Done", "#22c55e"))

                # ── Stage 3c: Gemini SRT mapping ─────────────────────────────
                self.root.after(0, lambda:
                    self._as_set_stage("S3c", "Calling Gemini…", "#d97706"))
                mapping_text = _call_gemini_mapping(en_srt, te_srt, script_text,
                                                     self._gemini_model_var.get(),
                                                     language=pipeline_language)
                mapping_path = base + "_sync_mapping.txt"
                with open(mapping_path, "w", encoding="utf-8") as f:
                    f.write(mapping_text)
                self.root.after(0, lambda:
                    self._as_set_stage("S3c", "✔ Done", "#22c55e"))

                # ── Stage 3d: Sync SRTs ──────────────────────────────────────
                self.root.after(0, lambda:
                    self._as_set_stage("S3d", "Syncing…", "#d97706"))
                synced_subs, orig_te_subs, sync_log = run_sync_from_strings(
                    en_srt, te_srt, mapping_text,
                    en_audio_duration=en_audio_duration)
                with open(base + "_sync_log.txt", "w", encoding="utf-8") as _f:
                    _f.write(sync_log)
                # Save synced Bengali SRT (used by the Captions exporter).
                try:
                    synced_srt_text = _write_srt_from_dict(synced_subs)
                    synced_srt_path = base + "_sync_synced.srt"
                    with open(synced_srt_path, "w", encoding="utf-8") as _f:
                        _f.write(synced_srt_text)
                    self.root.after(
                        0, lambda p=synced_srt_path: setattr(
                            self, "_last_synced_srt_path", p))
                except Exception:
                    pass
                ts_list = _build_timestamps(orig_te_subs, synced_subs)
                with open(base + "_sync_timestamps.txt", "w", encoding="utf-8") as f:
                    f.write(_format_timestamps_as_text(ts_list))
                self.root.after(0, lambda n=len(synced_subs):
                    self._as_set_stage("S3d", f"✔ {n} subtitles synced", "#22c55e"))

                # ── Stage 3e: Build synced audio ─────────────────────────────
                self.root.after(0, lambda:
                    self._as_set_stage("S3e", "Creating audio…", "#d97706"))
                synced_name = _tts_output_name(pipeline_language, en_path, "_synced")
                synced_path = os.path.join(outdir, synced_name)

                def _sync_status_cb(msg):
                    self.root.after(0, lambda m=msg:
                        self._as_set_stage("S3e", m, "#d97706"))

                sync_audio_with_timestamps(te_path, ts_list, synced_path,
                                           status_cb=_sync_status_cb)
                self.root.after(0, lambda n=synced_name:
                    self._as_set_stage("S3e", f"✔ Saved: {n}", "#22c55e"))

                self.root.after(0, lambda p=synced_path:
                    self._as_status.config(
                        text=f"All stages complete ✓ Synced audio → {p}",
                        fg=TR_ACCENT))
                self.root.after(0, lambda b=base, l=pipeline_language:
                    self._show_run_result(b, l))

            except Exception as exc:
                import traceback
                err = str(exc)
                tb  = traceback.format_exc()
                self.root.after(0, lambda e=err:
                    self._as_status.config(text=f"Error: {e[:80]}", fg="#f87171"))
                self.root.after(0, self._ensure_window_visible)
                self.root.after(0, lambda e=err, t=tb:
                    messagebox.showerror("Audio Syncing Error",
                                         f"{e}\n\n{t[:600]}"))
            finally:
                self.root.after(0, self._as_finish_run)

        threading.Thread(target=_worker, daemon=True).start()

    def _as_finish_run(self):
        self._as_running = False
        self._as_run_btn.config(state="normal")

    # ─────────────────────────────────────────────────────────────────────────
    #  Pipeline run helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _set_stage(self, tag, msg, colour=None):
        var = self._stage_vars.get(tag)
        lbl = self._stage_labels.get(tag)
        if var and lbl:
            self.root.after(0, lambda: var.set(msg))
            if colour:
                self.root.after(0, lambda: lbl.config(fg=colour))
        # Mirror progress onto the bottom stage bar
        try:
            self.root.after(0, lambda: self._stage_bar_pulse(tag, msg))
        except Exception:
            pass
        # Mirror every stage update into the Log view (dedup repeats)
        try:
            key = (tag, str(msg))
            if str(msg).strip() not in ("", "—") and \
                    key != getattr(self, "_sf_last_log_key", None):
                self._sf_last_log_key = key
                self._sf_log_append(f"[{tag}] {msg}")
        except Exception:
            pass

    def _cancel_pipeline(self):
        self._pipeline_cancel.set()
        self.btn_cancel_pipeline.config(state="disabled")
        self.tr_status.config(text="Cancelling…", fg="#f87171")
        self.status.config(text="Cancel requested — stopping after current step…")

    # ─────────────────────────────────────────────────────────────────────────
    #  Manual translation review (pipeline pause before dubbing)
    # ─────────────────────────────────────────────────────────────────────────
    def _show_translation_review(self, en_entries, tr_paragraphs, language,
                                 result_holder, done_event,
                                 audio=None, sr=None, audio_path=None):
        """
        Modal side-by-side review of the translation before dubbing.

        Left pane: the full English transcription in one read-only box.
        Right pane: the full translation in one editable box. When *audio*
        (numpy array) and *sr* are given, a Play/Pause/Stop bar plays the
        whole English audio in one go, highlighting the segment currently
        being spoken (double-click a paragraph to jump there). A video file
        can be imported into a third pane; it plays muted, frame-synced to
        the same clock, so the pipeline audio is the soundtrack. Must be
        called on the Tk main thread; the pipeline worker blocks on
        *done_event*.

        Fills *result_holder* with:
            action — "skip" (dub the script unchanged) or
                     "continue" (dub the edited text)
            text   — the edited script when action == "continue"
        """
        # The pipeline opens this dialog programmatically — often while the
        # user has the app minimized during the long LLM wait. On Windows a
        # transient child of an iconified window is created hidden, and
        # grab_set() on a hidden window raises "grab failed: window not
        # viewable", leaving a half-built dialog + invisible modal that
        # blocks the whole app from being restored. So: restore the main
        # window first, and only grab once the dialog is actually viewable.
        self._ensure_window_visible()

        win = tk.Toplevel(self.root)
        result_holder["win"] = win
        win.title(f"Review Translation — {language}")
        win.configure(bg=BG)
        win.geometry("1200x800")
        win.minsize(760, 480)
        win.transient(self.root)

        def _try_grab(attempt=0):
            try:
                win.grab_set()
            except tk.TclError:
                # Not viewable yet — Windows maps it a few ticks later.
                # Retry briefly; if it never becomes grabbable, the dialog
                # simply runs non-modal instead of crashing.
                if attempt < 20 and win.winfo_exists():
                    win.after(100, lambda: _try_grab(attempt + 1))
        win.after(50, _try_grab)

        lang_up = language.upper()

        # ── Header ────────────────────────────────────────────────────────
        hdr = tk.Frame(win, bg=PANEL, height=46)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text=f"REVIEW {lang_up} TRANSLATION", bg=PANEL,
                 fg=TR_ACCENT, font=(MONO_FONT, 10, "bold")
                 ).pack(side="left", padx=14, pady=12)
        tk.Label(hdr,
                 text=f"English is read-only — edit the {language} text on "
                      "the right, then Continue. Skip dubs the script "
                      "unchanged.",
                 bg=PANEL, fg=TEXT_FAINT, font=(MONO_FONT, 9)
                 ).pack(side="left", padx=8)

        # ── Footer (packed before the middle so it never collapses) ───────
        ftr = tk.Frame(win, bg=PANEL, height=52)
        ftr.pack(fill="x", side="bottom")
        ftr.pack_propagate(False)

        status_lbl = tk.Label(
            ftr,
            text=(f"{len(en_entries)} English segment(s) · "
                  f"{len(tr_paragraphs)} {language} paragraph(s)"),
            bg=PANEL, fg=TEXT_FAINT, font=(MONO_FONT, 9))
        status_lbl.pack(side="left", padx=14)

        has_audio = audio is not None and sr

        rows = (_pair_review_rows(en_entries, tr_paragraphs)
                or [("", "", None, None)])

        # ── Playback / video toolbar (widgets created after the panes) ────
        bar = tk.Frame(win, bg=BG)
        bar.pack(fill="x", padx=14, pady=(8, 0))

        # ── Full-width panes: [video] | English | translation ────────────
        paned = tk.PanedWindow(win, orient="horizontal", bg=BG,
                               sashwidth=6, bd=0, relief="flat")
        paned.pack(fill="both", expand=True, padx=14, pady=6)

        def _make_pane(title, fg):
            holder = tk.Frame(paned, bg=BG)
            tk.Label(holder, text=title, bg=BG, fg=fg,
                     font=(MONO_FONT, 9, "bold"), anchor="w"
                     ).pack(fill="x", pady=(0, 4))
            body = tk.Frame(holder, bg=BG)
            body.pack(fill="both", expand=True)
            return holder, body

        # Video pane — built up-front but only added to the PanedWindow
        # once a video is imported.
        vid_holder, vid_body = _make_pane(
            "VIDEO  (muted — pipeline audio is the soundtrack)", REG_LABEL)
        vid_name = tk.Label(vid_holder, text="", bg=BG, fg=TEXT_FAINT,
                            font=(MONO_FONT, 8), anchor="w")
        vid_name.pack(fill="x", side="bottom")
        vid_lbl = tk.Label(vid_body, bg="black", anchor="center")
        vid_lbl.pack(fill="both", expand=True)

        en_holder, en_body = _make_pane(
            "ENGLISH TRANSCRIPTION"
            + ("  (double-click a paragraph to play from it)"
               if has_audio else ""),
            REG_LABEL)
        en_box = tk.Text(en_body, wrap="word", bg=PANEL2, fg=TEXT,
                         relief="flat", bd=0, padx=10, pady=8,
                         font=(MONO_FONT, 10), highlightthickness=1,
                         highlightbackground=PANEL_BORDER)
        en_vsb = ttk.Scrollbar(en_body, orient="vertical",
                               command=en_box.yview,
                               style="Vertical.TScrollbar")
        en_box.configure(yscrollcommand=en_vsb.set)
        en_vsb.pack(side="right", fill="y")
        en_box.pack(side="left", fill="both", expand=True)

        tr_holder, tr_body = _make_pane(
            f"{lang_up} TRANSLATION (editable)", ACCENT)
        tr_box = tk.Text(tr_body, wrap="word", bg=INPUT_BG, fg=INPUT_FG,
                         insertbackground=INPUT_FG, relief="flat", bd=0,
                         padx=10, pady=8, font=(MONO_FONT, 11),
                         highlightthickness=1,
                         highlightbackground=PANEL_BORDER, undo=True)
        tr_vsb = ttk.Scrollbar(tr_body, orient="vertical",
                               command=tr_box.yview,
                               style="Vertical.TScrollbar")
        tr_box.configure(yscrollcommand=tr_vsb.set)
        tr_vsb.pack(side="right", fill="y")
        tr_box.pack(side="left", fill="both", expand=True)

        paned.add(en_holder, stretch="always", minsize=240)
        paned.add(tr_holder, stretch="always", minsize=240)

        # Fill the English pane, remembering each segment's char range so
        # playback can highlight the paragraph currently being spoken.
        seg_ranges = []          # (char_start, char_end, seg_start, seg_end)
        ofs = 0
        for i, (en_txt, _tr_txt, seg_s, seg_e) in enumerate(rows):
            if i:
                en_box.insert("end", "\n\n")
                ofs += 2
            en_box.insert("end", en_txt)
            seg_ranges.append((ofs, ofs + len(en_txt), seg_s, seg_e))
            ofs += len(en_txt)
        en_box.tag_configure("curseg", background="#1e3a8a",
                             foreground="#f8fafc")
        en_box.config(state="disabled")

        tr_box.insert("1.0", "\n\n".join(r[1] for r in rows if r[1]))
        tr_box.edit_reset()

        # ── One clock drives audio, segment highlight and video frames ────
        play = {"on": False, "pos": 0.0, "t0": 0.0, "after": None}
        vid = {"cap": None, "cv2": None, "fps": 25.0, "frames": 0,
               "last": -1, "photo": None, "offset": 0.0}
        mute_var = tk.BooleanVar(value=False)
        seek_var = tk.DoubleVar(value=0.0)   # 0..1000 across the timeline
        seek_drag = {"on": False}      # user dragging the seek slider

        def _total_dur():
            if has_audio:
                return len(audio) / float(sr)
            if vid["cap"] is not None and vid["fps"]:
                return vid["frames"] / vid["fps"]
            return 0.0

        def _cur_t():
            if play["on"]:
                return play["pos"] + (time.perf_counter() - play["t0"])
            return play["pos"]

        def _fmt_t(t):
            t = max(0, int(t))
            return f"{t // 60:02d}:{t % 60:02d}"

        def _update_time():
            t, total = _cur_t(), _total_dur()
            time_lbl.config(text=f"{_fmt_t(t)} / {_fmt_t(total)}")
            # Reflect playback position on the slider, but never fight the
            # user while they are dragging it.
            if not seek_drag["on"]:
                seek_var.set((t / total * 1000.0) if total > 0 else 0.0)

        def _cancel_tick():
            if play["after"] is not None:
                try:
                    win.after_cancel(play["after"])
                except Exception:
                    pass
                play["after"] = None

        def _highlight_seg(t):
            en_box.tag_remove("curseg", "1.0", "end")
            for cs, ce, s, e in seg_ranges:
                if s is not None and e is not None and s <= t < e:
                    i0, i1 = f"1.0 + {cs} chars", f"1.0 + {ce} chars"
                    en_box.tag_add("curseg", i0, i1)
                    en_box.see(i1)
                    en_box.see(i0)
                    break

        def _video_show(t):
            cap, cv2 = vid["cap"], vid["cv2"]
            if cap is None:
                return
            fps = vid["fps"] or 25.0
            idx = int((t + vid["offset"]) * fps)
            idx = max(0, idx)
            if vid["frames"]:
                idx = min(idx, vid["frames"] - 1)
            if idx == vid["last"]:
                return
            # Rewind or big forward jump → hard seek; otherwise read through.
            if idx < vid["last"] or idx - vid["last"] > fps * 2:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                vid["last"] = idx - 1
            frame = None
            while vid["last"] < idx:
                ok, frame = cap.read()
                if not ok:
                    return
                vid["last"] += 1
            if frame is None:
                return
            maxw = vid_lbl.winfo_width()
            maxh = vid_lbl.winfo_height()
            maxw = maxw - 8 if maxw > 60 else 420
            maxh = maxh - 8 if maxh > 60 else 320
            h, w = frame.shape[:2]
            k = min(maxw / w, maxh / h)
            if k < 1.0:
                frame = cv2.resize(
                    frame, (max(1, int(w * k)), max(1, int(h * k))),
                    interpolation=cv2.INTER_AREA)
            ppm = (b"P6 %d %d 255\n" % (frame.shape[1], frame.shape[0])
                   + frame[:, :, ::-1].tobytes())
            try:
                photo = tk.PhotoImage(data=ppm)
            except Exception:
                return
            vid["photo"] = photo        # keep a reference or tk drops it
            vid_lbl.config(image=photo)

        def _tick():
            play["after"] = None
            t, total = _cur_t(), _total_dur()
            if total and t >= total:
                _pause()
                play["pos"] = total
                _update_time()
                return
            _update_time()
            _highlight_seg(t)
            _video_show(t)
            if play["on"]:
                play["after"] = win.after(80, _tick)

        def _audio_start(t):
            if not has_audio or mute_var.get():
                return
            i0 = int(t * sr)
            if i0 >= len(audio):
                return
            try:
                sd.play(audio[i0:].astype(np.float32), samplerate=sr)
            except Exception as e:
                status_lbl.config(text=f"Playback error: {e}", fg="#f87171")

        def _play():
            if play["on"] or _total_dur() <= 0:
                return
            if play["pos"] >= _total_dur():
                play["pos"] = 0.0
            _audio_start(play["pos"])
            play["on"] = True
            play["t0"] = time.perf_counter()
            play_btn.config(text="⏸ Pause")
            _tick()

        def _pause():
            if play["on"]:
                play["pos"] = _cur_t()
            play["on"] = False
            try:
                sd.stop()
            except Exception:
                pass
            _cancel_tick()
            play_btn.config(text="▶ Play")

        def _toggle_play():
            (_pause if play["on"] else _play)()

        def _stop_all():
            _pause()
            play["pos"] = 0.0
            en_box.tag_remove("curseg", "1.0", "end")
            _update_time()
            _video_show(0.0)

        def _seek(t):
            was_on = play["on"]
            _pause()
            play["pos"] = max(0.0, t)
            _update_time()
            _highlight_seg(play["pos"])
            _video_show(play["pos"])
            if was_on:
                _play()

        def _skip(delta):
            total = _total_dur()
            if total <= 0:
                return
            _seek(min(total, max(0.0, _cur_t() + delta)))

        # ── Seek slider (0..1000 maps to 0..total) ────────────────────────
        def _seek_press(_e):
            if _total_dur() <= 0:
                return
            seek_drag["on"] = True
            seek_drag["was_on"] = play["on"]
            if play["on"]:               # stop audio/tick; scrub silently
                _pause()

        def _seek_drag(_e):
            # Live scrub while dragging: show the frame/time under the thumb.
            if not seek_drag["on"]:
                return
            total = _total_dur()
            if total <= 0:
                return
            t = seek_var.get() / 1000.0 * total
            play["pos"] = t
            time_lbl.config(text=f"{_fmt_t(t)} / {_fmt_t(total)}")
            _highlight_seg(t)
            _video_show(t)

        def _seek_release(_e):
            if not seek_drag["on"]:
                return
            seek_drag["on"] = False
            total = _total_dur()
            if total <= 0:
                return
            play["pos"] = seek_var.get() / 1000.0 * total
            _update_time()
            _highlight_seg(play["pos"])
            _video_show(play["pos"])
            if seek_drag.get("was_on"):
                _play()

        # ── Video sync offset — shift the video against the audio clock ───
        def _nudge_video(delta):
            vid["offset"] += delta
            vid["last"] = -1          # force a reseek on the next frame
            off_lbl.config(text=f"video {vid['offset']:+.2f}s")
            _video_show(_cur_t())

        def _reset_video_offset():
            vid["offset"] = 0.0
            vid["last"] = -1
            off_lbl.config(text="video +0.00s")
            _video_show(_cur_t())

        def _on_en_dclick(event):
            idx = en_box.index(f"@{event.x},{event.y}")
            n = en_box.count("1.0", idx, "chars")
            n = n[0] if n else 0
            for cs, ce, s, _e in seg_ranges:
                if cs <= n <= ce and s is not None:
                    _seek(s)
                    break
            return "break"

        if has_audio:
            en_box.bind("<Double-Button-1>", _on_en_dclick)

        def _on_mute():
            if not play["on"]:
                return
            try:
                sd.stop()
            except Exception:
                pass
            if not mute_var.get():
                _audio_start(_cur_t())

        def _shutdown_media():
            play["on"] = False
            try:
                sd.stop()
            except Exception:
                pass
            _cancel_tick()
            if vid["cap"] is not None:
                try:
                    vid["cap"].release()
                except Exception:
                    pass
                vid["cap"] = None

        # ── Video import / removal ────────────────────────────────────────
        def _remove_video():
            cap = vid["cap"]
            vid.update(cap=None, last=-1, photo=None)
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            try:
                paned.forget(vid_holder)
            except Exception:
                pass
            vid_lbl.config(image="")
            video_btn.config(text="🎬 Import Video…", command=_import_video)
            try:
                off_box.pack_forget()
            except Exception:
                pass
            if not has_audio:
                _stop_all()
                play_btn.config(state="disabled")

        def _import_video():
            try:
                import cv2
            except ImportError:
                messagebox.showwarning(
                    "OpenCV required",
                    "Video preview needs the opencv-python package.\n\n"
                    "Install it with:\n\n    pip install opencv-python\n\n"
                    "then import the video again.",
                    parent=win)
                return
            initdir = (os.path.dirname(os.path.abspath(audio_path))
                       if audio_path else os.path.expanduser("~"))
            path = filedialog.askopenfilename(
                parent=win, title="Import video",
                initialdir=initdir,
                filetypes=[("Video files",
                            "*.mp4 *.mov *.mkv *.avi *.webm *.m4v"),
                           ("All files", "*.*")])
            if not path:
                return
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                messagebox.showerror(
                    "Video", "Could not open this video file.", parent=win)
                return
            _remove_video()
            vid.update(cap=cap, cv2=cv2,
                       fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
                       frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                       last=-1, offset=0.0)
            off_lbl.config(text="video +0.00s")
            vid_name.config(text=os.path.basename(path))
            paned.add(vid_holder, before=en_holder, stretch="always",
                      width=430, minsize=240)
            video_btn.config(text="✕ Remove Video", command=_remove_video)
            off_box.pack(side="right", padx=(0, 10))
            play_btn.config(state="normal")
            win.after(80, lambda: _video_show(_cur_t()))

        # ── Toolbar: row 1 = seek slider, row 2 = transport controls ──────
        seek_row = tk.Frame(bar, bg=BG)
        seek_row.pack(fill="x")
        ctrl_row = tk.Frame(bar, bg=BG)
        ctrl_row.pack(fill="x", pady=(4, 0))

        seek = ttk.Scale(seek_row, from_=0, to=1000, orient="horizontal",
                         variable=seek_var, command=lambda _v: None,
                         style="Horizontal.TScale",
                         state="normal" if has_audio else "disabled")
        seek.pack(fill="x", expand=True)
        seek.bind("<ButtonPress-1>", _seek_press)
        seek.bind("<B1-Motion>", _seek_drag)
        seek.bind("<ButtonRelease-1>", _seek_release)

        def _tbtn(parent, label, cmd, bold=False, accent=False, **kw):
            return tk.Button(parent, text=label, command=cmd,
                             bg=PANEL3,
                             fg=_btn_fg(ACCENT if accent else TEXT),
                             activebackground=PANEL3,
                             activeforeground=_btn_fg(
                                 ACCENT if accent else TEXT),
                             font=(MONO_FONT, 10, "bold" if bold else
                                   "normal"),
                             relief="flat", cursor="hand2", bd=0, **kw)

        play_btn = _tbtn(ctrl_row, "▶ Play", _toggle_play, bold=True,
                         accent=True, padx=12)
        play_btn.config(state="normal" if has_audio else "disabled")
        play_btn.pack(side="left")
        _tbtn(ctrl_row, "⏹ Stop", _stop_all, padx=10).pack(
            side="left", padx=(6, 0))
        _tbtn(ctrl_row, "⏪ -5s", lambda: _skip(-5), padx=6).pack(
            side="left", padx=(6, 0))
        _tbtn(ctrl_row, "+5s ⏩", lambda: _skip(5), padx=6).pack(
            side="left", padx=(4, 0))
        time_lbl = tk.Label(ctrl_row, text="", bg=BG, fg=TEXT_FAINT,
                            font=(MONO_FONT, 10))
        time_lbl.pack(side="left", padx=12)
        tk.Checkbutton(ctrl_row, text="🔇 Mute audio", variable=mute_var,
                       command=_on_mute, bg=BG, fg=TEXT_FAINT,
                       activebackground=BG, activeforeground=TEXT_FAINT,
                       selectcolor=PANEL2,
                       state="normal" if has_audio else "disabled",
                       font=(MONO_FONT, 9)).pack(side="left", padx=4)

        video_btn = _tbtn(ctrl_row, "🎬 Import Video…", _import_video,
                          padx=12)
        video_btn.pack(side="right")

        # Video sync-offset controls — hidden until a video is imported.
        off_box = tk.Frame(ctrl_row, bg=BG)
        _tbtn(off_box, "◀", lambda: _nudge_video(-0.10), padx=6).pack(
            side="left")
        off_lbl = tk.Label(off_box, text="video +0.00s", bg=BG,
                           fg=TEXT_FAINT, font=(MONO_FONT, 9), width=13)
        off_lbl.pack(side="left", padx=2)
        _tbtn(off_box, "▶", lambda: _nudge_video(0.10), padx=6).pack(
            side="left")
        _tbtn(off_box, "⟲", _reset_video_offset, padx=6).pack(
            side="left", padx=(4, 0))

        _update_time()

        # ── Actions ───────────────────────────────────────────────────────
        def _gather_text():
            return tr_box.get("1.0", "end-1c").strip()

        def _copy(mode):
            if mode == "tr":
                payload = _gather_text()
            else:
                payload = (f"=== ENGLISH ===\n"
                           f"{en_box.get('1.0', 'end-1c').strip()}\n\n"
                           f"=== {lang_up} ===\n{_gather_text()}")
            win.clipboard_clear()
            win.clipboard_append(payload)
            status_lbl.config(text="✓ Copied to clipboard", fg=TR_ACCENT)

        def _finish(action):
            if done_event.is_set():
                return
            _shutdown_media()
            result_holder["action"] = action
            if action == "continue":
                txt = _gather_text()
                if txt.strip():
                    result_holder["text"] = txt
                else:
                    # Everything deleted — dub the original script instead.
                    result_holder["action"] = "skip"
            done_event.set()
            win.destroy()

        # Safety net: window destroyed any other way (app quit, Cmd+W)
        # must never leave the pipeline worker waiting forever.
        def _on_destroy(event):
            if event.widget is win and not done_event.is_set():
                _shutdown_media()
                result_holder.setdefault("action", "skip")
                done_event.set()
        win.bind("<Destroy>", _on_destroy)
        win.protocol("WM_DELETE_WINDOW", lambda: _finish("skip"))

        tk.Button(ftr, text="✔ Continue to Dubbing",
                  command=lambda: _finish("continue"),
                  bg="#2563eb", fg=_btn_fg("white"),
                  font=(MONO_FONT, 10, "bold"), relief="flat",
                  cursor="hand2", padx=16
                  ).pack(side="right", padx=(6, 14), pady=8)
        tk.Button(ftr, text="⏭ Skip — Dub As-Is",
                  command=lambda: _finish("skip"),
                  bg=PANEL3, fg=_btn_fg(TEXT), font=(MONO_FONT, 10),
                  relief="flat", cursor="hand2", padx=12
                  ).pack(side="right", padx=6, pady=8)
        tk.Button(ftr, text=f"📋 Copy English + {language}",
                  command=lambda: _copy("both"),
                  bg=PANEL3, fg=_btn_fg(TEXT), font=(MONO_FONT, 9),
                  relief="flat", cursor="hand2", padx=10
                  ).pack(side="right", padx=6, pady=8)
        tk.Button(ftr, text=f"📋 Copy {language}",
                  command=lambda: _copy("tr"),
                  bg=PANEL3, fg=_btn_fg(TEXT), font=(MONO_FONT, 9),
                  relief="flat", cursor="hand2", padx=10
                  ).pack(side="right", padx=6, pady=8)

        try:
            win.lift()
            win.focus_force()
        except Exception:
            pass

    def _ensure_window_visible(self):
        """Restore the main window if it is minimized. Any modal dialog or
        input grab created on top of an iconified window locks the app on
        Windows (invisible modal holds the grab; taskbar restore is dead)."""
        try:
            if self.root.state() == "iconic":
                self.root.deiconify()
        except Exception:
            pass

    def _run_full_pipeline(self):
        if not self.filepath:
            messagebox.showwarning("No file", "Open an audio file first.")
            return
        if not self.regions:
            messagebox.showwarning("No Regions",
                                   "No regions detected. Adjust threshold and Re-Apply first.")
            return
        try:
            _get_api_key()
        except Exception as e:
            messagebox.showerror("ElevenLabs API Key Error", str(e))
            return

        # LLM provider only needed when going past SRT-only
        if self._single_stop_step.get() != "English SRT":
            try:
                _validate_llm_config()
            except Exception as e:
                messagebox.showerror("LLM Provider Error", str(e))
                return

        self._pipeline_cancel.clear()
        self.btn_run_pipeline.config(state="disabled")
        self.btn_cancel_pipeline.config(state="normal")
        for tag in self._stage_vars:
            self._set_stage(tag, "—", TEXT_MUTED)
        self.tr_status.config(text="Running…", fg=TEXT_FAINT)
        self.status.config(text="Pipeline running — please wait…")
        self._sf_show_view("log")
        self._sf_log_append(
            f"=== Pipeline started: {os.path.basename(self.filepath)} "
            f"(up to: {self._single_stop_step.get()}) ===")

        filepath      = self.filepath
        regions       = list(self.regions)
        y_data        = self.audio_data
        sr            = self.sample_rate
        (tts_platform, tts_lang_code, tts_voice_name, el_voice_id,
         pipeline_language, el_model) = self._get_tts_params()
        en_thr, en_hys, en_min = self._get_en_region_params()
        te_thr, te_hys, te_min = self._get_bn_region_params()

        def _worker():
            def _chk():
                if self._pipeline_cancel.is_set():
                    raise InterruptedError("Pipeline cancelled by user.")

            try:
                api_key = _get_api_key()

                # Stage 1a: Transcription
                _chk()
                self.root.after(0, lambda: self._set_stage("S1a", "Transcribing…", "#d97706"))
                result  = _transcribe_audio(filepath, api_key)
                words   = result.get("words", [])
                raw_eng = result.get("text", "").strip()
                if not raw_eng and words:
                    raw_eng = " ".join(w.get("text", "").strip() for w in words
                                       if w.get("type", "word") == "word")
                if not words:
                    raise ValueError("No word data from ElevenLabs.")
                self.root.after(0, lambda: self._set_stage("S1a", "✔ Done", "#22c55e"))

                # Build SRT
                final_srt     = _build_subtitle_srt(regions, words)
                formatted_srt = _parse_srt_to_analysis_format(final_srt)

                # Auto-save SRT — create a per-file output folder named
                # after the audio file (without extension) and copy the
                # original audio into it. All subsequent outputs go there.
                outdir   = _prepare_output_dir(filepath)
                base     = os.path.join(
                    outdir, os.path.splitext(os.path.basename(filepath))[0])
                self._last_run_outdir   = outdir
                self._last_run_base     = base
                self._last_run_language = pipeline_language
                srt_path = base + ".srt"
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(final_srt)

                # Save formatted analysis text file
                analyzed_path = base + "_analyzed.txt"
                with open(analyzed_path, "w", encoding="utf-8") as f:
                    f.write(formatted_srt)

                # ── Early exit: SRT only ──────────────────────────────────────
                if self._single_stop_step.get() == "English SRT":
                    def _done_srt():
                        self.btn_run_pipeline.config(state="normal")
                        self.btn_cancel_pipeline.config(state="disabled")
                        self.tr_status.config(text="Done — SRT saved ✓", fg=TR_ACCENT)
                        self.status.config(text=f"Done! SRT → {srt_path}")
                    self.root.after(0, _done_srt)
                    return

                # Stage 1b: Gemini pipeline (translation memory first — an
                # exact proofed match skips the LLM entirely)
                _chk()
                tm_on = self._tm_enabled_var.get()
                en_entries_tm = _extract_srt_entries(final_srt)
                tm_cached = (_tm_lookup_full(pipeline_language, en_entries_tm)
                             if tm_on else None)
                if tm_cached:
                    tr_result = rev_result = punc_result = tm_cached
                    self.root.after(0, lambda: self._set_stage(
                        "S1b", "Reusing proofed translation…", "#22c55e"))
                    self.root.after(0, lambda: self.status.config(
                        text="🧠 Translation memory hit — reusing the "
                             "human-proofed script (no LLM call)."))
                else:
                    self.root.after(0, lambda: self._set_stage("S1b", "Running LLM…", "#d97706"))
                    tm_gloss = (_tm_glossary_block(pipeline_language, en_entries_tm)
                                if tm_on else "")
                    (tr_result, rev_result, punc_result,
                     _, _, _) = _run_gemini_pipeline(formatted_srt,
                                                      self._gemini_model_var.get(),
                                                      language=pipeline_language,
                                                      steps=self._translation_steps_var.get(),
                                                      tm_glossary=tm_gloss)

                # Save Step-1 Translation output (raw translation before review)
                try:
                    tr_step_path = base + "_TranslationStep.txt"
                    with open(tr_step_path, "w", encoding="utf-8") as f:
                        f.write(tr_result)
                except Exception:
                    pass

                # Save Step-2 Review output (after review pass, before punctuation)
                try:
                    rev_step_path = base + "_ReviewStep.txt"
                    with open(rev_step_path, "w", encoding="utf-8") as f:
                        f.write(rev_result)
                except Exception:
                    pass

                # Save FinalScript (Step-3 Punctuation result, paired with English)
                combined = (f"=== ENGLISH TRANSCRIPTION ===\n{raw_eng}\n\n"
                            f"=== {pipeline_language.upper()} TRANSLATION ===\n{punc_result}")
                final_path = base + "_FinalScript.txt"
                with open(final_path, "w", encoding="utf-8") as f:
                    f.write(combined)
                _history_record(filepath, outdir, base, pipeline_language,
                                "single")
                self.root.after(0, lambda: self._set_stage("S1b", "✔ Done", "#22c55e"))
                self.root.after(0, lambda p=punc_result: setattr(self, "punctuation_result", p))
                self.root.after(0, lambda s=final_srt: setattr(self, "final_srt", s))

                # ── Early exit: Translation only ─────────────────────────────
                if self._single_stop_step.get() == "Translation":
                    def _done_tr():
                        self.btn_run_pipeline.config(state="normal")
                        self.btn_cancel_pipeline.config(state="disabled")
                        self.tr_status.config(text="Done — Translation saved ✓", fg=TR_ACCENT)
                        self.status.config(text=f"Done! FinalScript → {final_path}")
                    self.root.after(0, _done_tr)
                    return

                # ── Optional manual review before dubbing ────────────────────
                if self._review_enabled_var.get():
                    _chk()
                    self.root.after(0, lambda: self._set_stage(
                        "S2", "Waiting for manual review…", "#d97706"))
                    self.root.after(0, lambda: self.status.config(
                        text="Review the translation — Skip or Continue to start dubbing."))
                    review_res = {}
                    review_done = threading.Event()
                    en_entries = _extract_srt_entries(final_srt)
                    tr_paras = _split_translation_paragraphs(punc_result)
                    self.root.after(0, lambda: self._show_translation_review(
                        en_entries, tr_paras, pipeline_language,
                        review_res, review_done,
                        audio=y_data, sr=sr, audio_path=filepath))
                    try:
                        while not review_done.wait(0.25):
                            _chk()
                    except InterruptedError:
                        w = review_res.get("win")
                        if w is not None:
                            self.root.after(0, w.destroy)
                        raise
                    if (review_res.get("action") == "continue"
                            and (review_res.get("text") or "").strip()):
                        punc_result = review_res["text"].strip()
                        # Rewrite FinalScript so the saved file matches what
                        # actually gets dubbed.
                        combined = (
                            f"=== ENGLISH TRANSCRIPTION ===\n{raw_eng}\n\n"
                            f"=== {pipeline_language.upper()} TRANSLATION ===\n"
                            f"{punc_result}")
                        with open(final_path, "w", encoding="utf-8") as f:
                            f.write(combined)
                        self.root.after(0, lambda p=punc_result: setattr(
                            self, "punctuation_result", p))
                        # Feedback loop: "Continue" means a human reviewed
                        # this script — save it to translation memory so
                        # identical content never hits the LLM again.
                        if tm_on:
                            n_pairs = _tm_capture(pipeline_language, en_entries,
                                                  punc_result, source=base)
                            if n_pairs:
                                self.root.after(0, lambda n=n_pairs: self.status.config(
                                    text=f"🧠 Proofed translation saved to memory "
                                         f"({n} pairs) — future identical content "
                                         f"reuses it for free."))

                # Stage 2: TTS
                _chk()
                self.root.after(0, lambda: self._set_stage("S2", "Emotion detection…", "#d97706"))
                tts_path = os.path.join(outdir, _tts_output_name(pipeline_language, filepath, "_tts"))

                def _tts_status_cb(msg):
                    self.root.after(0, lambda m=msg: self._set_stage("S2", m, "#d97706"))

                # Multi-speaker: a saved per-paragraph voice map switches the
                # ElevenLabs synthesis to speaker groups. Emotion enrichment
                # then runs per group so paragraph indices stay aligned with
                # the 🎭 Speakers dialog.
                spk_paras = _split_translation_paragraphs(punc_result)
                spk_map = (_speakers_voice_map(base)
                           if tts_platform == "ElevenLabs" else {})
                spk_map = {i: v for i, v in spk_map.items()
                           if 0 <= i < len(spk_paras)}

                self.root.after(0, lambda: self._set_stage("S2", "Synthesizing…", "#d97706"))
                if tts_platform == "ElevenLabs" and spk_map:
                    self._sf_log_append(
                        f"[S2] 🎭 Multi-speaker dub: {len(spk_map)} "
                        f"paragraph(s) use a custom voice "
                        f"(of {len(spk_paras)} total).")
                    enrich_cb = (
                        (lambda txt: _run_emotion_enrichment(
                            txt, language=pipeline_language,
                            model=GEMINI_DEFAULT_MODEL,
                            status_cb=_tts_status_cb))
                        if self._emotion_enabled_var.get() else None)
                    synthesize_tts_elevenlabs_multi(
                        spk_paras, spk_map, tts_path, api_key=api_key,
                        default_voice_id=el_voice_id, model_id=el_model,
                        status_cb=_tts_status_cb, enrich_cb=enrich_cb)
                else:
                    # Emotion enrichment — skipped when checkbox is unchecked
                    enriched_text = (
                        _run_emotion_enrichment(
                            punc_result, language=pipeline_language,
                            model=GEMINI_DEFAULT_MODEL, status_cb=_tts_status_cb)
                        if self._emotion_enabled_var.get() else punc_result
                    )
                    if tts_platform == "ElevenLabs":
                        synthesize_tts_elevenlabs(enriched_text, tts_path, api_key=api_key,
                                                  voice_id=el_voice_id, model_id=el_model,
                                                  status_cb=_tts_status_cb)
                    else:
                        # Strip ElevenLabs-specific tags before sending to Google TTS
                        synthesize_tts(_strip_emotion_tags(enriched_text), tts_path,
                                       status_cb=_tts_status_cb,
                                       lang_code=tts_lang_code, voice_name=tts_voice_name)
                self.root.after(0, lambda: self._set_stage("S2", "✔ Done", "#22c55e"))

                # ── Early exit: TTS Audio only ───────────────────────────────
                if self._single_stop_step.get() == "TTS Audio":
                    def _done_tts():
                        self.btn_run_pipeline.config(state="normal")
                        self.btn_cancel_pipeline.config(state="disabled")
                        self.tr_status.config(text="Done — TTS audio saved ✓", fg=TR_ACCENT)
                        self.status.config(text=f"Done! TTS → {tts_path}")
                    self.root.after(0, _done_tts)
                    return

                # Stage 3a: English SRT for sync — reuse Stage 1a transcription (no extra API call)
                _chk()
                self.root.after(0, lambda: self._set_stage("S3a", "Building EN SRT…", "#d97706"))
                if not words:
                    raise ValueError("No word data from ElevenLabs for English audio.")
                en_srt = _build_english_subtitle_srt(regions, words)
                en_srt_path = base + "_sync_en.srt"
                with open(en_srt_path, "w", encoding="utf-8") as f:
                    f.write(en_srt)
                self.root.after(0, lambda: self._set_stage("S3a", "✔ Done", "#22c55e"))

                # Stage 3b: Target-language SRT from TTS audio
                _chk()
                self.root.after(0, lambda: self._set_stage("S3b", "Loading TTS audio…", "#d97706"))
                te_y, te_sr = librosa.load(tts_path, sr=None, mono=True)
                te_regions  = _detect_regions_from_audio(te_y, te_sr, te_thr, te_hys, te_min)
                if not te_regions:
                    raise ValueError("No regions detected in TTS audio.")
                self.root.after(0, lambda n=len(te_regions):
                    self._set_stage("S3b", f"Transcribing ({n} regions)…", "#d97706"))
                te_result = _transcribe_audio(tts_path, api_key)
                te_words  = te_result.get("words", [])
                if not te_words:
                    raise ValueError("No word data from ElevenLabs for TTS audio.")
                te_srt = _build_target_subtitle_srt(te_regions, te_words)
                te_srt_path = base + "_sync_te.srt"
                with open(te_srt_path, "w", encoding="utf-8") as f:
                    f.write(te_srt)
                self.root.after(0, lambda: self._set_stage("S3b", "✔ Done", "#22c55e"))

                # Stage 3c: Gemini SRT mapping
                _chk()
                self.root.after(0, lambda: self._set_stage("S3c", "Calling Gemini…", "#d97706"))
                mapping_text = _call_gemini_mapping(en_srt, te_srt, punc_result,
                                                     self._gemini_model_var.get(),
                                                     language=pipeline_language)
                mapping_path = base + "_sync_mapping.txt"
                with open(mapping_path, "w", encoding="utf-8") as f:
                    f.write(mapping_text)
                self.root.after(0, lambda: self._set_stage("S3c", "✔ Done", "#22c55e"))

                # Stage 3d: Sync SRTs
                _chk()
                self.root.after(0, lambda: self._set_stage("S3d", "Syncing…", "#d97706"))
                try:
                    _en_audio_dur = float(len(y_data)) / float(sr) if sr else 0.0
                except Exception:
                    _en_audio_dur = 0.0
                synced_subs, orig_te_subs, _sync_log = run_sync_from_strings(
                    en_srt, te_srt, mapping_text,
                    en_audio_duration=_en_audio_dur)
                # Surface sync-quality info: bleed-over sections were longer
                # than their English slot and anchored to their English start
                # (the order sweep keeps them from overlapping); dropped-
                # from-mapping subs were kept at original timing.
                _n_fit  = _sync_log.count("[bleed-over]")
                _n_kept = 1 if "missing from the" in _sync_log else 0
                if _n_fit or _n_kept:
                    self.root.after(0, lambda a=_n_fit:
                        self._set_stage(
                            "S3d",
                            f"✔ synced — {a} long section(s) anchored to "
                            f"their English start (see sync log)",
                            "#f59e0b"))
                    self._sf_log_append(
                        f"[S3d] ⚠ {_n_fit} section(s) were longer than their "
                        "English slot — they start on time and bleed into "
                        "the following gap at natural pace. If a passage "
                        "sounds crowded, shorten its translation and Re-Dub, "
                        "or drag chunks in Compare to taste.")
                else:
                    self.root.after(0, lambda n=len(synced_subs):
                        self._set_stage("S3d", f"✔ {n} subtitles synced",
                                        "#22c55e"))

                # Save sync log
                with open(base + "_sync_log.txt", "w", encoding="utf-8") as _f:
                    _f.write(_sync_log)

                # Save synced Bengali SRT (used by the Captions exporter).
                try:
                    synced_srt_text = _write_srt_from_dict(synced_subs)
                    synced_srt_path = base + "_sync_synced.srt"
                    with open(synced_srt_path, "w", encoding="utf-8") as _f:
                        _f.write(synced_srt_text)
                    self.root.after(
                        0, lambda p=synced_srt_path: setattr(
                            self, "_last_synced_srt_path", p))
                except Exception:
                    pass

                # Save synced timestamps as text file
                _ts_list = _build_timestamps(orig_te_subs, synced_subs)
                sync_ts_path = base + "_sync_timestamps.txt"
                with open(sync_ts_path, "w", encoding="utf-8") as f:
                    f.write(_format_timestamps_as_text(_ts_list))

                # Stage 3e: Create synced audio
                _chk()
                self.root.after(0, lambda: self._set_stage("S3e", "Creating audio…", "#d97706"))
                timestamps  = _ts_list  # reuse list already built for timestamps file
                # Save synced audio inside the per-file output folder
                synced_name = _tts_output_name(pipeline_language, filepath, "_synced")
                synced_path = os.path.join(outdir, synced_name)

                def _sync_status_cb(msg):
                    self.root.after(0, lambda m=msg: self._set_stage("S3e", m, "#d97706"))

                sync_audio_with_timestamps(tts_path, timestamps, synced_path,
                                           status_cb=_sync_status_cb)
                self.root.after(0, lambda n=synced_name:
                    self._set_stage("S3e", f"✔ Saved: {n}", "#22c55e"))

                def _on_complete():
                    self.btn_run_pipeline.config(state="normal")
                    self.btn_cancel_pipeline.config(state="disabled")
                    self.tr_status.config(text="All stages complete ✓", fg=TR_ACCENT)
                    self.status.config(text=f"Done! Synced audio → {synced_path}")
                    self._sf_log_append(
                        "=== Pipeline complete — opening Compare view ===")
                    self._show_run_result(base, pipeline_language, jump=False)
                    self._sf_show_view("compare")
                    self._sf_cmp_load()

                self.root.after(0, _on_complete)

            except InterruptedError:
                def _on_cancel():
                    self.btn_run_pipeline.config(state="normal")
                    self.btn_cancel_pipeline.config(state="disabled")
                    self.tr_status.config(text="Cancelled", fg="#f87171")
                    self.status.config(text="Pipeline cancelled.")
                    self._sf_log_append("=== Pipeline cancelled by user ===")
                self.root.after(0, _on_cancel)

            except Exception as exc:
                import traceback
                err = str(exc)
                tb  = traceback.format_exc()

                def _on_error():
                    self.btn_run_pipeline.config(state="normal")
                    self.btn_cancel_pipeline.config(state="disabled")
                    self.tr_status.config(text=f"Error: {err[:60]}", fg="#f87171")
                    self.status.config(text=f"Error: {err[:100]}")
                    self._sf_log_append(f"=== Pipeline ERROR: {err} ===")
                    self._ensure_window_visible()
                    messagebox.showerror("Pipeline Error", f"{err}\n\n{tb[:600]}")

                self.root.after(0, _on_error)

        threading.Thread(target=_worker, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────────────
    #  Batch logic
    # ─────────────────────────────────────────────────────────────────────────
    def _batch_pick_folder(self):
        folder = filedialog.askdirectory(title="Select folder with audio files")
        if not folder:
            return
        self._batch_folder = folder
        self.batch_folder_label.config(text=folder, fg=TEXT)

        files = sorted([
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
        ])
        if not files:
            messagebox.showwarning("No Audio Files",
                                   f"No supported audio files found.\n"
                                   f"Supported: {', '.join(sorted(AUDIO_EXTENSIONS))}")
            return

        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)
        self._batch_files = files

        for fname in files:
            self.batch_tree.insert("", "end",
                                   values=(fname, "Pending", "Pending", "Pending", "-"),
                                   tags=("pending",))

        self.batch_progress_label.config(
            text=f"Found {len(files)} audio file(s). Starting processing automatically…")
        self.root.after(500, self._batch_start)

    def _batch_start(self):
        if self._batch_running:
            return

        # Always need ElevenLabs (transcription)
        try:
            _get_api_key()
        except Exception as e:
            messagebox.showerror("ElevenLabs API Key Error", str(e)); return

        # LLM provider only needed for Translation, TTS Audio, Full Pipeline
        stop_step = self._batch_stop_step.get()
        if stop_step != "English SRT":
            try:
                _validate_llm_config()
            except Exception as e:
                messagebox.showerror("LLM Provider Error", str(e)); return

        self._batch_running  = True
        self._batch_stop_req = False
        self.batch_stop_btn.config(state="normal")
        threading.Thread(target=self._batch_worker, daemon=True).start()

    def _batch_stop(self):
        self._batch_stop_req = True
        self.batch_progress_label.config(
            text="Stop requested — will halt after current file…")

    def _batch_worker(self):
        items = self.batch_tree.get_children()
        total = len(items)
        (tts_platform, tts_lang_code, tts_voice_name, el_voice_id,
         pipeline_language, el_model) = self._get_tts_params()
        en_thr, en_hys, en_min = self._get_en_region_params()
        te_thr, te_hys, te_min = self._get_bn_region_params()

        for idx, item in enumerate(items):
            if self._batch_stop_req:
                self._batch_set_row(item, tr_status="Stopped", tag="skipped")
                for rem in list(items)[idx+1:]:
                    self._batch_set_row(rem, tr_status="Skipped", tag="skipped")
                break

            fname  = self._batch_files[idx]
            fpath  = os.path.join(self._batch_folder, fname)
            base   = os.path.splitext(fname)[0]
            # Each batch file gets its own per-file output folder (named after
            # the file, without extension) inside the batch folder. The
            # original audio is also copied into that folder.
            outdir = _prepare_output_dir(fpath)

            self._batch_upd(idx+1, total, fname, "Loading audio…")
            self._batch_set_row(item, tr_status="Loading…", tag="running")

            try:
                y, sr = librosa.load(fpath, sr=None, mono=True)

                # Stage 1a: detect regions — use English panel params
                regions = _detect_regions_from_audio(y, sr, en_thr, en_hys, en_min)
                n_reg   = len(regions)
                if n_reg == 0:
                    self._batch_set_row(item, tr_status="No regions — skipped",
                                        tts_status="—", sync_status="—", tag="skipped")
                    continue

                self._batch_upd(idx+1, total, fname, f"{n_reg} regions — transcribing…")
                self._batch_set_row(item, tr_status=f"Transcribing ({n_reg} reg)", tag="running")

                api_key = _get_api_key()
                result  = _transcribe_audio(fpath, api_key)
                words   = result.get("words", [])
                raw_eng = result.get("text", "").strip()
                if not raw_eng and words:
                    raw_eng = " ".join(w.get("text","").strip() for w in words
                                      if w.get("type","word") == "word")
                if not words:
                    self._batch_set_row(item, tr_status="No ElevenLabs data", tag="error")
                    continue

                final_srt     = _build_subtitle_srt(regions, words)
                formatted_srt = _parse_srt_to_analysis_format(final_srt)

                srt_path = os.path.join(outdir, base + ".srt")
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(final_srt)

                # Save formatted analysis text file
                analyzed_path = os.path.join(outdir, base + "_analyzed.txt")
                with open(analyzed_path, "w", encoding="utf-8") as f:
                    f.write(formatted_srt)

                # ── Early exit: English SRT only ─────────────────────────────
                if self._batch_stop_step.get() == "English SRT":
                    self._batch_set_row(item, tr_status="✔ SRT saved",
                                        tts_status="—", sync_status="—",
                                        output=srt_path, tag="done")
                    self._batch_upd(idx+1, total, fname,
                                    f"Done (SRT only) ✓ → {os.path.basename(srt_path)}")
                    continue

                # Stage 1b: Gemini (translation memory first — an exact
                # proofed match skips the LLM entirely)
                tm_on = self._tm_enabled_var.get()
                en_entries_tm = _extract_srt_entries(final_srt)
                tm_cached = (_tm_lookup_full(pipeline_language, en_entries_tm)
                             if tm_on else None)
                if tm_cached:
                    tr_result = rev_result = punc_result = tm_cached
                    self._batch_upd(idx+1, total, fname,
                                    "🧠 Reusing proofed translation (memory)…")
                    self._batch_set_row(item, tr_status="✔ From memory", tag="running")
                else:
                    self._batch_upd(idx+1, total, fname, "Vertex AI translation…")
                    self._batch_set_row(item, tr_status="Vertex AI running…", tag="running")
                    tm_gloss = (_tm_glossary_block(pipeline_language, en_entries_tm)
                                if tm_on else "")
                    (tr_result, rev_result, punc_result,
                     _, _, _) = _run_gemini_pipeline(formatted_srt,
                                                      self._gemini_model_var.get(),
                                                      language=pipeline_language,
                                                      steps=self._translation_steps_var.get(),
                                                      tm_glossary=tm_gloss)

                # Save Step-1 Translation output (raw translation before review)
                try:
                    tr_step_path = os.path.join(outdir, base + "_TranslationStep.txt")
                    with open(tr_step_path, "w", encoding="utf-8") as f:
                        f.write(tr_result)
                except Exception:
                    pass

                # Save Step-2 Review output (after review pass, before punctuation)
                try:
                    rev_step_path = os.path.join(outdir, base + "_ReviewStep.txt")
                    with open(rev_step_path, "w", encoding="utf-8") as f:
                        f.write(rev_result)
                except Exception:
                    pass

                # Save FinalScript (Step-3 Punctuation result, paired with English)
                combined   = (f"=== ENGLISH TRANSCRIPTION ===\n{raw_eng}\n\n"
                              f"=== {pipeline_language.upper()} TRANSLATION ===\n{punc_result}")
                final_path = os.path.join(outdir, base + "_FinalScript.txt")
                with open(final_path, "w", encoding="utf-8") as f:
                    f.write(combined)
                _history_record(fpath, outdir, os.path.join(outdir, base),
                                pipeline_language, "batch")

                self._batch_set_row(item, tr_status="✔ Done", tag="running")

                # ── Early exit: Translation only ─────────────────────────────
                if self._batch_stop_step.get() == "Translation":
                    self._batch_set_row(item, tts_status="—", sync_status="—",
                                        output=final_path, tag="done")
                    self._batch_upd(idx+1, total, fname,
                                    f"Done (Translation) ✓ → {os.path.basename(final_path)}")
                    continue

                # Stage 2: TTS
                self._batch_upd(idx+1, total, fname, "TTS synthesis…")
                self._batch_set_row(item, tts_status="Synthesizing…", tag="running")

                tts_path = os.path.join(outdir, _tts_output_name(pipeline_language, fpath, "_tts"))
                def _batch_emo_cb(msg, _f=fname, _i=idx, _t=total):
                    self._batch_upd(_i+1, _t, _f, msg)
                batch_enriched = (
                    _run_emotion_enrichment(
                        punc_result, language=pipeline_language,
                        model=GEMINI_DEFAULT_MODEL, status_cb=_batch_emo_cb)
                    if self._emotion_enabled_var.get() else punc_result
                )
                if tts_platform == "ElevenLabs":
                    synthesize_tts_elevenlabs(batch_enriched, tts_path,
                                              api_key=api_key, voice_id=el_voice_id,
                                              model_id=el_model)
                else:
                    synthesize_tts(_strip_emotion_tags(batch_enriched), tts_path,
                                   lang_code=tts_lang_code, voice_name=tts_voice_name)
                self._batch_set_row(item, tts_status="✔ Done", tag="running")

                # ── Early exit: TTS Audio only ───────────────────────────────
                if self._batch_stop_step.get() == "TTS Audio":
                    self._batch_set_row(item, sync_status="—",
                                        output=tts_path, tag="done")
                    self._batch_upd(idx+1, total, fname,
                                    f"Done (TTS) ✓ → {os.path.basename(tts_path)}")
                    continue

                # Stage 3a: English SRT — reuse Stage 1a transcription, save file
                self._batch_upd(idx+1, total, fname, "Sync — English SRT…")
                self._batch_set_row(item, sync_status="EN SRT…", tag="running")

                if not words:
                    self._batch_set_row(item, sync_status="No EN word data", tag="error")
                    continue
                en_srt = _build_english_subtitle_srt(regions, words)
                with open(os.path.join(outdir, base + "_sync_en.srt"), "w", encoding="utf-8") as f:
                    f.write(en_srt)

                # Stage 3b: Target-language SRT from TTS — save file
                self._batch_upd(idx+1, total, fname,
                                f"Sync — {pipeline_language} SRT…")
                self._batch_set_row(item, sync_status="Target SRT…", tag="running")

                te_y, te_sr = librosa.load(tts_path, sr=None, mono=True)
                te_regions  = _detect_regions_from_audio(te_y, te_sr, te_thr, te_hys, te_min)
                if not te_regions:
                    self._batch_set_row(item, sync_status="No target regions",
                                        tag="error")
                    continue
                te_result = _transcribe_audio(tts_path, api_key)
                te_words  = te_result.get("words", [])
                if not te_words:
                    self._batch_set_row(item, sync_status="No target word data",
                                        tag="error")
                    continue
                te_srt = _build_target_subtitle_srt(te_regions, te_words)
                with open(os.path.join(outdir, base + "_sync_te.srt"), "w", encoding="utf-8") as f:
                    f.write(te_srt)

                # Stage 3c: Gemini mapping — save file
                self._batch_upd(idx+1, total, fname, "Sync — Gemini mapping…")
                self._batch_set_row(item, sync_status="Mapping…", tag="running")
                mapping_text = _call_gemini_mapping(en_srt, te_srt, punc_result,
                                                     self._gemini_model_var.get(),
                                                     language=pipeline_language)
                with open(os.path.join(outdir, base + "_sync_mapping.txt"), "w", encoding="utf-8") as f:
                    f.write(mapping_text)

                # Stage 3d: Sync SRTs
                self._batch_upd(idx+1, total, fname, "Sync — syncing SRTs…")
                self._batch_set_row(item, sync_status="Syncing…", tag="running")
                try:
                    _en_audio_dur = float(len(y)) / float(sr) if sr else 0.0
                except Exception:
                    _en_audio_dur = 0.0
                synced_subs, orig_te_subs, _sync_log = run_sync_from_strings(
                    en_srt, te_srt, mapping_text,
                    en_audio_duration=_en_audio_dur)

                # Save sync log
                with open(os.path.join(outdir, base + "_sync_log.txt"), "w", encoding="utf-8") as _f:
                    _f.write(_sync_log)

                # Save synced Bengali SRT (used by the Captions exporter).
                try:
                    synced_srt_text = _write_srt_from_dict(synced_subs)
                    synced_srt_path = os.path.join(outdir, base + "_sync_synced.srt")
                    with open(synced_srt_path, "w", encoding="utf-8") as _f:
                        _f.write(synced_srt_text)
                    self.root.after(
                        0, lambda p=synced_srt_path: setattr(
                            self, "_last_synced_srt_path", p))
                except Exception:
                    pass

                # Save synced timestamps as text file
                timestamps = _build_timestamps(orig_te_subs, synced_subs)
                sync_ts_path = os.path.join(outdir, base + "_sync_timestamps.txt")
                with open(sync_ts_path, "w", encoding="utf-8") as f:
                    f.write(_format_timestamps_as_text(timestamps))

                # Stage 3e: Create synced audio
                self._batch_upd(idx+1, total, fname, "Sync — creating audio…")
                self._batch_set_row(item, sync_status="Audio…", tag="running")
                synced_name = _tts_output_name(pipeline_language, fpath, "_synced")
                synced_path = os.path.join(outdir, synced_name)
                sync_audio_with_timestamps(tts_path, timestamps, synced_path)

                self._batch_set_row(item, sync_status="✔ Done",
                                    output=synced_path, tag="done")
                self._batch_upd(idx+1, total, fname, f"Done ✓ → {synced_name}")

            except Exception as exc:
                err_msg = str(exc)[:80]
                self._batch_set_row(item, tr_status=f"Error: {err_msg}", tag="error")
                self._batch_upd(idx+1, total, fname, f"Error: {err_msg}")

        self._batch_running = False
        done_count = sum(
            1 for i in self.batch_tree.get_children()
            if self.batch_tree.item(i, "tags")[0] == "done")
        self.root.after(0, lambda: self.batch_stop_btn.config(state="disabled"))
        self.root.after(0, lambda: self.batch_progress_label.config(
            text=f"Batch complete — {done_count}/{total} files fully processed.",
            fg=TR_ACCENT))
        if done_count:
            self.root.after(0, self._show_run_result)

    def _batch_set_row(self, item, tr_status=None, tts_status=None, sync_status=None,
                       output=None, tag=None):
        vals = list(self.batch_tree.item(item, "values"))
        if tr_status  is not None: vals[1] = tr_status
        if tts_status is not None: vals[2] = tts_status
        if sync_status is not None: vals[3] = sync_status
        if output     is not None: vals[4] = output
        self.root.after(0, lambda v=vals, t=tag: self._batch_tree_update(item, v, t))

    def _batch_tree_update(self, item, vals, tag):
        try:
            self.batch_tree.item(item, values=vals,
                                 tags=(tag,) if tag else self.batch_tree.item(item, "tags"))
            self.batch_tree.see(item)
        except Exception:
            pass

    def _batch_upd(self, current, total, fname, step_msg):
        msg = f"[{current}/{total}]  {fname}  →  {step_msg}"
        self.root.after(0, lambda m=msg: self.batch_progress_label.config(text=m, fg=ACCENT))

    # ─────────────────────────────────────────────────────────────────────────
    #  Project Manager (launch screen), stage bar & Scripts tab
    # ─────────────────────────────────────────────────────────────────────────

    _STAGES = [
        ("①  Setup & Run",  "setup"),
        ("②  Transcript",   "transcript"),
        ("③  Translation",  "translation"),
        ("④  Result",       "result"),
    ]

    def _build_stage_bar(self, parent):
        bar = tk.Frame(parent, bg="#0b1220", height=48, bd=0,
                       highlightbackground=PANEL_BORDER, highlightthickness=1)
        bar.pack(side="bottom", fill="x")
        bar.pack_propagate(False)

        self._proj_label = tk.Label(bar, text="", bg="#0b1220", fg="#a78bfa",
                                    font=(MONO_FONT, 9, "bold"))
        self._proj_label.pack(side="left", padx=(14, 18), pady=12)

        self._stage_btns: Dict[str, tk.Button] = {}
        for label, key in self._STAGES:
            b = tk.Button(bar, text=label,
                          command=lambda k=key: self._goto_stage(k),
                          bg="#0b1220", fg=_btn_fg(TEXT_MUTED),
                          activebackground="#1e293b",
                          activeforeground=_btn_fg(ACCENT),
                          relief="flat", bd=0, cursor="hand2",
                          font=(MONO_FONT, 10, "bold"), padx=14, pady=6)
            b.pack(side="left", padx=2, pady=6)
            self._stage_btns[key] = b

        self._btn(bar, "⌂ Projects", self._show_project_manager,
                  bg="#1e1b3a", fg="#a78bfa", abg="#312e81"
                  ).pack(side="right", padx=(4, 12), pady=8)

    def _refresh_stage_bar(self):
        e = self._current_project
        status = _project_stage_status(e) if e else {}
        for label, key in self._STAGES:
            btn = self._stage_btns.get(key)
            if not btn:
                continue
            done = status.get(key, key == "setup" and bool(e))
            btn.config(text=label + ("  ✓" if done and key != "setup" else ""),
                       fg=_btn_fg(TR_ACCENT if done else TEXT_MUTED))
        if e:
            self._proj_label.config(
                text=f"🎬 {e.get('name', '?')}  ·  {e.get('language', '')}")
        else:
            self._proj_label.config(text="")

    def _goto_stage(self, key: str):
        if key == "setup":
            self.notebook.select(self.main_tab)
        elif key in ("transcript", "translation"):
            self._scripts_load_project()
            self.notebook.select(self.scripts_tab)
            try:
                (self._scr_en_text if key == "transcript"
                 else self._scr_tr_text).focus_set()
            except Exception:
                pass
        elif key == "result":
            e = self._current_project
            if e:
                # reuse the auto-open path: select run + load compare
                auto = self._auto_open_var.get()
                self._auto_open_var.set(True)
                self._show_run_result(e.get("base"), e.get("language"))
                self._auto_open_var.set(auto)
            else:
                self.notebook.select(self.hist_tab)

    # ── Stage-bar live tint while the pipeline runs ───────────────────────────

    _PIPE_TO_STAGE = {"S1a": "transcript", "S1b": "translation",
                      "S2": "result", "S3a": "result", "S3b": "result",
                      "S3c": "result", "S3d": "result", "S3e": "result"}

    def _stage_bar_pulse(self, tag: str, msg: str):
        key = self._PIPE_TO_STAGE.get(tag.strip())
        btn = getattr(self, "_stage_btns", {}).get(key)
        if not btn:
            return
        try:
            if msg.startswith("✔"):
                self._refresh_stage_bar()
            elif msg not in ("—", ""):
                btn.config(fg=_btn_fg("#d97706"))   # running — amber
        except Exception:
            pass

    # ── Project Manager screen ────────────────────────────────────────────────

    def _build_project_manager(self):
        pm = self._pm_frame

        hdr = tk.Frame(pm, bg=PANEL, height=64, bd=0,
                       highlightbackground=PANEL_BORDER, highlightthickness=1)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="🎬 PROJECTS", bg=PANEL, fg="#a78bfa",
                 font=(MONO_FONT, 14, "bold")).pack(side="left",
                                                    padx=18, pady=16)
        self._btn(hdr, "➕ New Project", self._pm_new_project,
                  bg="#0f1d14", fg=TR_ACCENT, abg="#1f4d2e"
                  ).pack(side="left", padx=(8, 4), pady=14)
        self._btn(hdr, "↻ Refresh", self._pm_refresh,
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT
                  ).pack(side="left", padx=4, pady=14)
        self._btn(hdr, "🗑 Remove", self._pm_delete_selected,
                  bg="#1f1213", fg="#f87171", abg="#3a1414"
                  ).pack(side="right", padx=(4, 14), pady=14)
        self._btn(hdr, "📂 Open Folder", self._pm_open_folder,
                  bg="#172554", fg=REG_LABEL, abg="#1e3a8a"
                  ).pack(side="right", padx=4, pady=14)
        self._btn(hdr, "▶ Open Project", self._pm_open_selected,
                  bg="#0f1d14", fg=TR_ACCENT, abg="#1f4d2e"
                  ).pack(side="right", padx=4, pady=14)

        tk.Label(pm, text="Double-click a project to continue where you left "
                          "off — every stage output is saved automatically.",
                 bg=BG, fg=TEXT_FAINT, font=(MONO_FONT, 9),
                 anchor="w").pack(fill="x", padx=18, pady=(8, 4))

        wrap = tk.Frame(pm, bg=BG)
        wrap.pack(fill="both", expand=True, padx=14, pady=(4, 12))
        self._pm_canvas = tk.Canvas(wrap, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(wrap, orient="vertical",
                            command=self._pm_canvas.yview)
        self._pm_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._pm_canvas.pack(side="left", fill="both", expand=True)
        self._pm_inner = tk.Frame(self._pm_canvas, bg=BG)
        self._pm_win = self._pm_canvas.create_window(
            (0, 0), window=self._pm_inner, anchor="nw")
        self._pm_inner.bind("<Configure>", lambda _e: self._pm_canvas.configure(
            scrollregion=self._pm_canvas.bbox("all")))
        self._pm_canvas.bind("<Configure>", lambda ev: self._pm_canvas.itemconfigure(
            self._pm_win, width=ev.width))

        self._pm_status = tk.Label(pm, text="", bg=BG, fg=TEXT_FAINT,
                                   font=(MONO_FONT, 9), anchor="w")
        self._pm_status.pack(fill="x", padx=18, pady=(0, 10))

        self._pm_entries: List[dict] = []
        self._pm_selected: Optional[int] = None
        self._pm_cards: List[tk.Frame] = []

    def _pm_refresh(self):
        _projects_sync_from_history()
        self._pm_entries = _projects_load()
        self._pm_selected = None
        for w in self._pm_inner.winfo_children():
            w.destroy()
        self._pm_cards = []

        if not self._pm_entries:
            tk.Label(self._pm_inner,
                     text="\n\nNo projects yet.\n\nClick  ➕ New Project  to "
                          "start:  pick an English audio file and a language "
                          "— everything else is automatic.",
                     bg=BG, fg=TEXT_FAINT, font=(MONO_FONT, 11),
                     justify="center").pack(pady=60, fill="x")
            self._pm_status.config(text="0 projects.")
            return

        for idx, e in enumerate(self._pm_entries):
            st = _project_stage_status(e)
            card = tk.Frame(self._pm_inner, bg=PANEL, bd=0,
                            highlightbackground=PANEL_BORDER,
                            highlightthickness=1)
            card.pack(fill="x", pady=3)
            top = tk.Frame(card, bg=PANEL)
            top.pack(fill="x", padx=12, pady=(8, 0))
            tk.Label(top, text=e.get("name", "?"), bg=PANEL, fg=TEXT,
                     font=(MONO_FONT, 11, "bold")).pack(side="left")
            tk.Label(top, text=f"  ·  {e.get('language', '?')}",
                     bg=PANEL, fg="#a78bfa",
                     font=(MONO_FONT, 10)).pack(side="left")
            tk.Label(top, text=e.get("updated", e.get("created", "")),
                     bg=PANEL, fg=TEXT_FAINT,
                     font=(MONO_FONT, 9)).pack(side="right")
            prog = "   ".join(
                f"{lbl.split()[0]} {'✓' if st.get(key) else '·'}"
                for lbl, key in self._STAGES[1:])
            bot = tk.Frame(card, bg=PANEL)
            bot.pack(fill="x", padx=12, pady=(2, 8))
            tk.Label(bot, text=prog, bg=PANEL, fg=TR_ACCENT,
                     font=(MONO_FONT, 9)).pack(side="left")
            tk.Label(bot, text=e.get("outdir", ""), bg=PANEL, fg=TEXT_FAINT,
                     font=(MONO_FONT, 8)).pack(side="right")

            def _bind_all(w, i=idx):
                w.bind("<Button-1>",  lambda _e2, j=i: self._pm_select(j))
                w.bind("<Double-1>",  lambda _e2, j=i: self._pm_open(j))
                for c in w.winfo_children():
                    _bind_all(c, i)
            _bind_all(card)
            self._pm_cards.append(card)

        self._pm_status.config(text=f"{len(self._pm_entries)} project(s).")

    def _pm_select(self, idx: int):
        self._pm_selected = idx
        for i, card in enumerate(self._pm_cards):
            card.config(highlightbackground=(
                "#3b82f6" if i == idx else PANEL_BORDER),
                highlightthickness=(2 if i == idx else 1))

    def _pm_open(self, idx: int):
        try:
            self._open_project(self._pm_entries[idx])
        except Exception as ex:
            self._pm_status.config(text=f"Could not open project: {ex}",
                                   fg="#f87171")

    def _pm_open_selected(self):
        if self._pm_selected is None:
            self._pm_status.config(text="Select a project first "
                                   "(click a card).", fg="#fbbf24")
            return
        self._pm_open(self._pm_selected)

    def _pm_open_folder(self):
        if self._pm_selected is None:
            self._pm_status.config(text="Select a project first.",
                                   fg="#fbbf24")
            return
        path = self._pm_entries[self._pm_selected].get("outdir", "")
        if not os.path.isdir(path):
            self._pm_status.config(text=f"Folder missing: {path}",
                                   fg="#f87171")
            return
        import subprocess
        try:
            if IS_WINDOWS:
                os.startfile(path)          # noqa — Windows only
            elif IS_MAC:
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as ex:
            self._pm_status.config(text=f"Could not open folder: {ex}",
                                   fg="#f87171")

    def _pm_delete_selected(self):
        if self._pm_selected is None:
            self._pm_status.config(text="Select a project first.",
                                   fg="#fbbf24")
            return
        e = self._pm_entries[self._pm_selected]
        if not messagebox.askyesno(
                "Remove project",
                f"Remove “{e.get('name')}” from the project list?\n\n"
                "Files on disk are NOT deleted."):
            return
        _project_remove(e.get("base", ""), e.get("language", ""))
        self._pm_refresh()

    def _pm_new_project(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("New Project")
        dlg.configure(bg=PANEL)
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()

        body = tk.Frame(dlg, bg=PANEL, padx=16, pady=14)
        body.pack(fill="both", expand=True)
        tk.Label(body, text="NEW PROJECT", bg=PANEL, fg="#a78bfa",
                 font=(MONO_FONT, 11, "bold")).pack(anchor="w", pady=(0, 10))

        name_var  = tk.StringVar()
        audio_var = tk.StringVar()
        lang_var  = tk.StringVar(value=self._current_language())

        def _row(label):
            r = tk.Frame(body, bg=PANEL)
            r.pack(fill="x", pady=4)
            tk.Label(r, text=label, bg=PANEL, fg=TEXT_FAINT, width=14,
                     anchor="w", font=(MONO_FONT, 9)).pack(side="left")
            return r

        r = _row("Project name:")
        tk.Entry(r, textvariable=name_var, width=38, bg=INPUT_BG,
                 fg=INPUT_FG, insertbackground=INPUT_FG, relief="flat",
                 font=(MONO_FONT, 9)).pack(side="left", ipady=3)

        r = _row("English audio:")
        tk.Entry(r, textvariable=audio_var, width=30, bg=INPUT_BG,
                 fg=INPUT_FG, insertbackground=INPUT_FG, relief="flat",
                 font=(MONO_FONT, 9)).pack(side="left", ipady=3)

        def _browse():
            p = filedialog.askopenfilename(
                parent=dlg, title="Choose English audio",
                filetypes=[("Audio Files",
                            "*.wav *.mp3 *.flac *.ogg *.aiff *.aif *.m4a"),
                           ("All Files", "*.*")])
            if p:
                audio_var.set(p)
                if not name_var.get().strip():
                    name_var.set(os.path.splitext(os.path.basename(p))[0])
        tk.Button(r, text="Browse…", command=_browse, bg=PANEL3,
                  fg=_btn_fg(TEXT), relief="flat", cursor="hand2",
                  font=(MONO_FONT, 8)).pack(side="left", padx=(6, 0))

        r = _row("Language:")
        ttk.Combobox(r, textvariable=lang_var,
                     values=list(TTS_LANGUAGES.keys()), state="readonly",
                     width=16).pack(side="left")

        status = tk.Label(body, text="", bg=PANEL, fg="#f87171",
                          font=(MONO_FONT, 9), anchor="w")
        status.pack(fill="x", pady=(6, 4))

        def _create():
            audio = audio_var.get().strip()
            name  = name_var.get().strip() \
                or os.path.splitext(os.path.basename(audio))[0]
            lang  = lang_var.get()
            if not audio or not os.path.exists(audio):
                status.config(text="Pick a valid audio file first.")
                return
            outdir = _prepare_output_dir(audio)
            base   = os.path.join(
                outdir, os.path.splitext(os.path.basename(audio))[0])
            entry = {"name": name, "language": lang,
                     "audio_path": os.path.abspath(audio),
                     "outdir": os.path.abspath(outdir),
                     "base": os.path.abspath(base)}
            _project_upsert(entry)
            dlg.destroy()
            self._open_project(entry)

        btns = tk.Frame(body, bg=PANEL)
        btns.pack(fill="x", pady=(4, 0))
        tk.Button(btns, text="Create & Open", command=_create,
                  bg="#2563eb", fg=_btn_fg("white"),
                  font=(MONO_FONT, 9, "bold"), relief="flat",
                  cursor="hand2", padx=16).pack(side="right")
        tk.Button(btns, text="Cancel", command=dlg.destroy, bg=PANEL3,
                  fg=_btn_fg(TEXT), font=(MONO_FONT, 9), relief="flat",
                  cursor="hand2", padx=10).pack(side="right", padx=(0, 8))

    def _show_project_manager(self):
        try:
            self._workspace.pack_forget()
        except Exception:
            pass
        self._pm_refresh()
        self._pm_frame.pack(fill="both", expand=True)

    def _open_project(self, e: dict):
        """Open a project: restore language + audio + stage state."""
        self._pm_frame.pack_forget()
        self._workspace.pack(fill="both", expand=True)
        self._current_project = e

        # Language first (drives voice list + prompts + labels)
        lang = e.get("language", "")
        if lang in TTS_LANGUAGES and self._tts_language_var \
                and self._tts_language_var.get() != lang:
            self._tts_language_var.set(lang)
            self._on_tts_language_change()

        # Point every path helper at the project's saved run location BEFORE
        # anything derives paths from the loaded audio file. Without this,
        # reopening a project whose audio lives inside its own output folder
        # made _sf_run_paths()/_prepare_output_dir() nest a new folder inside
        # the old one on every reopen — and Compare/stage previews looked
        # for outputs in the wrong (empty) folder.
        if e.get("base"):
            self._last_run_base = e["base"]
        if e.get("outdir"):
            self._last_run_outdir = e["outdir"]
        if lang:
            self._last_run_language = lang

        # Load the project's audio into the Single File tab (prefer the copy
        # inside the project folder — it always travels with the project).
        audio = ""
        for cand in (e.get("base", "") + os.path.splitext(
                         e.get("audio_path", ""))[1] or "",
                     e.get("audio_path", "")):
            if cand and os.path.exists(cand):
                audio = cand
                break
        if audio and audio != (self.filepath or ""):
            self._load_audio_from_path(audio)

        _project_upsert({k: e.get(k, "") for k in
                         ("name", "language", "audio_path", "outdir", "base")})
        self._scripts_load_project()
        self._refresh_stage_bar()
        self.notebook.select(self.main_tab)

        # Restore pipeline-stage ticks from the files already on disk, and
        # reopen the Compare view when a synced dub exists — so an old
        # project looks exactly like it did right after its last run.
        self._sf_restore_stage_status(e)

    _STAGE_RESTORE_FILES = {
        "S1a": ("{base}.srt",),
        "S1b": ("{base}_FinalScript.txt",),
        "S2":  ("{tts_audio}",),
        "S3a": ("{base}_sync_en.srt",),
        "S3b": ("{base}_sync_te.srt",),
        "S3c": ("{base}_sync_mapping.txt",),
        "S3d": ("{base}_sync_synced.srt",),
        "S3e": ("{synced_audio}",),
    }

    def _sf_restore_stage_status(self, e: dict):
        """Mark pipeline stages Done based on output files found on disk."""
        base = e.get("base", "")
        if not base:
            return
        lang  = e.get("language", "") or self._current_language()
        audio = e.get("audio_path", "") or (base + ".wav")
        outdir = e.get("outdir", "") or os.path.dirname(base)
        try:
            tts_audio    = os.path.join(outdir, _tts_output_name(lang, audio, "_tts"))
            synced_audio = os.path.join(outdir, _tts_output_name(lang, audio, "_synced"))
        except Exception:
            tts_audio = synced_audio = ""
        n_done = 0
        for tag, patterns in self._STAGE_RESTORE_FILES.items():
            found = False
            for pat in patterns:
                p = pat.format(base=base, tts_audio=tts_audio,
                               synced_audio=synced_audio)
                if p and os.path.isfile(p):
                    found = True
                    break
            if found:
                self._set_stage(tag, "✔ Done (restored)", "#22c55e")
                n_done += 1
        if n_done:
            self.tr_status.config(text="Project restored ✓", fg=TR_ACCENT)
        # Synced dub on disk → open the Compare view with both waveforms,
        # once the audio file itself has finished loading in its thread.
        if synced_audio and os.path.isfile(synced_audio):
            def _open_compare(tries=40):
                if self.audio_data is None and tries > 0:
                    self.root.after(250, lambda: _open_compare(tries - 1))
                    return
                self._sf_show_view("compare")
                self._sf_cmp_load()
            self.root.after(400, _open_compare)

    # ── Scripts tab (② Transcript / ③ Translation) ───────────────────────────

    def _build_scripts_tab(self):
        tab = self.scripts_tab

        bar = tk.Frame(tab, bg=PANEL, height=48, bd=0,
                       highlightbackground=PANEL_BORDER, highlightthickness=1)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text="SCRIPTS", bg=PANEL, fg="#a78bfa",
                 font=(MONO_FONT, 10, "bold")).pack(side="left",
                                                    padx=(14, 12), pady=12)
        self._btn(bar, "💾 Save Translation", self._scripts_save_translation,
                  bg="#0f1d14", fg=TR_ACCENT, abg="#1f4d2e"
                  ).pack(side="left", padx=(0, 6), pady=8)
        self._btn(bar, "🔁 Re-Dub with this text…", self._scripts_redub,
                  bg="#172554", fg=REG_LABEL, abg="#1e3a8a"
                  ).pack(side="left", padx=(0, 6), pady=8)
        self._btn(bar, "↻ Reload", self._scripts_load_project,
                  bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT
                  ).pack(side="left", padx=(0, 6), pady=8)
        self._scr_status = tk.Label(bar, text="", bg=PANEL, fg=TEXT_FAINT,
                                    font=(MONO_FONT, 9), anchor="w")
        self._scr_status.pack(side="left", fill="x", expand=True, padx=8)

        panes = tk.Frame(tab, bg=BG)
        panes.pack(fill="both", expand=True, padx=8, pady=8)
        panes.columnconfigure(0, weight=1, uniform="scr")
        panes.columnconfigure(1, weight=1, uniform="scr")
        panes.rowconfigure(1, weight=1)

        tk.Label(panes, text="② ENGLISH TRANSCRIPT (read-only)",
                 bg=BG, fg=REG_EDGE, font=(MONO_FONT, 9, "bold"),
                 anchor="w").grid(row=0, column=0, sticky="ew",
                                  padx=(0, 4), pady=(0, 2))
        self._scr_tr_label = tk.Label(
            panes, text="③ TRANSLATION (editable)",
            bg=BG, fg=TR_ACCENT, font=(MONO_FONT, 9, "bold"), anchor="w")
        self._scr_tr_label.grid(row=0, column=1, sticky="ew",
                                padx=(4, 0), pady=(0, 2))

        self._scr_en_text = scrolledtext.ScrolledText(
            panes, wrap="word", bg=PANEL, fg=TEXT, insertbackground=TEXT,
            font=(MONO_FONT, 10), relief="flat")
        self._scr_en_text.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        self._scr_tr_text = scrolledtext.ScrolledText(
            panes, wrap="word", bg=PANEL, fg=TEXT, insertbackground=TEXT,
            font=(MONO_FONT, 11), relief="flat")
        self._scr_tr_text.grid(row=1, column=1, sticky="nsew", padx=(4, 0))

    def _scripts_paths(self):
        e = self._current_project or {}
        base = e.get("base", "")
        if not base:
            return "", ""
        srt = base + ".srt"
        if not os.path.exists(srt):
            srt = base + "_sync_en.srt"
        return srt, base + "_FinalScript.txt"

    def _scripts_load_project(self):
        srt_path, fs_path = self._scripts_paths()
        lang = (self._current_project or {}).get("language", "")
        self._scr_tr_label.config(
            text=f"③ {lang.upper() or 'TRANSLATION'} (editable)")

        self._scr_en_text.config(state="normal")
        self._scr_en_text.delete("1.0", "end")
        if srt_path and os.path.exists(srt_path):
            with open(srt_path, "r", encoding="utf-8") as f:
                self._scr_en_text.insert("1.0", f.read())
        else:
            self._scr_en_text.insert(
                "1.0", "No English transcript yet.\n\nGo to ① Setup & Run "
                       "and run the pipeline — the transcript appears here "
                       "automatically.")
        self._scr_en_text.config(state="disabled")

        self._scr_tr_text.delete("1.0", "end")
        if fs_path and os.path.exists(fs_path):
            with open(fs_path, "r", encoding="utf-8") as f:
                combined = f.read()
            text = _extract_translation_from_finalscript(combined, lang)
            self._scr_tr_text.insert("1.0", text or combined)
            self._scr_status.config(text=os.path.basename(fs_path),
                                    fg=TEXT_FAINT)
        else:
            self._scr_tr_text.insert(
                "1.0", "No translation yet — run the pipeline from "
                       "① Setup & Run.")
            self._scr_status.config(text="", fg=TEXT_FAINT)

    def _scripts_save_translation(self):
        e = self._current_project
        if not e:
            self._scr_status.config(text="Open a project first.",
                                    fg="#fbbf24")
            return
        _srt, fs_path = self._scripts_paths()
        new_text = self._scr_tr_text.get("1.0", "end-1c").strip()
        if not new_text or not fs_path:
            self._scr_status.config(text="Nothing to save.", fg="#fbbf24")
            return
        lang = e.get("language", "")
        raw_eng = ""
        if os.path.exists(fs_path):
            with open(fs_path, "r", encoding="utf-8") as f:
                combined_old = f.read()
            if combined_old.startswith("=== ENGLISH TRANSCRIPTION ==="):
                raw_eng = combined_old.split(
                    "=== ENGLISH TRANSCRIPTION ===", 1)[1]
                raw_eng = raw_eng.split("===", 1)[0].strip()
            # Version the old file before overwriting (same scheme as re-dub)
            rev = _history_next_redub_rev(e.get("outdir", "")) - 1
            backup = f"{e.get('base')}_FinalScript_v{max(rev, 1):02d}.txt"
            if not os.path.exists(backup):
                try:
                    shutil.copy2(fs_path, backup)
                except Exception:
                    pass
        header = (f"=== ENGLISH TRANSCRIPTION ===\n{raw_eng}\n\n"
                  if raw_eng else "")
        with open(fs_path, "w", encoding="utf-8") as f:
            f.write(header + f"=== {lang.upper()} TRANSLATION ===\n"
                    + new_text)
        self._scr_status.config(text=f"✓ Saved {os.path.basename(fs_path)}",
                                fg=TR_ACCENT)
        self._refresh_stage_bar()

    def _scripts_redub(self):
        """Save the edited translation, then run the existing re-dub flow
        (opens the review window, then TTS + sync — no re-translation)."""
        e = self._current_project
        if not e:
            self._scr_status.config(text="Open a project first.",
                                    fg="#fbbf24")
            return
        self._scripts_save_translation()
        # Select this project's run in History and trigger the re-dub flow.
        self._history_refresh()
        key = (os.path.abspath(e.get("base", "")), e.get("language"))
        for idx, h in enumerate(self._hist_entries):
            if (os.path.abspath(h.get("base", "")), h.get("language")) == key:
                self.hist_tree.selection_set(str(idx))
                self.notebook.select(self.hist_tab)
                self._history_redub_selected()
                return
        # Not in history yet (e.g. project created from an old folder):
        # register it, then retry once.
        _history_record(e.get("audio_path", ""), e.get("outdir", ""),
                        e.get("base", ""), e.get("language", ""), "project")
        self._history_refresh()
        if self.hist_tree.get_children():
            self.hist_tree.selection_set(self.hist_tree.get_children()[0])
            self.notebook.select(self.hist_tab)
            self._history_redub_selected()

    # ─────────────────────────────────────────────────────────────────────────
    #  File picker & audio load
    # ─────────────────────────────────────────────────────────────────────────
    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="Open Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg *.aiff *.aif *.m4a"),
                       ("All Files", "*.*")])
        if not path:
            return
        self._load_audio_from_path(path)

    def _load_audio_from_path(self, path: str):
        """Programmatic version of _pick_file — used by the Project Manager."""
        self.filepath = path
        short = path.replace("\\", "/").split("/")[-1]
        self.file_label.config(text=f"  {short}", fg=TEXT)

        self.transcription_text = ""
        self.transcription_raw  = None
        self.final_srt          = ""
        self.punctuation_result = ""

        for tag in self._stage_vars:
            self._set_stage(tag, "—", TEXT_MUTED)
        self.tr_status.config(text="")
        self.status.config(text=f"Loading audio: {short} …")
        self.root.update()
        threading.Thread(target=self._load_and_draw, daemon=True).start()

    def _load_and_draw(self):
        try:
            y, sr = librosa.load(self.filepath, sr=None, mono=True)
            self.audio_data      = y
            self.sample_rate     = sr
            self.duration        = len(y) / sr
            self.regions         = []
            self.cursor_pos      = 0.0
            self.zoom_slider_val = 50.0
            self.scroll_pos      = 0.0
            self.zoom_var.set(50.0)
            self.scroll_var.set(0.0)
            self.root.after(0, self._render_waveform)
            self.root.after(0, lambda: self.info_label.config(
                text=f"SR: {sr} Hz  |  {self.duration:.2f}s  |  Mono"))
            self.root.after(0, lambda: self.cursor_label.config(text="Cursor: 0.00s"))
            self.root.after(0, lambda: self.status.config(
                text="Waveform loaded — detecting regions automatically…"))
            self.root.after(100, self._auto_detect_regions)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Load Error", str(e)))

    def _auto_detect_regions(self):
        try:
            thr_db = float(self.thr_var.get())
            hys_db = float(self.hys_var.get())
            min_ms = int(self.minsilence_var.get())
        except ValueError:
            return
        threading.Thread(target=self._detect_regions,
                         args=(thr_db, hys_db, min_ms), daemon=True).start()

    def _detect_regions(self, thr_db, hys_db, min_ms):
        regions = _detect_regions_from_audio(self.audio_data, self.sample_rate,
                                             thr_db, hys_db, min_ms)
        self.regions = regions
        count = len(regions)
        self.root.after(0, self._render_waveform)
        self.root.after(0, lambda: self.region_count_label.config(
            text=f"{count} region{'s' if count != 1 else ''} found"))
        self.root.after(0, lambda: self.status.config(
            text=(f"{count} region{'s' if count != 1 else ''} detected  |  "
                  f"thr {thr_db:.0f} dBFS  |  hys {hys_db:.1f} dB  |  "
                  f"min sil {min_ms} ms  |  "
                  "Click '▶ Start Full Pipeline' to begin")))

    def _apply_regions(self):
        if self.audio_data is None:
            messagebox.showwarning("No audio", "Open an audio file first.")
            return
        try:
            thr_db = float(self.thr_var.get())
            hys_db = float(self.hys_var.get())
            min_ms = int(self.minsilence_var.get())
        except ValueError:
            return
        threading.Thread(target=self._detect_regions,
                         args=(thr_db, hys_db, min_ms), daemon=True).start()

    def _clear_regions(self):
        self.regions = []
        self.region_count_label.config(text="")
        self.status.config(text="Regions cleared.")
        self._render_waveform()

    # ─────────────────────────────────────────────────────────────────────────
    #  Waveform rendering
    # ─────────────────────────────────────────────────────────────────────────
    def _render_waveform(self):
        """Coalesced render: momentum-scroll / playhead ticks can fire dozens
        of times per second — schedule ONE draw on the idle loop instead of
        rendering synchronously for every event (this froze scrolling and
        made the whole app feel slow)."""
        if getattr(self, "_render_pending", False):
            return
        self._render_pending = True
        try:
            self.root.after_idle(self._render_waveform_now)
        except Exception:
            self._render_pending = False

    def _render_waveform_now(self):
        self._render_pending = False
        if self.audio_data is None:
            return
        self.ax.clear()
        self._style_axes()
        y, sr, dur = self.audio_data, self.sample_rate, self.duration
        visible    = min(self._slider_to_seconds(self.zoom_slider_val), dur)
        scrollable = max(0.0, dur - visible)
        start_t    = max(0.0, min(self.scroll_pos * scrollable, scrollable))
        end_t      = start_t + visible
        i0         = int(start_t * sr)
        i1         = min(int(end_t * sr), len(y))
        chunk      = y[i0:i1]
        n          = len(chunk)
        if n == 0:
            return

        for idx, (rs, re) in enumerate(self.regions):
            if re < start_t or rs > end_t:
                continue
            rx0 = max(rs, start_t); rx1 = min(re, end_t)
            self.ax.add_patch(Rectangle((rx0, -1.05), rx1-rx0, 2.10,
                              facecolor=REG_FILL, edgecolor="none", alpha=0.55, zorder=1))
            if rs >= start_t: self.ax.axvline(rs, color=REG_EDGE, linewidth=1.3, alpha=0.9, zorder=3)
            if re <= end_t:   self.ax.axvline(re, color=REG_EDGE, linewidth=1.3, alpha=0.9, zorder=3)
            lx = max(rx0, start_t) + (rx1-rx0)*0.02
            self.ax.text(lx, 1.01, f"R{idx+1}  {rs:.2f}s", color=REG_LABEL, fontsize=7,
                         va="bottom", fontfamily="monospace", zorder=4,
                         transform=self.ax.get_xaxis_transform(), clip_on=True)

        TARGET = 1000
        if n > TARGET:
            step   = n // TARGET; frames = n // step
            trimmed = chunk[:frames*step].reshape(frames, step)
            peaks_p = trimmed.max(axis=1); peaks_n = trimmed.min(axis=1)
            t_axis  = np.linspace(start_t, end_t, frames)
        else:
            peaks_p = peaks_n = chunk
            t_axis  = np.linspace(start_t, end_t, n)

        self.ax.fill_between(t_axis, peaks_p, peaks_n, color=WAVEFORM, alpha=0.85, linewidth=0, zorder=2)
        self.ax.plot(t_axis, peaks_p, color="#d97706", linewidth=0.6, alpha=0.9, zorder=2)
        self.ax.plot(t_axis, peaks_n, color="#b45309", linewidth=0.4, alpha=0.7, zorder=2)
        self.ax.axhline(0, color=TEXT_MUTED, linewidth=0.8, zorder=2)

        cp = self.cursor_pos
        if start_t <= cp <= end_t:
            self.ax.axvline(cp, color=CURSOR_C, linewidth=CURSOR_W, alpha=0.95, zorder=5)

        self.ax.set_xlim(start_t, end_t)
        self.ax.set_ylim(-1.05, 1.05)
        self.ax.set_xlabel("Time (s)", color=TEXT, fontsize=9, labelpad=4)

        def fmt_time(x, _):
            m, s = divmod(x, 60)
            return f"{int(m)}:{s:05.2f}" if m else f"{s:.2f}s"
        self.ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_time))
        fname = self.filepath.replace("\\", "/").split("/")[-1]
        self.ax.set_title(f"{fname}  ·  {dur:.2f}s  ·  {sr} Hz",
                          color=TEXT, fontsize=9, pad=6, fontfamily="monospace")
        self.zoom_readout.config(text=f"{visible:.1f}s")
        # NOTE: no tight_layout() here — it is very expensive and this runs
        # on every scroll/zoom/playhead tick. Margins are set once at build.
        self.canvas.draw()
        self._sb_redraw()

    def _draw_placeholder(self):
        self.ax.set_facecolor("#0f172a")
        self.ax.text(0.5, 0.5, "Open an audio file — waveform loads automatically",
                     ha="center", va="center", transform=self.ax.transAxes,
                     color=TEXT_MUTED, fontsize=12, fontfamily="monospace")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        for sp in self.ax.spines.values(): sp.set_edgecolor(GRID)
        self.canvas.draw()

    def _style_axes(self):
        self.ax.set_facecolor("#0f172a")
        self.ax.tick_params(colors=TEXT, labelsize=8)
        for sp in self.ax.spines.values(): sp.set_edgecolor(GRID)
        self.ax.grid(True, axis="x", color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)
        self.ax.grid(True, axis="y", color=GRID, linewidth=0.3, linestyle=":", alpha=0.4)

    # ─────────────────────────────────────────────────────────────────────────
    #  Playback
    # ─────────────────────────────────────────────────────────────────────────
    def _toggle_play(self):
        if self.is_playing: self._stop_playback()
        else:               self._start_playback()

    def _start_playback(self):
        if self.audio_data is None: return
        with self._play_lock:
            if self.is_playing: return
            self.is_playing = True
        self._play_start_pos = self.cursor_pos
        self._play_start_t   = time.perf_counter()
        start_sample = int(self.cursor_pos * self.sample_rate)
        audio_slice  = self.audio_data[start_sample:].astype(np.float32)
        self.play_btn.config(text=" Pause ", bg="#0f1d14")
        def _run():
            try: sd.play(audio_slice, samplerate=self.sample_rate); sd.wait()
            except Exception: pass
            finally: self.root.after(0, self._on_playback_finished)
        threading.Thread(target=_run, daemon=True).start()
        self._render_waveform()

    def _stop_playback(self):
        with self._play_lock:
            if not self.is_playing: return
            self.is_playing = False
        try: sd.stop()
        except Exception: pass
        elapsed = time.perf_counter() - self._play_start_t
        self.cursor_pos = min(self._play_start_pos + elapsed, self.duration)
        self.cursor_label.config(text=f"Cursor: {self.cursor_pos:.3f}s")
        self.play_btn.config(text="  Play  ", bg="#0f1d14")
        self._render_waveform()

    def _on_playback_finished(self):
        with self._play_lock:
            if not self.is_playing: return
            self.is_playing = False
        self.cursor_pos = self.duration
        self.cursor_label.config(text=f"Cursor: {self.cursor_pos:.3f}s")
        self.play_btn.config(text="  Play  ", bg="#0f1d14")
        self._render_waveform()

    def _playhead_tick(self):
        # Bail out cleanly if the window is being torn down — Tk raises
        # `invalid command name "<id>_playhead_tick"` otherwise.
        if getattr(self, "_closing", False):
            return
        try:
            if not self.root.winfo_exists():
                return
        except tk.TclError:
            return

        if self.is_playing:
            elapsed = time.perf_counter() - self._play_start_t
            pos     = min(self._play_start_pos + elapsed, self.duration)
            self.cursor_pos = pos
            self.cursor_label.config(text=f"Cursor: {pos:.3f}s")
            visible    = min(self._slider_to_seconds(self.zoom_slider_val), self.duration)
            scrollable = max(0.0, self.duration - visible)
            if scrollable > 0:
                win_end = self.scroll_pos * scrollable + visible
                if pos > win_end - visible * 0.1:
                    new_scroll = max(0.0, min((pos - visible*0.5) / scrollable, 1.0))
                    if abs(new_scroll - self.scroll_pos) > 0.001:
                        self.scroll_pos = new_scroll
                        self.scroll_var.set(new_scroll)
            self._render_waveform()
        try:
            # 10 fps is plenty for a moving cursor line and keeps the UI
            # responsive (33 ms full-figure redraws made the app crawl).
            self._playhead_after_id = self.root.after(100, self._playhead_tick)
        except tk.TclError:
            return

    # ─────────────────────────────────────────────────────────────────────────
    #  Canvas / waveform interaction
    # ─────────────────────────────────────────────────────────────────────────
    def _on_canvas_click(self, event):
        if self.audio_data is None or event.inaxes != self.ax or event.button != 1: return
        t = event.xdata
        if t is None: return
        t = max(0.0, min(t, self.duration))
        was_playing = self.is_playing
        self._stop_playback()
        self.cursor_pos = t
        self.cursor_label.config(text=f"Cursor: {t:.3f}s")
        self._render_waveform()
        if was_playing: self._start_playback()

    def _on_canvas_hover(self, event):
        if self.audio_data is None or event.inaxes != self.ax: return
        if event.xdata is not None:
            m, s = divmod(event.xdata, 60)
            self.status.config(text=f"  {int(m):02d}:{s:06.3f}  (click to place cursor)")

    def _on_waveform_scroll(self, event):
        if self.audio_data is None: return
        ctrl_held = bool(event.state & 0x4)
        num   = getattr(event, "num", 0)
        delta = getattr(event, "delta", 0)
        # Button-6/7 = X11 horizontal wheel; Shift+wheel = horizontal on
        # macOS trackpads (two-finger swipe) and Windows mice.
        up = (delta > 0) or (num in (4, 6))
        if ctrl_held: self._zoom_by_delta(5.0 if up else -5.0)
        else:         self._scroll_by(-0.05 if up else 0.05)

    # ─────────────────────────────────────────────────────────────────────────
    #  Scrollbar
    # ─────────────────────────────────────────────────────────────────────────
    def _sb_thumb_geometry(self):
        w = self.sb_canvas.winfo_width()
        if w < 2 or self.duration <= 0: return 0, w
        visible    = min(self._slider_to_seconds(self.zoom_slider_val), self.duration)
        ratio      = visible / self.duration
        thumb_w    = max(20, int(w * ratio))
        scrollable = max(1, w - thumb_w)
        x0         = int(self.scroll_pos * scrollable)
        return x0, x0 + thumb_w

    def _sb_redraw(self):
        w = self.sb_canvas.winfo_width(); h = self.sb_canvas.winfo_height()
        if w < 2: return
        self.sb_canvas.delete("all")
        self.sb_canvas.create_rectangle(0, 0, w, h, fill="#334155", outline="")
        x0, x1 = self._sb_thumb_geometry()
        self.sb_canvas.create_rectangle(x0+1, 2, x1-1, h-2, fill=TEXT_MUTED,
                                        outline=TEXT_MUTED, width=1)

    def _sb_on_configure(self, _event=None): self._sb_redraw()
    def _sb_on_press(self, event):
        x0, x1 = self._sb_thumb_geometry()
        if x0 <= event.x <= x1:
            self._sb_drag_start_x   = event.x
            self._sb_drag_start_pos = self.scroll_pos
        else:
            w = self.sb_canvas.winfo_width()
            visible  = min(self._slider_to_seconds(self.zoom_slider_val), self.duration)
            thumb_w  = max(20, int(w * (visible / max(self.duration, 0.001))))
            scrollable_px = max(1, w - thumb_w)
            new_pos  = max(0.0, min(1.0, (event.x - thumb_w/2) / scrollable_px))
            self.scroll_pos = new_pos; self.scroll_var.set(new_pos)
            self._sb_drag_start_x = event.x; self._sb_drag_start_pos = new_pos
            self._render_waveform()

    def _sb_on_drag(self, event):
        if self._sb_drag_start_x is None: return
        w = self.sb_canvas.winfo_width()
        visible  = min(self._slider_to_seconds(self.zoom_slider_val), self.duration)
        thumb_w  = max(20, int(w * (visible / max(self.duration, 0.001))))
        scrollable_px = max(1, w - thumb_w)
        dx = event.x - self._sb_drag_start_x
        new_pos = max(0.0, min(1.0, self._sb_drag_start_pos + dx / scrollable_px))
        self.scroll_pos = new_pos; self.scroll_var.set(new_pos)
        self._render_waveform()

    def _sb_on_release(self, _event=None): self._sb_drag_start_x = None

    # ─────────────────────────────────────────────────────────────────────────
    #  Zoom / scroll
    # ─────────────────────────────────────────────────────────────────────────
    def _init_zoom_scroll_vars(self):
        self.zoom_var    = tk.DoubleVar(value=50.0)
        self.scroll_var  = tk.DoubleVar(value=0.0)
        self.zoom_readout = tk.Label(self.root, bg=BG)

    def _slider_to_seconds(self, val):
        t = val / 100.0
        return math.exp(math.log(60.0) + (math.log(1.0) - math.log(60.0)) * t)

    def _zoom_by_delta(self, delta):
        new_val = max(0.0, min(100.0, self.zoom_slider_val + delta))
        self.zoom_slider_val = new_val; self.zoom_var.set(new_val)
        self._render_waveform()

    def _scroll_by(self, delta):
        self.scroll_pos = max(0.0, min(1.0, self.scroll_pos + delta))
        self.scroll_var.set(self.scroll_pos); self._render_waveform(); self._sb_redraw()

    def _reset_view(self):
        self.zoom_slider_val = 50.0; self.scroll_pos = 0.0
        self.zoom_var.set(50.0); self.scroll_var.set(0.0)
        self._render_waveform(); self._sb_redraw()

    # ─────────────────────────────────────────────────────────────────────────
    #  Utility
    # ─────────────────────────────────────────────────────────────────────────
    # ── Border colours for known button backgrounds (slightly darker shades).
    # These give every button a thin outline matching its tone, which (along
    # with relief="raised" + a 2-px bevel) approximates a soft gradient look
    # within the limits of plain tk.Button.
    _BTN_BORDER_BY_BG = {
        BTN_BG:    "#334155",   # default slate button   → slate-700 border
        "#0f1d14": "#1f4d2e",   # success green dark     → green-900 border
        "#1f1213": "#7f1d1d",   # alert red dark         → red-900 border
        "#172554": "#1e3a8a",   # info blue dark         → blue-900 border
        "#0f1d20": "#0e3a35",   # sync teal dark         → teal-900 border
        "#1e1b3a": "#5b4fbf",   # tts violet dark        → violet-500 border
    }

    def _btn(self, parent, text, cmd, bg=BTN_BG, fg=BTN_FG, abg=BTN_ACT,
             border=None):
        if border is None:
            border = self._BTN_BORDER_BY_BG.get(bg, "#334155")
        fg = _btn_fg(fg)
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg=fg, activebackground=abg,
                         activeforeground=fg, relief="raised", bd=2,
                         highlightbackground=border,
                         highlightcolor=border,
                         highlightthickness=1,
                         font=(MONO_FONT, 10, "bold"),
                         padx=10, pady=4, cursor="hand2")

    def _check_api_key_badge(self):
        try:
            key = _get_api_key()
            self.api_badge.config(
                text=f"ElevenLabs: {_redact_api_key(key)} ✔",
                fg=TR_ACCENT)
        except Exception:
            self.api_badge.config(
                text="ElevenLabs: paste API key above",
                fg="#f87171")

    def _on_close(self):
        # Mark the app as closing so any in-flight `after` callbacks bail out
        # before they try to touch destroyed widgets.
        self._closing = True
        try:
            after_id = getattr(self, "_playhead_after_id", None)
            if after_id is not None:
                self.root.after_cancel(after_id)
        except Exception:
            pass
        try: sd.stop()
        except Exception: pass
        try:
            self.root.destroy()
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    # Log + show GUI errors instead of printing to a console nobody sees.
    root.report_callback_exception = _tk_exception_handler
    app  = EndToEndApp(root)

    if FFMPEG_PATH is None:
        if IS_WINDOWS:
            _ff_hint = ("Open PowerShell and run:\n    winget install ffmpeg\n"
                        "then restart the computer (so PATH updates).")
        elif IS_MAC:
            _ff_hint = "Open Terminal and run:\n    brew install ffmpeg"
        else:
            _ff_hint = "Install it with your package manager, e.g. apt install ffmpeg"
        root.after(800, lambda: messagebox.showwarning(
            "FFmpeg Not Found",
            "FFmpeg is not installed (or not on PATH).\n\n"
            "MP3 loading and audio export will fail without it.\n\n" + _ff_hint))

    root.mainloop()
