"""Shared helpers for LTX Video Studio: folders, model files, the resumable downloader and the Python-environment setup.

Only the Python standard library is used here, so this works before any packages are installed.

Folders (all relative to the program folder unless you override them):
  LTX_HOME    program folder (default: the folder this file is in)
  LTX_MODELS  where the video models live (default: <LTX_HOME>/models)
  LTX_PYLIBS  optional extra folder of Python packages to put on sys.path (advanced)
"""
import importlib.metadata
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

NO_WINDOW = 0x08000000 if os.name == "nt" else 0

HOME = os.path.abspath(os.environ.get("LTX_HOME") or os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.abspath(os.environ.get("LTX_MODELS") or os.path.join(HOME, "models"))
OUT = os.path.join(HOME, "outputs")
PROJECTS = os.path.join(HOME, "projects")
SETTINGS = os.path.join(HOME, "app_settings.json")
VENV = os.path.join(HOME, ".venv")
LTX_DIR = os.path.join(MODELS, "ltx")
CFG_DIR = os.path.join(MODELS, "ltx-cfg")
CKPT = os.path.join(LTX_DIR, "ltx-video-2b-v0.9.1.safetensors")
T5_FILE = os.path.join(LTX_DIR, "t5xxl_fp8_e4m3fn.safetensors")
T5_CFG = os.path.join(LTX_DIR, "text_encoder")
TOKENIZER = os.path.join(LTX_DIR, "tokenizer")

for extra in (os.path.join(HOME, "pylibs"), os.environ.get("LTX_PYLIBS", "")):
    if extra and os.path.isdir(extra) and extra not in sys.path:
        sys.path.insert(0, extra)

HF = "https://huggingface.co"
# (path under the models folder, download address, exact size in bytes or None for small text files)
FILES = [
    ("ltx/ltx-video-2b-v0.9.1.safetensors", f"{HF}/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.1.safetensors",
     5716863844),
    ("ltx/t5xxl_fp8_e4m3fn.safetensors", f"{HF}/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
     4893934904),
    ("ltx/text_encoder/config.json", f"{HF}/Lightricks/LTX-Video/resolve/main/text_encoder/config.json", None),
    ("ltx/tokenizer/added_tokens.json", f"{HF}/Lightricks/LTX-Video/resolve/main/tokenizer/added_tokens.json", None),
    ("ltx/tokenizer/special_tokens_map.json",
     f"{HF}/Lightricks/LTX-Video/resolve/main/tokenizer/special_tokens_map.json", None),
    ("ltx/tokenizer/spiece.model", f"{HF}/Lightricks/LTX-Video/resolve/main/tokenizer/spiece.model", None),
    ("ltx/tokenizer/tokenizer_config.json", f"{HF}/Lightricks/LTX-Video/resolve/main/tokenizer/tokenizer_config.json",
     None),
    ("ltx-cfg/model_index.json", f"{HF}/diffusers/LTX-Video-0.9.1/resolve/main/model_index.json", None),
    ("ltx-cfg/transformer/config.json", f"{HF}/diffusers/LTX-Video-0.9.1/resolve/main/transformer/config.json", None),
    ("ltx-cfg/vae/config.json", f"{HF}/diffusers/LTX-Video-0.9.1/resolve/main/vae/config.json", None),
    ("ltx-cfg/scheduler/scheduler_config.json",
     f"{HF}/diffusers/LTX-Video-0.9.1/resolve/main/scheduler/scheduler_config.json", None),
]
TOTAL_BYTES = sum(s for _, _, s in FILES if s)
USER_AGENT = "ltx-video-studio/4"


def fmt_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024


def fmt_time(sec):
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


# ------------------------------------------------------------------ model files
def _path(rel):
    return os.path.join(MODELS, *rel.split("/"))


def file_ok(rel, size):
    p = _path(rel)
    if not os.path.isfile(p):
        return False
    n = os.path.getsize(p)
    if size:
        return n == size
    if n <= 0:
        return False
    if rel.endswith(".json"):
        try:
            with open(p, encoding="utf-8") as f:
                json.load(f)
        except (OSError, ValueError):
            return False
    return True


def model_status():
    """Returns (everything_present, [(rel, size, ok, bytes_on_disk_including_partial)])."""
    rows = []
    for rel, _url, size in FILES:
        ok = file_ok(rel, size)
        have = os.path.getsize(_path(rel)) if ok else 0
        part = _path(rel) + ".part"
        if not ok and os.path.isfile(part):
            have = os.path.getsize(part)
        rows.append((rel, size, ok, have))
    return all(r[2] for r in rows), rows


def bytes_missing():
    _ok, rows = model_status()
    return sum((size or 0) - have for _rel, size, ok, have in rows if not ok and size)


class Cancelled(Exception):
    pass


def _download_one(url, dest, expected, on_bytes, cancel):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    attempt = 0
    while True:
        if cancel.is_set():
            raise Cancelled()
        have = os.path.getsize(part) if os.path.isfile(part) else 0
        if expected and have > expected:  # a broken partial file of ours: start this file again
            os.remove(part)
            have = 0
        if expected and have == expected:
            break
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
                if have and r.status != 206:  # the server ignored the resume request: start over
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        if cancel.is_set():
                            raise Cancelled()
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        on_bytes(len(chunk))
            break
        except Cancelled:
            raise
        except urllib.error.HTTPError as e:
            if e.code == 416 and expected and os.path.isfile(part) and os.path.getsize(part) >= expected:
                break
            if e.code in (401, 403, 404):
                raise RuntimeError(f"The server refused the download ({e.code}) for {url}")
            attempt += 1
        except (urllib.error.URLError, OSError, socket.timeout, TimeoutError, ConnectionError):
            attempt += 1
        except Exception:  # e.g. a connection cut mid-read (http.client.IncompleteRead)
            attempt += 1
        if attempt > 40:
            raise RuntimeError("The connection keeps failing. Check your internet and press Download again - "
                               "what is already downloaded is kept.")
        for _ in range(min(30, 2 * attempt)):  # wait, then resume from where the file stopped
            if cancel.is_set():
                raise Cancelled()
            time.sleep(1)
    got = os.path.getsize(part)
    if expected and got != expected:
        os.remove(part)
        raise RuntimeError(f"{os.path.basename(dest)} downloaded with the wrong size ({got} instead of {expected}). "
                           f"Press Download again.")
    os.replace(part, dest)


def download_models(progress=None, cancel=None):
    """Downloads every missing model file. Safe to stop and start again: it carries on from the partial file.

    progress(info) gets a dict: file, done, total, speed (bytes/s), eta (seconds or None).
    Returns True when everything is present. Raises Cancelled, RuntimeError.
    """
    cancel = cancel or threading.Event()
    ok, rows = model_status()
    if ok:
        return True
    need = bytes_missing()
    probe = MODELS
    while not os.path.isdir(probe):
        probe = os.path.dirname(probe)
    free = shutil.disk_usage(probe).free
    if free < need + (1 << 30):
        raise RuntimeError(f"Not enough free disk space: the models need {fmt_bytes(need)} more but only "
                           f"{fmt_bytes(free)} is free on that drive.")
    total_need = sum(s for _r, s, ok_, _h in rows if s and not ok_)
    state = {"done": sum(h for _r, s, ok_, h in rows if s and not ok_), "file": "", "t": time.time(),
             "marks": [(time.time(), 0)], "sent": 0}
    state["base"] = state["done"]

    def on_bytes(n):
        state["done"] += n
        state["sent"] += n
        now = time.time()
        state["marks"].append((now, state["sent"]))
        while len(state["marks"]) > 2 and now - state["marks"][0][0] > 10:
            state["marks"].pop(0)
        if progress and now - state["t"] >= 0.25:
            state["t"] = now
            (t0, b0), (t1, b1) = state["marks"][0], state["marks"][-1]
            speed = (b1 - b0) / (t1 - t0) if t1 > t0 else 0
            left = max(0, total_need - state["done"]) if total_need else 0
            progress({"file": state["file"], "done": state["done"], "total": total_need,
                      "speed": speed, "eta": (left / speed) if speed > 1 else None})

    for rel, url, size in FILES:
        if file_ok(rel, size):
            continue
        state["file"] = rel
        _download_one(url, _path(rel), size, on_bytes, cancel)
    if progress:
        progress({"file": "", "done": total_need, "total": total_need, "speed": 0, "eta": 0})
    return model_status()[0]


# ---------------------------------------------------------- python environment
# Exact versions this program was tested with.
TORCH_SPEC = "torch==2.1.2"
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
PIP_PACKAGES = [
    "numpy==1.26.2", "pillow>=9.5", "imageio==2.38.0", "imageio-ffmpeg==0.8.1", "diffusers==0.32.2",
    "transformers==4.46.1", "tokenizers==0.20.3", "accelerate==0.31.0", "safetensors==0.4.2",
    "sentencepiece==0.2.3", "protobuf>=3.20,<5", "huggingface-hub==0.26.2",
]
REQUIRED_MODULES = {"torch": "torch", "numpy": "numpy", "PIL": "pillow", "imageio": "imageio",
                    "imageio_ffmpeg": "imageio-ffmpeg", "diffusers": "diffusers", "transformers": "transformers",
                    "accelerate": "accelerate", "safetensors": "safetensors", "sentencepiece": "sentencepiece"}


def missing_packages():
    missing = [mod for mod in REQUIRED_MODULES if importlib.util.find_spec(mod) is None]
    if "diffusers" not in missing:
        try:
            if tuple(int(x) for x in importlib.metadata.version("diffusers").split(".")[:2]) < (0, 32):
                missing.append("diffusers (needs 0.32 or newer)")
        except Exception:
            missing.append("diffusers")
    return missing


def venv_python(windowed=False):
    return os.path.join(VENV, "Scripts", "pythonw.exe" if windowed else "python.exe")


def engine_python():
    """A console python.exe in the same environment as the running program (pythonw has no output pipes)."""
    exe = sys.executable
    if os.path.basename(exe).lower() == "pythonw.exe":
        sibling = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.isfile(sibling):
            return sibling
    return exe


def has_nvidia():
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=15, creationflags=NO_WINDOW)
        return r.returncode == 0 and b"GPU" in r.stdout
    except Exception:
        return False


def _stream(cmd, log, cancel):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
    for raw in iter(p.stdout.readline, b""):
        if cancel.is_set():
            p.kill()
            raise Cancelled()
        line = raw.decode("utf-8", "replace").rstrip()
        if line:
            log(line)
    return p.wait()


def bootstrap_env(log, cancel=None):
    """Creates <program folder>/.venv and installs everything the program needs. Returns the venv's pythonw path."""
    cancel = cancel or threading.Event()
    if sys.version_info < (3, 10) or sys.version_info >= (3, 12):
        raise RuntimeError(f"This is Python {sys.version_info.major}.{sys.version_info.minor}. The video engine needs "
                           f"Python 3.10 or 3.11 (64-bit). Install Python 3.11 from python.org and start again.")
    py = venv_python()
    if not os.path.isfile(py):
        log("Creating a private Python environment (.venv)...")
        if _stream([sys.executable, "-m", "venv", VENV], log, cancel) != 0:
            raise RuntimeError("Could not create the Python environment (is the 'venv' part of Python installed?).")
    log("Updating pip...")
    _stream([py, "-m", "pip", "install", "--upgrade", "pip", "--progress-bar", "off", "--disable-pip-version-check"],
            log, cancel)
    if _stream([py, "-c", "import torch"], lambda s: None, cancel) != 0:
        log("Downloading PyTorch with NVIDIA support (about 2.4 GB). This is the longest step - please wait...")
        if _stream([py, "-m", "pip", "install", TORCH_SPEC, "--index-url", TORCH_INDEX, "--progress-bar", "off",
                    "--disable-pip-version-check"], log, cancel) != 0:
            raise RuntimeError("Installing PyTorch failed. See the log above, check your internet and try again.")
    log("Installing the other packages...")
    if _stream([py, "-m", "pip", "install", *PIP_PACKAGES, "--progress-bar", "off", "--disable-pip-version-check"],
               log, cancel) != 0:
        raise RuntimeError("Installing the packages failed. See the log above, check your internet and try again.")
    log("Checking the installation...")
    out = []
    code = _stream([py, "-c", "import torch,diffusers,transformers;print('torch',torch.__version__,'| NVIDIA GPU:',"
                    "torch.cuda.is_available(),'| diffusers',diffusers.__version__)"], out.append, cancel)
    if code != 0:
        raise RuntimeError("The packages installed but could not be loaded: " + " ".join(out)[-300:])
    log(out[-1] if out else "ok")
    return venv_python(windowed=True)


def bootstrap_window():
    """Small window that sets up the Python environment. Returns the venv's pythonw path, or None if not finished."""
    import queue
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("LTX Video Studio - first-time setup")
    root.geometry("760x420")
    q = queue.Queue()
    cancel = threading.Event()
    ttk.Label(root, text="First-time setup", font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=14, pady=(12, 0))
    ttk.Label(root, wraplength=720, justify="left", text=(
        "The program needs a few Python packages (the biggest is PyTorch, about 2.4 GB). They are installed into a "
        "private folder next to the program and nothing else on your PC is changed. When this finishes the program "
        "starts by itself. You can close this window and start the program again later to carry on.")).pack(
        anchor="w", padx=14, pady=6)
    bar = ttk.Progressbar(root, mode="indeterminate")
    bar.pack(fill="x", padx=14, pady=(4, 6))
    bar.start(12)
    txt = tk.Text(root, height=14, wrap="word", state="disabled", font=("Consolas", 9), bg="#101010", fg="#d0d0d0")
    txt.pack(fill="both", expand=True, padx=14, pady=(0, 12))
    state = {"done": False, "error": None, "py": None}

    def work():
        try:
            state["py"] = bootstrap_env(q.put, cancel)
            state["done"] = True
        except Cancelled:
            state["error"] = "Cancelled."
        except Exception as e:
            state["error"] = str(e)
        q.put(None)

    def poll():
        try:
            while True:
                line = q.get_nowait()
                if line is None:
                    bar.stop()
                    if state["done"]:
                        root.destroy()
                        return
                    bar.configure(mode="determinate", value=0)
                    txt.configure(state="normal")
                    txt.insert("end", "\nSETUP DID NOT FINISH: " + str(state["error"]) + "\n")
                    txt.configure(state="disabled")
                    txt.see("end")
                    return
                txt.configure(state="normal")
                txt.insert("end", line + "\n")
                txt.configure(state="disabled")
                txt.see("end")
        except queue.Empty:
            pass
        root.after(150, poll)

    def on_close():
        cancel.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    threading.Thread(target=work, daemon=True).start()
    root.after(150, poll)
    root.mainloop()
    return state["py"] if state["done"] else None
