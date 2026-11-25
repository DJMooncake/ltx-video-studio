"""LTX Video Studio v4 - shot list, batch generation, last-frame chaining and joining (offline).

Builds long videos from many short shots. The heavy models load ONCE (ltx_batch.py) and then every shot in the list
is made in turn. Finished shots are remembered, so a stopped or crashed run carries on where it left off.

New in v4: the program sets itself up. On the first start it creates its own Python environment and offers to
download the video models (about 10.6 GB) with a progress bar. See README.md.
"""
import datetime
import hashlib
import json
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import ltx_common as C  # standard library only, so it works before any package is installed

ROOT = C.HOME
BATCH = os.path.join(ROOT, "ltx_batch.py")
OUT = C.OUT
PROJECTS = C.PROJECTS
SETTINGS = C.SETTINGS
NO_WINDOW = C.NO_WINDOW
APP_NAME = "LTX Video Studio v4"
os.makedirs(OUT, exist_ok=True)
os.makedirs(PROJECTS, exist_ok=True)

SIZES = {
    "Landscape 512x320 (fastest)": (512, 320),
    "Square 512x512": (512, 512),
    "Wide 704x384": (704, 384),
    "Landscape 768x512 (needs more memory)": (768, 512),
    "Portrait 320x512": (320, 512),
}
TYPES = {"t2v": "Text to video", "i2v": "Picture to video", "cont": "Continue previous shot"}
TYPE_TAG = {"t2v": "TEXT", "i2v": "PIC ", "cont": "NEXT"}
DEFAULT_NEG = "worst quality, inconsistent motion, blurry, jittery, distorted"
PROGRESS_RE = re.compile(r"(\d+)/(\d+)\s*\[")
FPS = 24

DARK = dict(bg="#1e1f22", fg="#e6e6e6", dim="#9aa0a6", field="#383a40", btn="#3b3f46", btn_hover="#4a4f58",
            border="#4a4d55", trough="#15161a", accent="#4f8cff", log_bg="#101010", log_fg="#d0d0d0")
LIGHT = dict(bg="#f0f0f0", fg="#000000", dim="#555555", field="#ffffff", btn="#e1e1e1", btn_hover="#d0d0d0",
             border="#adadad", trough="#d9d9d9", accent="#3b78e7", log_bg="#101010", log_fg="#d0d0d0")


def frames_for(seconds):
    return max(9, round(seconds * FPS / 8) * 8 + 1)


def fmt_time(sec):
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def est_seconds(frames, steps, w, h):
    # measured on one PC (RTX 3060 12 GB): about 0.6 s per frame at 512x320 with 30 steps
    return frames * 0.6 * (steps / 30) * (w * h / (512 * 320))


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME + "  (offline)")
        self.geometry("1720x940")
        self.minsize(1400, 780)
        self.proc = None
        self.q = queue.Queue()
        self.reader = None
        self.play_job = None
        self.preview_path = None
        self.shot_img = None
        self.thumb = None
        self.t_start = 0
        self.shots = []
        self.sel = None
        self.loading = False
        self.done_times = []
        self.run_total = 0
        self.run_error = None
        self.cur_dir = None
        self.last_joined = None
        self.models_ok = False
        self.dl_cancel = None
        self.dl_thread = None

        st = ttk.Style(self)
        st.theme_use("clam")

        self.project = tk.StringVar(value="my_video")
        self.size = tk.StringVar(value=list(SIZES)[0])
        self.seconds = tk.DoubleVar(value=5)
        self.steps = tk.DoubleVar(value=30)
        self.cfg = tk.DoubleVar(value=3.0)
        self.seed = tk.StringVar(value="1")
        self.rand_seed = tk.BooleanVar(value=False)
        self.resume = tk.BooleanVar(value=True)
        self.autojoin = tk.BooleanVar(value=True)
        self.dark = tk.BooleanVar(value=True)
        self.s_type = tk.StringVar(value="t2v")
        self.s_image = tk.StringVar(value="")
        self.status = tk.StringVar(value="Ready.")
        self.summary = tk.StringVar(value="")
        self._sliders = []

        self.build()
        self.load_settings()
        if not self.shots:
            self.shots = [{"type": "t2v", "prompt": "A small campfire at night, sparks rising into the dark sky, "
                           "cinematic", "image": ""}]
        self.refresh_list(select=0)
        self.sync_sliders()
        self.apply_theme()
        self.refresh_history()
        self.update_summary()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self.poll)
        self.after(300, self.check_models)

    # ---------------------------------------------------------------- layout
    def build(self):
        self.setup_bar = ttk.LabelFrame(self, text="First-time setup: the video models are not on this PC yet",
                                        padding=8)
        sb_top = ttk.Frame(self.setup_bar)
        sb_top.pack(fill="x")
        self.dl_text = tk.StringVar(value="")
        ttk.Label(sb_top, textvariable=self.dl_text, wraplength=1100, justify="left").pack(side="left", fill="x",
                                                                                          expand=True)
        self.dl_btn = ttk.Button(sb_top, text="Download the models now", style="Big.TButton",
                                 command=self.start_download)
        self.dl_btn.pack(side="right", padx=(8, 0))
        self.dl_pause = ttk.Button(sb_top, text="Pause", command=self.pause_download, state="disabled")
        self.dl_pause.pack(side="right")
        self.dl_bar = ttk.Progressbar(self.setup_bar, maximum=100)
        self.dl_bar.pack(fill="x", pady=(6, 0))
        self.dl_status = tk.StringVar(value="")
        ttk.Label(self.setup_bar, textvariable=self.dl_status, style="Dim.TLabel").pack(anchor="w", pady=(2, 0))

        colA = self.colA = ttk.Frame(self, padding=(10, 10, 4, 10))
        colA.pack(side="left", fill="y")
        colB = ttk.Frame(self, padding=(4, 10, 4, 10))
        colB.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=(4, 10, 10, 10))
        right.pack(side="left", fill="both", expand=True)

        # ---- column A: project + shot list + shot editor
        pbox = ttk.LabelFrame(colA, text="1. Your project", padding=8)
        pbox.pack(fill="x")
        row = ttk.Frame(pbox)
        row.pack(fill="x")
        ttk.Label(row, text="Name").pack(side="left")
        ttk.Entry(row, textvariable=self.project, width=22).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(row, text="Open...", command=self.open_project).pack(side="left")
        ttk.Button(row, text="Save...", command=self.save_project).pack(side="left", padx=(4, 0))
        ttk.Label(pbox, text=f"Every shot is saved in a folder with this name, inside {OUT}",
                  style="Dim.TLabel", wraplength=460, justify="left").pack(anchor="w", pady=(4, 0))

        lbox = ttk.LabelFrame(colA, text="2. Shot list (each shot is one short clip)", padding=8)
        lbox.pack(fill="x", pady=(8, 0))
        lf = ttk.Frame(lbox)
        lf.pack(fill="x")
        self.lst = tk.Listbox(lf, height=9, width=58, exportselection=False, font=("Consolas", 9))
        sb = ttk.Scrollbar(lf, command=self.lst.yview)
        self.lst.configure(yscrollcommand=sb.set)
        self.lst.pack(side="left", fill="x", expand=True)
        sb.pack(side="left", fill="y")
        self.lst.bind("<<ListboxSelect>>", self.on_select)
        br = ttk.Frame(lbox)
        br.pack(fill="x", pady=(6, 0))
        for text, cmd in (("Add shot", self.add_shot), ("Copy", self.dup_shot), ("Remove", self.del_shot),
                          ("Up", lambda: self.move_shot(-1)), ("Down", lambda: self.move_shot(1))):
            ttk.Button(br, text=text, command=cmd, width=9).pack(side="left", padx=(0, 4))

        ebox = ttk.LabelFrame(colA, text="3. The selected shot", padding=8)
        ebox.pack(fill="x", pady=(8, 0))
        self.type_radios = []
        for key in ("t2v", "i2v", "cont"):
            r = ttk.Radiobutton(ebox, text=TYPES[key], variable=self.s_type, value=key, command=self.commit)
            r.pack(anchor="w")
            self.type_radios.append(r)
        ttk.Label(ebox, text="\"Continue previous shot\" starts from the last picture of the shot above it,\n"
                  "so the scene flows on.  \"Picture to video\" uses a keyframe you made in Forge.",
                  style="Dim.TLabel").pack(anchor="w", pady=(2, 4))
        self.img_row = ttk.Frame(ebox)
        self.img_row.pack(fill="x")
        ttk.Label(self.img_row, text="Picture").pack(side="left")
        ttk.Entry(self.img_row, textvariable=self.s_image, width=34).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(self.img_row, text="Browse...", command=self.pick_image).pack(side="left")
        self.thumb_label = ttk.Label(ebox)
        self.thumb_label.pack(anchor="w", pady=(4, 0))
        ttk.Label(ebox, text="What happens in this shot (English works best):").pack(anchor="w", pady=(4, 0))
        self.prompt = tk.Text(ebox, width=58, height=5, wrap="word", font=("Segoe UI", 10))
        self.prompt.pack(fill="x")
        self.prompt.bind("<KeyRelease>", lambda e: self.commit())
        ttk.Label(ebox, text="Things to avoid (all shots):").pack(anchor="w", pady=(6, 0))
        self.neg = tk.Text(ebox, width=58, height=2, wrap="word", font=("Segoe UI", 9))
        self.neg.insert("1.0", DEFAULT_NEG)
        self.neg.pack(fill="x")

        # ---- column B: settings, options, buttons
        obox = ttk.LabelFrame(colB, text="4. Settings (for every shot)", padding=8)
        obox.pack(fill="x")
        grid = ttk.Frame(obox)
        grid.pack(fill="x")
        ttk.Label(grid, text="Size").grid(row=0, column=0, sticky="w", pady=3)
        self.size_box = ttk.Combobox(grid, textvariable=self.size, values=list(SIZES), state="readonly", width=30)
        self.size_box.grid(row=0, column=1, columnspan=2, sticky="we", pady=3)
        self.size_box.bind("<<ComboboxSelected>>", lambda e: self.update_summary())
        self.slider(grid, 1, "Seconds per shot", self.seconds, 1, 10, 1, "{:.0f} s")
        self.slider(grid, 2, "Quality steps", self.steps, 10, 60, 1, "{:.0f}")
        self.slider(grid, 3, "Prompt strength", self.cfg, 1, 7, 0.5, "{:.1f}")
        ttk.Label(grid, text="Seed").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Entry(grid, textvariable=self.seed, width=8).grid(row=4, column=1, sticky="w", padx=6)
        grid.columnconfigure(1, weight=1)
        ttk.Checkbutton(obox, text="new random seed each run (also redoes every shot)",
                        variable=self.rand_seed).pack(anchor="w", pady=(2, 0))
        ttk.Label(obox, textvariable=self.summary, style="Dim.TLabel", wraplength=330, justify="left").pack(
            anchor="w", pady=(8, 0))

        xbox = ttk.LabelFrame(colB, text="5. Options", padding=8)
        xbox.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(xbox, text="Skip shots that are already made\n(so a stopped run carries on)",
                        variable=self.resume).pack(anchor="w")
        ttk.Checkbutton(xbox, text="Join all shots into one video\nwhen the last shot is done",
                        variable=self.autojoin).pack(anchor="w", pady=(4, 0))

        bar = ttk.Frame(colB)
        bar.pack(fill="x", pady=(12, 0))
        self.go = ttk.Button(bar, text="Generate all shots", style="Big.TButton", command=self.generate)
        self.go.pack(fill="x")
        row2 = ttk.Frame(bar)
        row2.pack(fill="x", pady=(6, 0))
        self.stop = ttk.Button(row2, text="Stop", command=self.stop_run, state="disabled")
        self.stop.pack(side="left", fill="x", expand=True)
        self.join_btn = ttk.Button(row2, text="Join shots now", command=self.join_now)
        self.join_btn.pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Checkbutton(colB, text="Dark mode", variable=self.dark, command=self.apply_theme).pack(anchor="w",
                                                                                                   pady=(10, 0))

        # ---- right: viewer, progress, log, files
        self.preview = tk.Label(right, text="Pick a made shot in the list, or generate, to watch it here.",
                                bg="#202020", fg="#aaa", width=60, height=24, font=("Segoe UI", 11))
        self.preview.pack(fill="both", expand=True)
        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=(6, 0))
        self.open_btn = ttk.Button(btns, text="Open in video player", command=self.open_video, state="disabled")
        self.open_btn.pack(side="left")
        ttk.Button(btns, text="Open project folder", command=self.open_folder).pack(side="left", padx=6)
        self.out_label = ttk.Label(btns, text="", style="Dim.TLabel")
        self.out_label.pack(side="left", padx=6)
        self.progress = ttk.Progressbar(right, maximum=100)
        self.progress.pack(fill="x", pady=(8, 0))
        ttk.Label(right, textvariable=self.status, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 0))
        mid = ttk.Frame(right)
        mid.pack(fill="x", pady=(6, 0))
        logf = ttk.LabelFrame(mid, text="What is happening (live log)", padding=4)
        logf.pack(side="left", fill="both", expand=True)
        self.log = tk.Text(logf, height=4, wrap="word", state="disabled", font=("Consolas", 9))
        lsb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=lsb.set)
        lsb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        histf = ttk.LabelFrame(mid, text="Files in this project (double-click)", padding=4)
        histf.pack(side="left", fill="y", padx=(8, 0))
        self.hist = tk.Listbox(histf, width=30, height=4, exportselection=False)
        self.hist.pack(fill="both", expand=True)
        self.hist.bind("<Double-Button-1>", self.pick_history)

    # --------------------------------------------------------------- sliders and theme
    def slider(self, parent, row, text, var, lo, hi, step, fmt):
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", pady=3)
        readout = ttk.Label(parent, width=6, anchor="e")
        readout.grid(row=row, column=2, sticky="e")

        def on_move(raw):
            val = round(float(raw) / step) * step
            if abs(float(raw) - val) > 1e-6:
                sc.set(val)
                return
            var.set(val)
            readout.configure(text=fmt.format(val))
            self.update_summary()

        sc = ttk.Scale(parent, from_=lo, to=hi, orient="horizontal", command=on_move, length=190)
        sc.grid(row=row, column=1, sticky="we", padx=6)
        self._sliders.append((sc, var, step, lo, hi))

    def sync_sliders(self):
        for sc, var, step, lo, hi in self._sliders:
            try:
                v = min(hi, max(lo, round(float(var.get()) / step) * step))
            except (tk.TclError, ValueError):
                v = lo
            sc.set(v)

    def apply_theme(self):
        P = DARK if self.dark.get() else LIGHT
        self.configure(bg=P["bg"])
        st = ttk.Style(self)
        st.configure(".", background=P["bg"], foreground=P["fg"], fieldbackground=P["field"],
                     bordercolor=P["border"], lightcolor=P["bg"], darkcolor=P["bg"], troughcolor=P["trough"],
                     insertcolor=P["fg"], focuscolor=P["bg"])
        for name in ("TFrame", "TLabel", "TLabelframe", "TRadiobutton", "TCheckbutton"):
            st.configure(name, background=P["bg"], foreground=P["fg"])
        st.configure("TLabelframe.Label", background=P["bg"], foreground=P["fg"])
        st.configure("Dim.TLabel", background=P["bg"], foreground=P["dim"])
        st.configure("TButton", background=P["btn"], foreground=P["fg"], bordercolor=P["border"])
        st.configure("Big.TButton", font=("Segoe UI", 12, "bold"), padding=8)
        st.map("TButton", background=[("active", P["btn_hover"]), ("disabled", P["bg"])],
               foreground=[("disabled", P["dim"])])
        for w in ("TRadiobutton", "TCheckbutton"):
            st.map(w, background=[("active", P["bg"])], foreground=[("disabled", P["dim"])],
                   indicatorcolor=[("selected", P["accent"]), ("!selected", P["field"])])
        st.configure("TEntry", fieldbackground=P["field"], foreground=P["fg"], insertcolor=P["fg"])
        st.map("TEntry", fieldbackground=[("disabled", P["bg"])], foreground=[("disabled", P["dim"])])
        st.configure("TCombobox", fieldbackground=P["field"], background=P["btn"], foreground=P["fg"],
                     arrowcolor=P["fg"], selectbackground=P["field"], selectforeground=P["fg"])
        st.map("TCombobox", fieldbackground=[("readonly", P["field"]), ("disabled", P["bg"])],
               foreground=[("disabled", P["dim"])])
        self.option_add("*TCombobox*Listbox.background", P["field"])
        self.option_add("*TCombobox*Listbox.foreground", P["fg"])
        self.option_add("*TCombobox*Listbox.selectBackground", P["accent"])
        st.configure("Horizontal.TScale", background=P["bg"], troughcolor=P["trough"])
        st.configure("Horizontal.TProgressbar", background=P["accent"], troughcolor=P["trough"],
                     bordercolor=P["border"])
        for sb_style in ("TScrollbar", "Vertical.TScrollbar", "Horizontal.TScrollbar"):
            st.configure(sb_style, background=P["btn"], troughcolor=P["trough"], arrowcolor=P["fg"],
                         bordercolor=P["border"], lightcolor=P["btn"], darkcolor=P["btn"])
            st.map(sb_style, background=[("active", P["btn_hover"]), ("disabled", P["trough"])])
        for t in (self.prompt, self.neg):
            t.configure(bg=P["field"], fg=P["fg"], insertbackground=P["fg"], selectbackground=P["accent"],
                        highlightthickness=1, highlightbackground=P["border"], highlightcolor=P["accent"],
                        relief="flat")
        self.log.configure(bg=P["log_bg"], fg=P["log_fg"], highlightthickness=0)
        for lb in (self.lst, self.hist):
            lb.configure(bg=P["field"], fg=P["fg"], selectbackground=P["accent"], selectforeground="#ffffff",
                         highlightthickness=1, highlightbackground=P["border"], relief="flat")
        try:
            import ctypes
            self.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            val = ctypes.c_int(1 if self.dark.get() else 0)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(val), ctypes.sizeof(val))
        except Exception:
            pass

    # ------------------------------------------------------------ shot list
    def line_for(self, i):
        s = self.shots[i]
        text = " ".join(s["prompt"].split())
        text = (text[:44] + "...") if len(text) > 47 else text
        done = ""
        if self.shot_file(i + 1) and os.path.isfile(self.shot_file(i + 1)):
            done = "  [made]"
        return f"{i + 1:>3}  {TYPE_TAG[s['type']]}  {text or '(empty - describe this shot)'}{done}"

    def refresh_list(self, select=None):
        self.lst.delete(0, "end")
        for i in range(len(self.shots)):
            self.lst.insert("end", self.line_for(i))
        if select is not None and self.shots:
            select = max(0, min(select, len(self.shots) - 1))
            self.lst.selection_clear(0, "end")
            self.lst.selection_set(select)
            self.lst.see(select)
            self.load_shot(select)
        self.update_summary()

    def on_select(self, _e=None):
        sel = self.lst.curselection()
        if sel and sel[0] != self.sel:
            self.load_shot(sel[0])
            self.preview_made_shot(sel[0])

    def load_shot(self, i):
        self.loading = True
        self.sel = i
        s = self.shots[i]
        self.s_type.set(s["type"])
        self.s_image.set(s["image"])
        self.prompt.delete("1.0", "end")
        self.prompt.insert("1.0", s["prompt"])
        self.loading = False
        self.show_thumb(s["image"])
        self.update_editor_state()

    def update_editor_state(self):
        i2v = self.s_type.get() == "i2v"
        for w in self.img_row.winfo_children():
            if isinstance(w, ttk.Label):
                continue  # a disabled label turns into a bright box in dark mode
            try:
                w.configure(state="normal" if i2v else "disabled")
            except tk.TclError:
                pass
        if self.sel == 0:  # the first shot cannot continue from nothing
            self.type_radios[2].configure(state="disabled")
        else:
            self.type_radios[2].configure(state="normal")

    def commit(self):
        if self.loading or self.sel is None or not self.shots:
            return
        i = self.sel
        self.shots[i] = {"type": self.s_type.get(), "prompt": self.prompt.get("1.0", "end").strip(),
                         "image": self.s_image.get().strip()}
        self.lst.delete(i)
        self.lst.insert(i, self.line_for(i))
        self.lst.selection_set(i)
        self.update_editor_state()
        self.update_summary()

    def add_shot(self):
        kind = "cont" if self.shots else "t2v"
        self.shots.append({"type": kind, "prompt": "", "image": ""})
        self.refresh_list(select=len(self.shots) - 1)
        self.prompt.focus_set()

    def dup_shot(self):
        if self.sel is None:
            return
        s = dict(self.shots[self.sel])
        self.shots.insert(self.sel + 1, s)
        self.refresh_list(select=self.sel + 1)

    def del_shot(self):
        if self.sel is None or len(self.shots) <= 1:
            messagebox.showinfo(APP_NAME, "A project needs at least one shot.")
            return
        del self.shots[self.sel]
        if self.shots[0]["type"] == "cont":
            self.shots[0]["type"] = "t2v"
        self.refresh_list(select=min(self.sel, len(self.shots) - 1))

    def move_shot(self, d):
        i = self.sel
        if i is None or not (0 <= i + d < len(self.shots)):
            return
        self.shots[i], self.shots[i + d] = self.shots[i + d], self.shots[i]
        if self.shots[0]["type"] == "cont":
            self.shots[0]["type"] = "t2v"
        self.refresh_list(select=i + d)

    def pick_image(self):
        p = filedialog.askopenfilename(title="Choose the picture to animate",
                                       filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")])
        if p:
            self.s_image.set(p)
            self.show_thumb(p)
            self.commit()

    def show_thumb(self, p):
        self.thumb_label.configure(image="")
        self.thumb = None
        if not p or not os.path.isfile(p):
            return
        try:
            from PIL import Image, ImageTk
            im = Image.open(p)
            if im.mode == "RGBA":
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.getchannel("A"))
                im = bg
            im.thumbnail((110, 110))
            self.thumb = ImageTk.PhotoImage(im.convert("RGB"))
            self.thumb_label.configure(image=self.thumb)
        except Exception:
            pass

    # ------------------------------------------------------------- paths / job
    def pname(self):
        return re.sub(r"[^A-Za-z0-9_\-]+", "_", self.project.get().strip()).strip("_") or "my_video"

    def proj_dir(self):
        return os.path.join(OUT, self.pname())

    def shot_file(self, i):
        return os.path.join(self.proj_dir(), f"shot_{i:03d}.mp4")

    def size_wh(self):
        return SIZES.get(self.size.get(), SIZES[list(SIZES)[0]])

    def signatures(self, seed):
        w, h = self.size_wh()
        frames = frames_for(float(self.seconds.get()))
        neg = self.neg.get("1.0", "end").strip() or DEFAULT_NEG
        sigs, prev = [], ""
        for i, s in enumerate(self.shots, 1):
            img_info = ""
            if s["type"] == "i2v" and os.path.isfile(s["image"]):
                img_info = f"{s['image']}|{os.path.getmtime(s['image'])}"
            raw = json.dumps([s["type"], s["prompt"], img_info, w, h, frames, int(self.steps.get()),
                              float(self.cfg.get()), seed + i, neg, FPS, prev if s["type"] == "cont" else ""])
            prev = hashlib.sha1(raw.encode("utf-8")).hexdigest()
            sigs.append(prev)
        return sigs

    def manifest_path(self):
        return os.path.join(self.proj_dir(), "manifest.json")

    def read_manifest(self):
        try:
            with open(self.manifest_path(), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def write_manifest(self, m):
        os.makedirs(self.proj_dir(), exist_ok=True)
        tmp = self.manifest_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=1)
        os.replace(tmp, self.manifest_path())

    def update_summary(self):
        if not hasattr(self, "lst") or not self.shots:
            return
        w, h = self.size_wh()
        secs = float(self.seconds.get())
        n = len(self.shots)
        frames = frames_for(secs)
        total = n * (frames - 1) / FPS
        est = n * est_seconds(frames, int(self.steps.get()), w, h) + 60
        self.summary.set(f"{n} shot(s) x {secs:.0f} s = about {fmt_time(total)} of video.\n"
                         f"Rough time to make it: {fmt_time(est)} (this PC, plus about a minute to load).")

    # ---------------------------------------------------------------- run
    def validate(self):
        self.commit()
        if not self.shots:
            return "Add at least one shot."
        for i, s in enumerate(self.shots, 1):
            if not s["prompt"].strip():
                return f"Shot {i} has no description yet."
            if s["type"] == "cont" and i == 1:
                return "Shot 1 cannot continue a previous shot. Make it Text to video or Picture to video."
            if s["type"] == "i2v" and not os.path.isfile(s["image"]):
                return f"Shot {i} is set to Picture to video but its picture was not found."
        try:
            int(self.seed.get())
        except ValueError:
            return "The seed must be a whole number."
        return None

    def generate(self):
        if not self.models_ok:
            self.check_models()
            if not self.models_ok:
                messagebox.showinfo(APP_NAME, "The video models are not downloaded yet.\n\nPress \"Download the models "
                                    "now\" in the bar at the top of the window first.")
                return
        err = self.validate()
        if err:
            messagebox.showinfo(APP_NAME, err)
            return
        if self.rand_seed.get():
            self.seed.set(str(random.randint(1, 900000)))
        seed = int(self.seed.get())
        w, h = self.size_wh()
        frames = frames_for(float(self.seconds.get()))
        sigs = self.signatures(seed)
        manifest = self.read_manifest()
        skip = []
        if self.resume.get():
            skip = [i for i in range(1, len(self.shots) + 1)
                    if manifest.get(str(i)) == sigs[i - 1] and os.path.isfile(self.shot_file(i))
                    and os.path.getsize(self.shot_file(i)) > 1000]
        self.sigs = sigs
        os.makedirs(self.proj_dir(), exist_ok=True)
        self.cur_dir = self.proj_dir()
        self.run_total = len(self.shots)
        self.save_settings()
        self.add_log("=" * 60)
        self.add_log(f"Project '{self.pname()}': {len(self.shots)} shots, {len(skip)} already made and kept.")
        if len(skip) == len(self.shots):
            self.status.set("Every shot is already made.")
            if self.autojoin.get():
                self.join_now()
            return
        job = {"out_dir": self.cur_dir, "w": w, "h": h, "frames": frames, "steps": int(self.steps.get()),
               "cfg": float(self.cfg.get()), "seed": seed, "fps": FPS,
               "negative": self.neg.get("1.0", "end").strip() or DEFAULT_NEG, "skip": skip, "resume": False,
               "shots": [dict(s) for s in self.shots]}
        job_path = os.path.join(self.cur_dir, "job.json")
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(job, f, indent=1)
        self.progress.configure(value=0)
        self.status.set("Starting... (loading the models takes about a minute, once for all shots)")
        self.go.configure(state="disabled")
        self.join_btn.configure(state="disabled")
        self.stop.configure(state="normal")
        self.done_times = []
        self.run_error = None
        self.n_todo = len(self.shots) - len(skip)
        self.n_done = 0
        self.cur_shot = 0
        self.t_start = time.time()
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen([C.engine_python(), "-u", BATCH, job_path], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, cwd=ROOT, env=env, creationflags=NO_WINDOW)
        threading.Thread(target=self.reader_thread, args=(self.proc,), daemon=True).start()

    def reader_thread(self, proc):
        buf = ""
        while True:
            chunk = proc.stdout.read1(4096)
            if not chunk:
                break
            buf += chunk.decode("utf-8", "replace")
            parts = re.split(r"[\r\n]+", buf)
            buf = parts.pop()
            for line in parts:
                if line.strip():
                    self.q.put(("line", line.rstrip()))
        proc.wait()
        self.q.put(("done", proc.returncode))

    def poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "line":
                    self.handle_line(val)
                elif kind == "done":
                    self.finished(val)
                elif kind == "joined":
                    self.joined(*val)
                elif kind == "dl":
                    self.on_dl(val)
                elif kind == "dl_end":
                    self.on_dl_end(val)
                elif kind == "note":
                    self.add_log(val)
        except queue.Empty:
            pass
        self.after(100, self.poll)

    # ------------------------------------------------------------ model download
    def check_models(self):
        ok, rows = C.model_status()
        self.models_ok = ok
        if ok:
            if getattr(self, "_bar_shown", False):
                self.setup_bar.pack_forget()
                self._bar_shown = False
            return
        partial = any(have for _r, size, done, have in rows if size and not done)
        self.dl_text.set(
            f"The program needs the LTX-Video model and a text encoder: about {C.fmt_bytes(C.TOTAL_BYTES)} in total "
            f"({C.fmt_bytes(C.bytes_missing())} still to download). They are saved in {C.MODELS}. You only do this "
            f"once - after that everything runs offline. By downloading you accept the model licenses (see README).")
        self.dl_btn.configure(text="Resume download" if partial else "Download the models now")
        if not getattr(self, "_bar_shown", False):
            self.setup_bar.pack(side="top", fill="x", padx=10, pady=(8, 0), before=self.colA)
            self._bar_shown = True
        if not getattr(self, "_gpu_noted", False):
            self._gpu_noted = True
            if not C.has_nvidia():
                self.add_log("Note: no NVIDIA graphics card was detected. The video engine needs one (12 GB or more "
                             "of video memory is recommended).")

    def start_download(self):
        if self.dl_thread and self.dl_thread.is_alive():
            return
        self.dl_cancel = threading.Event()
        self.dl_btn.configure(state="disabled")
        self.dl_pause.configure(state="normal")
        self.dl_status.set("Starting...")
        self.add_log("Downloading the models...")

        def work(cancel=self.dl_cancel):
            try:
                C.download_models(lambda info: self.q.put(("dl", info)), cancel)
                self.q.put(("dl_end", None))
            except C.Cancelled:
                self.q.put(("dl_end", "paused"))
            except Exception as e:
                self.q.put(("dl_end", f"error: {e}"))

        self.dl_thread = threading.Thread(target=work, daemon=True)
        self.dl_thread.start()

    def pause_download(self):
        if self.dl_cancel:
            self.dl_cancel.set()
            self.dl_status.set("Pausing...")

    def on_dl(self, info):
        total = max(1, info["total"])
        self.dl_bar.configure(value=100 * info["done"] / total)
        eta = f"about {C.fmt_time(info['eta'])} left" if info.get("eta") else "working out the time left..."
        name = os.path.basename(info["file"]) if info["file"] else ""
        self.dl_status.set(f"{name}   {C.fmt_bytes(info['done'])} of {C.fmt_bytes(total)}   |   "
                           f"{C.fmt_bytes(info['speed'])}/s   |   {eta}")

    def on_dl_end(self, result):
        self.dl_pause.configure(state="disabled")
        self.dl_btn.configure(state="normal")
        if result is None:
            self.dl_bar.configure(value=100)
            self.add_log("The models are downloaded. You can generate videos now.")
            self.status.set("Models ready. You can generate videos now.")
            self.check_models()
        elif result == "paused":
            self.dl_btn.configure(text="Resume download")
            self.dl_status.set("Paused. Everything downloaded so far is kept - press Resume to carry on.")
        else:
            self.dl_btn.configure(text="Try again")
            self.dl_status.set(result)
            self.add_log(result)

    def eta_text(self):
        left = self.n_todo - self.n_done
        if self.done_times:
            per = sum(self.done_times) / len(self.done_times)
            return f"about {fmt_time(left * per)} left"
        return "working out the time left..."

    def handle_line(self, line):
        if "triton" in line.lower():
            return
        m = PROGRESS_RE.search(line)
        if m and "|" in line:
            a, b = int(m.group(1)), int(m.group(2))
            frac = (self.n_done + a / max(1, b)) / max(1, self.n_todo)
            self.progress.configure(value=100 * frac)
            self.status.set(f"Shot {self.cur_shot} of {self.run_total}: step {a}/{b}   "
                            f"({self.n_done}/{self.n_todo} made, {self.eta_text()})")
            return
        if not line.startswith("@@"):
            self.add_log(line)
            return
        tag, _, rest = line[2:].partition(" ")
        if tag == "STAGE":
            self.status.set(rest)
            self.add_log(rest)
        elif tag == "SHOT":
            self.cur_shot = int(rest.split()[0])
        elif tag == "SKIP":
            self.add_log(f"Shot {rest.split()[0]} already made - kept.")
        elif tag == "DONE":
            parts = rest.split(" ")
            i, secs = int(parts[0]), float(parts[-1])
            self.n_done += 1
            self.done_times.append(secs)
            manifest = self.read_manifest()
            manifest[str(i)] = self.sigs[i - 1]
            self.write_manifest(manifest)
            self.progress.configure(value=100 * self.n_done / max(1, self.n_todo))
            self.add_log(f"Shot {i} of {self.run_total} done in {secs:.0f}s.")
            self.status.set(f"{self.n_done}/{self.n_todo} shots made, {self.eta_text()}")
            self.line_refresh(i - 1)
            self.refresh_history()
        elif tag == "NOTE":
            self.add_log("Note: " + rest)
        elif tag == "ERROR":
            self.run_error = rest
            self.add_log("ERROR: " + rest)
        elif tag == "FINISHED":
            self.add_log("All shots finished.")

    def line_refresh(self, i):
        if 0 <= i < len(self.shots):
            keep = self.sel
            self.lst.delete(i)
            self.lst.insert(i, self.line_for(i))
            if keep is not None:
                self.lst.selection_clear(0, "end")
                self.lst.selection_set(keep)

    def finished(self, code):
        self.proc = None
        self.go.configure(state="normal")
        self.join_btn.configure(state="normal")
        self.stop.configure(state="disabled")
        took = time.time() - self.t_start
        self.refresh_list(select=self.sel)
        self.refresh_history()
        if code == 0 and not self.run_error:
            self.progress.configure(value=100)
            self.status.set(f"All {self.n_done} new shot(s) made in {fmt_time(took)}.")
            if self.autojoin.get():
                self.join_now()
        else:
            msg = self.run_error or "It stopped before finishing."
            self.status.set(f"Stopped after {self.n_done} shot(s). Finished shots are kept - press Generate again "
                            f"to carry on.")
            self.add_log(msg)

    def stop_run(self):
        if self.proc:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], creationflags=NO_WINDOW,
                           capture_output=True)
            self.add_log("Stopped by you. Finished shots are kept.")

    # -------------------------------------------------------------- joining
    def join_now(self):
        d = self.proj_dir()
        files = [self.shot_file(i) for i in range(1, len(self.shots) + 1) if os.path.isfile(self.shot_file(i))]
        if not files:
            messagebox.showinfo(APP_NAME, "No shots have been made yet for this project.")
            return
        missing = len(self.shots) - len(files)
        if missing:
            self.add_log(f"Note: {missing} shot(s) are not made yet, so they are left out of the joined video.")
        self.status.set("Joining the shots into one video...")
        self.join_btn.configure(state="disabled")
        threading.Thread(target=self.join_thread, args=(d, files), daemon=True).start()

    def join_thread(self, d, files):
        try:
            name = self.pname()
            out = os.path.join(d, f"{name}_joined.mp4")
            n = 2
            while os.path.exists(out):  # never overwrite an earlier joined video
                out = os.path.join(d, f"{name}_joined_{n}.mp4")
                n += 1
            lst = os.path.join(d, "join_list.txt")
            with open(lst, "w", encoding="utf-8") as f:
                for p in files:
                    f.write("file '" + p.replace("\\", "/").replace("'", "'\\''") + "'\n")
            r = subprocess.run([ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out],
                               capture_output=True, creationflags=NO_WINDOW)
            if r.returncode != 0 or not os.path.isfile(out):
                self.q.put(("joined", (None, r.stderr.decode("utf-8", "replace")[-400:], len(files))))
            else:
                self.q.put(("joined", (out, "", len(files))))
        except Exception as e:
            self.q.put(("joined", (None, str(e), len(files))))

    def joined(self, out, err, n):
        self.join_btn.configure(state="normal" if self.proc is None else "disabled")
        if out:
            self.last_joined = out
            secs = n * (frames_for(float(self.seconds.get())) - 1) / FPS
            self.status.set(f"Joined {n} shot(s) into {os.path.basename(out)}")
            self.add_log(f"Joined video saved: {out}")
            self.refresh_history()
            self.load_video(out)
        else:
            self.status.set("Joining failed - see the log.")
            self.add_log("Join failed: " + err)

    # -------------------------------------------------------------- viewer
    def preview_made_shot(self, i):
        p = self.shot_file(i + 1)
        if os.path.isfile(p):
            self.load_video(p)

    def refresh_history(self):
        self.hist.delete(0, "end")
        d = self.proj_dir()
        if not os.path.isdir(d):
            return
        files = [f for f in os.listdir(d) if f.lower().endswith(".mp4") and ".tmp" not in f]
        for f in sorted(files, key=lambda f: ("joined" not in f, f)):
            self.hist.insert("end", f)

    def pick_history(self, _e=None):
        sel = self.hist.curselection()
        if sel:
            self.load_video(os.path.join(self.proj_dir(), self.hist.get(sel[0])))

    def load_video(self, path):
        try:
            import imageio.v2 as iio
            self.stop_play()
            self.reader = iio.get_reader(path)
            self.fps = self.reader.get_meta_data().get("fps", FPS)
        except Exception as e:
            self.add_log(f"Could not preview {os.path.basename(path)}: {e}")
            return
        self.preview_path = path
        self.preview_size = None
        self.out_label.configure(text=os.path.basename(path))
        self.open_btn.configure(state="normal")
        self.preview.configure(text="", width=1, height=1)
        self.play()

    def stop_play(self):
        if self.play_job:
            self.after_cancel(self.play_job)
            self.play_job = None
        if self.reader:
            try:
                self.reader.close()
            except Exception:
                pass
            self.reader = None

    def play(self):
        # frames are read one at a time, so even a 30-minute video previews with almost no memory
        from PIL import Image, ImageTk
        if not self.reader:
            return
        try:
            frame = self.reader.get_next_data()
        except Exception:
            import imageio.v2 as iio
            try:
                self.reader.close()
                self.reader = iio.get_reader(self.preview_path)
                frame = self.reader.get_next_data()
            except Exception:
                return
        im = Image.fromarray(frame)
        maxw, maxh = max(300, self.preview.winfo_width() - 6), max(200, self.preview.winfo_height() - 6)
        s = min(maxw / im.width, maxh / im.height)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BILINEAR)
        self.frame_img = ImageTk.PhotoImage(im)
        self.preview.configure(image=self.frame_img)
        self.play_job = self.after(int(1000 / max(1, self.fps)), self.play)

    def open_video(self):
        if self.preview_path and os.path.isfile(self.preview_path):
            os.startfile(self.preview_path)

    def open_folder(self):
        d = self.proj_dir()
        os.makedirs(d, exist_ok=True)
        os.startfile(d)

    def add_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ------------------------------------------------------------- settings
    def collect(self):
        self.commit()
        return {"project": self.project.get(), "shots": self.shots, "neg": self.neg.get("1.0", "end").strip(),
                "size": self.size.get(), "seconds": float(self.seconds.get()), "steps": float(self.steps.get()),
                "cfg": float(self.cfg.get()), "seed": self.seed.get(), "rand": self.rand_seed.get(),
                "resume": self.resume.get(), "autojoin": self.autojoin.get(), "dark": self.dark.get()}

    def apply(self, d):
        self.project.set(d.get("project", "my_video"))
        self.shots = [{"type": s.get("type", "t2v"), "prompt": s.get("prompt", ""), "image": s.get("image", "")}
                      for s in d.get("shots", [])]
        self.neg.delete("1.0", "end")
        self.neg.insert("1.0", d.get("neg", DEFAULT_NEG))
        self.size.set(d.get("size", list(SIZES)[0]) if d.get("size") in SIZES else list(SIZES)[0])
        for var, key, default in ((self.seconds, "seconds", 5), (self.steps, "steps", 30), (self.cfg, "cfg", 3.0)):
            try:
                var.set(float(d.get(key, default)))
            except (TypeError, ValueError):
                var.set(default)
        self.seed.set(str(d.get("seed", "1")))
        self.rand_seed.set(bool(d.get("rand", False)))
        self.resume.set(bool(d.get("resume", True)))
        self.autojoin.set(bool(d.get("autojoin", True)))
        self.dark.set(bool(d.get("dark", True)))

    def save_settings(self):
        try:
            with open(SETTINGS, "w", encoding="utf-8") as f:
                json.dump(self.collect(), f, indent=1)
        except OSError:
            pass

    def load_settings(self):
        try:
            with open(SETTINGS, encoding="utf-8") as f:
                self.apply(json.load(f))
        except (OSError, ValueError):
            pass

    def save_project(self):
        p = filedialog.asksaveasfilename(title="Save project", initialdir=PROJECTS, defaultextension=".json",
                                         initialfile=self.pname() + ".json", filetypes=[("Project", "*.json")])
        if p:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.collect(), f, indent=1)
            self.add_log(f"Project saved: {p}")

    def open_project(self):
        p = filedialog.askopenfilename(title="Open project", initialdir=PROJECTS, filetypes=[("Project", "*.json")])
        if not p:
            return
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            self.apply(d)
        except (OSError, ValueError) as e:
            messagebox.showinfo(APP_NAME, f"Could not open that project: {e}")
            return
        if not self.shots:
            self.shots = [{"type": "t2v", "prompt": "", "image": ""}]
        self.sel = None
        self.refresh_list(select=0)
        self.sync_sliders()
        self.apply_theme()
        self.refresh_history()
        self.add_log(f"Project opened: {p}")

    def on_close(self):
        if self.proc and not messagebox.askyesno(APP_NAME,
                                                 "Shots are still being made. Stop and close?\n"
                                                 "(Finished shots are kept.)"):
            return
        if self.dl_cancel:
            self.dl_cancel.set()  # the partly downloaded files stay, so the download carries on next time
        self.stop_run()
        self.stop_play()
        self.save_settings()
        self.destroy()


def make_sample_picture():
    """A small transparent test picture, used only by the self-test."""
    from PIL import Image, ImageDraw
    p = os.path.join(OUT, "selftest_picture.png")
    im = Image.new("RGBA", (512, 320), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((100, 90, 412, 260), radius=24, fill=(200, 40, 40, 255), outline=(255, 215, 0, 255), width=10)
    d.ellipse((230, 150, 282, 202), fill=(255, 215, 0, 255))
    im.save(p)
    return p


def main():
    relaunched = os.environ.get("LTX_RELAUNCHED") == "1"
    if C.missing_packages():
        if relaunched:
            messagebox.showerror(APP_NAME, "The Python packages are still missing after setup:\n\n" +
                                 "\n".join(C.missing_packages()))
            return
        vpy = C.bootstrap_window()  # first start: builds the private environment (quick if it already exists)
        if not vpy:
            return  # the user closed the setup window; starting the program again carries on
        subprocess.Popen([vpy, os.path.abspath(__file__), *sys.argv[1:]], cwd=ROOT,
                         env=dict(os.environ, LTX_RELAUNCHED="1"), creationflags=NO_WINDOW)
        return
    global SETTINGS
    if "--selftest" in sys.argv:  # tests must never overwrite the user's saved settings
        SETTINGS = os.path.join(ROOT, "selftest_settings.json")
    app = App()
    if "--autostart" in sys.argv:  # start generating right away with the saved project
        app.after(1500, app.generate)
    if "--selftest" in sys.argv:
        report = os.path.join(OUT, "v4_selftest_edit.txt" if "--edit" in sys.argv else "v4_selftest.txt")
        picture = make_sample_picture()

        def check():
            if "--edit" in sys.argv:  # second pass: change ONE shot and run again; only that shot should be redone
                app.shots[1]["prompt"] = "the camera slowly pulls back from the fire, glowing embers drift away"
                app.refresh_list(select=1)
                app.generate()
                return
            app.project.set("v4_selftest")
            app.shots = [
                {"type": "t2v", "prompt": "a small campfire at night, sparks rising, cinematic", "image": ""},
                {"type": "cont", "prompt": "the camera slowly moves closer to the fire", "image": ""},
                {"type": "i2v", "prompt": "a red treasure chest with a gold lock, sparkles drift upward, pixel art",
                 "image": picture},
            ]
            app.sel = None
            app.refresh_list(select=0)
            app.seconds.set(1)
            app.steps.set(8)
            app.sync_sliders()
            app.generate()

        def watch():
            if app.proc is None and app.t_start and time.time() - app.t_start > 5:
                joined = app.last_joined
                ok = bool(joined and os.path.isfile(joined))
                shots = [os.path.isfile(app.shot_file(i)) for i in (1, 2, 3)]
                with open(report, "w") as f:
                    f.write(f"{'ok' if ok and all(shots) else 'FAILED'} shots_made={shots} joined={joined} "
                            f"status={app.status.get()}")
                app.after(2500, app.destroy)
            else:
                app.after(1500, watch)

        app.after(1500, check)
        app.after(6000, watch)
    app.mainloop()


if __name__ == "__main__":
    main()
