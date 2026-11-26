"""
╔══════════════════════════════════════════════════════════════════════════╗
║  TEMPEST EXPLORER · Electromagnetic Leakage Explorer                     ║
║  PerceptionLab  ·  Data Scanner Edition                                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  New in this edition:                                                    ║
║   • LIVE DATA SCANNER – Baseband envelope demodulation & UART decoder    ║
║   • CALM MODE – temporal frame blending, soft colormaps, ~12 fps         ║
║   • Device Library – one-click profiles: phone, laptop, monitor, mains   ║
║   • Known-frequency markers on waterfall (50 Hz, 217 Hz GSM, etc.)       ║
╚══════════════════════════════════════════════════════════════════════════╝

Setup:
  pip install sounddevice soundfile numpy scipy matplotlib

Usage:
  python tempest_explorer.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
import scipy.signal
import threading
import datetime
import time
import warnings
import json
import os
from collections import deque

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────
BG       = "#060810"
PANEL    = "#0c1220"
PANEL2   = "#101828"
SIDEBAR  = "#0a1018"
BORDER   = "#1a2840"
BORDER2  = "#243040"
CYAN     = "#00e5ff"
CYAN2    = "#00bcd4"
GREEN    = "#00ff88"
RED      = "#ff2244"
YELLOW   = "#ffd600"
ORANGE   = "#ff7000"
GREY     = "#3a4a5a"
TXT      = "#c8d8f0"
TXT2     = "#6080a0"
MAGENTA  = "#e040fb"
TEAL     = "#00bfa5"
LIME     = "#aeea00"

FONT_MONO  = ("Consolas", 9)
FONT_HEAD  = ("Consolas", 10, "bold")
FONT_LARGE = ("Consolas", 12, "bold")
FONT_SMALL = ("Consolas", 8)

# ─────────────────────────────────────────────────────────────────
#  DEVICE PROFILES
# ─────────────────────────────────────────────────────────────────
DEVICE_PROFILES = {
    "📱 Phone GSM":    dict(band=217.0,  q=20, width=220, dec=1,
                            note="GSM TDMA frame burst: 217 Hz exactly"),
    "📱 Phone LTE":    dict(band=100.0,  q=15, width=480, dec=1,
                            note="LTE sub-frame 1 ms → 1000 Hz, look for 100 Hz harmonics"),
    "💻 Laptop PSU":   dict(band=130.0,  q=10, width=369, dec=1,
                            note="Switching supply ripple, often 100–200 Hz range"),
    "🖥  Monitor PWM": dict(band=1000.0, q=25, width=48,  dec=1,
                            note="LED backlight PWM: 200 Hz – 5 kHz depending on brand"),
    "⌨  USB Device":  dict(band=1000.0, q=15, width=48,  dec=1,
                            note="USB SOF packet: 1 kHz (Full Speed) or 8 kHz (Hi-Speed)"),
    "🔌 Mains 50 Hz":  dict(band=50.0,   q=30, width=960, dec=1,
                            note="European mains 50 Hz + harmonics (100, 150, 200 Hz…)"),
    "🔌 Mains 60 Hz":  dict(band=60.0,   q=30, width=800, dec=1,
                            note="American mains 60 Hz + harmonics"),
    "📡 WiFi burst":   dict(band=250.0,  q=12, width=192, dec=1,
                            note="OFDM beacon bursts, look for rhythmic pattern"),
    "🎛  Audio gear":  dict(band=19000.0, q=40, width=3,  dec=1,
                            note="19 kHz pilot tone (FM stereo), DAC clocks"),
}

KNOWN_MARKERS = [
    (50.0,    "50Hz mains",  ORANGE),
    (60.0,    "60Hz mains",  ORANGE),
    (100.0,   "100Hz harm",  ORANGE),
    (217.0,   "GSM 217Hz",  "#ff80ab"),
    (1000.0,  "USB SOF 1k", TEAL),
    (8000.0,  "USB SOF 8k", TEAL),
    (15734.0, "CRT H-sync", TXT2),
    (19000.0, "FM pilot",   TXT2),
]

FINGERPRINTS = [
    ("Phone GSM",   [217, 434, 651, 868],    "#ff80ab"),
    ("Mains 50 Hz", [50, 100, 150, 200, 250], ORANGE),
    ("Mains 60 Hz", [60, 120, 180, 240],      ORANGE),
    ("USB FS",      [1000, 2000, 4000],        TEAL),
    ("USB HS",      [8000, 16000],             TEAL),
    ("Monitor PWM", [200, 400, 800, 1600],     CYAN2),
    ("FM Pilot",    [19000, 38000],            TXT2),
]


# ─────────────────────────────────────────────────────────────────
#  SONIFICATION ENGINE
# ─────────────────────────────────────────────────────────────────
class SonificationEngine:
    def __init__(self, sample_rate, output_device=None):
        self.fs = sample_rate
        self.output_device = output_device
        self.output_stream = None
        self.audio_queue = deque(maxlen=10)
        self.is_playing = False
        self.lock = threading.Lock()

        self.pitch_shift = 1.0
        self.volume = 0.5
        self.normalize = True
        self.monitor_band = False
        self.band_center = 217.0
        self.band_q = 20.0

        self._start_output_stream()

    def _start_output_stream(self):
        try:
            self.output_stream = sd.OutputStream(
                device=self.output_device,
                samplerate=self.fs,
                channels=1,
                dtype='float32',
                blocksize=1024,
                callback=self._audio_callback
            )
            self.output_stream.start()
            self.is_playing = True
        except Exception as e:
            print(f"Output stream error: {e}")
            self.is_playing = False

    def _audio_callback(self, outdata, frames, time_info, status):
        with self.lock:
            if len(self.audio_queue) > 0:
                data = self.audio_queue.popleft()
                if len(data) >= frames:
                    outdata[:] = data[:frames].reshape(-1, 1)
                else:
                    outdata[:len(data)] = data.reshape(-1, 1)
                    outdata[len(data):] = 0
            else:
                outdata.fill(0)

    def process_audio(self, audio_chunk):
        if not self.is_playing:
            return
        processed = audio_chunk.copy().astype(np.float32)
        if self.monitor_band:
            processed = self._bandpass(processed, self.band_center, self.band_q)
        if self.pitch_shift != 1.0:
            processed = self._pitch_shift(processed, self.pitch_shift)
        if self.normalize and len(processed) > 0:
            mx = np.max(np.abs(processed))
            if mx > 0:
                processed = processed / mx * 0.9
        processed = processed * self.volume
        processed = np.tanh(processed * 0.8) / 0.8
        with self.lock:
            self.audio_queue.append(processed.astype(np.float32))

    def _bandpass(self, sig, fc, q):
        nyq = self.fs / 2
        if fc < 10 or fc > nyq - 10:
            return sig
        bw = fc / q
        lo = max(10, fc - bw/2) / nyq
        hi = min(nyq - 10, fc + bw/2) / nyq
        sos = scipy.signal.butter(4, [lo, hi], btype='band', output='sos')
        return scipy.signal.sosfilt(sos, sig)

    def _pitch_shift(self, sig, shift):
        new_len = int(len(sig) / shift)
        idx = np.linspace(0, len(sig) - 1, new_len)
        shifted = np.interp(idx, np.arange(len(sig)), sig)
        if len(shifted) < len(sig):
            shifted = np.pad(shifted, (0, len(sig) - len(shifted)))
        else:
            shifted = shifted[:len(sig)]
        return shifted

    def update_params(self, pitch=None, volume=None, normalize=None,
                      monitor_band=None, band_center=None, band_q=None):
        if pitch is not None:       self.pitch_shift  = np.clip(pitch, 0.5, 4.0)
        if volume is not None:      self.volume       = np.clip(volume, 0.0, 1.0)
        if normalize is not None:   self.normalize    = normalize
        if monitor_band is not None: self.monitor_band = monitor_band
        if band_center is not None: self.band_center  = np.clip(band_center, 10, self.fs/2 - 100)
        if band_q is not None:      self.band_q       = np.clip(band_q, 1, 100)

    def stop(self):
        if self.output_stream:
            try:
                self.output_stream.stop()
                self.output_stream.close()
            except Exception:
                pass
        self.is_playing = False


# ─────────────────────────────────────────────────────────────────
#  DEVICE PICKER
# ─────────────────────────────────────────────────────────────────
class DevicePicker(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("TEMPEST EXPLORER · Select Audio Input")
        self.configure(bg=BG)
        self.geometry("720x480")
        self.resizable(False, False)
        self.result = None

        devices = sd.query_devices()
        input_devs = [(i, d) for i, d in enumerate(devices)
                      if d['max_input_channels'] > 0]

        tk.Label(self, text="TEMPEST EXPLORER",
                 fg=CYAN, bg=BG, font=("Consolas", 16, "bold")).pack(pady=(18, 2))
        tk.Label(self, text="Electromagnetic Leakage Explorer  ·  PerceptionLab",
                 fg=TXT2, bg=BG, font=FONT_MONO).pack()
        tk.Label(self, text="Select the audio input device (Focusrite, built-in mic, RTL-SDR, etc.)",
                 fg=TXT2, bg=BG, font=FONT_MONO).pack(pady=(8, 8))

        lf = tk.Frame(self, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        lf.pack(fill=tk.BOTH, expand=True, padx=24, pady=4)

        sc = tk.Scrollbar(lf, bg=PANEL)
        sc.pack(side=tk.RIGHT, fill=tk.Y)

        self.lb = tk.Listbox(lf, bg=PANEL, fg=TXT, font=FONT_MONO,
                             selectbackground=CYAN, selectforeground=BG,
                             activestyle='none', highlightthickness=0,
                             yscrollcommand=sc.set, borderwidth=0,
                             width=82, height=12)
        self.lb.pack(fill=tk.BOTH, expand=True)
        sc.config(command=self.lb.yview)

        self.dev_ids = []
        default_dev = sd.default.device[0]
        default_idx = None
        for i, (did, d) in enumerate(input_devs):
            mark = "★ DEFAULT  " if did == default_dev else "           "
            lbl = f" {mark}[{did:2d}]  {d['name'][:52]:<52}  {int(d['default_samplerate'])} Hz"
            self.lb.insert(tk.END, lbl)
            self.dev_ids.append(did)
            if did == default_dev:
                default_idx = i
        if default_idx is not None:
            self.lb.selection_set(default_idx)
            self.lb.see(default_idx)

        sr_frame = tk.Frame(self, bg=BG)
        sr_frame.pack(fill=tk.X, padx=24, pady=8)
        tk.Label(sr_frame, text="Sample Rate:", fg=TXT2, bg=BG,
                 font=FONT_MONO).pack(side=tk.LEFT)
        self.sr_var = tk.StringVar(value="48000")
        for sr in ["22050", "44100", "48000", "96000"]:
            tk.Radiobutton(sr_frame, text=sr, variable=self.sr_var, value=sr,
                           fg=CYAN, bg=BG, selectcolor=BG,
                           activebackground=BG, font=FONT_MONO).pack(side=tk.LEFT, padx=8)

        tk.Label(sr_frame, text="Output:", fg=TXT2, bg=BG,
                 font=FONT_MONO).pack(side=tk.LEFT, padx=(20, 4))
        self._out_devs = []
        out_names = []
        for i, d in enumerate(devices):
            if d['max_output_channels'] > 0:
                self._out_devs.append(i)
                out_names.append(f"[{i}] {d['name'][:28]}")
        self.out_var = tk.StringVar(value=out_names[0] if out_names else "Default")
        ttk.Combobox(sr_frame, textvariable=self.out_var, values=out_names,
                     width=26, state="readonly").pack(side=tk.LEFT, padx=4)

        btn = tk.Frame(self, bg=BG)
        btn.pack(pady=12)
        tk.Button(btn, text="▶  LAUNCH EXPLORER", bg=CYAN, fg=BG,
                  font=("Consolas", 11, "bold"), padx=20, pady=6,
                  relief=tk.FLAT, cursor="hand2",
                  command=self._confirm).pack(side=tk.LEFT, padx=10)
        tk.Button(btn, text="USE DEFAULT", bg=PANEL2, fg=TXT,
                  font=FONT_MONO, padx=12, pady=6, relief=tk.FLAT,
                  cursor="hand2", command=self._use_default).pack(side=tk.LEFT, padx=10)

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._use_default)

    def _confirm(self):
        sel = self.lb.curselection()
        if not sel:
            return
        did = self.dev_ids[sel[0]]
        fs  = int(self.sr_var.get())
        out = None
        for i, nm in enumerate(self.out_var['values'] if hasattr(self.out_var, '__getitem__') else []):
            if nm == self.out_var.get():
                out = self._out_devs[i]
                break
        self.result = (did, fs, out)
        self.destroy()

    def _use_default(self):
        self.result = (sd.default.device[0], 48000, sd.default.device[1])
        self.destroy()


# ─────────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────
class TempestExplorer:

    # ── refresh rates (ms)
    RATE_LIVE  = 30    # ~33 fps  – standard mode
    RATE_CALM  = 90    # ~11 fps  – calm mode (low flicker)

    # ── temporal EMA alphas for raster display
    ALPHA_LIVE = 1.0   # no blending
    ALPHA_CALM = 0.18  # heavy smoothing

    def __init__(self, root, device_id, fs, output_device):
        self.root        = root
        self.device_id   = device_id
        self.fs          = fs
        self.output_dev  = output_device

        dev_name = sd.query_devices(device_id)['name']
        self.root.title(f"TEMPEST EXPLORER  ·  {dev_name}  ·  {fs} Hz")
        self.root.geometry("1680x980")
        self.root.configure(bg=BG)
        self.root.minsize(1100, 700)

        # ── audio ring buffer
        self.buf_sec  = 2.5
        self.buf_len  = int(fs * self.buf_sec)
        self.audio_buf = np.zeros(self.buf_len, dtype=np.float32)
        self._alock   = threading.Lock()
        self._apos    = 0

        # ── sonification
        self.sonifier = SonificationEngine(fs, output_device)

        # ── UI variables
        self.width_var       = tk.IntVar(value=220)
        self.roll_var        = tk.DoubleVar(value=0.0)
        self.dec_var         = tk.IntVar(value=1)
        self.squelch_var     = tk.DoubleVar(value=2.5)
        self.edge_var        = tk.BooleanVar(value=False)
        self.square_var      = tk.BooleanVar(value=True)
        self.negative_var    = tk.BooleanVar(value=False)
        self.raster_off_var  = tk.BooleanVar(value=False)
        self.anomaly_var     = tk.BooleanVar(value=False)
        self.show_waterfall  = tk.BooleanVar(value=True)
        self.show_history    = tk.BooleanVar(value=True)
        self.calm_mode_var   = tk.BooleanVar(value=False)
        self.exact_freq_var  = tk.StringVar()

        self.sonify_enabled  = tk.BooleanVar(value=True)
        self.sonify_pitch    = tk.DoubleVar(value=1.0)
        self.sonify_volume   = tk.DoubleVar(value=0.5)
        self.sonify_norm     = tk.BooleanVar(value=True)
        self.sonify_band     = tk.BooleanVar(value=True)
        self.sonify_fc       = tk.DoubleVar(value=217.0)
        self.sonify_q        = tk.DoubleVar(value=20.0)

        # ── state
        self.is_running       = True
        self.stream           = None
        self.auto_sync_running = False
        self.snr_smooth       = 1.0
        self.display_frame    = None   # for temporal EMA in calm mode
        self.pixel_mean = self.pixel_m2 = self.anomaly_map = None
        self.pixel_n    = 0

        # ── plot artists (set during rebuild)
        self.img_raster  = self.img_fft = self.img_wfall = None
        self.scatter_peak = self.hline_peak = self.vline_peak = None
        self.wfall_vline  = None
        self.hist_line    = self.hist_fill = None
        self.wfall_markers = []

        # ── waterfall buffer
        self.wfall_rows  = 200
        self.wfall_bins  = 512
        self.wfall_buf   = deque(maxlen=self.wfall_rows)
        for _ in range(self.wfall_rows):
            self.wfall_buf.append(np.full(self.wfall_bins, -80.0))

        # ── SNR history
        self.hist_maxlen = 400
        self.peak_snr_hist = deque([0.0] * self.hist_maxlen, maxlen=self.hist_maxlen)

        # ── fingerprint state
        self.fp_label_var = tk.StringVar(value="—")
        self.fp_conf_var  = tk.StringVar(value="")
        self.fp_buf       = deque(maxlen=60)   # 60 spectrum frames for fingerprint

        self._build_ui()
        self._bind_keys()
        self._start_audio()

    # ═══════════════════════════════════════════════════════════════
    #  UI BUILD
    # ═══════════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── top bar ──────────────────────────────────────────────
        top = tk.Frame(self.root, bg=PANEL, height=36)
        top.pack(fill=tk.X, side=tk.TOP)
        top.pack_propagate(False)

        self.btn_pause = tk.Button(top, text="⏸  PAUSE", bg=RED, fg="#fff",
                                   font=FONT_HEAD, padx=10, relief=tk.FLAT,
                                   cursor="hand2", command=self._toggle_stream)
        self.btn_pause.pack(side=tk.LEFT, padx=(8, 4), pady=4)

        self.btn_sync = tk.Button(top, text="⚡ AUTO-SYNC", bg=PANEL2, fg=YELLOW,
                                  font=FONT_HEAD, padx=8, relief=tk.FLAT,
                                  cursor="hand2", command=self._start_auto_sync)
        self.btn_sync.pack(side=tk.LEFT, padx=4, pady=4)

        tk.Button(top, text="💾 SAVE PNG", bg=PANEL2, fg=TXT2,
                  font=FONT_HEAD, padx=8, relief=tk.FLAT,
                  cursor="hand2", command=self._save_png).pack(side=tk.LEFT, padx=4, pady=4)

        tk.Frame(top, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=6)

        # calm mode toggle
        self.calm_btn = tk.Button(top, text="🌙 CALM MODE: OFF",
                                  bg=PANEL2, fg=TXT2,
                                  font=FONT_HEAD, padx=10, relief=tk.FLAT,
                                  cursor="hand2", command=self._toggle_calm)
        self.calm_btn.pack(side=tk.LEFT, padx=4, pady=4)
        
        # ── NEW DATA SCANNER BUTTON ──
        tk.Button(top, text="🕵️ DATA SCANNER", bg="#e040fb", fg="white",
                  font=FONT_HEAD, padx=10, relief=tk.FLAT,
                  command=self._launch_data_scanner).pack(side=tk.LEFT, padx=4, pady=4)

        tk.Frame(top, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=6)

        for lbl, var, color in [
            ("WATERFALL", self.show_waterfall, CYAN2),
            ("HISTORY",   self.show_history,   CYAN2),
            ("RASTER OFF",self.raster_off_var, ORANGE),
            ("NEGATIVE",  self.negative_var,   CYAN2),
            ("EDGE",      self.edge_var,        CYAN2),
            ("SQUARE PX", self.square_var,      CYAN2),
            ("ANOMALY",   self.anomaly_var,     MAGENTA),
        ]:
            cmd = self._layout_changed if lbl in ("WATERFALL", "HISTORY", "RASTER OFF") else None
            tk.Checkbutton(top, text=lbl, variable=var,
                           fg=color, bg=PANEL, selectcolor=BG,
                           activebackground=PANEL, activeforeground=color,
                           font=FONT_MONO, command=cmd).pack(side=tk.LEFT, padx=3)

        # right side of top bar
        self.snr_label = tk.Label(top, text="SNR: --- dB",
                                  fg=RED, bg=PANEL, font=FONT_HEAD, width=14)
        self.snr_label.pack(side=tk.RIGHT, padx=8)
        self.clock_label = tk.Label(top, text="LOOP: --- Hz",
                                    fg=YELLOW, bg=PANEL, font=FONT_HEAD, width=16)
        self.clock_label.pack(side=tk.RIGHT, padx=6)
        self.scan_canvas = tk.Canvas(top, width=110, height=18, bg=PANEL,
                                     highlightthickness=0)
        self.scan_canvas.pack(side=tk.RIGHT, padx=8, pady=9)

        # ── controls row ─────────────────────────────────────────
        ctrl = tk.Frame(self.root, bg=PANEL2)
        ctrl.pack(fill=tk.X, side=tk.TOP)

        row1 = tk.Frame(ctrl, bg=PANEL2)
        row1.pack(fill=tk.X, padx=8, pady=(5, 2))
        self._make_slider_row(row1)

        row2 = tk.Frame(ctrl, bg=PANEL2)
        row2.pack(fill=tk.X, padx=8, pady=(0, 2))
        self._make_sonification_row(row2)

        self.status_bar = tk.Label(ctrl, text="Ready. Place a device near your pickup and press AUTO-SYNC.",
                                   fg=TXT2, bg=PANEL2, font=FONT_MONO, anchor="w")
        self.status_bar.pack(fill=tk.X, padx=8, pady=(0, 4))

        # ── main content: sidebar + plots ────────────────────────
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill=tk.BOTH, expand=True)

        # sidebar
        self.sidebar = tk.Frame(content, bg=SIDEBAR, width=220)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        # plot frame
        self.plot_frame = tk.Frame(content, bg=BG)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._rebuild_figure()

    def _launch_data_scanner(self):
        if not hasattr(self, 'data_scanner') or not self.data_scanner.win.winfo_exists():
            self.data_scanner = LiveDataScanner(self)
            self.data_scanner.start()
        else:
            self.data_scanner.win.lift()

    # ── SIDEBAR ──────────────────────────────────────────────────
    def _build_sidebar(self):
        s = self.sidebar

        def section(label, color=CYAN2):
            f = tk.Frame(s, bg=BORDER, height=1)
            f.pack(fill=tk.X, padx=0)
            tk.Label(s, text=f"  {label}", fg=color, bg=SIDEBAR,
                     font=("Consolas", 9, "bold"), anchor="w").pack(
                fill=tk.X, padx=0, pady=(6, 2))

        # ── Device Library
        section("DEVICE LIBRARY", CYAN)
        tk.Label(s, text="  One-click profiles", fg=TXT2, bg=SIDEBAR,
                 font=FONT_SMALL, anchor="w").pack(fill=tk.X, padx=4)

        for name, profile in DEVICE_PROFILES.items():
            btn = tk.Button(s, text=name, bg=PANEL2, fg=TXT,
                            font=FONT_MONO, anchor="w", padx=8,
                            relief=tk.FLAT, cursor="hand2",
                            command=lambda p=profile, n=name: self._apply_profile(p, n))
            btn.pack(fill=tk.X, padx=4, pady=1)
            btn.bind("<Enter>", lambda e, w=btn: w.config(bg=BORDER2))
            btn.bind("<Leave>", lambda e, w=btn: w.config(bg=PANEL2))

        # ── Fingerprint
        section("FINGERPRINT", MAGENTA)
        tk.Label(s, text="  Detected source:", fg=TXT2, bg=SIDEBAR,
                 font=FONT_SMALL, anchor="w").pack(fill=tk.X, padx=6)
        self.fp_name_lbl = tk.Label(s, textvariable=self.fp_label_var,
                                    fg=MAGENTA, bg=SIDEBAR,
                                    font=("Consolas", 11, "bold"), anchor="w")
        self.fp_name_lbl.pack(fill=tk.X, padx=8, pady=(2, 0))
        self.fp_conf_lbl = tk.Label(s, textvariable=self.fp_conf_var,
                                    fg=TXT2, bg=SIDEBAR,
                                    font=FONT_SMALL, anchor="w")
        self.fp_conf_lbl.pack(fill=tk.X, padx=8)
        tk.Button(s, text="⟳  Fingerprint Now", bg=PANEL2, fg=CYAN2,
                  font=FONT_MONO, padx=8, relief=tk.FLAT, cursor="hand2",
                  command=self._run_fingerprint).pack(fill=tk.X, padx=4, pady=4)

        # ── Harmonic table
        section("KNOWN MARKERS", YELLOW)
        tk.Label(s, text="  (shown on waterfall)", fg=TXT2, bg=SIDEBAR,
                 font=FONT_SMALL, anchor="w").pack(fill=tk.X, padx=6, pady=(0, 2))
        for freq, label, color in KNOWN_MARKERS:
            row = tk.Frame(s, bg=SIDEBAR)
            row.pack(fill=tk.X, padx=6, pady=0)
            tk.Label(row, text=f"  {freq:>8.1f} Hz", fg=color, bg=SIDEBAR,
                     font=FONT_SMALL, width=13, anchor="w").pack(side=tk.LEFT)
            tk.Label(row, text=label, fg=TXT2, bg=SIDEBAR,
                     font=FONT_SMALL, anchor="w").pack(side=tk.LEFT)

        # ── Tips
        section("TIPS", TXT2)
        tips = [
            "• INST button = high-Z",
            "  makes strings an antenna",
            "• Phone on strings, data ON",
            "• 217 Hz = GSM heartbeat",
            "• AUTO-SYNC finds period",
            "• CALM: reduces flicker",
            "• SPACE = pause / resume",
            "• Arrow keys = fine tune",
        ]
        for t in tips:
            tk.Label(s, text=t, fg=TXT2, bg=SIDEBAR,
                     font=FONT_SMALL, anchor="w").pack(fill=tk.X, padx=6)

    # ── SLIDER CONTROLS ROW ──────────────────────────────────────
    def _make_slider_row(self, parent):
        sliders = [
            ("DECIMATION",   self.dec_var,     1, 100,  1,   7),
            ("RASTER WIDTH", self.width_var,   10, 4000, 1,   7),
            ("PHASE SHIFT %",self.roll_var,    0,  100,  0.1, 6),
            ("SQUELCH",      self.squelch_var, 0.1, 49, 0.1, 5),
        ]
        for lbl, var, lo, hi, res, ew in sliders:
            f = tk.Frame(parent, bg=PANEL2)
            f.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
            tk.Label(f, text=lbl, fg=TXT2, bg=PANEL2,
                     font=FONT_SMALL, anchor="w").pack(fill=tk.X)
            row = tk.Frame(f, bg=PANEL2)
            row.pack(fill=tk.X)
            tk.Scale(row, from_=lo, to=hi, variable=var,
                     orient=tk.HORIZONTAL, resolution=res,
                     bg=PANEL2, fg=CYAN, troughcolor=BORDER,
                     highlightthickness=0, showvalue=False,
                     sliderlength=14, sliderrelief=tk.FLAT
                     ).pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Entry(row, textvariable=var, width=ew,
                     bg=BG, fg=CYAN, insertbackground=CYAN,
                     font=FONT_MONO, relief=tk.FLAT,
                     highlightbackground=BORDER,
                     highlightthickness=1).pack(side=tk.LEFT, padx=2)

        # Target Hz
        tf = tk.Frame(parent, bg=PANEL2)
        tf.pack(side=tk.RIGHT, padx=8)
        tk.Label(tf, text="TARGET Hz", fg=TXT2, bg=PANEL2,
                 font=FONT_SMALL, anchor="w").pack(fill=tk.X)
        er = tk.Frame(tf, bg=PANEL2)
        er.pack(fill=tk.X)
        en = tk.Entry(er, textvariable=self.exact_freq_var, width=8,
                      bg=BG, fg=CYAN, insertbackground=CYAN,
                      font=FONT_MONO, relief=tk.FLAT,
                      highlightbackground=BORDER, highlightthickness=1)
        en.pack(side=tk.LEFT, padx=2)
        en.bind('<Return>', lambda e: self._apply_exact_freq())
        tk.Button(er, text="SET", bg=CYAN, fg=BG, font=FONT_HEAD,
                  padx=6, relief=tk.FLAT, cursor="hand2",
                  command=self._apply_exact_freq).pack(side=tk.LEFT)

        # Harmonic display
        self.harmonic_label = tk.Label(parent, text="", fg=YELLOW,
                                       bg=PANEL2, font=FONT_SMALL)
        self.harmonic_label.pack(side=tk.RIGHT, padx=10)

        # Colormap
        cf = tk.Frame(parent, bg=PANEL2)
        cf.pack(side=tk.RIGHT, padx=6)
        tk.Label(cf, text="CMAP", fg=TXT2, bg=PANEL2,
                 font=FONT_SMALL).pack(fill=tk.X)
        self.cmap_var = tk.StringVar(value="viridis")
        cm = ttk.Combobox(cf, textvariable=self.cmap_var,
                          values=["viridis", "plasma", "inferno", "magma",
                                  "cividis", "bone", "hot", "turbo"],
                          width=8, state="readonly")
        cm.pack()
        cm.bind("<<ComboboxSelected>>",
                lambda e: self._update_wfall_cmap(self.cmap_var.get()))

    # ── SONIFICATION ROW ─────────────────────────────────────────
    def _make_sonification_row(self, parent):
        tk.Label(parent, text="🎧", fg=MAGENTA, bg=PANEL2,
                 font=FONT_HEAD).pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(parent, text="SONIFICATION", fg=MAGENTA, bg=PANEL2,
                 font=FONT_HEAD).pack(side=tk.LEFT, padx=(0, 8))

        tk.Checkbutton(parent, text="ENABLE", variable=self.sonify_enabled,
                       fg=GREEN, bg=PANEL2, selectcolor=BG,
                       activebackground=PANEL2, font=FONT_HEAD,
                       command=self._update_sonification
                       ).pack(side=tk.LEFT, padx=4)

        tk.Frame(parent, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        for lbl, var, lo, hi, res, w in [
            ("Pitch", self.sonify_pitch, 0.5, 4.0, 0.05, 90),
            ("Vol",   self.sonify_volume, 0.0, 1.0, 0.02, 70),
        ]:
            tk.Label(parent, text=lbl+":", fg=TXT2, bg=PANEL2,
                     font=FONT_MONO).pack(side=tk.LEFT, padx=(4, 1))
            tk.Scale(parent, from_=lo, to=hi, variable=var,
                     orient=tk.HORIZONTAL, resolution=res, length=w,
                     bg=PANEL2, fg=CYAN2, troughcolor=BORDER,
                     highlightthickness=0, showvalue=False,
                     command=lambda x: self._update_sonification()
                     ).pack(side=tk.LEFT)
            tk.Label(parent, textvariable=var, width=4,
                     fg=CYAN2, bg=PANEL2, font=FONT_SMALL).pack(side=tk.LEFT)

        tk.Checkbutton(parent, text="Norm", variable=self.sonify_norm,
                       fg=CYAN2, bg=PANEL2, selectcolor=BG,
                       activebackground=PANEL2, font=FONT_MONO,
                       command=self._update_sonification).pack(side=tk.LEFT, padx=4)

        tk.Frame(parent, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        tk.Checkbutton(parent, text="Band Focus @", variable=self.sonify_band,
                       fg=YELLOW, bg=PANEL2, selectcolor=BG,
                       activebackground=PANEL2, font=FONT_MONO,
                       command=self._update_sonification).pack(side=tk.LEFT, padx=(4, 1))

        fc_en = tk.Entry(parent, textvariable=self.sonify_fc, width=7,
                         bg=BG, fg=CYAN, insertbackground=CYAN,
                         font=FONT_MONO, relief=tk.FLAT,
                         highlightbackground=BORDER, highlightthickness=1)
        fc_en.pack(side=tk.LEFT, padx=2)
        fc_en.bind('<Return>', lambda e: self._update_sonification())
        tk.Label(parent, text="Hz  Q:", fg=TXT2, bg=PANEL2,
                 font=FONT_MONO).pack(side=tk.LEFT)
        tk.Scale(parent, from_=1, to=80, variable=self.sonify_q,
                 orient=tk.HORIZONTAL, resolution=1, length=60,
                 bg=PANEL2, fg=CYAN2, troughcolor=BORDER,
                 highlightthickness=0, showvalue=False,
                 command=lambda x: self._update_sonification()
                 ).pack(side=tk.LEFT)
        tk.Label(parent, textvariable=self.sonify_q, width=3,
                 fg=CYAN2, bg=PANEL2, font=FONT_SMALL).pack(side=tk.LEFT)

    # ─────────────────────────────────────────────────────────────
    #  FIGURE
    # ─────────────────────────────────────────────────────────────
    def _rebuild_figure(self):
        for w in self.plot_frame.winfo_children():
            w.destroy()

        wfall    = self.show_waterfall.get()
        hist     = self.show_history.get()
        roff     = self.raster_off_var.get()
        calm     = self.calm_mode_var.get()

        # choose raster colormap
        raster_cmap = "bone" if calm else "gray"

        self.fig = plt.Figure(figsize=(13, 7), dpi=95, facecolor=BG)
        self.fig.subplots_adjust(left=0.01, right=0.99,
                                 top=0.96, bottom=0.04,
                                 wspace=0.07, hspace=0.10)

        if roff:
            if wfall and hist:
                gs = GridSpec(1, 3, figure=self.fig,
                              width_ratios=[2, 3, 1], wspace=0.06)
                self.ax_fft   = self.fig.add_subplot(gs[0, 0])
                self.ax_wfall = self.fig.add_subplot(gs[0, 1])
                self.ax_hist  = self.fig.add_subplot(gs[0, 2])
                self.ax_raster = None
            elif wfall:
                gs = GridSpec(1, 2, figure=self.fig,
                              width_ratios=[1, 2], wspace=0.06)
                self.ax_fft   = self.fig.add_subplot(gs[0, 0])
                self.ax_wfall = self.fig.add_subplot(gs[0, 1])
                self.ax_hist  = None
                self.ax_raster = None
            else:
                self.ax_fft   = self.fig.add_subplot(111)
                self.ax_wfall = self.ax_hist = self.ax_raster = None
        else:
            if wfall and hist:
                gs = GridSpec(2, 2, figure=self.fig,
                              width_ratios=[3, 2], height_ratios=[3, 2],
                              wspace=0.07, hspace=0.10)
                self.ax_raster = self.fig.add_subplot(gs[0, 0])
                self.ax_fft   = self.fig.add_subplot(gs[0, 1])
                self.ax_wfall = self.fig.add_subplot(gs[1, 0])
                self.ax_hist  = self.fig.add_subplot(gs[1, 1])
            elif wfall:
                gs = GridSpec(2, 2, figure=self.fig,
                              width_ratios=[3, 2], height_ratios=[3, 2],
                              wspace=0.07, hspace=0.10)
                self.ax_raster = self.fig.add_subplot(gs[0, 0])
                self.ax_fft   = self.fig.add_subplot(gs[0, 1])
                self.ax_wfall = self.fig.add_subplot(gs[1, :])
                self.ax_hist  = None
            else:
                gs = GridSpec(1, 2, figure=self.fig, wspace=0.07)
                self.ax_raster = self.fig.add_subplot(gs[0, 0])
                self.ax_fft   = self.fig.add_subplot(gs[0, 1])
                self.ax_wfall = self.ax_hist = None

        for ax in filter(None, [self.ax_raster, self.ax_fft,
                                 self.ax_wfall,  self.ax_hist]):
            ax.set_facecolor(BG)
            ax.tick_params(colors=TXT2, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(BORDER)

        if self.ax_raster:
            self.ax_raster.set_title("Hardware Raster", color=CYAN,
                                     fontsize=9, pad=4)
            self.ax_raster.axis('off')
            self.img_raster = self.ax_raster.imshow(
                np.zeros((64, 64)), cmap=raster_cmap, aspect='auto',
                interpolation='nearest')
        else:
            self.img_raster = None

        self.ax_fft.set_title("2D Spatial Spectrum", color=CYAN,
                              fontsize=9, pad=4)
        self.ax_fft.axis('off')
        fft_cmap = "cividis" if calm else "magma"
        self.img_fft = self.ax_fft.imshow(
            np.zeros((64, 64)), cmap=fft_cmap, aspect='auto')

        self.scatter_peak = self.ax_fft.scatter(
            [], [], color=RED, s=90, facecolors='none',
            edgecolors=RED, linewidth=1.5, zorder=5)
        self.hline_peak = self.ax_fft.axhline(-999, color=RED, lw=0.6,
                                               ls=':', alpha=0.6)
        self.vline_peak = self.ax_fft.axvline(-999, color=RED, lw=0.6,
                                               ls=':', alpha=0.6)

        if self.ax_wfall:
            self.ax_wfall.set_title(
                "Waterfall / Time-Frequency  (EMI Analysis)",
                color=CYAN, fontsize=9, pad=4)
            self.ax_wfall.set_xlabel("kHz", color=TXT2, fontsize=7)
            self.ax_wfall.set_ylabel("Time →", color=TXT2, fontsize=7)
            self.img_wfall = self.ax_wfall.imshow(
                np.zeros((self.wfall_rows, self.wfall_bins)),
                cmap=self.cmap_var.get(), aspect='auto', origin='upper',
                extent=[0, self.fs/2/1000, 0, self.wfall_rows],
                vmin=-80, vmax=0)
            self.wfall_vline = self.ax_wfall.axvline(
                0, color=RED, lw=0.8, ls='--', alpha=0.7)
            # Draw known-frequency markers
            self.wfall_markers = []
            for freq, label, color in KNOWN_MARKERS:
                fkhz = freq / 1000.0
                if fkhz < self.fs/2/1000:
                    ln  = self.ax_wfall.axvline(
                        fkhz, color=color, lw=0.6, ls=':', alpha=0.45)
                    txt = self.ax_wfall.text(
                        fkhz, 5, f" {label}", color=color,
                        fontsize=6, va='top', alpha=0.7,
                        rotation=90)
                    self.wfall_markers.append((ln, txt))
        else:
            self.img_wfall = None
            self.wfall_markers = []

        if self.ax_hist:
            self.ax_hist.set_title("Peak SNR History", color=CYAN,
                                   fontsize=9, pad=4)
            self.ax_hist.set_ylabel("dB", color=TXT2, fontsize=7)
            self.ax_hist.set_ylim(0, 20)
            xs = list(range(self.hist_maxlen))
            ys = [0.0] * self.hist_maxlen
            self.hist_line, = self.ax_hist.plot(xs, ys, color=CYAN, lw=0.8)
            self.hist_fill  = self.ax_hist.fill_between(
                xs, ys, alpha=0.2, color=CYAN)
            self.ax_hist.axhline(10*np.log10(1.5), color=GREEN,
                                 lw=0.6, ls='--', alpha=0.6)
        else:
            self.hist_line = self.hist_fill = None

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self._on_click)

        # reset display accumulator
        self.display_frame = None

    def _layout_changed(self):
        self._rebuild_figure()

    # ─────────────────────────────────────────────────────────────
    #  AUDIO
    # ─────────────────────────────────────────────────────────────
    def _audio_cb(self, indata, frames, time_info, status):
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        with self._alock:
            space = len(mono)
            pos   = self._apos
            if pos + space <= self.buf_len:
                self.audio_buf[pos:pos+space] = mono
            else:
                first = self.buf_len - pos
                self.audio_buf[pos:]   = mono[:first]
                self.audio_buf[:space-first] = mono[first:]
            self._apos = (pos + space) % self.buf_len

    def _start_audio(self):
        try:
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=1,
                samplerate=self.fs,
                callback=self._audio_cb,
                dtype='float32',
                blocksize=1024
            )
            self.stream.start()
            self._update_loop()
        except Exception as e:
            messagebox.showerror("Audio Error", str(e))

    def _toggle_stream(self):
        if self.is_running:
            self.stream.stop()
            self.is_running = False
            self.btn_pause.config(text="▶  RESUME", bg=GREEN, fg=BG)
            self.sonifier.stop()
        else:
            self.stream.start()
            self.is_running = True
            self.btn_pause.config(text="⏸  PAUSE", bg=RED, fg="#fff")
            self.sonifier._start_output_stream()
            self._update_loop()

    # ─────────────────────────────────────────────────────────────
    #  MAIN RENDER LOOP
    # ─────────────────────────────────────────────────────────────
    def _update_loop(self):
        if not self.is_running:
            return

        calm  = self.calm_mode_var.get()
        alpha = self.ALPHA_CALM if calm else self.ALPHA_LIVE
        rate  = self.RATE_CALM  if calm else self.RATE_LIVE

        w   = max(10, self.width_var.get())
        dec = max(1,  self.dec_var.get())
        h   = min(600, self.buf_len // (w * dec))

        if h < 6:
            self.root.after(50, self._update_loop)
            return

        with self._alock:
            pos   = self._apos
            total = h * w * dec
            if total > self.buf_len:
                self.root.after(50, self._update_loop)
                return
            if pos >= total:
                raw = self.audio_buf[pos-total:pos].copy()
            else:
                raw = np.concatenate([self.audio_buf[pos-total:],
                                       self.audio_buf[:pos]])

        chunk = raw[::dec][:h*w]

        # Edge detection
        if self.edge_var.get():
            chunk = np.diff(chunk, prepend=chunk[0])

        # Squelch / normalization
        c = self.squelch_var.get()
        lo, hi = np.percentile(chunk, c), np.percentile(chunk, 100-c)
        chunk = np.clip(chunk, lo, hi)
        rng = hi - lo
        mat = 255.0 * (chunk - lo) / rng if rng > 0 else chunk
        mat = mat.reshape(h, w)

        # Phase roll
        roll_px = int(w * self.roll_var.get() / 100.0)
        if roll_px:
            mat = np.roll(mat, roll_px, axis=1)

        # Negative
        if self.negative_var.get():
            mat = 255.0 - mat

        # Anomaly mode
        if self.anomaly_var.get() and not self.raster_off_var.get():
            self.pixel_n += 1
            if self.pixel_mean is None or self.pixel_mean.shape != mat.shape:
                self.pixel_mean = mat.copy()
                self.pixel_m2   = np.zeros_like(mat)
                self.pixel_n    = 1
                self.anomaly_map = None
            else:
                delta = mat - self.pixel_mean
                self.pixel_mean += delta / self.pixel_n
                self.pixel_m2   += delta * (mat - self.pixel_mean)
            if self.pixel_n > 30:
                var = self.pixel_m2 / (self.pixel_n - 1)
                std = np.sqrt(var + 1e-6)
                self.anomaly_map = np.abs(mat - self.pixel_mean) / std
            mat_raw = (self.anomaly_map
                       if self.anomaly_map is not None
                       and self.anomaly_map.shape == mat.shape
                       else mat)
        else:
            mat_raw = mat

        # ── CALM MODE temporal blending ──────────────────────────
        if alpha < 1.0:
            if self.display_frame is None or self.display_frame.shape != mat_raw.shape:
                self.display_frame = mat_raw.copy()
            else:
                self.display_frame = (alpha * mat_raw.astype(np.float32)
                                      + (1.0 - alpha) * self.display_frame)
            mat_display = self.display_frame
        else:
            mat_display = mat_raw

        # ── Waterfall update ─────────────────────────────────────
        self._update_waterfall(raw[::dec])

        # ── Sonification ─────────────────────────────────────────
        if self.sonify_enabled.get() and self.is_running:
            self.sonifier.process_audio(raw[::dec])

        # ── 2D FFT ───────────────────────────────────────────────
        F  = np.fft.fft2(mat - np.mean(mat))
        Fs = np.fft.fftshift(F)
        mag = 20 * np.log10(np.abs(Fs) + 1e-9)

        cy, cx = h//2, w//2
        gy, gx = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask = (gx**2 + gy**2) > 5**2
        mm = mag * mask

        peak_idx = np.unravel_index(np.argmax(mm), mm.shape)
        snr_raw  = mm[peak_idx] / (np.mean(mm[mask]) + 1e-9)

        # smoother in calm mode
        snr_alpha = 0.08 if calm else 0.18
        self.snr_smooth = ((1.0 - snr_alpha) * self.snr_smooth
                           + snr_alpha * snr_raw)

        eff_fs    = self.fs / dec
        loop_freq = eff_fs / w

        # ── Update display labels ────────────────────────────────
        harmonics = [f"{loop_freq * n:.1f}" for n in [2, 3, 4, 5]]
        self.harmonic_label.config(
            text="Harmonics: " + "  ".join(harmonics) + " Hz")
        self.clock_label.config(text=f"LOOP: {loop_freq:.2f} Hz")

        snr_db = 10 * np.log10(self.snr_smooth + 1e-9)
        locked = self.snr_smooth > 1.5
        snr_col = GREEN if locked else (YELLOW if self.snr_smooth > 1.1 else RED)
        self.snr_label.config(text=f"SNR: {snr_db:.1f} dB", fg=snr_col)
        self._draw_snr_bar(self.snr_smooth, snr_col)

        self.peak_snr_hist.append(self.snr_smooth)

        # ── Fingerprint accumulation (background) ────────────────
        N = self.wfall_bins * 2
        seg = raw[::dec][-N:]
        if len(seg) == N:
            spec = np.abs(np.fft.rfft(seg * np.hanning(N), n=N))
            self.fp_buf.append(spec)

        # ── Update matplotlib artists ────────────────────────────
        if self.img_raster is not None and not self.raster_off_var.get():
            self.img_raster.set_data(mat_display)
            self.img_raster.set_clim(np.min(mat_display), np.max(mat_display))
            if self.square_var.get():
                self.ax_raster.set_aspect('equal')
            else:
                self.ax_raster.set_aspect('auto')

        self.img_fft.set_data(mag)
        self.img_fft.set_clim(np.min(mag), np.max(mag))
        self.img_fft.set_cmap("cividis" if calm else "magma")
        self.ax_fft.set_title("2D Spatial Spectrum", color=CYAN, fontsize=9, pad=4)

        if self.snr_smooth > 1.1:
            self.scatter_peak.set_offsets(np.c_[peak_idx[1], peak_idx[0]])
            self.hline_peak.set_ydata([peak_idx[0]])
            self.vline_peak.set_xdata([peak_idx[1]])
        else:
            self.scatter_peak.set_offsets(np.c_[-999, -999])

        if self.img_wfall is not None:
            self.img_wfall.set_data(np.array(self.wfall_buf))
            loop_khz = loop_freq / 1000.0
            self.wfall_vline.set_xdata([loop_khz])

        if self.hist_line is not None:
            ys = [10 * np.log10(v + 1e-9) for v in self.peak_snr_hist]
            xs = list(range(len(ys)))
            self.hist_line.set_data(xs, ys)
            if self.hist_fill:
                self.hist_fill.remove()
            self.hist_fill = self.ax_hist.fill_between(
                xs, ys, alpha=0.2, color=CYAN)
            if ys:
                self.ax_hist.set_ylim(min(0, min(ys)-1),
                                      max(12, max(ys)+2))

        self.canvas.draw_idle()
        self.root.after(rate, self._update_loop)

    def _update_waterfall(self, chunk):
        N   = self.wfall_bins * 2
        seg = chunk[-N:] * np.hanning(min(len(chunk), N))
        if len(seg) < N:
            seg = np.pad(seg, (N - len(seg), 0))
        spec = 20 * np.log10(
            np.abs(np.fft.rfft(seg, n=N))[:self.wfall_bins] + 1e-9)
        self.wfall_buf.append(spec)

    # ─────────────────────────────────────────────────────────────
    #  CALM MODE
    # ─────────────────────────────────────────────────────────────
    def _toggle_calm(self):
        val = not self.calm_mode_var.get()
        self.calm_mode_var.set(val)
        if val:
            self.calm_btn.config(text="🌙 CALM MODE: ON",
                                 bg="#0d2040", fg=CYAN)
            self.status_bar.config(
                text="Calm mode ON — temporal blending active, refresh ~11 fps",
                fg=CYAN2)
        else:
            self.calm_btn.config(text="🌙 CALM MODE: OFF",
                                 bg=PANEL2, fg=TXT2)
            self.display_frame = None
            self.status_bar.config(text="Calm mode OFF — live refresh", fg=TXT2)
        # rebuild so colormaps update
        self._rebuild_figure()

    # ─────────────────────────────────────────────────────────────
    #  DEVICE PROFILES
    # ─────────────────────────────────────────────────────────────
    def _apply_profile(self, profile, name):
        self.width_var.set(profile['width'])
        self.dec_var.set(profile.get('dec', 1))
        self.sonify_fc.set(profile['band'])
        self.sonify_q.set(profile['q'])
        self.sonify_band.set(True)
        self.exact_freq_var.set(str(profile['band']))
        self._apply_exact_freq()
        self._update_sonification()
        self.status_bar.config(
            text=f"Profile: {name}  —  {profile['note']}",
            fg=YELLOW)
        # trigger auto-sync after brief delay
        self.root.after(300, self._start_auto_sync)

    # ─────────────────────────────────────────────────────────────
    #  FINGERPRINT
    # ─────────────────────────────────────────────────────────────
    def _run_fingerprint(self):
        if len(self.fp_buf) < 10:
            self.fp_label_var.set("Need more data")
            self.fp_conf_var.set("keep scanning…")
            return

        # Average spectrum
        avg = np.mean(np.array(self.fp_buf), axis=0)
        freqs = np.fft.rfftfreq(self.wfall_bins * 2, d=1.0/self.fs)[:len(avg)]

        def peak_power_at(target_hz, bw_hz=10):
            mask = np.abs(freqs - target_hz) < bw_hz
            if not np.any(mask):
                return 0.0
            return float(np.mean(avg[mask]))

        noise_floor = np.percentile(avg, 20)

        best_name  = "Unknown"
        best_score = 0.0
        best_color = TXT2

        for name, harmonics, color in FINGERPRINTS:
            scores = []
            for hf in harmonics:
                pw = peak_power_at(hf)
                scores.append(max(0.0, pw - noise_floor))
            score = float(np.mean(scores)) if scores else 0.0
            if score > best_score:
                best_score = score
                best_name  = name
                best_color = color

        confidence = min(100, int(best_score / (noise_floor + 0.1) * 10))
        self.fp_label_var.set(best_name)
        self.fp_conf_var.set(f"confidence: {confidence}%")
        self.fp_name_lbl.config(fg=best_color)
        self.status_bar.config(
            text=f"Fingerprint: {best_name}  (confidence ~{confidence}%)",
            fg=best_color)

    # ─────────────────────────────────────────────────────────────
    #  AUTO-SYNC
    # ─────────────────────────────────────────────────────────────
    def _start_auto_sync(self):
        if self.auto_sync_running:
            return
        self.auto_sync_running = True
        self.btn_sync.config(bg=YELLOW, fg=BG, text="⚡ SCANNING…")
        self.status_bar.config(text="Auto-Sync: searching for periodicity…",
                               fg=YELLOW)
        threading.Thread(target=self._auto_sync_worker, daemon=True).start()

    def _auto_sync_worker(self):
        with self._alock:
            pos = self._apos
            chunk = (np.concatenate([self.audio_buf[pos:], self.audio_buf[:pos]])
                     if pos > 0 else self.audio_buf.copy())

        win   = np.hanning(len(chunk))
        spec  = np.fft.rfft(chunk * win)
        lspec = np.log(np.abs(spec) + 1e-9)
        cep   = np.abs(np.fft.irfft(lspec))
        lo, hi = 20, min(len(cep)//2, 4000)
        peaks, props = scipy.signal.find_peaks(
            cep[lo:hi], height=np.percentile(cep[lo:hi], 90))

        dec      = self.dec_var.get()
        eff_fs   = self.fs / dec
        best_w   = None
        best_sc  = -1

        if len(peaks) > 0:
            heights  = props['peak_heights']
            top      = peaks[np.argsort(heights)[::-1][:6]] + lo
            for p in top:
                ww = max(10, min(4000, int(round(p / dec)) if dec > 1 else p))
                sc = self._score_width(chunk, ww, dec)
                if sc > best_sc:
                    best_sc, best_w = sc, ww

        cur = self.width_var.get()
        for ww in range(max(10, int(cur*0.85)), min(4001, int(cur*1.15))):
            sc = self._score_width(chunk, ww, dec)
            if sc > best_sc:
                best_sc, best_w = sc, ww

        def _apply():
            if best_w:
                self.width_var.set(best_w)
                freq = eff_fs / best_w
                self.status_bar.config(
                    text=f"Auto-Sync: locked to {best_w} px → {freq:.2f} Hz  (score {best_sc:.3f})",
                    fg=GREEN)
            else:
                self.status_bar.config(
                    text="Auto-Sync: no strong periodicity found — try adjusting width manually",
                    fg=ORANGE)
            self.btn_sync.config(bg=PANEL2, fg=YELLOW, text="⚡ AUTO-SYNC")
            self.auto_sync_running = False

        self.root.after(0, _apply)

    def _score_width(self, chunk, w, dec):
        h = min(64, len(chunk) // (w * dec))
        if h < 4:
            return 0.0
        total = h * w * dec
        sub   = chunk[-total::dec][:h*w].reshape(h, w)
        F     = np.abs(np.fft.fft2(sub - np.mean(sub)))
        F[0, 0] = 0
        flat  = F.flatten()
        flat.sort()
        total_e = np.sum(flat) + 1e-9
        return np.sum(flat[-5:]) / total_e

    # ─────────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────────
    def _apply_exact_freq(self):
        try:
            f = float(self.exact_freq_var.get())
            if f > 0:
                dec = self.dec_var.get()
                w   = int(round(self.fs / dec / f))
                self.width_var.set(max(10, min(4000, w)))
                self.status_bar.config(
                    text=f"Raster width → {w} px  for {f:.2f} Hz",
                    fg=CYAN)
        except ValueError:
            pass

    def _update_sonification(self):
        if not self.sonify_enabled.get():
            self.sonifier.volume = 0
            return
        self.sonifier.update_params(
            pitch=self.sonify_pitch.get(),
            volume=self.sonify_volume.get(),
            normalize=self.sonify_norm.get(),
            monitor_band=self.sonify_band.get(),
            band_center=self.sonify_fc.get(),
            band_q=self.sonify_q.get()
        )

    def _update_wfall_cmap(self, cmap):
        if self.img_wfall:
            self.img_wfall.set_cmap(cmap)

    def _draw_snr_bar(self, snr, color):
        c = self.scan_canvas
        c.delete("all")
        c.create_rectangle(0, 2, 108, 14, outline=BORDER, fill=PANEL)
        pct   = min(1.0, max(0.0, (snr - 1.0) / 3.0))
        wfill = int(106 * pct)
        if wfill > 0:
            c.create_rectangle(1, 3, wfill, 13, fill=color, outline="")
        label = "LOCKED" if snr > 1.5 else "SCANNING"
        c.create_text(54, 8, text=label,
                      fill=BG if wfill > 54 else color,
                      font=("Consolas", 7, "bold"))

    def _save_png(self):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"tempest_{ts}.png",
            filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if fname:
            self.fig.savefig(fname, dpi=150, facecolor=BG)
            meta = {
                "timestamp": ts,
                "device_id": self.device_id,
                "sample_rate": self.fs,
                "raster_width": self.width_var.get(),
                "decimation": self.dec_var.get(),
                "loop_freq_hz": self.fs / self.dec_var.get() / self.width_var.get(),
                "band_center_hz": self.sonify_fc.get(),
                "snr_smooth": float(self.snr_smooth),
                "calm_mode": self.calm_mode_var.get(),
            }
            with open(fname.replace(".png", "_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            self.status_bar.config(
                text=f"Saved: {os.path.basename(fname)}", fg=GREEN)

    def _on_click(self, event):
        if event.button != 1 or event.xdata is None:
            return

        # ── Raster click = phase shift only
        if (event.inaxes == self.ax_raster
                and not self.raster_off_var.get()):
            w = self.width_var.get()
            pct = (event.xdata / w) * 100.0
            self.roll_var.set(pct % 100)
            return

        # ── Waterfall click = set Data Scanner carrier only
        if event.inaxes == self.ax_wfall:
            clicked_hz = event.xdata * 1000.0

            # launch if missing
            if not hasattr(self, "data_scanner") or self.data_scanner is None:
                self._launch_data_scanner()

            self.data_scanner.carrier_hz.set(clicked_hz)

            self.status_bar.config(
                text=f"Carrier selected: {clicked_hz:.1f} Hz",
                fg=CYAN
            )
            return

    def _bind_keys(self):
        r = self.root
        r.bind('<space>',  lambda e: self._toggle_stream())
        r.bind('s',        lambda e: self._start_auto_sync())
        r.bind('S',        lambda e: self._start_auto_sync())
        r.bind('c',        lambda e: self._toggle_calm())
        r.bind('C',        lambda e: self._toggle_calm())
        r.bind('f',        lambda e: self._run_fingerprint())
        r.bind('n',        lambda e: self.negative_var.set(not self.negative_var.get()))
        r.bind('<Left>',   lambda e: self.roll_var.set(max(0, self.roll_var.get() - 1)))
        r.bind('<Right>',  lambda e: self.roll_var.set(min(100, self.roll_var.get() + 1)))
        r.bind('<Up>',     lambda e: self.width_var.set(self.width_var.get() + 1))
        r.bind('<Down>',   lambda e: self.width_var.set(max(10, self.width_var.get() - 1)))

    def __del__(self):
        if hasattr(self, 'sonifier'):
            self.sonifier.stop()


# ─────────────────────────────────────────────────────────────────
#  LIVE DATA SCANNER
#  Demodulates audio-band EMI into binary data using envelope 
#  detection and UART bit-slicing.
# ─────────────────────────────────────────────────────────────────
class LiveDataScanner:
    DISPLAY_RATE = 100  # ms per frame

    def __init__(self, parent_app):
        self.app = parent_app
        self.fs = parent_app.fs
        self._abuf = parent_app.audio_buf
        self._alock = parent_app._alock
        self._get_pos = lambda: parent_app._apos

        # DSP Parameters
        self.carrier_hz = tk.DoubleVar(value=8000.0)
        self.bw_hz = tk.DoubleVar(value=2000.0)
        self.baud_rate = tk.IntVar(value=1200)
        self.threshold_pct = tk.DoubleVar(value=70.0)
        self.invert_bits = tk.BooleanVar(value=False)
        self.capture_len = tk.DoubleVar(value=0.25) # seconds to process per frame

        self.is_running = False
        self._build_window()

    def _build_window(self):
        self.win = tk.Toplevel(self.app.root)
        self.win.title("TEMPEST EXPLORER · Live Data Scanner")
        self.win.geometry("1400x800")
        self.win.configure(bg=BG)
        self.win.protocol("WM_DELETE_WINDOW", self._close)

        # ── TOP BAR
        top = tk.Frame(self.win, bg=PANEL, height=40)
        top.pack(fill=tk.X)
        top.pack_propagate(False)

        tk.Label(top, text="🕵️  LIVE DATA SCANNER", fg=MAGENTA, bg=PANEL, font=FONT_LARGE).pack(side=tk.LEFT, padx=10)
        tk.Label(top, text="Baseband Envelope Demodulation & UART Decoding", fg=TXT2, bg=PANEL, font=FONT_MONO).pack(side=tk.LEFT, padx=6)

        # ── CONTROLS
        ctrl = tk.Frame(self.win, bg=SIDEBAR, width=280)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)
        ctrl.pack_propagate(False)

        def slider(parent, label, var, lo, hi, res):
            tk.Label(parent, text=label, fg=TXT2, bg=SIDEBAR, font=FONT_SMALL, anchor="w").pack(fill=tk.X, padx=8, pady=(8,0))
            row = tk.Frame(parent, bg=SIDEBAR)
            row.pack(fill=tk.X, padx=8)
            tk.Scale(row, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL, resolution=res,
                     bg=SIDEBAR, fg=CYAN, troughcolor=BORDER, highlightthickness=0, showvalue=False).pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Label(row, textvariable=var, width=6, fg=CYAN, bg=SIDEBAR, font=FONT_SMALL).pack(side=tk.LEFT)

        tk.Label(ctrl, text="RADIO FRONT-END", fg=CYAN2, bg=SIDEBAR, font=FONT_HEAD, anchor="w").pack(fill=tk.X, padx=8, pady=(12,4))
        slider(ctrl, "Carrier Freq (Hz)", self.carrier_hz, 100, 24000, 100)
        slider(ctrl, "Bandwidth (Hz)", self.bw_hz, 100, 10000, 100)
        
        tk.Label(ctrl, text="DIGITAL DECODER", fg=MAGENTA, bg=SIDEBAR, font=FONT_HEAD, anchor="w").pack(fill=tk.X, padx=8, pady=(20,4))
        slider(ctrl, "Baud Rate (bps)", self.baud_rate, 50, 19200, 50)
        slider(ctrl, "Slicer Threshold %", self.threshold_pct, 10, 99, 1)
        tk.Checkbutton(ctrl, text="Invert Bit Polarity", variable=self.invert_bits, fg=TXT, bg=SIDEBAR, selectcolor=BG, activebackground=SIDEBAR).pack(anchor="w", padx=8, pady=4)

        tk.Button(ctrl, text="🗑️ CLEAR TERMINAL", bg=PANEL2, fg=TXT, font=FONT_MONO, relief=tk.FLAT, command=self._clear_term).pack(fill=tk.X, padx=12, pady=30)

        # ── PLOTS & TERMINAL
        right = tk.Frame(self.win, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Plot area
        self.fig = plt.Figure(figsize=(10, 5), dpi=95, facecolor=BG)
        self.fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.08, hspace=0.3)
        self.ax_env = self.fig.add_subplot(211)
        self.ax_bit = self.fig.add_subplot(212, sharex=self.ax_env)

        for ax in [self.ax_env, self.ax_bit]:
            ax.set_facecolor(BG)
            ax.tick_params(colors=TXT2, labelsize=7)
            for sp in ax.spines.values(): sp.set_color(BORDER)

        self.ax_env.set_title("Baseband Envelope (AM Demodulation)", color=CYAN, fontsize=9, pad=3)
        self.line_env, = self.ax_env.plot([], [], color=CYAN, lw=1.0)
        self.line_thresh, = self.ax_env.plot([], [], color=RED, lw=1.0, ls='--')

        self.ax_bit.set_title("Sliced Digital Waveform (0s and 1s)", color=MAGENTA, fontsize=9, pad=3)
        self.line_bit, = self.ax_bit.plot([], [], color=MAGENTA, lw=1.5, drawstyle='steps-post')
        self.ax_bit.set_ylim(-0.2, 1.2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Terminal area
        term_frame = tk.Frame(right, bg=PANEL2, height=180)
        term_frame.pack(fill=tk.X, padx=8, pady=8)
        term_frame.pack_propagate(False)
        tk.Label(term_frame, text="DECODED DATA (8-N-1 UART)", fg=YELLOW, bg=PANEL2, font=FONT_HEAD, anchor="w").pack(fill=tk.X, padx=4, pady=2)
        
        self.terminal = tk.Text(term_frame, bg=BG, fg=GREEN, font=("Consolas", 11), wrap=tk.WORD, borderwidth=1, highlightthickness=0)
        self.terminal.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.terminal.insert(tk.END, "Listening for EMI data bursts...\n")

    def _clear_term(self):
        self.terminal.delete(1.0, tk.END)

    def _process_audio(self):
        n_samples = int(self.fs * self.capture_len.get())
        with self._alock:
            pos = self._get_pos()
            if pos >= n_samples:
                raw = self._abuf[pos - n_samples: pos].copy()
            else:
                raw = np.concatenate([self._abuf[pos - n_samples:], self._abuf[:pos]])

        fc = self.carrier_hz.get()
        bw = self.bw_hz.get()
        baud = self.baud_rate.get()

        # 1. Bandpass Filter (Isolate the carrier)
        nyq = self.fs / 2
        lo = max(10, fc - bw/2) / nyq
        hi = min(nyq - 10, fc + bw/2) / nyq
        
        try:
            sos_bp = scipy.signal.butter(4, [lo, hi], btype='bandpass', output='sos')
            filtered = scipy.signal.sosfilt(sos_bp, raw)

            # 2. Envelope Detection (Rectify + Lowpass)
            rectified = np.abs(filtered)
            lp_cutoff = min(nyq - 10, baud * 2.0) / nyq
            sos_lp = scipy.signal.butter(4, lp_cutoff, btype='lowpass', output='sos')
            envelope = scipy.signal.sosfilt(sos_lp, rectified)

            # 3. Bit Slicing
            thresh_val = np.percentile(envelope, self.threshold_pct.get())
            bits = (envelope > thresh_val).astype(np.int8)
            if self.invert_bits.get():
                bits = 1 - bits

            # 4. UART Decoding (8-N-1)
            # Find start bits (transition from 1 to 0)
            samples_per_symbol = int(self.fs / baud)
            chars_found = []
            
            if samples_per_symbol > 2:
                i = 0
                while i < len(bits) - (samples_per_symbol * 10):
                    # Look for idle (1) to active (0) transition
                    if bits[i] == 1 and bits[i+1] == 0:
                        # Jump to middle of first data bit
                        scan_idx = i + int(samples_per_symbol * 1.5)
                        byte_val = 0
                        valid = True
                        for bit_num in range(8):
                            if scan_idx >= len(bits): 
                                valid = False; break
                            b = bits[scan_idx]
                            byte_val |= (b << bit_num) # LSB first
                            scan_idx += samples_per_symbol
                        
                        # Check Stop bit (should be 1)
                        if valid and scan_idx < len(bits) and bits[scan_idx] == 1:
                            chars_found.append(byte_val)
                            i = scan_idx # skip to end of frame
                        else:
                            i += 1 # False alarm, step forward
                    else:
                        i += 1

            return envelope, bits, thresh_val, chars_found

        except Exception as e:
            return None, None, None, []

    def _update_loop(self):
        if not self.is_running: return

        env, bits, thresh, chars = self._process_audio()

        if env is not None:
            # Update plots
            xs = np.arange(len(env))
            self.line_env.set_data(xs, env)
            self.line_thresh.set_data([0, len(env)], [thresh, thresh])
            self.ax_env.set_xlim(0, len(env))
            self.ax_env.set_ylim(0, max(np.max(env), thresh * 1.5) + 1e-6)

            self.line_bit.set_data(xs, bits)
            
            # Print decoded characters
            if chars:
                out_str = ""
                for c in chars:
                    if 32 <= c <= 126: # Printable ASCII
                        out_str += chr(c)
                    else:
                        out_str += f"[{hex(c)}]"
                
                self.terminal.insert(tk.END, out_str)
                self.terminal.see(tk.END)

            self.canvas.draw_idle()

        self.win.after(self.DISPLAY_RATE, self._update_loop)

    def start(self):
        self.is_running = True
        self.win.after(self.DISPLAY_RATE, self._update_loop)

    def _close(self):
        self.is_running = False
        self.win.destroy()


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    root.withdraw()

    picker = DevicePicker(root)
    root.wait_window(picker)

    if picker.result is None:
        root.destroy()
        return

    device_id, fs, output_device = picker.result
    root.deiconify()

    app = TempestExplorer(root, device_id, fs, output_device)

    def _on_close():
        app.is_running = False
        if app.stream:
            try:
                app.stream.stop()
                app.stream.close()
            except Exception:
                pass
        app.sonifier.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()


if __name__ == "__main__":
    main()