"""
gms/app.py -- GuitarMicScanner GUI.

tkinter + numpy only in the display loop (no matplotlib redraws): images go
to Tk as PPM bytes, curves are canvas lines.  The display runs at ~10 fps and
never touches the listening path (that lives in the audio callback).

Panels
  Raster        the signal folded at a fractional lap P; LOCK pulls P into
                step with the source; PERSIST averages laps for the eye
  Lap profile   coherent stack of the laps in the raster
  Waterfall     scrolling spectrum, lap fundamental and harmonics marked
  Field (H^2)   two coils -> polarisation ellipse of the field, its path on
                the Poincare disk, and the invariant speed sigma / turning
                kappa.  One coil: a delay-pair view instead (labelled)
"""
from __future__ import annotations

import argparse
import time
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import scipy.signal

from . import dsp
from .audio import AudioEngine, list_devices, sd

BG, PANEL, INK, DIM, LINE = "#0b0e12", "#141a22", "#e6e8eb", "#8b949e", "#2a313c"
BLUE, GOLD, RED, GREEN = "#58a6ff", "#e3b341", "#ff7b72", "#7ee787"
FONT = ("Consolas", 9)
FONT_B = ("Consolas", 10, "bold")

# lap search ranges (Hz) and listening bands per source
PROFILES = {
    "Phone GSM (216.68 Hz)": dict(lo=200, hi=235, band=217.0, q=20, bandpass=False),
    "Phone LTE (1 kHz)": dict(lo=950, hi=1050, band=1000.0, q=15, bandpass=True),
    "Laptop PSU": dict(lo=90, hi=220, band=130.0, q=10, bandpass=True),
    "Monitor PWM": dict(lo=180, hi=2500, band=1000.0, q=25, bandpass=True),
    "USB full speed (1 kHz)": dict(lo=990, hi=1010, band=1000.0, q=15, bandpass=True),
    "Mains 50 Hz": dict(lo=45, hi=55, band=50.0, q=30, bandpass=True),
    "Mains 60 Hz": dict(lo=55, hi=65, band=60.0, q=30, bandpass=True),
    "Speaker audio (wide)": dict(lo=80, hi=400, band=600.0, q=0.7, bandpass=False),
}


def photo(rgb, W, H):
    """Nearest-neighbour resize to W x H and hand to Tk as PPM."""
    h, w, _ = rgb.shape
    yi = (np.arange(H) * h // H).clip(0, h - 1)
    xi = (np.arange(W) * w // W).clip(0, w - 1)
    img = np.ascontiguousarray(rgb[yi][:, xi])
    return tk.PhotoImage(data=f"P6 {W} {H} 255 ".encode() + img.tobytes(), format="PPM")


class Scanner:
    RATE_MS = 100

    def __init__(self, root, engine: AudioEngine):
        self.root = root
        self.eng = engine
        self.fs = engine.fs
        self.lut = dsp.make_lut("viridis")
        self.lut_gray = dsp.make_lut("gray")
        self.lock = dsp.LapLock(self.fs / 216.68, w=220)
        self.phase0 = 0.0
        self.wf_rows = []
        self.h2_trail = []
        self.paused = False
        self._frame_times = []
        self._build()
        self._apply_profile("Phone GSM (216.68 Hz)")
        self.root.after(300, self._loop)

    # ------------------------------------------------------------------ UI
    def _build(self):
        r = self.root
        r.configure(bg=BG)
        r.title(f"GuitarMicScanner · {self.eng.mode} · {self.fs} Hz · {self.eng.channels} ch")
        top = tk.Frame(r, bg=PANEL)
        top.pack(fill=tk.X)

        def btn(text, cmd, fg=INK):
            b = tk.Button(top, text=text, command=cmd, bg=PANEL, fg=fg, activebackground=LINE,
                          relief=tk.FLAT, font=FONT_B, padx=8)
            b.pack(side=tk.LEFT, padx=2, pady=4)
            return b

        def lab(text):
            tk.Label(top, text=text, bg=PANEL, fg=DIM, font=FONT).pack(side=tk.LEFT, padx=(8, 2))

        self.b_pause = btn("PAUSE", self._toggle_pause, RED)
        self.v_listen = tk.BooleanVar(value=True)
        tk.Checkbutton(top, text="LISTEN", variable=self.v_listen, command=self._listen_changed,
                       bg=PANEL, fg=GREEN, selectcolor=BG, activebackground=PANEL,
                       font=FONT_B).pack(side=tk.LEFT, padx=4)
        self.v_bandpass = tk.BooleanVar(value=False)
        tk.Checkbutton(top, text="band", variable=self.v_bandpass, command=self._band_changed,
                       bg=PANEL, fg=INK, selectcolor=BG, activebackground=PANEL,
                       font=FONT).pack(side=tk.LEFT)
        self.v_band = tk.DoubleVar(value=217.0)
        e = tk.Entry(top, textvariable=self.v_band, width=7, bg=BG, fg=GOLD, insertbackground=INK,
                     font=FONT, relief=tk.FLAT)
        e.pack(side=tk.LEFT)
        e.bind("<Return>", lambda ev: self._band_changed())
        lab("Q")
        self.v_q = tk.DoubleVar(value=20.0)
        tk.Scale(top, from_=0.5, to=60, resolution=0.5, variable=self.v_q, orient=tk.HORIZONTAL,
                 length=80, showvalue=False, bg=PANEL, troughcolor=LINE, highlightthickness=0,
                 command=lambda v: self._band_changed()).pack(side=tk.LEFT)
        lab("vol")
        self.v_vol = tk.DoubleVar(value=0.5)
        tk.Scale(top, from_=0, to=1, resolution=0.01, variable=self.v_vol, orient=tk.HORIZONTAL,
                 length=80, showvalue=False, bg=PANEL, troughcolor=LINE, highlightthickness=0,
                 command=lambda v: setattr(self.eng.band, "volume", float(v))).pack(side=tk.LEFT)

        lab("source")
        self.v_prof = tk.StringVar(value="Phone GSM (216.68 Hz)")
        cb = ttk.Combobox(top, textvariable=self.v_prof, values=list(PROFILES), width=22,
                          state="readonly")
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>", lambda ev: self._apply_profile(self.v_prof.get()))
        btn("ACQUIRE", self._acquire, GOLD)
        self.v_lock = tk.BooleanVar(value=True)
        tk.Checkbutton(top, text="LOCK", variable=self.v_lock, bg=PANEL, fg=GOLD, selectcolor=BG,
                       activebackground=PANEL, font=FONT_B).pack(side=tk.LEFT, padx=4)
        lab("px/lap")
        self.v_w = tk.IntVar(value=220)
        tk.Spinbox(top, from_=32, to=1024, increment=4, textvariable=self.v_w, width=5, bg=BG,
                   fg=INK, font=FONT, relief=tk.FLAT, buttonbackground=PANEL).pack(side=tk.LEFT)
        lab("laps")
        self.v_h = tk.IntVar(value=240)
        tk.Spinbox(top, from_=40, to=1200, increment=20, textvariable=self.v_h, width=5, bg=BG,
                   fg=INK, font=FONT, relief=tk.FLAT, buttonbackground=PANEL).pack(side=tk.LEFT)
        lab("persist")
        self.v_persist = tk.IntVar(value=1)
        tk.Spinbox(top, from_=1, to=100, textvariable=self.v_persist, width=4, bg=BG, fg=INK,
                   font=FONT, relief=tk.FLAT, buttonbackground=PANEL).pack(side=tk.LEFT)
        btn("SAVE", self._save, DIM)

        stat = tk.Frame(r, bg=BG)
        stat.pack(fill=tk.X)
        self.l_lap = tk.Label(stat, text="", bg=BG, fg=GOLD, font=FONT_B)
        self.l_lap.pack(side=tk.LEFT, padx=10)
        self.l_lock = tk.Label(stat, text="", bg=BG, fg=DIM, font=FONT_B)
        self.l_lock.pack(side=tk.LEFT, padx=10)
        self.l_info = tk.Label(stat, text="", bg=BG, fg=DIM, font=FONT)
        self.l_info.pack(side=tk.RIGHT, padx=10)

        body = tk.Frame(r, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        left = tk.Frame(body, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = tk.Frame(body, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH)

        def titled(parent, text, w, h):
            f = tk.Frame(parent, bg=BG)
            f.pack(fill=tk.X, pady=(0, 6))
            tk.Label(f, text=text, bg=BG, fg=BLUE, font=FONT_B, anchor="w").pack(fill=tk.X)
            c = tk.Canvas(f, width=w, height=h, bg="#05070a", highlightthickness=0)
            c.pack()
            return c

        self.c_raster = titled(left, "RASTER  (signal folded at lap P)", 620, 400)
        self.c_prof = titled(left, "LAP PROFILE  (coherent stack of the laps above)", 620, 120)
        self.c_wf = titled(right, "WATERFALL  0–2 kHz  (gold: lap fundamental & harmonics)", 520, 250)
        h2 = tk.Frame(right, bg=BG)
        h2.pack(fill=tk.X)
        self.h2_title = tk.Label(h2, text="", bg=BG, fg=BLUE, font=FONT_B, anchor="w")
        self.h2_title.pack(fill=tk.X)
        row = tk.Frame(h2, bg=BG)
        row.pack(fill=tk.X)
        self.c_disk = tk.Canvas(row, width=250, height=250, bg="#05070a", highlightthickness=0)
        self.c_disk.pack(side=tk.LEFT)
        self.c_strip = tk.Canvas(row, width=266, height=250, bg="#05070a", highlightthickness=0)
        self.c_strip.pack(side=tk.LEFT, padx=(4, 0))
        self.l_h2 = tk.Label(h2, text="", bg=BG, fg=DIM, font=FONT, anchor="w", justify=tk.LEFT)
        self.l_h2.pack(fill=tk.X)
        self._img_refs = {}

    # ------------------------------------------------------------------ actions
    def _toggle_pause(self):
        self.paused = not self.paused
        self.b_pause.config(text="RESUME" if self.paused else "PAUSE",
                            fg=GREEN if self.paused else RED)

    def _listen_changed(self):
        self.eng.listen = bool(self.v_listen.get())

    def _band_changed(self):
        try:
            self.eng.band.set_band(float(self.v_band.get()), float(self.v_q.get()))
        except (tk.TclError, ValueError):
            return
        self.eng.band.enabled = bool(self.v_bandpass.get())

    def _apply_profile(self, name):
        p = PROFILES[name]
        self.v_band.set(p["band"])
        self.v_q.set(p["q"])
        self.v_bandpass.set(p["bandpass"])
        self._band_changed()
        self.lock.P = self.fs / (0.5 * (p["lo"] + p["hi"]))
        self.root.after(1500, self._acquire)

    def _acquire(self):
        p = PROFILES[self.v_prof.get()]
        x, _ = self.eng.latest(int(3.0 * self.fs))
        if len(x) < self.fs:
            self.root.after(500, self._acquire)
            return
        P, gain = dsp.acquire(x, self.fs, p["lo"], p["hi"])
        self.lock.P = P
        self.l_info.config(text=f"acquired {self.fs / P:.3f} Hz (comb +{gain:.1f} dB)")

    def _save(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path and "raster" in self._img_refs:
            self._img_refs["raster"].write(path, format="png")

    # ------------------------------------------------------------------ loop
    def _loop(self):
        t0 = time.perf_counter()
        if not self.paused:
            try:
                self._frame()
            except Exception as e:  # keep the loop alive, show the error
                self.l_info.config(text=f"frame error: {e}")
        dt = time.perf_counter() - t0
        self._frame_times = (self._frame_times + [dt])[-20:]
        self.root.after(max(10, self.RATE_MS - int(dt * 1000)), self._loop)

    def _frame(self):
        fs = self.fs
        w = max(16, int(self.v_w.get()))
        h = max(20, int(self.v_h.get()))
        self.lock.w = w
        need = int(self.lock.P * (h + 3))
        x, x0 = self.eng.latest(min(need, self.eng.ring.n))
        if len(x) < self.lock.P * 8:
            return
        h_eff = min(h, int(len(x) / self.lock.P) - 2)
        R, _ = dsp.fold(x, x0, self.lock.P, w, h_eff, self.phase0)
        if self.v_lock.get():
            self.lock.update(R)
        # ---- raster
        k = max(1, int(self.v_persist.get()))
        D = R
        if k > 1 and R.shape[0] > k:
            c = np.cumsum(np.vstack([np.zeros((1, w)), R]), axis=0)
            D = (c[k:] - c[:-k]) / k
        self._put("raster", self.c_raster, dsp.to_rgb(D, self.lut_gray), 620, 400)
        # ---- lap profile
        prof = dsp.stack_profile(R)
        snr = dsp.profile_snr(prof, R)
        self._curve(self.c_prof, prof, GOLD, 620, 120)
        # ---- status
        locked = self.lock.locked and self.v_lock.get()
        self.l_lap.config(text=f"LAP {self.lock.P:9.4f} samples = {fs / self.lock.P:9.4f} Hz")
        self.l_lock.config(text=(f"LOCKED  q {self.lock.quality:.2f}" if locked else
                                 f"hunting  q {self.lock.quality:.2f}  slant {self.lock.lag:+.3f} px/lap"),
                           fg=GREEN if locked else DIM)
        ft = np.mean(self._frame_times) * 1000 if self._frame_times else 0
        self.l_info.config(text=f"stack {snr:+.1f} dB over {R.shape[0]} laps · frame {ft:.0f} ms · {self.eng.note}")
        # ---- waterfall
        seg, _ = self.eng.latest(4096)
        if len(seg) == 4096:
            S = np.abs(np.fft.rfft((seg - seg.mean()) * np.hanning(4096)))
            nb = int(2000 / (fs / 4096))
            self.wf_rows.append(20 * np.log10(S[:nb] + 1e-9))
            self.wf_rows = self.wf_rows[-160:]
            img = np.array(self.wf_rows)[::-1]
            lo, hi = np.percentile(img, 5), np.percentile(img, 99.5)
            self._put("wf", self.c_wf, dsp.to_rgb(img, self.lut, lo, hi), 520, 250)
            f0 = fs / self.lock.P
            for m in range(1, 12):
                xm = m * f0 / 2000 * 520
                if xm > 520:
                    break
                self.c_wf.create_line(xm, 0, xm, 8, fill=GOLD, width=2, tags="ov")
        # ---- field / H^2
        self._field()

    def _field(self):
        fs = self.fs
        two = self.eng.channels >= 2
        m = int(0.8 * fs)
        x, _ = self.eng.latest(m, 0)
        if len(x) < m:
            return
        band_c = float(self.v_band.get()) if self.v_bandpass.get() else None
        if band_c:
            sos = scipy.signal.butter(2, [max(20, band_c * 0.7) / (fs / 2),
                                          min(fs / 2 - 50, band_c * 1.3) / (fs / 2)],
                                      btype="band", output="sos")
        else:
            sos = scipy.signal.butter(2, [80 / (fs / 2), 3000 / (fs / 2)], btype="band", output="sos")
        xf = scipy.signal.sosfiltfilt(sos, x)
        if two:
            y, _ = self.eng.latest(m, 1)
            yf = scipy.signal.sosfiltfilt(sos, y)
            self.h2_title.config(text="FIELD ELLIPSE IN H²  (two coils, mixing-invariant σ, κ)")
        else:
            xf, yf = dsp.delay_pair(xf, fs, band_c or 400.0)
            self.h2_title.config(text="FIELD VIEW  (one coil: delay-pair, NOT mixing-invariant)")
        rd = dsp.H2Reader(fs).read(xf, yf)
        c = self.c_disk
        c.delete("all")
        R0 = 115
        c.create_oval(10, 10, 10 + 2 * R0, 10 + 2 * R0, outline=LINE)
        if rd is None:
            return
        zd = dsp.to_disk(rd["z1"], rd["z2"])
        ok = rd["ellipticity"] > 0.03
        pts = [(125 + R0 * p.real, 125 - R0 * p.imag) for p, g in zip(zd, ok) if g]
        for i in range(1, len(pts)):
            a = i / len(pts)
            col = f"#{int(40 + 200 * a):02x}{int(80 + 120 * a):02x}{int(160 + 80 * a):02x}"
            c.create_line(*pts[i - 1], *pts[i], fill=col)
        # current ellipse, drawn small at the centre
        a, b, d = (np.mean(v[-5:]) for v in (np.exp(rd["z1"]), np.exp(rd["z1"]) * rd["z2"],
                                             np.exp(-rd["z1"]) + np.exp(rd["z1"]) * rd["z2"] ** 2))
        ev, evec = np.linalg.eigh(np.array([[a, b], [b, d]]))
        th = np.linspace(0, 2 * np.pi, 60)
        pts_e = (evec @ (np.sqrt(np.maximum(ev, 1e-6))[:, None] * np.vstack([np.cos(th), np.sin(th)])))
        sc = 22 / max(np.sqrt(ev.max()), 1e-6)
        c.create_line(*[v for p in zip(125 + sc * pts_e[0], 125 - sc * pts_e[1]) for v in p],
                      fill=GOLD, width=2)
        # strips
        s = self.c_strip
        s.delete("all")
        sig = rd["sigma"]
        kap = np.clip(rd["kappa"], -20, 20)
        self._strip(s, sig, 0, 120, BLUE, "speed σ")
        self._strip(s, kap, 126, 120, GOLD, "turning κ")
        ell = float(np.median(rd["ellipticity"]))
        warn = "  (field ≈ linear: ellipse degenerate, σ/κ unreliable)" if ell < 0.05 else ""
        self.l_h2.config(text=f"ellipticity {ell:.3f}   median σ {np.median(sig):.2f}/s   "
                              f"median |κ| {np.median(np.abs(kap)):.2f}{warn}")

    # ------------------------------------------------------------------ drawing
    def _put(self, key, canvas, rgb, W, H):
        ph = photo(rgb, W, H)
        canvas.delete("all")
        canvas.create_image(0, 0, image=ph, anchor="nw")
        self._img_refs[key] = ph

    @staticmethod
    def _curve(canvas, y, col, W, H):
        canvas.delete("all")
        lo, hi = np.min(y), np.max(y)
        u = (y - lo) / (hi - lo + 1e-12)
        xs = np.linspace(0, W, len(y))
        canvas.create_line(*[v for p in zip(xs, H - 6 - u * (H - 12)) for v in p], fill=col, width=2)

    @staticmethod
    def _strip(canvas, y, y0, H, col, label):
        W = int(canvas["width"])
        if len(y) < 2:
            return
        lo, hi = np.percentile(y, 1), np.percentile(y, 99)
        u = np.clip((y - lo) / (hi - lo + 1e-12), 0, 1)
        xs = np.linspace(0, W, len(y))
        canvas.create_line(*[v for p in zip(xs, y0 + H - 4 - u * (H - 8)) for v in p], fill=col)
        canvas.create_text(6, y0 + 4, text=label, fill=col, anchor="nw", font=FONT)


# ---------------------------------------------------------------------- start
def pick_devices(root, default_fs):
    """Small start dialog: input device, output device, channels, rate."""
    devs = list_devices()
    win = tk.Toplevel(root)
    win.title("GuitarMicScanner · devices")
    win.configure(bg=PANEL)
    ins = [d for d in devs if d[2] > 0]
    outs = [d for d in devs if d[3] > 0]
    v_in = tk.StringVar(value=f"{ins[0][0]}: {ins[0][1]}" if ins else "")
    v_out = tk.StringVar(value=f"{outs[0][0]}: {outs[0][1]}" if outs else "")
    v_ch = tk.IntVar(value=1)
    v_fs = tk.IntVar(value=default_fs)
    for i, (lbl, var, opts) in enumerate([("input", v_in, ins), ("output", v_out, outs)]):
        tk.Label(win, text=lbl, bg=PANEL, fg=DIM, font=FONT).grid(row=i, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(win, textvariable=var, width=60, state="readonly",
                     values=[f"{d[0]}: {d[1]}  (in {d[2]}, out {d[3]}, {d[4]} Hz)" for d in opts]
                     ).grid(row=i, column=1, padx=8, pady=4)
    tk.Label(win, text="coils (input channels)", bg=PANEL, fg=DIM, font=FONT).grid(row=2, column=0, sticky="w", padx=8)
    fr = tk.Frame(win, bg=PANEL)
    fr.grid(row=2, column=1, sticky="w")
    for n in (1, 2):
        tk.Radiobutton(fr, text=str(n), variable=v_ch, value=n, bg=PANEL, fg=INK, selectcolor=BG,
                       font=FONT).pack(side=tk.LEFT)
    tk.Label(win, text="sample rate", bg=PANEL, fg=DIM, font=FONT).grid(row=3, column=0, sticky="w", padx=8)
    tk.Entry(win, textvariable=v_fs, width=8, bg=BG, fg=INK, font=FONT).grid(row=3, column=1, sticky="w", padx=8)
    out = {}

    def ok():
        out["in"] = int(v_in.get().split(":")[0]) if v_in.get() else None
        out["out"] = int(v_out.get().split(":")[0]) if v_out.get() else None
        out["ch"] = int(v_ch.get())
        out["fs"] = int(v_fs.get())
        win.destroy()

    tk.Button(win, text="START", command=ok, bg=GOLD, fg=BG, font=FONT_B, relief=tk.FLAT
              ).grid(row=4, column=1, sticky="e", padx=8, pady=8)
    root.wait_window(win)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="GuitarMicScanner")
    ap.add_argument("--demo", action="store_true", help="synthetic phone-on-pickup source")
    ap.add_argument("--wav", help="play a recording (1 or 2 channels) instead of a device")
    ap.add_argument("--channels", type=int, default=None, help="1 = one coil, 2 = two coils")
    ap.add_argument("--fs", type=int, default=22050)
    ap.add_argument("--in-dev", type=int, default=None)
    ap.add_argument("--out-dev", type=int, default=None)
    ap.add_argument("--no-dialog", action="store_true")
    a = ap.parse_args(argv)

    root = tk.Tk()
    root.withdraw()
    if a.demo or a.wav:
        ch = a.channels or 2
        eng = AudioEngine(a.fs, ch)
        if a.demo:
            eng.start_demo(a.out_dev)
        else:
            eng.start_wav(a.wav, a.out_dev)
    else:
        if sd is None:
            raise SystemExit("sounddevice/PortAudio not available: pip install sounddevice, or use --demo")
        if a.no_dialog:
            pick = dict(**{"in": a.in_dev, "out": a.out_dev, "ch": a.channels or 1, "fs": a.fs})
        else:
            pick = pick_devices(root, a.fs)
            if not pick:
                return
        eng = AudioEngine(pick["fs"], pick["ch"])
        eng.start_device(pick["in"], pick["out"])
    root.deiconify()
    Scanner(root, eng)
    root.protocol("WM_DELETE_WINDOW", lambda: (eng.stop(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
