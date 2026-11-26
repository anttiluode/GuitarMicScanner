"""
bench/bench.py -- everything that can be checked without the guitar.

  1. Listening path: the old Tempest Explorer sonification (queue the whole
     raster window each GUI frame, play its first 1024 samples) vs the new
     streaming band filter inside the audio callback.  Measured: delay,
     how much of the input is actually heard, CPU per block.
  2. Fractional lap lock vs the best integer raster width, on a synthetic
     GSM-style 216.68 Hz buzz (period 101.763 samples at 22050 Hz) in noise.
  3. Two coils -> H^2: sigma and kappa of the field-ellipse path before and
     after a random mixing of the coils (rotation, gain mismatch, crosstalk).

Run:  python bench/bench.py      (writes bench/results.json and bench/*.png)
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from gms import dsp  # noqa: E402

rng = np.random.default_rng(1)
fs = 22050
res = {}

# ============================================================== 1. listening
# Feed a ramp: input sample i has value i.  Whatever comes out of the speaker
# then says exactly which input sample it is and how late it is.
dur = 8.0
n = int(fs * dur)
ramp = np.arange(n, dtype=np.float64)


def old_path(x, frame_s, w=220, buf_s=2.5, start_s=3.0):
    """Tempest Explorer as written: every GUI frame (period frame_s) the last
    h*w samples of the 2.5 s ring are queued (deque maxlen 10); the output
    callback pops one queued block per 1024-sample period and plays its first
    1024 samples."""
    buf_len = int(fs * buf_s)
    h = min(600, buf_len // w)
    win = h * w
    out = np.full(len(x), np.nan)
    queue = []
    next_frame = start_s
    for start in range(int(start_s * fs), len(x) - 1024, 1024):
        now = start / fs
        while next_frame <= now:
            p = int(next_frame * fs)
            queue.append(x[p - win:p])
            queue = queue[-10:]
            next_frame += frame_s
        if queue:
            out[start:start + 1024] = queue.pop(0)[:1024]
    return out


def new_path(x, block=256, start_s=3.0):
    """New: each input block is filtered and written to the output in the
    same duplex callback; it reaches the speaker one block later."""
    out = np.full(len(x), np.nan)
    times = []
    band = dsp.StreamingBand(fs, 217.0, 20.0)
    for s0 in range(int(start_s * fs), len(x) - 2 * block, block):
        blk = x[s0:s0 + block]
        t0 = time.perf_counter()
        band.process(rng.normal(size=block))   # the real cost per block
        times.append(time.perf_counter() - t0)
        out[s0 + block:s0 + 2 * block] = blk    # identity of what is played
    return out, np.array(times)


def describe(out):
    played = np.isfinite(out)
    idx = np.arange(len(out))[played]
    delay = (idx - out[played]) / fs
    span = int(np.sum(played))
    heard_fraction = span / (len(out) - int(3.0 * fs))
    unique_inputs = len(np.unique(out[played].astype(np.int64)))
    return dict(delay_median_s=float(np.median(delay)), delay_max_s=float(np.max(delay)),
                speaker_busy_fraction=float(heard_fraction),
                input_coverage=float(unique_inputs / (len(out) - int(3.0 * fs))))


rows1 = {}
for frame_s in [0.033, 0.08, 0.15]:
    rows1[f"old_gui_{int(frame_s * 1000)}ms"] = describe(old_path(ramp, frame_s))
o_new, ptimes = new_path(ramp)
rows1["new_streaming"] = describe(o_new)
rows1["new_streaming"]["cpu_us_per_256_block_bandpass"] = float(np.median(ptimes) * 1e6)
res["listening"] = rows1
for k, v in rows1.items():
    print("1.", k, {kk: round(vv, 4) for kk, vv in v.items()})

# ============================================================== 2. lock
T = fs / 216.68                     # GSM TDMA frame rate -> fractional period
n2 = int(fs * 3.0)
tt2 = np.arange(n2)
burst = ((tt2 % T) / T < 0.125).astype(float)          # 1 of 8 slots on
x2 = 0.25 * burst + rng.normal(size=n2)                 # buried: per-sample SNR -12 dB
W, H = 128, 600
# best integer width (what an integer raster can do)
best = None
for P in range(95, 110):
    R, _ = dsp.fold(x2, 0, float(P), W, H)
    sc = dsp.profile_snr(dsp.stack_profile(R), R)
    if best is None or sc > best[1]:
        best = (P, sc)
P_int = best[0]
R_int, _ = dsp.fold(x2, 0, float(P_int), W, H)
# fractional: harmonic-comb acquisition, then the lap lock
P_acq, comb_db = dsp.acquire(x2, fs, 150, 300)
L = dsp.LapLock(P_acq, w=W)
trace = []
for it in range(30):
    R, _ = dsp.fold(x2, 0, L.P, W, H)
    L.update(R)
    trace.append(L.P)
R_frac, _ = dsp.fold(x2, 0, L.P, W, H)
snr_int = dsp.profile_snr(dsp.stack_profile(R_int), R_int)
snr_frac = dsp.profile_snr(dsp.stack_profile(R_frac), R_frac)
prof_true = ((np.arange(W) / W) < 0.125).astype(float) * 0.25
drift_int = (T - P_int) / T * W * H  # pixels the burst walks across the picture
res["lock"] = dict(true_period=T, best_integer_width=P_int, acquired_period=P_acq,
                   comb_gain_db=comb_db, locked_period=L.P,
                   period_error_ppm=float((L.P - T) / T * 1e6),
                   stacked_snr_db_integer=snr_int, stacked_snr_db_locked=snr_frac,
                   burst_walk_px_integer_over_600_laps=float(drift_int),
                   lock_quality=L.quality, trace=trace)
print(f"2. true lap {T:.4f}; best integer {P_int}; locked {L.P:.4f}"
      f" ({res['lock']['period_error_ppm']:.1f} ppm); stacked SNR {snr_int:.1f} -> {snr_frac:.1f} dB;"
      f" integer raster walks {drift_int:.0f} px over 600 laps")

# ============================================================== 3. H2
n3 = int(fs * 2.0)
t3 = np.arange(n3) / fs
f0 = 1000.0
# polarisation that changes with a slow "voice": axis ratio and phase
a = 1.0 + 0.6 * np.sin(2 * np.pi * 3.0 * t3)
b = 0.8 + 0.5 * np.sin(2 * np.pi * 5.0 * t3 + 1.0)
phi = 0.9 + 0.7 * np.sin(2 * np.pi * 2.0 * t3)
fx = a * np.cos(2 * np.pi * f0 * t3)
fy = b * np.cos(2 * np.pi * f0 * t3 + phi)
reader = dsp.H2Reader(fs)
base_sig = reader.read(fx + 0.01 * rng.normal(size=n3), fy + 0.01 * rng.normal(size=n3))
rows3 = []
for trial in range(10):
    M = rng.normal(size=(2, 2))
    if np.linalg.det(M) < 0:
        M[:, 0] *= -1
    mx, my = M @ np.vstack([fx, fy])
    r2 = reader.read(mx + 0.01 * rng.normal(size=n3), my + 0.01 * rng.normal(size=n3))
    m = slice(10, -10)
    # kappa = turning per unit speed: only meaningful where the ellipse moves
    moving = base_sig["sigma"][m] > np.percentile(base_sig["sigma"][m], 40)
    corr_sigma = np.corrcoef(base_sig["sigma"][m], r2["sigma"][m])[0, 1]
    corr_kappa = np.corrcoef(base_sig["kappa"][m][moving], r2["kappa"][m][moving])[0, 1]
    corr_raw = np.corrcoef(base_sig["z1"][m], r2["z1"][m])[0, 1]
    if trial == 0 or abs(corr_raw) < min(abs(r["corr_raw_z1"]) for r in rows3):
        mix_example = (base_sig, r2)   # plot the mix that scrambles the raw ellipse most
    rows3.append(dict(cond_M=float(np.linalg.cond(M)), corr_sigma=float(corr_sigma),
                      corr_kappa=float(corr_kappa), corr_raw_z1=float(corr_raw),
                      ellipticity_mean=float(np.mean(r2["ellipticity"]))))
    print(f"3. mix cond={np.linalg.cond(M):5.1f}  corr sigma {corr_sigma:.4f}  kappa {corr_kappa:.4f}"
          f"  raw z1 {corr_raw:+.3f}")
# degenerate case: linearly polarised field (both coils see the same signal)
lin = reader.read(fx, 0.5 * fx + 1e-4 * rng.normal(size=n3))
res["h2"] = dict(rows=rows3, linear_case_ellipticity=float(np.median(lin["ellipticity"])))
print(f"3. linear polarisation -> median ellipticity {res['h2']['linear_case_ellipticity']:.4f} (degenerate)")

with open(os.path.join(HERE, "results.json"), "w") as fh:
    json.dump(res, fh, indent=1, default=float)

# ============================================================== figures
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
keys = list(rows1.keys())
xs = np.arange(len(keys))
ax[0, 0].bar(xs, [rows1[k]["delay_median_s"] for k in keys], color=["#c44536"] * 3 + ["#2a6f97"])
ax[0, 0].set_xticks(xs, ["old\n33 ms GUI", "old\n80 ms GUI", "old\n150 ms GUI", "new\nstreaming"])
ax[0, 0].set_ylabel("delay input → speaker (s)")
ax[0, 0].set_title("1. what you hear, and when")
for i, k in enumerate(keys):
    ax[0, 0].text(i, rows1[k]["delay_median_s"] + 0.05,
                  f"{rows1[k]['input_coverage'] * 100:.0f}% of input\nheard", ha="center", fontsize=8)
ax[0, 0].set_ylim(0, 3.4)
def vsmooth(R, k=80):
    ker = np.ones(k) / k
    return np.apply_along_axis(lambda c: np.convolve(c, ker, mode="valid"), 0, R)


ax[0, 1].imshow(vsmooth(R_int), aspect="auto", cmap="gray")
ax[0, 1].set_title(f"2a. best integer width {P_int}: the burst walks")
ax[0, 1].set_xlabel("pixel in lap")
ax[0, 1].set_ylabel("lap (80-lap running mean, display only)")
ax[0, 2].imshow(vsmooth(R_frac), aspect="auto", cmap="gray")
ax[0, 2].set_title(f"2b. fractional lock P = {L.P:.3f}: it stands still")
ax[0, 2].set_xlabel("pixel in lap")
ax[1, 0].plot(dsp.stack_profile(R_int), color="#c44536", label=f"integer, {snr_int:.1f} dB")
ax[1, 0].plot(dsp.stack_profile(R_frac), color="#2a6f97", label=f"locked, {snr_frac:.1f} dB")
ax[1, 0].plot(prof_true, color="#999", ls="--", label="true burst")
ax[1, 0].set_title("2c. 600 laps stacked (input per-sample SNR −12 dB)")
ax[1, 0].legend(fontsize=8)
bs, r2 = mix_example
m = slice(10, -10)
ax[1, 1].plot(bs["t"][m], bs["sigma"][m], color="#2a6f97", lw=1.5, label="coils as placed")
ax[1, 1].plot(r2["t"][m], r2["sigma"][m], color="#e3b341", lw=1, ls="--", label="coils mixed by random M")
ax[1, 1].set_title("3. ellipse speed σ is blind to how the coils sit")
ax[1, 1].set_xlabel("time (s)")
ax[1, 1].legend(fontsize=8)
ax[1, 2].plot(bs["t"][m], bs["z1"][m], color="#2a6f97", lw=1.5, label="raw z1, as placed")
ax[1, 2].plot(r2["t"][m], r2["z1"][m], color="#e3b341", lw=1, ls="--", label="raw z1, mixed")
ax[1, 2].set_title("3'. the raw ellipse is not")
ax[1, 2].set_xlabel("time (s)")
ax[1, 2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "bench.png"), dpi=120)
print("wrote bench/bench.png and bench/results.json")
