"""
gms/dsp.py -- the signal core of GuitarMicScanner.  numpy + scipy only.

Pieces
  StreamingBand   band filter with persistent state, block by block: the
                  low-latency listening path (runs inside the audio callback)
  Ring            sample ring buffer with an absolute sample counter
  LapLock         the raster as a ring: fold the signal at a *fractional* lap
                  P (samples), measure the slant between consecutive laps,
                  and pull P into step with the source (a phase-lock loop)
  stack_profile   coherent average of the last N laps
  H2Reader        two coils -> windowed 2x2 field covariance = area x
                  unimodular ellipse = a point in the hyperbolic plane.
                  Reads speed sigma and turning kappa of the ellipse path:
                  invariant to any linear mixing of the two coils
                  (rotation of the phone, coil gain mismatch, crosstalk).
"""
from __future__ import annotations

import numpy as np
import scipy.signal

# ----------------------------------------------------------------------------
# listening path
# ----------------------------------------------------------------------------


class StreamingBand:
    """Band-pass (or pass-through) with filter state carried across blocks,
    a slow AGC instead of per-block normalisation, and a soft clip."""

    def __init__(self, fs, center=217.0, q=20.0, enabled=True, order=4):
        self.fs = fs
        self.order = order
        self.enabled = enabled
        self.gain = 1.0
        self.volume = 0.5
        self.agc = True
        self._env = 1e-3
        self.set_band(center, q)

    def set_band(self, center, q):
        nyq = self.fs / 2
        center = float(np.clip(center, 20.0, nyq - 50.0))
        bw = center / max(q, 0.5)
        lo = max(10.0, center - bw / 2) / nyq
        hi = min(nyq - 10.0, center + bw / 2) / nyq
        self.center, self.q = center, q
        self.sos = scipy.signal.butter(self.order, [lo, hi], btype="band", output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2))

    def process(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.enabled:
            y, self.zi = scipy.signal.sosfilt(self.sos, x, zi=self.zi)
        else:
            y = x
        if self.agc:
            # slow envelope follower (~0.3 s), applied smoothly
            e = np.sqrt(np.mean(y * y) + 1e-12)
            self._env = 0.9 * self._env + 0.1 * e
            y = y * (0.25 / max(self._env, 1e-6))
        y = y * self.gain * self.volume
        return np.tanh(y).astype(np.float32)


# ----------------------------------------------------------------------------
# buffer
# ----------------------------------------------------------------------------


class Ring:
    """Multi-channel ring buffer; `count` is the absolute number of samples
    ever written, so laps can be anchored to absolute time."""

    def __init__(self, n, channels=1):
        self.n = int(n)
        self.buf = np.zeros((self.n, channels), dtype=np.float32)
        self.count = 0

    def write(self, block):
        block = np.asarray(block, dtype=np.float32)
        if block.ndim == 1:
            block = block[:, None]
        m = block.shape[0]
        if m >= self.n:
            block = block[-self.n:]
            m = self.n
        p = self.count % self.n
        first = min(m, self.n - p)
        self.buf[p:p + first, : block.shape[1]] = block[:first]
        if first < m:
            self.buf[: m - first, : block.shape[1]] = block[first:]
        self.count += m

    def latest(self, m, ch=0):
        """Last m samples of one channel, oldest first, and the absolute index
        of the first returned sample."""
        m = int(min(m, self.n, self.count))
        end = self.count % self.n
        idx = (np.arange(end - m, end)) % self.n
        return self.buf[idx, ch].astype(np.float64), self.count - m


# ----------------------------------------------------------------------------
# the raster as a ring: fractional lap and lock
# ----------------------------------------------------------------------------


def fold(x, x0_abs, P, w, h, phase0=0.0):
    """Fold samples x (absolute index of x[0] = x0_abs) into h laps of w
    pixels at a fractional lap P (samples).  Laps start at absolute times
    phase0 + k*P (integer k), so a locked picture does not jump between
    frames.  Returns (raster h x w, index k of the first lap)."""
    n_end = x0_abs + len(x)
    k_last = int(np.floor((n_end - 1 - phase0) / P)) - 1  # last complete lap
    k_first = k_last - h + 1
    ks = np.arange(k_first, k_last + 1)
    t = phase0 + ks[:, None] * P + np.arange(w)[None, :] * (P / w) - x0_abs
    t = np.clip(t, 0, len(x) - 1.001)
    i = t.astype(np.int64)
    f = t - i
    return x[i] * (1 - f) + x[i + 1] * f, k_first


def row_drift(raster, sep=1):
    """Average slant between laps `sep` apart, returned in pixels per lap, by
    the cross-power spectrum of the row pairs (circular), with parabolic
    sub-pixel refinement.  Also returns the normalised peak correlation.
    A long baseline (sep >> 1) multiplies the drift before it is measured,
    which removes the sub-pixel dead zone near lock."""
    R = raster - raster.mean(axis=1, keepdims=True)
    sep = int(max(1, min(sep, R.shape[0] - 1)))
    F = np.fft.rfft(R, axis=1)
    cross = np.sum(F[sep:] * np.conj(F[:-sep]), axis=0)
    norm = np.sqrt(np.sum(R[sep:] ** 2) * np.sum(R[:-sep] ** 2)) + 1e-12
    cc = np.fft.irfft(cross, n=R.shape[1])  # = sum_rows sum_t R1(t) R0(t - lag)
    w = R.shape[1]
    j = int(np.argmax(cc))
    y0, y1, y2 = cc[(j - 1) % w], cc[j], cc[(j + 1) % w]
    den = y0 - 2 * y1 + y2
    frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    lag = j + frac
    if lag > w / 2:
        lag -= w
    return float(lag / sep), float(y1 / norm)


class LapLock:
    """Phase-lock loop on the lap length.  If row k+1 is the same as row k
    shifted by `lag` pixels, the lap is off by lag * P / w samples; pull P
    towards that.  Gain < 1 keeps it calm."""

    def __init__(self, P, w=220, gain=0.35, max_step=0.02, min_quality=0.15):
        self.min_quality = min_quality
        self.P = float(P)
        self.w = int(w)
        self.gain = gain
        self.max_step = max_step
        self.locked = False
        self.quality = 0.0
        self.lag = 0.0

    def update(self, raster, group=None):
        """raster: h x w laps folded at the current P.  Laps are first averaged
        in consecutive groups (coherent sub-stacks) so the slant can be seen
        even when one lap is buried in noise; the slant between groups is
        divided by the group length."""
        h = raster.shape[0]
        g = group or self.group_for(raster)
        ng = h // g
        if ng < 3:
            g, ng = 1, h
        G = raster[h - ng * g:].reshape(ng, g, -1).mean(axis=1)
        lag, q = row_drift(G, 1)
        lag /= g
        if abs(lag) * g < 0.5 and q > 0.2:
            # near lock: long baseline for sub-pixel precision
            sep = max(2, min(ng // 2, 32))
            lag_long, q_long = row_drift(G, sep)
            if abs(lag_long * sep) < self.w / 4:
                lag, q = lag_long / g, max(q, q_long)
        self.lag, self.quality = lag, q
        if q < self.min_quality:   # nothing to lock to: hold, don't wander
            self.locked = False
            return self.P
        step = self.gain * lag * self.P / self.w
        step = float(np.clip(step, -self.max_step * self.P, self.max_step * self.P))
        self.P += step
        self.locked = q > 0.4 and abs(lag) < 0.05
        return self.P

    @staticmethod
    def group_for(raster):
        """Pick a sub-stack length so one group's mean has usable SNR:
        estimate per-lap SNR from the lap-to-lap correlation of neighbours."""
        R = raster - raster.mean(axis=1, keepdims=True)
        c = np.sum(R[1:] * R[:-1]) / (np.sum(R * R) + 1e-12)
        c = float(np.clip(c, 1e-3, 0.99))
        snr = c / (1 - c)               # signal/noise power per lap
        g = int(np.clip(np.ceil(2.0 / snr), 1, raster.shape[0] // 4))
        return max(1, g)


def acquire(x, fs, f_lo, f_hi, harmonics=12, step_hz=0.005, pad=4):
    """Coarse lock: find the fundamental f0 in [f_lo, f_hi] whose harmonic
    comb holds the most power (zero-padded spectrum, linear interpolation
    between bins).  Returns (lap P = fs/f0 in samples, comb score in dB over
    the median candidate).  Hands over to LapLock for the fine lock."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    N = int(2 ** np.ceil(np.log2(n * pad)))
    S = np.abs(np.fft.rfft(x * np.hanning(n), N)) ** 2
    df = fs / N
    f0 = np.arange(f_lo, f_hi, step_hz)
    score = np.zeros_like(f0)
    for k in range(1, harmonics + 1):
        fk = k * f0
        ok = fk < fs / 2 - df
        b = fk / df
        i = b.astype(np.int64)
        fr = b - i
        val = np.zeros_like(f0)
        val[ok] = S[i[ok]] * (1 - fr[ok]) + S[i[ok] + 1] * fr[ok]
        score += val
    j = int(np.argmax(score))
    gain_db = 10 * np.log10(score[j] / (np.median(score) + 1e-30) + 1e-30)
    return fs / f0[j], float(gain_db)


def stack_profile(raster, n=None):
    """Coherent average of the last n laps (all if None)."""
    R = raster if n is None else raster[-n:]
    return R.mean(axis=0)


def profile_snr(profile, raster):
    """Periodic power of the stacked lap vs the per-sample residual power."""
    resid = raster - profile[None, :]
    sig = np.var(profile)
    noise = np.mean(resid ** 2) / raster.shape[0]
    return float(10 * np.log10((sig + 1e-18) / (noise + 1e-18)))


# ----------------------------------------------------------------------------
# two coils -> hyperbolic plane
# ----------------------------------------------------------------------------


class H2Reader:
    """Field covariance of two (band-passed) coil signals over short windows.

        C = [[<xx>, <xy>], [<xy>, <yy>]]  = sqrt(det C) * qbar,  det qbar = 1

    qbar is the polarisation ellipse of the field (orientation + shape), a
    point in H^2 (Kartiokello coordinates: z1 = log qbar11, z2 = qbar12/qbar11).
    sqrt(det C) is its area (loudness of the elliptical part).

    Mixing the coils by any matrix M (det > 0) maps C -> M C M^T, i.e. an
    isometry of H^2 after removing det: the ellipse *path* keeps its speed
    sigma and turning kappa.  (det < 0, i.e. one coil reversed, flips the
    sign of kappa only.)

    ellipticity e = 2 sqrt(det C) / tr C in [0, 1]: 0 = linear polarisation
    (degenerate: the ellipse is a line and lives at the edge of H^2),
    1 = circular.  sigma/kappa are only meaningful where e is not tiny."""

    def __init__(self, fs, win_ms=16.0, hop_ms=4.0):
        self.fs = fs
        self.win = max(8, int(fs * win_ms / 1000))
        self.hop = max(1, int(fs * hop_ms / 1000))

    def covariances(self, x, y):
        nwin = (len(x) - self.win) // self.hop + 1
        if nwin < 3:
            return np.zeros((0, 3))
        idx = np.arange(self.win)[None, :] + self.hop * np.arange(nwin)[:, None]
        X, Y = x[idx], y[idx]
        X = X - X.mean(1, keepdims=True)
        Y = Y - Y.mean(1, keepdims=True)
        return np.stack([np.mean(X * X, 1), np.mean(X * Y, 1), np.mean(Y * Y, 1)], 1)

    @staticmethod
    def to_h2(c):
        a, b, d = c[:, 0], c[:, 1], c[:, 2]
        det = np.maximum(a * d - b * b, 1e-30)
        s = np.sqrt(det)
        q11, q12 = a / s, b / s
        z1 = np.log(np.maximum(q11, 1e-30))
        z2 = q12 / np.maximum(q11, 1e-30)
        tr = a + d
        ell = 2 * s / np.maximum(tr, 1e-30)
        return z1, z2, s, ell

    def read(self, x, y, smooth=5):
        """Returns dict: z1, z2, area, ellipticity, sigma, kappa, t (s)."""
        c = self.covariances(np.asarray(x, float), np.asarray(y, float))
        if len(c) < 5:
            return None
        z1, z2, area, ell = self.to_h2(c)
        dt = self.hop / self.fs
        if smooth > 1:
            k = np.ones(smooth) / smooth
            z1s = np.convolve(z1, k, mode="same")
            z2s = np.convolve(z2, k, mode="same")
        else:
            z1s, z2s = z1, z2
        d1 = np.gradient(z1s, dt)
        d2 = np.gradient(z2s, dt)
        e = np.exp(z1s)
        sigma = np.sqrt(d1 ** 2 + (e * d2) ** 2)
        theta = np.unwrap(np.arctan2(e * d2, d1))
        kappa = np.gradient(theta, dt) / np.maximum(sigma, 1e-9) + np.sin(theta)
        t = (np.arange(len(z1)) * self.hop + self.win / 2) / self.fs
        return dict(z1=z1, z2=z2, area=area, ellipticity=ell, sigma=sigma,
                    kappa=kappa, t=t)


def delay_pair(x, fs, f_band):
    """One-coil fallback: (x(t), x(t - tau)) with tau a quarter period of the
    focus band.  A clean tone gives a circle; modulation bends the ellipse.
    NOT mixing-invariant in any physical sense -- just a 2D view of 1 coil."""
    tau = max(1, int(round(fs / (4.0 * f_band))))
    return x[tau:], x[:-tau]


def to_disk(z1, z2):
    """H^2 point -> Poincare disk, for drawing.  Upper half plane w = z2 + i e^{-z1}."""
    w = z2 + 1j * np.exp(-z1)
    return (w - 1j) / (w + 1j)


# ----------------------------------------------------------------------------
# colour
# ----------------------------------------------------------------------------


def make_lut(name="viridis"):
    try:
        import matplotlib
        cmap = matplotlib.colormaps[name]
        lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    except Exception:
        g = np.arange(256, dtype=np.uint8)
        lut = np.stack([g, g, g], 1)
    return lut


def to_rgb(img, lut, lo=None, hi=None):
    lo = np.percentile(img, 2) if lo is None else lo
    hi = np.percentile(img, 98) if hi is None else hi
    u = np.clip((img - lo) / (hi - lo + 1e-12), 0, 1)
    return lut[(u * 255).astype(np.uint8)]
