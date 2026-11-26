"""
gms/audio.py -- where samples come from and where the listening goes.

The fix for the old lag lives here: the listening path runs inside the audio
callback, on the newest block only, with filter state carried across blocks.
The GUI never touches it, so a slow frame can no longer delay or chop sound.

Sources
  device : sounddevice duplex Stream (input 1-2 channels -> output 1 channel);
           if the host refuses duplex, separate input/output streams with a
           tiny FIFO of *fresh* blocks (never the old raster window)
  demo   : synthetic "phone on a guitar pickup" (GSM buzz + speaker voice +
           mains hum + noise, two coils with different field directions)
  wav    : a recording played at real-time pace (1 or 2 channels)
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from .dsp import Ring, StreamingBand

try:
    import sounddevice as sd
except Exception:  # PortAudio missing etc.
    sd = None


class AudioEngine:
    def __init__(self, fs=22050, channels=1, ring_s=4.0, block=256):
        self.fs = int(fs)
        self.channels = int(channels)
        self.block = int(block)
        self.ring = Ring(int(fs * ring_s), channels)
        self.lock = threading.Lock()
        self.band = StreamingBand(self.fs)
        self.listen = True
        self.listen_channel = 0
        self.running = False
        self._streams = []
        self._thread = None
        self._fifo = deque(maxlen=4)
        self.mode = None
        self.note = ""

    # ------------------------------------------------------------ shared
    def _ingest(self, block):
        with self.lock:
            self.ring.write(block)

    def _listen_block(self, mono):
        if not self.listen:
            return np.zeros(len(mono), np.float32)
        return self.band.process(mono)

    def latest(self, m, ch=0):
        with self.lock:
            return self.ring.latest(m, ch)

    # ------------------------------------------------------------ device
    def start_device(self, in_dev=None, out_dev=None):
        if sd is None:
            raise RuntimeError("sounddevice / PortAudio not available")
        self.mode = "device"

        def duplex_cb(indata, outdata, frames, t, status):
            self._ingest(indata[:, : self.channels])
            outdata[:, 0] = self._listen_block(indata[:, self.listen_channel])
            if outdata.shape[1] > 1:
                outdata[:, 1:] = outdata[:, :1]

        try:
            s = sd.Stream(device=(in_dev, out_dev), samplerate=self.fs, blocksize=self.block,
                          channels=(self.channels, 1), dtype="float32", callback=duplex_cb,
                          latency="low")
            s.start()
            self._streams = [s]
            self.note = "duplex stream"
        except Exception as e:  # fall back: separate streams + fresh-block FIFO
            def in_cb(indata, frames, t, status):
                self._ingest(indata[:, : self.channels])
                self._fifo.append(self._listen_block(indata[:, self.listen_channel]))

            def out_cb(outdata, frames, t, status):
                outdata.fill(0)
                if self._fifo:
                    b = self._fifo.popleft()
                    k = min(len(b), frames)
                    outdata[:k, 0] = b[:k]

            si = sd.InputStream(device=in_dev, samplerate=self.fs, blocksize=self.block,
                                channels=self.channels, dtype="float32", callback=in_cb,
                                latency="low")
            si.start()
            self._streams = [si]
            try:
                so = sd.OutputStream(device=out_dev, samplerate=self.fs, blocksize=self.block,
                                     channels=1, dtype="float32", callback=out_cb, latency="low")
                so.start()
                self._streams.append(so)
            except Exception:
                pass
            self.note = f"separate streams (duplex refused: {e.__class__.__name__})"
        self.running = True

    # ------------------------------------------------------------ demo / wav
    def _start_output_fifo(self, out_dev=None):
        if sd is None:
            return
        def out_cb(outdata, frames, t, status):
            outdata.fill(0)
            if self._fifo:
                b = self._fifo.popleft()
                k = min(len(b), frames)
                outdata[:k, 0] = b[:k]
        try:
            so = sd.OutputStream(device=out_dev, samplerate=self.fs, blocksize=self.block,
                                 channels=1, dtype="float32", callback=out_cb, latency="low")
            so.start()
            self._streams.append(so)
        except Exception:
            pass

    def _pump(self, gen):
        """Feed blocks from a generator at real-time pace."""
        t_next = time.perf_counter()
        dt = self.block / self.fs
        while self.running:
            blk = next(gen, None)
            if blk is None:
                break
            self._ingest(blk)
            self._fifo.append(self._listen_block(blk[:, self.listen_channel]))
            t_next += dt
            slack = t_next - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            elif slack < -0.5:
                t_next = time.perf_counter()

    def start_demo(self, out_dev=None, seed=0):
        self.mode = "demo"
        self.note = "synthetic phone-on-pickup"
        self.running = True
        self._start_output_fifo(out_dev)
        gen = demo_blocks(self.fs, self.block, self.channels, seed)
        self._thread = threading.Thread(target=self._pump, args=(gen,), daemon=True)
        self._thread.start()

    def start_wav(self, path, out_dev=None, loop=True):
        import scipy.io.wavfile as wf
        fs, data = wf.read(path)
        data = data.astype(np.float32)
        if data.dtype.kind in "iu" or np.max(np.abs(data)) > 2:
            data = data / (np.max(np.abs(data)) + 1e-9)
        if data.ndim == 1:
            data = data[:, None]
        if fs != self.fs:
            import scipy.signal
            data = scipy.signal.resample_poly(data, self.fs, fs, axis=0).astype(np.float32)
        if data.shape[1] < self.channels:
            data = np.repeat(data[:, :1], self.channels, axis=1)
        data = data[:, : self.channels]
        self.mode = "wav"
        self.note = path
        self.running = True
        self._start_output_fifo(out_dev)

        def gen():
            while True:
                for s in range(0, len(data) - self.block, self.block):
                    yield data[s:s + self.block]
                if not loop:
                    return
        self._thread = threading.Thread(target=self._pump, args=(gen(),), daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        for s in self._streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self._streams = []


def demo_blocks(fs, block, channels, seed=0):
    """Synthetic phone lying on a guitar pickup.
      - GSM TDMA buzz: 1 slot in 8 at 216.68 Hz (period 101.76 samples at 22050)
      - speaker voice: gliding harmonic voice with syllable envelope
      - 50 Hz mains hum, noise
    Coil 2 (if any) sees the buzz and the voice along different field
    directions and with a phase offset, so the field ellipse moves."""
    rng = np.random.default_rng(seed)
    f_gsm = 216.68
    n = 0
    phase_v = 0.0
    while True:
        t = (n + np.arange(block)) / fs
        gsm = ((t * f_gsm) % 1.0 < 0.125).astype(np.float64)
        gsm = gsm - 0.125
        f0 = 140 + 30 * np.sin(2 * np.pi * 0.7 * t) + 15 * np.sin(2 * np.pi * 3.1 * t)
        ph = phase_v + 2 * np.pi * np.cumsum(f0) / fs
        phase_v = ph[-1]
        syll = np.clip(np.sin(2 * np.pi * 2.3 * t) + 0.3, 0, None)
        voice = syll * sum(np.sin(k * ph) / k for k in range(1, 9))
        voice_q = syll * sum(np.cos(k * ph) / k for k in range(1, 9))   # 90 deg field lag
        hum = np.sin(2 * np.pi * 50 * t)
        c1 = 0.35 * gsm + 0.25 * voice + 0.05 * hum + 0.05 * rng.normal(size=block)
        if channels == 1:
            out = c1[:, None]
        else:
            c2 = (0.20 * gsm + 0.10 * voice + 0.22 * voice_q + 0.04 * hum
                  + 0.05 * rng.normal(size=block))
            out = np.stack([c1, c2], 1)
        n += block
        yield out.astype(np.float32)


def list_devices():
    if sd is None:
        return []
    out = []
    for i, d in enumerate(sd.query_devices()):
        out.append((i, d["name"], d["max_input_channels"], d["max_output_channels"],
                    int(d["default_samplerate"])))
    return out
