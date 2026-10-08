'use strict';

/* Pendulum clock tick-tock analyser.
 *
 * Pipeline: microphone -> high-pass filter -> AudioWorklet (raw sample blocks)
 *   -> Detector (finds each tick/tock, timestamped in audio-clock seconds)
 *   -> BeatAnalyser (beat error + rate) -> canvas UI.
 *
 * All timing comes from the audio sample counter, not from JS timers, so it
 * is not affected by main-thread jitter.
 */

// ---------------------------------------------------------------- Detector

class Detector {
  constructor(sampleRate, onEvent) {
    this.sr = sampleRate;
    this.onEvent = onEvent;
    this.n = 0;                 // absolute index of the next sample
    this.env = 0;               // smoothed rectified signal
    this.floor = 0;             // noise-floor estimate (median of recent peaks)
    this.mult = 3.4;            // threshold = floor * mult
    this.minThr = 1e-5;         // never trigger on digital silence
    this.atk = 1 - Math.exp(-1 / (0.0003 * sampleRate));
    this.rel = 1 - Math.exp(-1 / (0.002 * sampleRate));
    this.warmup = Math.round(0.5 * sampleRate);
    this.winLen = Math.round(0.03 * sampleRate);   // event analysis window
    this.lockLen = Math.round(0.12 * sampleRate);  // dead time after onset
    this.inWindow = false;
    this.win = [];
    this.winStart = 0;
    this.lockUntil = 0;
    this.binLen = Math.round(0.005 * sampleRate);  // scope resolution: 5 ms
    this.binMax = 0;
    this.binCount = 0;
    this.bins = [];             // decimated envelope for the scope and period finding
    this.maxBins = 1600;        // 8 s
    this.raw = [];              // recent emitted event times, for the scope

    // Gating: once the beat period is known, only listen around where the
    // next beat should be, with a lower threshold, and take the loudest peak.
    this.period = 0;            // seconds per beat, set by Bank; 0 = free-running
    this.nextBeat = null;       // predicted sample index of the next beat
    this.misses = 0;
    this.steady = 0;            // consecutive beats found inside the gate
    this.times = [];            // last three emitted event sample indices
    this.best = null;           // loudest candidate in the current gate
  }

  get threshold() { return Math.max(this.floor * this.mult, this.minThr); }
  get gateThreshold() { return Math.max(this.floor * Math.max(1.25, this.mult * 0.5), this.minThr); }
  // Wide gate while acquiring; narrow once 8 beats in a row landed in it.
  get gateHalf() { return (this.steady >= 8 ? 0.08 : 0.25) * this.period * this.sr; }

  setPeriod(p) {
    if (!p || !this.period || Math.abs(p / this.period - 1) > 0.15) this.unlock();
    this.period = p || 0;
  }

  unlock() {
    this.nextBeat = null;
    this.misses = 0;
    this.steady = 0;
    this.best = null;
  }

  // Feed filtered samples; `start` is the absolute index of samples[0].
  push(samples, start) {
    if (start !== undefined) this.n = start;
    for (let i = 0; i < samples.length; i++) this.step(Math.abs(samples[i]));
  }

  step(a) {
    const n = this.n++;
    this.env += (a - this.env) * (a > this.env ? this.atk : this.rel);
    const env = this.env;

    if (env > this.binMax) this.binMax = env;
    if (++this.binCount >= this.binLen) {
      this.bins.push(this.binMax);
      if (this.bins.length > this.maxBins) this.bins.shift();
      this.binCount = 0;
      this.binMax = 0;
      if (this.bins.length % 20 === 0) this.updateFloor();
    }

    if (n < this.warmup) return;

    if (this.inWindow) {
      this.win.push(env);
      if (this.win.length >= this.winLen) this.finishWindow();
      return;
    }

    if (this.nextBeat === null) {
      // Free-running: first crossing of the normal threshold is an event.
      if (env > this.threshold && n >= this.lockUntil) this.openWindow(n, env);
      return;
    }

    // Gated: ignore everything outside the predicted window.
    const gh = this.gateHalf;
    if (n < this.nextBeat - gh) return;
    if (n <= this.nextBeat + gh) {
      if (env > this.gateThreshold && n >= this.lockUntil) this.openWindow(n, env);
      return;
    }
    this.closeGate();
  }

  openWindow(n, env) {
    this.inWindow = true;
    this.winStart = n;
    this.win = [env];
  }

  finishWindow() {
    const ev = this.measureWindow();
    if (this.nextBeat === null) {
      this.lockUntil = this.winStart + this.lockLen;
      this.emit(ev);
    } else {
      // Inside a gate: keep looking. Best = loudest, discounted by distance
      // from where the beat was predicted.
      this.lockUntil = this.winStart + this.winLen;
      const off = (ev.idx - this.nextBeat) / this.gateHalf;
      ev.score = ev.peak * (1 - 0.5 * off * off);
      if (!this.best || ev.score > this.best.score) this.best = ev;
    }
  }

  closeGate() {
    const best = this.best;
    this.best = null;
    if (best) {
      this.misses = 0;
      this.steady++;
      this.emit(best);
      return;
    }
    // Nothing heard: widen the gate and coast on to the next expected beat;
    // give up after 4 misses in a row.
    this.steady = 0;
    if (++this.misses >= 4) this.unlock();
    else this.nextBeat += this.period * this.sr;
  }

  emit(ev) {
    const t = this.times;
    t.push(ev.idx);
    if (t.length > 3) t.shift();
    if (this.period) {
      // Beats alternate A, B, A, B: the next interval should match the one
      // before last. Fall back to the nominal period if that looks wrong.
      const P = this.period * this.sr;
      const prev = t.length === 3 ? t[1] - t[0] : 0;
      const step = prev > 0.75 * P && prev < 1.25 * P ? prev : P;
      this.nextBeat = ev.idx + step;
    }
    const sec = ev.idx / this.sr;
    this.raw.push(sec);
    while (this.raw.length && this.raw[0] < sec - 10) this.raw.shift();
    this.onEvent(sec, ev.peak);
  }

  // Noise floor = median of the last ~4 s of 5 ms peak-envelope bins. Ticks
  // fill only a small part of that time, so the median ignores them and
  // tracks the background noise however quiet the ticks are.
  updateFloor() {
    const recent = this.bins.slice(-800).sort((a, b) => a - b);
    this.floor = recent[recent.length >> 1];
  }

  // Timestamp = where the envelope first reaches 50% of its peak, with linear
  // interpolation. That is independent of how loud the tick is.
  // Find where the loudest pulse starts (walk back from its peak to 25%),
  // then the first 50% crossing after that. An earlier, smaller click in
  // the same window doesn't set the timestamp, and envelope ripple on a
  // low-pitched tick doesn't either.
  measureWindow() {
    const win = this.win;
    let peak = 0, ip = 0;
    for (let k = 0; k < win.length; k++) if (win[k] > peak) { peak = win[k]; ip = k; }
    const half = peak * 0.5;
    let s = ip;
    while (s > 0 && win[s - 1] >= peak * 0.25) s--;
    let i = s;
    while (i < ip && win[i] < half) i++;
    let frac = 0;
    if (i > 0) frac = (win[i] - half) / (win[i] - win[i - 1] || 1);
    this.inWindow = false;
    this.win = [];
    return { idx: this.winStart + i - frac, peak };
  }
}

// Beat period from the envelope by autocorrelation, independent of any
// threshold. Bins are log-compressed against the noise floor so a quiet tock
// still counts next to a loud tick. Returns { period (s), strength (0-1) };
// period 0 if nothing periodic.
function estimatePeriod(bins, binSec, floor) {
  const none = { period: 0, strength: 0 };
  const N = bins.length;
  if (N < 400 || !(floor > 0)) return none;
  const x = new Float64Array(N);
  let m = 0;
  for (let i = 0; i < N; i++) { x[i] = Math.log(Math.max(bins[i], floor) / floor); m += x[i]; }
  m /= N;
  let r0 = 0;
  for (let i = 0; i < N; i++) { x[i] -= m; r0 += x[i] * x[i]; }
  if (r0 === 0) return none;
  r0 /= N;
  const lo = Math.round(0.15 / binSec), hi = Math.min(Math.round(3 / binSec), N >> 1);
  const r = new Float64Array(hi + 2);
  for (let L = lo; L <= hi + 1; L++) {
    let s = 0;
    for (let i = 0; i + L < N; i++) s += x[i] * x[i + L];
    r[L] = s / (N - L) / r0;
  }
  const isPeak = L => L > lo && L <= hi && r[L] >= r[L - 1] && r[L] >= r[L + 1];
  let best = 0;
  for (let L = lo + 1; L <= hi; L++) if (isPeak(L) && (!best || r[L] > r[best])) best = L;
  if (!best || r[best] < 0.15) return none;
  // The strongest peak can be a multiple of the beat: tick-to-tick (2 beats)
  // when the tock is quiet, or 3 beats when an uneven tick-tock splits the
  // single-beat peak in two. Take the shortest sub-multiple that still shows.
  const top = best;
  for (let k = 4; k >= 2; k--) {
    const h = top / k, tol = Math.max(2, Math.round(h * 0.04));
    let cand = 0;
    for (let L = Math.round(h) - tol; L <= Math.round(h) + tol; L++) if (isPeak(L) && (!cand || r[L] > r[cand])) cand = L;
    if (cand && r[cand] >= 0.2 * r[top]) { best = cand; break; }
  }
  return { period: best * binSec, strength: r[top] };
}

// ------------------------------------------------------------ BeatAnalyser

const BEAT_PERIODS = [0.25, 0.4, 0.5, 0.6, 0.75, 1, 1.25, 1.5, 2];  // seconds per tick OR tock
const SHORT_WINDOW = 60;     // seconds
const LONG_WINDOW = 600;
const KEEP_SECONDS = 1800;

class BeatAnalyser {
  constructor(manualPeriod = null) {
    this.manualPeriod = manualPeriod;   // seconds per beat, or null for auto
    this.nominal = null;
    this.pending = [];                  // events before the nominal period is known
    this.events = [];                   // {t, amp, n}; n = beat number, parity = tick/tock
    this.history = [];                  // {t, rate} for the drift chart
    this.rejected = 0;
    this.rejectRun = 0;
  }

  add(t, amp) {
    const e = { t, amp, n: 0 };
    if (!this.nominal) {
      if (this.manualPeriod) this.nominal = this.manualPeriod;
      else {
        this.pending.push(e);
        if (this.pending.length < 6) return;
        this.nominal = snapPeriod(medianInterval(this.pending));
      }
      const first = this.pending.splice(0);
      for (const p of first) this.place(p);
      if (!first.length) this.place(e);
      return;
    }
    this.place(e);
  }

  // Number the beat relative to the previous one. Missed beats (gaps of
  // several periods) keep the tick/tock parity correct; off-grid events are
  // treated as noise and dropped.
  //
  // The fine check compares against the last beat of the same parity (tick
  // to tick, tock to tock). That spacing doesn't depend on beat error, so the
  // tolerance can be tight enough to catch a noise click near a real beat.
  // After 4 rejections in a row, accept anyway: the rhythm really changed
  // (e.g. the beat was just adjusted) or the anchor itself was noise.
  place(e) {
    const last = this.events[this.events.length - 1];
    if (last) {
      const P = this.measuredPeriod();
      const dt = e.t - last.t;
      const k = Math.round(dt / P);
      let ok = k >= 1 && Math.abs(dt - k * P) <= 0.3 * P;
      const n = last.n + Math.max(1, k);
      if (ok) {
        const same = this.events.slice(-4).reverse().find(ev => (n - ev.n) % 2 === 0);
        if (same && Math.abs(e.t - same.t - (n - same.n) * P) > 0.05 * P) ok = false;
      }
      if (!ok && ++this.rejectRun < 4) {
        this.rejected++;
        return;
      }
      e.n = n;
    }
    this.rejectRun = 0;
    this.events.push(e);
    while (this.events.length && e.t - this.events[0].t > KEEP_SECONDS) this.events.shift();

    const lastH = this.history[this.history.length - 1];
    if (!lastH || e.t - lastH.t >= 5) {
      const r = this.rate(SHORT_WINDOW);
      if (r && r.count >= 20) this.history.push({ t: e.t, rate: r.secPerDay });
    }
  }

  // Average beat period over the last 20 beats, or the nominal at first.
  measuredPeriod() {
    const ev = this.events.slice(-20);
    if (ev.length >= 5) {
      const a = ev[0], b = ev[ev.length - 1];
      if (b.n - a.n >= 4) return (b.t - a.t) / (b.n - a.n);
    }
    return this.nominal;
  }

  window(seconds) {
    const last = this.events[this.events.length - 1];
    if (!last) return [];
    return this.events.filter(ev => ev.t >= last.t - seconds);
  }

  rate(seconds) {
    return this.fit(this.window(seconds), 8);
  }

  rateBeats(beats) {
    return this.fit(this.events.slice(-beats), 6);
  }

  // Least-squares fit t = a + period * n + c * (±1 by tick/tock parity).
  // The parity term absorbs the beat error, so an uneven tick-tock doesn't
  // bias the period or inflate its uncertainty, even over a few beats.
  // Outliers (a noise click numbered as a beat) are dropped and the fit
  // repeated, so one bad event can't skew the rate.
  fit(evs, minCount) {
    if (evs.length < minCount || !this.nominal) return null;
    const n0 = evs[0].n, t0 = evs[0].t;
    let rows = evs.map(e => [[1, e.n - n0, e.n % 2 ? -1 : 1], e.t - t0]);
    let sol = lsq3(rows);
    if (!sol) return null;
    const absRes = sol.res.map(Math.abs);
    const limit = Math.max(5 * median(absRes), 0.0005);
    if (absRes.some(v => v > limit)) {
      const kept = rows.filter((_, i) => absRes[i] <= limit);
      if (kept.length < minCount) return null;
      rows = kept;
      sol = lsq3(rows);
      if (!sol) return null;
    }
    const m = rows.length;
    const period = sol.coef[1];
    const periodErr = m > 3 ? Math.sqrt(sol.ss / (m - 3) * sol.varUnit) : 0;
    return {
      period,
      beatsPerHour: 3600 / period,
      secPerDay: (this.nominal / period - 1) * 86400,
      secPerDayErr: this.nominal / (period * period) * periodErr * 86400,
      count: m,
    };
  }

  // Compare tick->tock (A) with tock->tick (B) intervals.
  beatError(beats = 20) {
    const evs = this.events.slice(-beats);
    const A = [], B = [];
    for (let i = 1; i < evs.length; i++) {
      if (evs[i].n - evs[i - 1].n !== 1) continue;
      (evs[i - 1].n % 2 === 0 ? A : B).push(evs[i].t - evs[i - 1].t);
    }
    if (A.length < 3 || B.length < 3) return null;
    const mA = mean(A), mB = mean(B);
    const sd = Math.sqrt((variance(A, mA) + variance(B, mB)) / 2);
    return { meanA: mA, meanB: mB, errorMs: (mA - mB) / 2 * 1000, jitterMs: sd * 1000, pairs: Math.min(A.length, B.length) };
  }

  // Consecutive intervals as deviation from nominal in ms, for the chart.
  intervals(beats = 20) {
    const evs = this.events.slice(-beats);
    const out = [];
    for (let i = 1; i < evs.length; i++) {
      if (evs[i].n - evs[i - 1].n !== 1) continue;
      out.push({ t: evs[i].t, dev: (evs[i].t - evs[i - 1].t - this.nominal) * 1000, isA: evs[i - 1].n % 2 === 0 });
    }
    return out;
  }
}

// Least squares for rows of [[x0, x1, x2], y]. Returns coefficients,
// residuals, residual sum of squares, and (S⁻¹)[1][1] for x1's variance.
function lsq3(rows) {
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], r = [0, 0, 0];
  for (const [x, y] of rows) {
    for (let i = 0; i < 3; i++) {
      r[i] += x[i] * y;
      for (let j = 0; j < 3; j++) S[i][j] += x[i] * x[j];
    }
  }
  const coef = solve3(S, r);
  const unit = solve3(S, [0, 1, 0]);
  if (!coef || !unit) return null;
  const res = rows.map(([x, y]) => y - coef[0] - coef[1] * x[1] - coef[2] * x[2]);
  return { coef, res, ss: res.reduce((s, v) => s + v * v, 0), varUnit: unit[1] };
}

function median(a) {
  const s = a.slice().sort((x, y) => x - y);
  return s[s.length >> 1];
}

// Solve a 3x3 linear system by Gaussian elimination with partial pivoting.
function solve3(A, b) {
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < 3; col++) {
    let p = col;
    for (let i = col + 1; i < 3; i++) if (Math.abs(M[i][col]) > Math.abs(M[p][col])) p = i;
    if (Math.abs(M[p][col]) < 1e-12) return null;
    [M[col], M[p]] = [M[p], M[col]];
    for (let i = 0; i < 3; i++) {
      if (i === col) continue;
      const f = M[i][col] / M[col][col];
      for (let j = col; j < 4; j++) M[i][j] -= f * M[col][j];
    }
  }
  return M.map((row, i) => row[3] / row[i]);
}

function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }
function variance(a, m) { return a.reduce((s, v) => s + (v - m) ** 2, 0) / Math.max(1, a.length - 1); }

function medianInterval(evs) {
  const d = [];
  for (let i = 1; i < evs.length; i++) d.push(evs[i].t - evs[i - 1].t);
  d.sort((a, b) => a - b);
  return d[Math.floor(d.length / 2)];
}

function snapPeriod(p) {
  let best = BEAT_PERIODS[0];
  for (const c of BEAT_PERIODS) if (Math.abs(Math.log(c / p)) < Math.abs(Math.log(best / p))) best = c;
  return best;
}

// ------------------------------------------------------- Synthetic source
// Fake clock for ?demo and for testing: a double click per beat over noise.
// Intervals alternate A, B, A, B...

// Options: tockGain (loudness of every other beat), dropEvery (silence every
// Nth beat), spikes (random noise clicks per second), spikeGain.
class SynthSource {
  constructor(sr, A, B, noise = 0.004, freq = 3000, hum = 0, opts = {}) {
    this.sr = sr; this.A = A; this.B = B; this.noise = noise; this.freq = freq; this.hum = hum;
    this.tockGain = opts.tockGain ?? 1;
    this.dropEvery = opts.dropEvery || 0;
    this.spikes = opts.spikes || 0;
    this.spikeGain = opts.spikeGain ?? 0.5;
    this.pos = 0; this.k = 0; this.nextT = 0.3; this.active = [];
  }
  read(count) {
    const out = new Float32Array(count);
    const w = 2 * Math.PI * this.freq;
    for (let i = 0; i < count; i++) {
      const t = (this.pos + i) / this.sr;
      while (t >= this.nextT) {
        const k = this.k++;
        const dropped = this.dropEvery && k % this.dropEvery === this.dropEvery - 1;
        if (!dropped) this.active.push({ t: this.nextT, g: k % 2 ? this.tockGain : 1, double: true });
        this.nextT += k % 2 === 0 ? this.A : this.B;
      }
      if (this.spikes && Math.random() < this.spikes / this.sr) this.active.push({ t, g: this.spikeGain, double: false });
      let v = (Math.random() * 2 - 1) * this.noise + this.hum * Math.sin(2 * Math.PI * 50 * t);
      for (let j = this.active.length - 1; j >= 0; j--) {
        const { t: t0, g, double } = this.active[j];
        const dt = t - t0;
        if (dt > 0.1) { this.active.splice(j, 1); continue; }
        v += g * 0.3 * Math.exp(-dt / 0.004) * Math.sin(w * dt);
        if (double && dt > 0.025) v += g * 0.15 * Math.exp(-(dt - 0.025) / 0.004) * Math.sin(w * dt);
      }
      out[i] = v;
    }
    this.pos += count;
    return out;
  }
}

// -------------------------------------------------------------- Auto-tuning
// Runs the detector at several low-cut filters and thresholds at once, and
// picks the combination whose events look most like a steady clock.

class Biquad {   // RBJ high-pass
  constructor(sr, f, Q = 0.707) {
    const w = 2 * Math.PI * f / sr, c = Math.cos(w), al = Math.sin(w) / (2 * Q), a0 = 1 + al;
    this.b0 = (1 + c) / 2 / a0; this.b1 = -(1 + c) / a0; this.b2 = this.b0;
    this.a1 = -2 * c / a0; this.a2 = (1 - al) / a0;
    this.x1 = this.x2 = this.y1 = this.y2 = 0;
  }
  run(inp, out) {
    for (let i = 0; i < inp.length; i++) {
      const x = inp[i];
      const y = this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
      this.x2 = this.x1; this.x1 = x; this.y2 = this.y1; this.y1 = y;
      out[i] = y;
    }
  }
}

// How clock-like are these event times? 1 = every beat found, evenly spaced,
// nothing extra. Penalises both noise events and missed beats. `period` is
// the beat period found independently (autocorrelation or user setting), so
// a detector that hears only the loud tick scores about 50%, not 100%.
const SCORE_WINDOW = 20;   // seconds

function regularity(times, now, period) {
  const span = Math.min(SCORE_WINDOW, now - 0.5);
  const ts = times.filter(t => t >= now - SCORE_WINDOW);
  if (ts.length < 6) return { score: 0, period: 0, median: 0 };
  const d = [];
  for (let i = 1; i < ts.length; i++) d.push(ts[i] - ts[i - 1]);
  const sorted = d.slice().sort((a, b) => a - b);
  const median = sorted[sorted.length >> 1];
  period = period || median;
  if (period < 0.15 || period > 3) return { score: 0, period, median };
  let good = 0;
  for (const x of d) if (Math.abs(x - period) <= 0.2 * period) good++;
  return { score: Math.min(1, good / Math.max(d.length, span / period - 1)), period, median };
}

// Combine per-filter period estimates. Ignore filters much less periodic
// than the best one (mostly hum or noise). Some filters may only see the
// tick (2x period), so take the shortest that divides the longest.
function combinePeriods(ests) {
  const strongest = Math.max(0, ...ests.map(e => e.strength));
  const valid = ests.filter(e => e.period > 0 && e.strength >= 0.5 * strongest).map(e => e.period).sort((a, b) => a - b);
  if (!valid.length) return 0;
  const longest = valid[valid.length - 1];
  for (const p of valid) {
    const k = longest / p;
    if (Math.abs(k - Math.round(k)) < 0.12 && Math.round(k) <= 4) return p;
  }
  return longest;
}

const samePeriod = (a, b) => a / b > 0.85 && a / b < 1.18;
const isMultiple = (a, b) => { const k = a / b, n = Math.round(k); return n >= 2 && n <= 4 && Math.abs(k - n) < 0.12; };

class Bank {
  constructor(sampleRate, onActiveEvent, onSwitch) {
    this.sr = sampleRate;
    this.onActiveEvent = onActiveEvent;
    this.onSwitch = onSwitch;          // (detector, periodChanged)
    this.cuts = [150, 400, 1000, 2500];
    this.mults = [1.8, 2.5, 3.5, 5, 8];
    this.auto = true;
    this.fixedPeriod = null;
    this.firstEval = 12;               // seconds of audio before the first decision
    this.nextEval = this.firstEval;
    this.period = 0;                   // beat period in use (s), 0 = unknown
    this.nextPeriodAt = 4;
    this.info = { score: 0, searching: true };
    this.branches = this.cuts.map(cut => {
      const br = { cut, filter: new Biquad(sampleRate, cut), buf: new Float32Array(0), dets: [] };
      br.dets = this.mults.map(m => {
        const d = new Detector(sampleRate, (t, amp) => this.handle(d, t, amp));
        d.baseMult = d.mult = m;
        d.recent = [];
        d.branch = br;
        return d;
      });
      return br;
    });
    this.active = this.branches[2].dets[2];   // 1 kHz, x3.5 until auto decides
  }

  get now() { return this.active.n / this.sr; }

  handle(d, t, amp) {
    d.recent.push(t);
    while (d.recent.length && d.recent[0] < t - 40) d.recent.shift();
    if (d === this.active) this.onActiveEvent(t, amp);
  }

  push(samples, start) {
    for (const br of this.branches) {
      if (br.buf.length !== samples.length) br.buf = new Float32Array(samples.length);
      br.filter.run(samples, br.buf);
      for (const d of br.dets) d.push(br.buf, start);
    }
    if (this.now >= this.nextPeriodAt) {
      this.nextPeriodAt = this.now + 1;
      this.updatePeriod();
    }
    if (this.auto && this.now >= this.nextEval) {
      this.nextEval = this.now + 3;
      this.evaluate();
    }
  }

  // All detectors in a branch see the same envelope, so one estimate each.
  updatePeriod() {
    for (const br of this.branches) {
      const d = br.dets[0];
      br.est = estimatePeriod(d.bins, d.binLen / this.sr, d.floor);
    }
    const est = this.fixedPeriod || combinePeriods(this.branches.map(br => br.est));
    const old = this.period;
    if (old && est && samePeriod(old, est)) {
      this.period = est;
    } else if (old && est && isMultiple(est, old)) {
      // 2-4x the current beat is the same rhythm seen coarser (e.g. tick-to-
      // tick when the tock is momentarily quiet): keep the shorter beat.
      this.candidate = 0;
      return;
    } else if (!this.fixedPeriod && (this.candidate && est && samePeriod(this.candidate, est) ? ++this.confirmations : (this.confirmations = 1)) < 3) {
      // A different period must show up 3 times in a row before it's used.
      this.candidate = est;
      return;
    } else {
      this.period = est;
    }
    this.candidate = 0;
    for (const br of this.branches) for (const d of br.dets) d.setPeriod(this.period);
    // New or changed beat period: beats numbered so far may be wrong.
    if (this.period && !(old && samePeriod(old, this.period))) this.onSwitch(this.active, true);
  }

  evaluate() {
    const now = this.now;
    const all = [];
    for (const br of this.branches) {
      for (const d of br.dets) all.push({ d, ...regularity(d.recent, now, this.period) });
    }
    const cur = all.find(r => r.d === this.active);
    const best = Math.max(...all.map(r => r.score));
    this.info = { score: cur.score, searching: best < 0.5 };
    if (best < 0.5) return;

    // Among near-best candidates prefer the one hearing the most beats per
    // second (so tick *and* tock), then the highest threshold (most robust).
    const near = all.filter(r => r.score >= best - 0.08);
    near.sort((a, b) => samePeriod(a.period, b.period) ? b.d.baseMult - a.d.baseMult : a.period - b.period);
    const pick = near[0];
    if (pick.d === this.active) return;
    if (pick.score > cur.score + 0.1 || pick.period < cur.period * 0.75 || cur.score < 0.5) {
      this.active = pick.d;
      this.info = { score: pick.score, searching: false };
      // Reset the analyser if the new detector hears beats at a different
      // spacing (e.g. tick and tock instead of tick only).
      this.onSwitch(pick.d, !(cur.median && samePeriod(pick.median, cur.median)));
    }
  }

  setAuto() {
    this.auto = true;
    for (const br of this.branches) for (const d of br.dets) d.mult = d.baseMult;
    this.nextEval = this.now;
  }

  setManual(cutIndex, mult) {
    this.auto = false;
    const d = this.branches[cutIndex].dets[0];
    d.mult = mult;
    if (d !== this.active) {
      this.active = d;
      this.onSwitch(d, false);
    }
  }
}

if (typeof module !== 'undefined') module.exports = { Detector, BeatAnalyser, SynthSource, Bank, estimatePeriod };

// ---------------------------------------------------------------------- UI

const WORKLET_SRC = `
class TickCapture extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(2048); this.fill = 0; this.pos = 0; }
  process(inputs) {
    const ch = inputs[0][0];
    if (ch) for (let i = 0; i < ch.length; i++) {
      this.buf[this.fill++] = ch[i];
      if (this.fill === this.buf.length) {
        this.port.postMessage({ start: this.pos, samples: this.buf }, [this.buf.buffer]);
        this.pos += this.fill;
        this.buf = new Float32Array(2048);
        this.fill = 0;
      }
    }
    return true;
  }
}
registerProcessor('tick-capture', TickCapture);`;

function initUI() {
  const $ = id => document.getElementById(id);
  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const params = new URLSearchParams(location.search);

  let ctx = null, stream = null, bank = null, detector = null, analyser = null, synth = null, demoTimer = null;
  let lastEventAudioTime = -Infinity;
  let scopeMax = 0.01;

  const sensitivity = $('sensitivity');
  const periodSel = $('period');
  const beatWindow = () => parseInt($('beatWindow').value, 10);

  const isAuto = () => $('mode').value === 'auto';
  const currentPeriod = () => periodSel.value === 'auto' ? null : parseFloat(periodSel.value);
  const multFromSlider = () => 10 * Math.pow(1.15 / 10, parseFloat(sensitivity.value) / 100);  // 0 -> x10, 100 -> x1.15
  const sliderFromMult = m => Math.max(0, Math.min(100, Math.round(100 * Math.log(m / 10) / Math.log(1.15 / 10))));

  function newAnalyser() {
    analyser = new BeatAnalyser(currentPeriod());
  }

  function newSession(sampleRate) {
    newAnalyser();
    bank = new Bank(sampleRate, (t, amp) => {
      analyser.add(t, amp);
      lastEventAudioTime = t;
    }, (det, periodChanged) => {
      detector = det;
      if (periodChanged) newAnalyser();
    });
    bank.fixedPeriod = currentPeriod();
    detector = bank.active;
    applyMode();
  }

  // Auto: the bank picks low cut + threshold. Manual: sliders drive them.
  function applyMode() {
    if (!bank) return;
    if (isAuto()) bank.setAuto();
    else bank.setManual(bank.cuts.indexOf(parseFloat($('lowcut').value)), multFromSlider());
    detector = bank.active;
  }

  function showAutoState() {
    const auto = isAuto();
    sensitivity.disabled = $('lowcut').disabled = auto;
    const info = $('autoInfo');
    if (!bank || !auto) { info.textContent = ''; return; }
    const d = bank.active;
    $('lowcut').value = String(d.branch.cut);
    sensitivity.value = String(sliderFromMult(d.mult));
    const wait = Math.ceil(bank.firstEval - bank.now);
    if (wait > 0) info.textContent = 'Auto: listening for ' + wait + ' s to choose settings…';
    else if (bank.info.searching) info.textContent = 'Auto: no regular ticking found yet. Move the mic closer to the clock and keep the room quiet.';
    else info.textContent = 'Auto: beat every ≈' + Math.round(bank.period * 1000) + ' ms, using ' + d.branch.cut + ' Hz low cut, threshold ×' + d.mult +
      (d.nextBeat !== null ? ', locked on' : '') + ' (' + Math.round(bank.info.score * 100) + '% of beats steady).';
  }

  async function start() {
    $('start').disabled = true;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      ctx = new AudioContext({ latencyHint: 'interactive' });
      const url = URL.createObjectURL(new Blob([WORKLET_SRC], { type: 'text/javascript' }));
      await ctx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);

      newSession(ctx.sampleRate);
      const src = ctx.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(ctx, 'tick-capture');
      const mute = ctx.createGain(); mute.gain.value = 0;
      src.connect(node); node.connect(mute); mute.connect(ctx.destination);
      node.port.onmessage = e => bank.push(e.data.samples, e.data.start);

      $('start').textContent = 'Stop';
      $('start').disabled = false;
      $('start').onclick = stop;
    } catch (err) {
      $('start').disabled = false;
      setStatus('Could not open the microphone: ' + err.message, 'bad');
      stop(true);
    }
  }

  function stop(keepStatus) {
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (ctx) ctx.close();
    stream = ctx = null;
    $('start').textContent = 'Start listening';
    $('start').onclick = start;
    if (keepStatus !== true) setStatus('Stopped.', '');
  }

  function startDemo() {
    // 120 s of synthetic audio instantly, then live. A=490.05 ms, B=510.05 ms:
    // expect beat error 10 ms, period 500.05 ms, about -8.6 s/day.
    const sr = 48000;
    newSession(sr);
    synth = new SynthSource(sr, 0.49005, 0.51005);
    for (let i = 0; i < 120; i++) bank.push(synth.read(sr));
    demoTimer = setInterval(() => bank.push(synth.read(sr / 10)), 100);
    $('start').disabled = true;
    $('start').textContent = 'Demo running';
  }

  function setStatus(text, cls) {
    const el = $('status');
    el.textContent = text;
    el.className = 'status ' + (cls || '');
  }

  // ---- readouts

  const fmt = (x, d = 1) => (x === null || x === undefined || !isFinite(x)) ? '–' : x.toFixed(d);
  const sign = x => (x > 0 ? '+' : '') + fmt(x, 1);

  function updateReadouts() {
    if (!analyser) return;
    showAutoState();
    const now = detector.n / detector.sr;
    const heard = now - lastEventAudioTime < 3;
    const r1 = analyser.rate(SHORT_WINDOW);
    const r10 = analyser.rate(LONG_WINDOW);
    const be = analyser.beatError(beatWindow());

    if (now < 1) setStatus('Calibrating noise floor…', '');
    else if (!analyser.events.length && !analyser.pending.length) setStatus('Listening – no ticks heard yet. Move the mic closer or raise sensitivity.', 'warn');
    else if (!heard) setStatus('Lost the tick – nothing heard for 3 s.', 'warn');
    else if (!r1) setStatus('Hearing ticks, collecting data…', '');
    else setStatus('Tracking · ' + analyser.events.length + ' beats', 'good');

    const main = analyser.rateBeats(beatWindow());
    $('rate').textContent = main ? sign(main.secPerDay) : '–';
    $('rateBeatsLabel').textContent = 'last ' + beatWindow() + ' beats' + (main ? ', ± ' + fmt(main.secPerDayErr, 1) : '');
    $('rateNote').textContent = main ? (Math.abs(main.secPerDay) <= Math.max(0.5, main.secPerDayErr) ? 'on time' : main.secPerDay > 0 ? 'running fast' : 'running slow') : 'seconds per day';
    $('rate1').textContent = r1 ? sign(r1.secPerDay) + ' s/day' : '–';
    $('rate10').textContent = r10 ? sign(r10.secPerDay) + ' s/day' : '–';

    $('beatErr').textContent = be ? fmt(Math.abs(be.errorMs), 1) : '–';
    let note = 'milliseconds';
    if (be) {
      const a = Math.abs(be.errorMs);
      note = a < 2 ? 'even' : a < 5 ? 'slightly uneven' : 'uneven';
      $('beatErr').className = 'big ' + (a < 2 ? 'good' : a < 5 ? 'warn' : 'bad');
    }
    $('beatErrNote').textContent = note;
    $('meanA').textContent = be ? fmt(be.meanA * 1000, 1) + ' ms' : '–';
    $('meanB').textContent = be ? fmt(be.meanB * 1000, 1) + ' ms' : '–';
    $('jitter').textContent = be ? fmt(be.jitterMs, 2) + ' ms' : '–';

    const period = (r10 || r1);
    $('period1').textContent = period ? fmt(period.period * 1000, 2) + ' ms' : '–';
    $('bph').textContent = period ? fmt(period.beatsPerHour, 0) : '–';
    $('nominal').textContent = analyser.nominal ? fmt(analyser.nominal * 1000, 0) + ' ms' : '–';
    $('rejected').textContent = analyser.rejected;

    // Beats heard vs. expected over the last 20 s, from the measured period.
    const beatP = analyser.nominal ? analyser.measuredPeriod() : bank.period;
    const span = Math.min(20, now - 1);
    if (beatP && span > 2) {
      const heardCount = analyser.events.filter(e => e.t > now - span).length;
      const expected = Math.max(1, Math.round(span / beatP));
      $('heard').textContent = heardCount + ' of ' + expected + ' (' + Math.min(100, Math.round(100 * heardCount / expected)) + '%)';
    } else {
      $('heard').textContent = '–';
    }
  }

  // ---- drawing

  function prep(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const c = canvas.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);
    c.font = '11px system-ui, sans-serif';
    return { c, w, h };
  }

  function drawScope() {
    const { c, w, h } = prep($('scope'));
    if (!detector) return;
    const bins = detector.bins;
    const binSec = detector.binLen / detector.sr;
    const span = detector.maxBins * binSec;
    const now = detector.n / detector.sr;

    let peak = detector.threshold * 2;
    for (const v of bins) if (v > peak) peak = v;
    scopeMax += (peak * 1.1 - scopeMax) * 0.1;
    const y = v => h - 14 - Math.min(1, v / scopeMax) * (h - 24);
    const x = t => (t - (now - span)) / span * w;

    c.fillStyle = css('--accent-soft');
    c.beginPath(); c.moveTo(w, h - 14);
    for (let i = bins.length - 1; i >= 0; i--) c.lineTo(x(now - (bins.length - 1 - i) * binSec), y(bins[i]));
    c.lineTo(x(now - (bins.length - 1) * binSec), h - 14);
    c.closePath(); c.fill();

    // Locked on: the lower in-window threshold is the one that matters.
    const locked = detector.nextBeat !== null;
    const thr = locked ? detector.gateThreshold : detector.threshold;
    c.strokeStyle = css('--muted'); c.setLineDash([4, 4]);
    c.beginPath(); c.moveTo(0, y(thr)); c.lineTo(w, y(thr)); c.stroke();
    c.setLineDash([]);
    c.fillStyle = css('--muted');
    c.fillText(locked ? 'trigger threshold (locked on to the beat)' : 'trigger threshold', 6, y(thr) - 4);

    if (analyser) {
      // Grey: picked up by the detector but not used (rejected as noise).
      const used = new Set(analyser.events.map(e => e.t));
      c.strokeStyle = css('--muted');
      for (const t of detector.raw) {
        if (t < now - span || used.has(t)) continue;
        c.beginPath(); c.moveTo(x(t), 18); c.lineTo(x(t), h - 14); c.stroke();
      }
      for (const e of analyser.events) {
        if (e.t < now - span) continue;
        c.strokeStyle = e.n % 2 === 0 ? css('--tick') : css('--tock');
        c.lineWidth = 2;
        c.beginPath(); c.moveTo(x(e.t), 6); c.lineTo(x(e.t), h - 14); c.stroke();
      }
      c.lineWidth = 1;
    }
    c.fillStyle = css('--muted');
    c.fillText('last ' + span.toFixed(0) + ' s', 6, h - 2);
  }

  function drawIntervals() {
    const { c, w, h } = prep($('intervals'));
    if (!analyser || !analyser.nominal) return;
    const pts = analyser.intervals(beatWindow());
    let range = 5;
    for (const p of pts) range = Math.max(range, Math.abs(p.dev) * 1.15);
    const y = v => h / 2 - v / range * (h / 2 - 14);
    const x = i => 10 + i / Math.max(1, pts.length - 1) * (w - 20);

    c.strokeStyle = css('--grid');
    c.beginPath(); c.moveTo(0, y(0)); c.lineTo(w, y(0)); c.stroke();
    c.fillStyle = css('--muted');
    c.fillText('+' + range.toFixed(0) + ' ms', 4, 12);
    c.fillText('−' + range.toFixed(0) + ' ms', 4, h - 4);
    c.fillText('nominal ' + (analyser.nominal * 1000).toFixed(0) + ' ms', w - 100, y(0) - 4);

    pts.forEach((p, i) => {
      c.fillStyle = p.isA ? css('--tick') : css('--tock');
      c.beginPath(); c.arc(x(i), y(p.dev), 3, 0, 7); c.fill();
    });
    c.fillStyle = css('--tick'); c.fillText('● tick→tock', w - 170, 12);
    c.fillStyle = css('--tock'); c.fillText('● tock→tick', w - 85, 12);
  }

  function drawDrift() {
    const { c, w, h } = prep($('drift'));
    if (!analyser || analyser.history.length < 2) {
      c.fillStyle = css('--muted');
      c.fillText('Rate history appears after about a minute of ticks.', 10, h / 2);
      return;
    }
    const hist = analyser.history;
    const t0 = hist[0].t, t1 = hist[hist.length - 1].t;
    let lo = Infinity, hi = -Infinity;
    for (const p of hist) { lo = Math.min(lo, p.rate); hi = Math.max(hi, p.rate); }
    lo = Math.min(lo, 0); hi = Math.max(hi, 0);
    const pad = Math.max(1, (hi - lo) * 0.1);
    lo -= pad; hi += pad;
    const x = t => 40 + (t - t0) / Math.max(1, t1 - t0) * (w - 50);
    const y = v => h - 16 - (v - lo) / (hi - lo) * (h - 28);

    c.strokeStyle = css('--grid');
    c.beginPath(); c.moveTo(40, y(0)); c.lineTo(w, y(0)); c.stroke();
    c.fillStyle = css('--muted');
    c.fillText('0', 28, y(0) + 4);
    c.fillText(hi.toFixed(0), 4, 12);
    c.fillText(lo.toFixed(0), 4, h - 18);
    c.fillText('s/day, 60 s window · ' + Math.round((t1 - t0) / 60) + ' min shown', 40, h - 2);

    c.strokeStyle = css('--accent'); c.lineWidth = 2;
    c.beginPath();
    hist.forEach((p, i) => (i ? c.lineTo(x(p.t), y(p.rate)) : c.moveTo(x(p.t), y(p.rate))));
    c.stroke(); c.lineWidth = 1;
  }

  function frame() {
    drawScope(); drawIntervals(); drawDrift();
    requestAnimationFrame(frame);
  }

  // ---- wiring

  $('start').onclick = start;
  $('reset').onclick = () => {
    if (!bank) return;
    newAnalyser();
    bank.fixedPeriod = currentPeriod();
  };
  sensitivity.oninput = applyMode;
  $('lowcut').onchange = applyMode;
  $('mode').onchange = applyMode;
  periodSel.onchange = $('reset').onclick;

  if (!navigator.mediaDevices) setStatus('Microphone access needs a secure page – use https://www.oliverlorton.co.uk/tick-tock/ or open index.html in Chrome/Firefox.', 'bad');
  if (params.has('demo')) startDemo();

  setInterval(updateReadouts, 250);
  requestAnimationFrame(frame);
}

if (typeof document !== 'undefined') initUI();
