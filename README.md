# Tick-Tock

Browser tool that listens to a pendulum clock and reports beat error (how even the tick-tock is) and rate (seconds per day fast/slow). Vanilla JS, no build step.

```
python3 -m http.server 8000
```

Open <http://localhost:8000> and click **Start listening**. (Microphone access needs `localhost` or https, so opening `index.html` directly may not work.)

- `?demo` shows synthetic data (expected: beat error 10 ms, about −8.6 s/day).
- Adjust **Sensitivity** if ticks aren't detected, or set **Beat every** if auto-detect picks the wrong period.
