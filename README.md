# Tick-Tock

Browser tool that listens to a pendulum clock and reports beat error (how even the tick-tock is) and rate (seconds per day fast/slow). Vanilla JS, no build step.

**Live:** <https://www.oliverlorton.co.uk/tick-tock/> ([demo](https://www.oliverlorton.co.uk/tick-tock/?demo))

Click **Start listening** and allow microphone access.

- `?demo` shows synthetic data (expected: beat error 10 ms, about −8.6 s/day).
- Adjust **Sensitivity** if ticks aren't detected, or set **Beat every** if auto-detect picks the wrong period.

## Local development

Open `index.html` directly in Chrome or Firefox (both allow microphone access from `file://`). For the demo, add `?demo` to the end of the `file://…/index.html` URL. Safari may block the microphone on local files, so use the live page there.
