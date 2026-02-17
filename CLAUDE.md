# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MIDI-to-PPM Teddy Ruxpin Controller — a real-time Python tool that listens for MIDI note input and generates PPM (Pulse Position Modulation) audio signals on the right stereo channel to control an original Teddy Ruxpin toy's servos.

## Running

```bash
source venv/bin/activate
python ppm_tool.py
```

Dependencies are in `requirements.txt` (mido, python-rtmidi, numpy, sounddevice). Install with `pip install -r requirements.txt`.

## Architecture

The entire application lives in `ppm_tool.py` (~300 lines) with four key components:

- **ServoState** — Thread-safe servo position tracker for three channels (mouth, eyes, nose). Uses a threading lock since MIDI input and audio output run on separate threads.
- **PPMGenerator** — Builds 22.5ms PPM audio frames in 8-channel RC format matching the original Teddy Ruxpin cassette encoding. PPM data is placed at the end of each frame with zero-padding at the start.
- **Audio callback** — Stereo output: silence on left channel, PPM signal on right channel. One PPM frame per audio block.
- **MIDI listener** — Daemon thread mapping MIDI notes to servo positions: C3→mouth (CH4), D3→eyes (CH2), E3→nose (CH3).

## Key Domain Details

- PPM frame structure: start pulse (400µs) → 8× [marker (700µs) + servo pulse (variable) + gap (400µs)] → zero padding
- Signal amplitude is 0.618 (~20262/32767)
- "Deadening filter" zeros out values ≤1.5 to prevent servo endpoint buzz
- Servo rest positions offset by 3 units to avoid hard stops
- Nose is toggled (rest=3 ↔ open=80) rather than velocity-scaled
