#!/usr/bin/env python3
"""
MIDI-to-PPM Teddy Ruxpin Controller

Listens for MIDI note input and generates a real-time PPM audio signal
on the right audio channel to control an original Teddy Ruxpin.

Uses the same PPM format as the original cassette tapes — 8-channel RC-style
PPM where the gap between marker pulses encodes servo position.

Teddy Ruxpin channel assignments:
  CH1: unused
  CH2: Eyes
  CH3: Upper jaw
  CH4: Lower jaw
  CH5–CH8: unused (Grubby / audio routing)
"""

import os
import sys
import threading
import time
import wave

import mido
import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# Configuration — adjust these to match your setup
# ---------------------------------------------------------------------------

# Higher sample rates give finer PPM timing resolution. T-Rux recommends 192000.
# At 44100 there are only ~33 servo positions per channel; 96000 gives ~72.
SAMPLE_RATE = 44100

# MIDI note assignments (default: C3, D3, E3)
NOTE_EYES = 60   # C3 — Note On = close eyes, Note Off = open
NOTE_NOSE = 62  # D3 — Note On = open jaw, Note Off = close
NOTE_MOUTH = 64   # E3 — Note On = wiggle upper jaw, Note Off = stop

# Servo rest positions (0–100 scale matching the original TR format).
# 0 = 700µs (servo min), 100 = 1500µs (servo max).
# Small offset from 0 prevents servo endpoint buzz at rest.
MOUTH_REST = 3       # lower jaw closed (slightly off hard stop)
EYES_REST = 3        # eyes open (slightly off hard stop)
NOSE_REST = 3        # upper jaw at rest (slightly off hard stop)

# Nose open position (0–100) when MIDI note is held
NOSE_OPEN = 80

# ---------------------------------------------------------------------------
# Teddy Ruxpin PPM timing (matches original cassette format / T-Rux project)
# ---------------------------------------------------------------------------

PPM_FRAME_WIDTH = 0.0225        # 22.5ms per frame
PPM_AMPLITUDE = 0.618           # ~20262/32767 — matches original signal level
PPM_NUM_CHANNELS = 8            # 8-channel RC-style PPM
SERVO_SCALE = 0.90              # servo range multiplier (0.75 = T-Rux default, higher = more travel)

# Derived timing unit — all PPM element durations are multiples of this
MMDIV = SAMPLE_RATE / 10000.0

# Per-channel PPM element durations (in samples)
MARKER_SAMPLES = int(MMDIV * 7)     # negative marker pulse (700µs)
GAP_SAMPLES = int(MMDIV * 4)        # positive gap between channels (400µs)
START_SAMPLES = int(MMDIV * 4)      # positive start-of-frame pulse (400µs)

FRAME_SAMPLES = int(PPM_FRAME_WIDTH * SAMPLE_RATE)
AUDIO_BLOCKSIZE = FRAME_SAMPLES     # one PPM frame per audio block


class ServoState:
    """Tracks the target position (0–100) of each servo channel."""

    def __init__(self):
        self.lock = threading.Lock()
        self.mouth = MOUTH_REST        # lower jaw (CH4)
        self.eyes = EYES_REST          # eyes (CH2)
        self.nose_active = False       # upper jaw (CH3)
        self.nose = NOSE_REST

    def note_on(self, note: int, velocity: int):
        amount = velocity / 127.0 * 100.0  # scale to 0–100
        with self.lock:
            if note == NOTE_MOUTH:
                self.mouth = amount
            elif note == NOTE_EYES:
                self.eyes = amount
            elif note == NOTE_NOSE:
                self.nose_active = True

    def note_off(self, note: int):
        with self.lock:
            if note == NOTE_MOUTH:
                self.mouth = MOUTH_REST
            elif note == NOTE_EYES:
                self.eyes = EYES_REST
            elif note == NOTE_NOSE:
                self.nose_active = False

    def get_channels(self, dt: float) -> list[float]:
        """Return 8-channel list (0–100 scale) matching TR PPM channel order."""
        with self.lock:
            mouth = self.mouth
            eyes = self.eyes
            nose = float(NOSE_OPEN) if self.nose_active else float(NOSE_REST)

        # 8 channels: [unused, eyes, upper_jaw, lower_jaw, unused, unused, unused, unused]
        return [0.0, eyes, nose, mouth, 0.0, 0.0, 0.0, 0.0]


class PPMGenerator:
    """Generates Teddy Ruxpin PPM audio matching the original cassette format.

    Frame structure (per the T-Rux reverse-engineering):
      [+start pulse]
      For each of 8 channels:
        [-marker pulse][+servo gap (variable)][+gap pulse]
      [zero padding to fill frame]
    """

    def __init__(self):
        self.phase = 0
        self.frame = self._build_frame([0.0] * 8)

    @staticmethod
    def _build_frame(channels: list[float]) -> np.ndarray:
        """Build one PPM frame matching the T-Rux inverse format.

        The PPM data is built into a list, then placed at the END of the
        frame with zero-padding at the START (matching T-Rux lines 294-299).

        Per-channel structure (T-Rux inverse, lines 281-288):
          [-marker][-servo][+gap]
        Marker and servo are BOTH negative, merging into one continuous
        negative pulse whose width encodes the channel position.
        """
        amp = PPM_AMPLITUDE
        clist: list[float] = []

        # Start-of-frame pulse (positive)
        clist += [amp] * START_SAMPLES

        for ch_value in channels:
            # Deadening filter (matches T-Rux line 309): near-zero → zero
            ch_value = float(np.clip(ch_value, 0.0, 100.0))
            if ch_value <= 1.5:
                ch_value = 0.0

            # Negative marker pulse
            clist += [-amp] * MARKER_SAMPLES

            # Negative servo pulse (extends the marker — T-Rux line 287)
            servo = ch_value * SERVO_SCALE / 100.0
            servo_samples = int(MMDIV * 10 * servo)
            clist += [-amp] * servo_samples

            # Positive gap (fixed width — channel separator)
            clist += [amp] * GAP_SAMPLES

        # Build frame: zero padding FIRST, PPM data at END (T-Rux lines 294-299)
        frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        ppm_len = len(clist)
        pad_len = FRAME_SAMPLES - ppm_len
        if pad_len >= 0:
            frame[pad_len:] = np.array(clist, dtype=np.float32)
        else:
            # PPM data exceeds frame — truncate (shouldn't happen at normal values)
            frame[:] = np.array(clist[:FRAME_SAMPLES], dtype=np.float32)

        return frame

    def generate(self, num_samples: int, servo_state: "ServoState", frame_dt: float) -> np.ndarray:
        """Generate exactly num_samples of continuous PPM audio.

        Fetches fresh servo positions at each PPM frame boundary so that
        continuously changing values (like the nose wiggle) update smoothly
        at the frame rate rather than once per audio callback.
        """
        out = np.empty(num_samples, dtype=np.float32)
        written = 0

        while written < num_samples:
            remaining_in_frame = FRAME_SAMPLES - self.phase
            needed = num_samples - written
            chunk = min(remaining_in_frame, needed)

            out[written : written + chunk] = self.frame[self.phase : self.phase + chunk]
            written += chunk
            self.phase += chunk

            if self.phase >= FRAME_SAMPLES:
                self.phase = 0
                channels = servo_state.get_channels(frame_dt)
                self.frame = self._build_frame(channels)

        return out


def make_audio_callback(servo_state: ServoState, recorded_frames=None):
    """Create a sounddevice output stream callback."""
    ppm_gen = PPMGenerator()
    frame_dt = PPM_FRAME_WIDTH

    def callback(outdata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        ppm = ppm_gen.generate(frames, servo_state, frame_dt)

        # Stereo: left = silence, right = PPM signal
        outdata[:, 0] = 0.0
        outdata[:, 1] = ppm

        if recorded_frames is not None:
            recorded_frames.append(outdata.copy())

    return callback


def list_midi_ports():
    ports = mido.get_input_names()
    if not ports:
        print("No MIDI input ports found.")
        print("Connect a MIDI device or create a virtual MIDI port and try again.")
        return None
    print("Available MIDI input ports:")
    for i, name in enumerate(ports):
        print(f"  [{i}] {name}")
    return ports


def midi_listener(port_name: str, servo_state: ServoState):
    """Run the MIDI listener loop in a background thread."""
    print(f"Listening on MIDI port: {port_name}")
    print(f"  Mouth (lower jaw): note {NOTE_MOUTH} (C3) → CH4")
    print(f"  Eyes:               note {NOTE_EYES} (D3) → CH2")
    print(f"  Nose (upper jaw):   note {NOTE_NOSE} (E3) → CH3")
    print()

    with mido.open_input(port_name) as port:
        for msg in port:
            if msg.type == "note_on" and msg.velocity > 0:
                servo_state.note_on(msg.note, msg.velocity)
                label = {NOTE_MOUTH: "Mouth", NOTE_EYES: "Eyes", NOTE_NOSE: "Nose"}.get(msg.note)
                if label:
                    print(f"  {label} ON  vel={msg.velocity}")
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                servo_state.note_off(msg.note)
                label = {NOTE_MOUTH: "Mouth", NOTE_EYES: "Eyes", NOTE_NOSE: "Nose"}.get(msg.note)
                if label:
                    print(f"  {label} OFF")


def midi_file_player(file_path: str, servo_state: ServoState, done_event: threading.Event):
    """Play a MIDI file, sending note events to the servo state with real-time timing."""
    print(f"Playing MIDI file: {file_path}")
    print(f"  Mouth (lower jaw): note {NOTE_MOUTH} (C3) → CH4")
    print(f"  Eyes:               note {NOTE_EYES} (D3) → CH2")
    print(f"  Nose (upper jaw):   note {NOTE_NOSE} (E3) → CH3")
    print()

    midi_file = mido.MidiFile(file_path)
    for msg in midi_file.play():
        if msg.type == "note_on" and msg.velocity > 0:
            servo_state.note_on(msg.note, msg.velocity)
            label = {NOTE_MOUTH: "Mouth", NOTE_EYES: "Eyes", NOTE_NOSE: "Nose"}.get(msg.note)
            if label:
                print(f"  {label} ON  vel={msg.velocity}")
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            servo_state.note_off(msg.note)
            label = {NOTE_MOUTH: "Mouth", NOTE_EYES: "Eyes", NOTE_NOSE: "Nose"}.get(msg.note)
            if label:
                print(f"  {label} OFF")

    print("\nMIDI file playback complete.")
    done_event.set()


def main():
    print("=== MIDI-to-PPM Teddy Ruxpin Controller ===")
    print()

    midi_file = None
    wav_output = None
    if len(sys.argv) > 1:
        midi_file = sys.argv[1]
        if not os.path.isfile(midi_file):
            print(f"Error: MIDI file not found: {midi_file}")
            sys.exit(1)
    if len(sys.argv) > 2:
        wav_output = sys.argv[2]

    servo_state = ServoState()
    recorded_frames = [] if wav_output else None

    if midi_file:
        # MIDI file playback mode
        done_event = threading.Event()
        midi_thread = threading.Thread(
            target=midi_file_player, args=(midi_file, servo_state, done_event), daemon=True
        )
    else:
        # Live MIDI input mode
        ports = list_midi_ports()
        if ports is None:
            sys.exit(1)

        if len(ports) == 1:
            choice = 0
        else:
            try:
                choice = int(input(f"\nSelect port [0-{len(ports)-1}]: "))
            except (ValueError, EOFError):
                choice = 0

        if choice < 0 or choice >= len(ports):
            print("Invalid selection.")
            sys.exit(1)

        port_name = ports[choice]
        done_event = None
        midi_thread = threading.Thread(
            target=midi_listener, args=(port_name, servo_state), daemon=True
        )

    midi_thread.start()

    # Start audio output stream
    print(f"Starting PPM audio output (right channel, {SAMPLE_RATE} Hz)...")
    print("Press Ctrl+C to quit.\n")

    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=AUDIO_BLOCKSIZE,
        channels=2,
        dtype="float32",
        callback=make_audio_callback(servo_state, recorded_frames),
    )

    try:
        with stream:
            if done_event:
                # File mode: wait for playback to finish, then flush remaining audio
                while not done_event.is_set():
                    time.sleep(0.1)
                time.sleep(0.5)  # let final PPM frames flush
            else:
                # Live mode: run until Ctrl+C
                while True:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping.")

    if wav_output and recorded_frames:
        audio_data = np.concatenate(recorded_frames, axis=0)
        # Convert float32 [-1.0, 1.0] to int16
        int_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        with wave.open(wav_output, "w") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(int_data.tobytes())
        print(f"Saved WAV output to: {wav_output}")


if __name__ == "__main__":
    main()
