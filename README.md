# RuxPPM — MIDI-to-PPM Teddy Ruxpin Controller

A real-time Python tool that converts MIDI note input into PPM (Pulse Position Modulation) audio signals to control an original Teddy Ruxpin toy's servos. Input can come from a live MIDI device or a MIDI file.

The PPM signal is output on the **right stereo channel**, matching the 8-channel RC-style format used by the original Teddy Ruxpin cassette tapes.

## Requirements

- Python 3.10+
- A Teddy Ruxpin with its cassette mechanism wired to accept line-level audio input
- An audio interface with stereo output (right channel carries the PPM signal)
- For live mode: a MIDI controller or virtual MIDI port

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Live MIDI Input

Connect a MIDI controller and run:

```bash
python ruxppm.py
```

The tool will list available MIDI input ports and prompt you to select one. If only one port is found, it is selected automatically.

### MIDI File Playback

Pass a `.mid` file as an argument:

```bash
python ruxppm.py song.mid
```

The file plays back in real time, driving the servos with proper timing. The program exits automatically when playback finishes.

### Controls

Both modes can be stopped at any time with **Ctrl+C**.

## MIDI Note Mapping

| Note | Name | Servo        | PPM Channel | Behavior                          |
|------|------|--------------|-------------|-----------------------------------|
| 60   | C3   | Lower jaw    | CH4         | Velocity-scaled open; release closes |
| 62   | D3   | Eyes         | CH2         | Velocity-scaled close; release opens |
| 64   | E3   | Upper jaw    | CH3         | Toggle on/off (not velocity-scaled)  |

## Configuration

Edit the constants at the top of `ruxppm.py` to adjust:

| Constant       | Default | Description                                      |
|----------------|---------|--------------------------------------------------|
| `SAMPLE_RATE`  | 44100   | Audio sample rate. Higher rates (96000, 192000) give finer servo resolution. |
| `NOTE_MOUTH`   | 60      | MIDI note number for lower jaw                   |
| `NOTE_EYES`    | 62      | MIDI note number for eyes                        |
| `NOTE_NOSE`    | 64      | MIDI note number for upper jaw                   |
| `SERVO_SCALE`  | 0.90    | Servo travel multiplier (0.0–1.0)                |
| `NOSE_OPEN`    | 80      | Upper jaw open position (0–100 scale)            |

## License

MIT
