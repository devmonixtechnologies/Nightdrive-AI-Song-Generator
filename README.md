# Nightdrive AI Song Generator

> Generate royalty-free instrumentals from pure text prompts using Meta's MusicGen family, with long-form stitching, live progress feedback, and built-in audio controls.

Nightdrive wraps the Hugging Face `text-to-audio` pipeline in a friendly CLI and Tkinter GUI. You get streamlined model caching, streaming writes for ultra-long tracks, prompt history, built-in playback, tempo presets, and loop-ready exports—no external services required.



<img width="2160" height="1367" alt="nightdrive" src="https://github.com/user-attachments/assets/96cd3430-34cb-4c30-b1c2-014149fcfd8f" />





## Highlights

- **All-local generation** – Runs on CPU or GPU using open MusicGen checkpoints (`facebook/musicgen-*`) or any custom Hugging Face ID.
- **Two interaction modes** – Scriptable CLI and a feature-rich desktop GUI with progress bar, cancel button, playback, prompt history, and event log.
- **Performance tuned** – Pipeline caching, optional streaming writer, and peak normalization keep memory down and loudness consistent.
- **Creative controls** – Tempo presets, loopable endings, MP3/WAV export, batch variations, negative prompts, and adjustable crossfades.
- **Extensible design** – Modular scheduling, normalization, and IO helpers make it easy to extend for new features, APIs, or workflows.

## Architecture at a Glance

1. **Prompt ingestion** – CLI/GUI collects the prompt, model choice, duration, and advanced toggles into a `GenerationConfig` dataclass.
2. **Pipeline caching** – `load_pipeline` reuses MusicGen models keyed by (model_id, device, dtype) to avoid redundant downloads.
3. **Segment scheduler** – Durations are split into ≤30 s segments with optional crossfade overlap for seamless stitching.
4. **Generation loop** – Each segment is rendered via the Hugging Face pipeline, streamed to disk if requested, or crossfaded in-memory.
5. **Post-processing** – Optional peak normalization, loopable endings, and MP3 conversion (via `pydub`) finalize the audio.
6. **Progress callbacks** – GUI receives `start`, `segment_complete`, `variation_start`, `variation_finished`, `finished`, and `error` events for responsive updates.

## Installation

### Prerequisites

- Python **3.10+**
- A modern CPU (GPU optional but recommended for speed)
- [`ffmpeg`](https://ffmpeg.org/) (recommended for any downstream format conversions)
- Optional playback/tooling dependencies:
  - [`pydub`](https://github.com/jiaaro/pydub) for MP3 export
  - [`simpleaudio`](https://simpleaudio.readthedocs.io/) for GUI playback (Linux users may need `sudo apt install libasound2-dev` before installing)
  - Please note that, you need to download a 6.5GB of file contains all the models (it will start to auto download at the first prompt. One time installation! after that all models works perfectly!!!)

### Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional extras
pip install pydub simpleaudio
```

MusicGen weights download automatically on first use. Expect several gigabytes and a multi-minute wait the first time you run a new model, especially on CPU.

## Quick Start (CLI)

```bash
# 30-second CPU-friendly preview
python src/song_generator.py \
  --prompt "nocturnal synthwave cruise" \
  --model small \
  --duration 30 \
  --output outputs/nightdrive_preview.wav

# 90-second MP3 export with tempo boost (requires pydub)
python src/song_generator.py \
  --prompt "lofi study beats" \
  --duration 90 \
  --tempo energetic \
  --format mp3 \
  --output outputs/study_session.mp3
```

Output defaults to 32 kHz, 16-bit PCM WAV. MP3 exports are encoded at 192 kbps CBR.

## GUI Workflow

```bash
python src/gui.py
```

1. **Prompt** – Enter your theme or story; optional negative prompt steers away from unwanted elements.
2. **Model & duration** – Choose an alias (`small`, `medium`, `large`, `melody`, `v3`, `v4`, `v4.5`, `v5`) or a custom model ID.
3. **Advanced controls** – Toggle normalization, streaming writes, loopable endings, tempo presets, MP3/WAV output, and batch variations.
4. **Generate** – Progress is reported per segment and per variation; cancel at any time.
5. **Playback & history** – Preview the most recent file with `simpleaudio`, and re-use prior prompts from the history panel.

The GUI writes to `outputs/gui_song.wav` (or `.mp3`) by default and remembers your prompt history for quick iteration.

## Advanced Controls

### Tempo presets

- `slow` (0.7×), `chill` (0.85×), `default` (1.0×), `energetic` (1.15×)
- Presets append descriptive hints to the prompt to bias the generation speed/energy.

### Loopable endings

- Enable `--loopable` (CLI) or the **Loopable** checkbox (GUI) to blend the start/end of the final track, useful for background loops.
- Loop smoothing uses ~0.75 s of material; disable if precise endings are required.

### Batch variations

- Generate multiple takes in a single run with `--batch-variations N` (CLI) or the GUI spinbox.
- Files are suffixed numerically (`track_1.wav`, `track_2.wav`, ...).

### Streaming writes vs. normalization

- `--stream` writes segments directly to disk to save memory (ideal for >5 min tracks), but skips normalization since samples are not held in memory.
- Without streaming, the full waveform is assembled in-memory, normalized to avoid clipping, then written.

### MP3 export

- Pass `--format mp3` or choose **MP3** in the GUI; requires `pydub` (and `ffmpeg` available on PATH).
- Streaming mode is WAV-only to avoid partial MP3 corruption.

### Custom models & prompts

- Provide `--model-id` (CLI) or enter a custom Hugging Face repo in the GUI model ID field to try non-default checkpoints.
- Negative prompts help suppress artifacts (e.g., `--negative "high distortion, glitch"`).

## CLI Reference

```text
usage: song_generator.py [-h] --prompt PROMPT [--model {small,medium,large,melody,v3,v4,v4.5,v5}] [--model-id MODEL_ID] [--list-models]
                         [--duration DURATION] [--guidance GUIDANCE] [--negative NEGATIVE] [--crossfade CROSSFADE] [--output OUTPUT]
                         [--no-normalize] [--stream] [--tempo {chill,default,energetic,slow}] [--batch-variations BATCH_VARIATIONS]
                         [--loopable] [--format {wav,mp3}]

optional arguments:
  -h, --help            show this help message and exit
  --prompt PROMPT       Text prompt describing the music to generate.
  --model {small,medium,large,melody,v3,v4,v4.5,v5}
                        MusicGen checkpoint alias to use (default: small).
  --model-id MODEL_ID   Override the Hugging Face model identifier (e.g., suno/bark-small).
  --list-models         List available model aliases and exit.
  --duration DURATION   Total desired duration in seconds (default: 180, range: 5–600).
  --guidance GUIDANCE   Classifier-free guidance scale (default: 3.5).
  --negative NEGATIVE   Negative text prompt to steer the model away from unwanted qualities.
  --crossfade CROSSFADE Seconds of overlap when stitching segments (default: 0.75, max: 15).
  --output OUTPUT       Output path (format-sensitive extension is enforced automatically).
  --no-normalize        Skip automatic peak normalization on the final audio file.
  --stream              Write audio incrementally to disk (WAV only; normalization disabled).
  --tempo {chill,default,energetic,slow}
                        Apply a tempo preset hint to the prompt.
  --batch-variations BATCH_VARIATIONS
                        Number of variations to synthesize (files are suffixed _1, _2, …).
  --loopable            Blend the beginning/end for seamless looping.
  --format {wav,mp3}    Output audio format (requires pydub for MP3).
```

## Tips for Faster Iteration

- **First run patience** – Model downloads can take several minutes; allow the first generation to finish so weights are cached.
- **Start small** – Use the `small` model with 20–30 s durations to validate settings before rendering multi-minute tracks.
- **CPU vs. GPU** – GPU dramatically speeds up generation. If available, ensure PyTorch detects CUDA and rerun the CLI/GUI.
- **Reuse prompts** – The GUI history list and CLI shell history help you iterate quickly on variations.

## Troubleshooting & FAQ

**Generation is extremely slow or stalls at 0 %.**
: The model is downloading. Keep the window open; future runs will start immediately once cached.

**`simpleaudio` fails to install with `asoundlib.h` missing.**
: Install ALSA dev headers first: `sudo apt install libasound2-dev`, then reinstall inside the virtualenv: `pip install simpleaudio`.

**MP3 export fails complaining about `pydub` or ffmpeg.**
: Run `pip install pydub` and ensure `ffmpeg` is installed and on your PATH (`ffmpeg -version`).

**Where are files saved?**
: CLI outputs to the path given by `--output` (extension enforced). GUI defaults to `outputs/gui_song.wav` or `.mp3` and adds numbered suffixes for variations.

**How do I cancel a long render?**
: Click **Cancel** in the GUI or press `Ctrl+C` in the CLI. Partial streaming outputs will contain whatever was generated up to the cancel point.

**Can I resume after cancelling during a download?**
: Yes. Hugging Face caches partial downloads; re-running will continue until all shards are cached.

## Release Notes

- **Phase 3 – Audio Control Suite**
  - Tempo presets, loopable endings, batch variations, MP3 export, enhanced GUI controls.
- **Phase 2 – UX Upgrades**
  - Asynchronous generation, progress bar, cancel button, playback (simpleaudio), prompt history, event log.
- **Phase 1 – Performance Enhancements**
  - Pipeline caching, streaming WAV writer, optional normalization, CLI flags for streaming/normalization.

## Licensing & Attribution

- MusicGen checkpoints are governed by the [Meta MusicGen License](https://ai.meta.com/resources/models-and-libraries/musicgen/).
- Audio exported with Nightdrive is royalty-free under the MusicGen license terms.
- This project itself is distributed under the license specified in the repository root.

Happy composing and safe night drives! 🎶
