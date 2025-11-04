"""CLI tool to generate royalty-free songs using Meta's MusicGen models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import sys

import numpy as np
import soundfile as sf
import torch
from threading import Event
from transformers import pipeline


MODEL_CHOICES = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
    "melody": "facebook/musicgen-melody",
    "v3": "facebook/musicgen-small",
    "v4": "facebook/musicgen-medium",
    "v4.5": "facebook/musicgen-large",
    "v5": "facebook/musicgen-melody",
}

TEMPO_PRESETS = {
    "chill": 0.85,
    "default": 1.0,
    "energetic": 1.15,
    "slow": 0.7,
}

DEFAULT_TOTAL_DURATION = 180.0
MIN_TOTAL_DURATION = 5.0
MAX_TOTAL_DURATION = 600.0
MAX_SEGMENT_DURATION = 30.0
DEFAULT_GUIDANCE = 3.5
DEFAULT_CROSSFADE = 0.75
TOKENS_PER_SECOND = 25.6
MIN_TOKENS = 32
MAX_TOKENS = 1024
MAX_LOOP_GAP_SECONDS = 0.75


_PIPELINE_CACHE: dict[tuple[str, str, str], Any] = {}


@dataclass
class GenerationConfig:
    prompt: str
    model_choice: str
    model_id: str
    total_duration: float
    guidance: float
    negative_prompt: Optional[str]
    output_path: Path
    crossfade: float
    normalize: bool
    stream: bool
    tempo_preset: str
    batch_variations: int
    loopable: bool
    audio_format: str


def parse_args() -> GenerationConfig:
    parser = argparse.ArgumentParser(description="Generate music using MusicGen models.")
    parser.add_argument("--prompt", required=True, help="Text prompt describing the music to generate.")
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES.keys(),
        default="small",
        help="MusicGen checkpoint to use (default: small).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override the model identifier (e.g., suno/bark-small).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model aliases and exit.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_TOTAL_DURATION,
        help=(
            "Total desired duration in seconds. The track is stitched from segments if needed "
            f"(default: {DEFAULT_TOTAL_DURATION}, range: {MIN_TOTAL_DURATION}-{MAX_TOTAL_DURATION})."
        ),
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=DEFAULT_GUIDANCE,
        help="Classifier-free guidance scale (default: 3.5).",
    )
    parser.add_argument(
        "--negative",
        type=str,
        default=None,
        help="Optional negative prompt to avoid unwanted qualities.",
    )
    parser.add_argument(
        "--crossfade",
        type=float,
        default=DEFAULT_CROSSFADE,
        help="Seconds of overlap between stitched segments (default: 0.75).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./generated_song.wav",
        help="Output WAV path (default: ./generated_song.wav).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip automatic peak normalization on the final audio file.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Write audio incrementally to disk to reduce memory usage (normalization disabled).",
    )
    parser.add_argument(
        "--tempo",
        choices=TEMPO_PRESETS.keys(),
        default="default",
        help="Tempo preset multiplier for generation.",
    )
    parser.add_argument(
        "--batch-variations",
        type=int,
        default=1,
        help="Number of variations to synthesize (writes numbered files).",
    )
    parser.add_argument(
        "--loopable",
        action="store_true",
        help="Apply loop-friendly crossfade by matching start/end segments.",
    )
    parser.add_argument(
        "--format",
        choices=("wav", "mp3"),
        default="wav",
        help="Output audio container format (default: wav).",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available model choices:")
        for alias, identifier in MODEL_CHOICES.items():
            print(f"  {alias:8s} -> {identifier}")
        if args.model_id:
            print(f"Custom identifier provided: {args.model_id}")
        sys.exit(0)

    model_id = args.model_id or MODEL_CHOICES[args.model]

    total_duration = max(MIN_TOTAL_DURATION, min(MAX_TOTAL_DURATION, args.duration))
    if total_duration != args.duration:
        print(
            f"[nightdrive] Adjusted duration to {total_duration:.1f}s to stay within supported range "
            f"({MIN_TOTAL_DURATION}-{MAX_TOTAL_DURATION}s)."
        )

    crossfade = max(0.0, min(args.crossfade, MAX_SEGMENT_DURATION / 2))

    output_path = Path(args.output).expanduser().resolve()
    requested_suffix = ".mp3" if args.format == "mp3" else ".wav"
    if output_path.suffix.lower() != requested_suffix:
        output_path = output_path.with_suffix(requested_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.stream and args.format != "wav":
        raise ValueError("Streaming mode currently supports only WAV output.")

    return GenerationConfig(
        prompt=args.prompt,
        model_choice=args.model,
        model_id=model_id,
        total_duration=total_duration,
        guidance=args.guidance,
        negative_prompt=args.negative,
        output_path=output_path,
        crossfade=crossfade,
        normalize=False if args.stream else not args.no_normalize,
        stream=args.stream,
        tempo_preset=args.tempo,
        batch_variations=max(1, args.batch_variations),
        loopable=args.loopable,
        audio_format=args.format,
    )


def duration_to_tokens(duration_seconds: float) -> int:
    tokens = int(round(duration_seconds * TOKENS_PER_SECOND))
    return max(MIN_TOKENS, min(MAX_TOKENS, tokens))


def select_device() -> tuple[int | str, Optional[torch.dtype]]:
    if torch.cuda.is_available():
        return 0, torch.float16
    if torch.backends.mps.is_available():  # pragma: no cover - macOS GPU optional
        return "mps", None
    return "cpu", None


def load_pipeline(model_id: str):
    device, torch_dtype = select_device()
    cache_key = (model_id, str(device), str(torch_dtype))

    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    generator = pipeline(
        "text-to-audio",
        model=model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    _PIPELINE_CACHE[cache_key] = generator
    return generator


def compute_segment_schedule(total_duration: float, crossfade: float) -> list[float]:
    segments: list[float] = []
    accumulated = 0.0
    while accumulated < total_duration - 1e-6:
        remaining = total_duration - accumulated
        overlap = crossfade if segments else 0.0
        duration = min(MAX_SEGMENT_DURATION, remaining + overlap)
        segments.append(duration)
        accumulated += duration - overlap
    return segments


def normalize_audio(raw_audio: Any) -> np.ndarray:
    audio = raw_audio
    if isinstance(audio, list):
        audio = audio[0]
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    elif not isinstance(audio, np.ndarray):
        audio = np.asarray(audio)

    if audio.ndim == 3:
        audio = np.squeeze(audio, axis=0)
    if audio.ndim == 1:
        audio = audio[:, None]
    elif audio.ndim == 2:
        if audio.shape[0] < audio.shape[1]:
            audio = audio.T
    else:
        raise ValueError(f"[nightdrive] Unexpected audio shape {audio.shape}")

    return audio.astype(np.float32)


def apply_peak_normalization(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-6:
        return audio
    return (audio / peak * 0.98).astype(np.float32)


def crossfade_audio(base: np.ndarray, nxt: np.ndarray, crossfade_seconds: float, sample_rate: int) -> np.ndarray:
    crossfade_samples = int(round(crossfade_seconds * sample_rate))
    if crossfade_samples <= 0:
        return np.concatenate([base, nxt], axis=0)

    crossfade_samples = min(crossfade_samples, base.shape[0], nxt.shape[0])
    if crossfade_samples == 0:
        return np.concatenate([base, nxt], axis=0)

    fade_out = np.linspace(1.0, 0.0, crossfade_samples, endpoint=False, dtype=base.dtype)[:, None]
    fade_in = 1.0 - fade_out

    overlapped = base[-crossfade_samples:] * fade_out + nxt[:crossfade_samples] * fade_in
    return np.concatenate([base[:-crossfade_samples], overlapped, nxt[crossfade_samples:]], axis=0)


def run_segment_generation(generator, prompt: str, segment_duration: float, guidance: float, negative_prompt: Optional[str]) -> tuple[np.ndarray, int]:
    generate_kwargs: dict[str, object] = {"max_new_tokens": duration_to_tokens(segment_duration)}
    optional_keys = {
        "guidance_scale": guidance,
        "negative_prompt": negative_prompt,
    }
    for key, value in optional_keys.items():
        if value is not None:
            generate_kwargs[key] = value

    while True:
        try:
            result = generator(prompt, generate_kwargs=generate_kwargs)
            break
        except TypeError as exc:
            message = str(exc)
            unsupported_key = next((key for key in list(generate_kwargs.keys()) if key in message), None)
            if unsupported_key:
                print(f"[nightdrive] '{unsupported_key}' not supported by this transformers build; retrying without.")
                generate_kwargs.pop(unsupported_key, None)
                continue
            raise

    output: Any = result[0] if isinstance(result, list) else result
    audio = normalize_audio(output["audio"])
    sample_rate = output["sampling_rate"]
    return audio, sample_rate


def apply_tempo_to_prompt(prompt: str, tempo_preset: str, variation_idx: int) -> str:
    tempo_multiplier = TEMPO_PRESETS.get(tempo_preset, 1.0)
    descriptor = {
        "slow": "slow and spacious",
        "chill": "relaxed and laid-back",
        "default": "balanced tempo",
        "energetic": "high-energy and fast",
    }.get(tempo_preset, "balanced tempo")
    variation_hint = "" if variation_idx == 0 else f" variation {variation_idx + 1}"
    return f"{prompt} ({descriptor},{variation_hint} {tempo_multiplier:.2f}x tempo)".strip()


def enforce_loopable(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    gap_samples = int(round(MAX_LOOP_GAP_SECONDS * sample_rate))
    fade_samples = max(1, gap_samples // 2)
    if fade_samples * 2 >= audio.shape[0]:
        return audio

    fade_out = np.linspace(1.0, 0.0, fade_samples, endpoint=False, dtype=audio.dtype)[:, None]
    fade_in = 1.0 - fade_out

    start = audio[:fade_samples]
    end = audio[-fade_samples:]
    loop_fade = start * fade_out + end * fade_in

    return np.concatenate([audio[:-fade_samples], loop_fade], axis=0)


def write_audio_file(audio: np.ndarray, sample_rate: int, path: Path, audio_format: str) -> None:
    if audio_format == "wav":
        sf.write(path, audio, sample_rate)
        return

    if audio_format == "mp3":
        try:
            from pydub import AudioSegment  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "MP3 export requires pydub. Install it with 'pip install pydub'."
            ) from exc

        channels = audio.shape[1] if audio.ndim == 2 else 1
        if channels == 1 and audio.ndim == 2:
            audio = audio[:, 0]
        if audio.ndim == 1:
            audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
        else:
            audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)

        segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=channels,
        )
        segment.export(str(path), format="mp3", bitrate="192k")
        return

    raise ValueError(f"Unsupported audio format: {audio_format}")


def generate_song(
    config: GenerationConfig,
    progress_callback: Optional[Callable[[str, Any], None]] = None,
    cancel_event: Optional[Event] = None,
) -> Optional[Path]:
    try:
        generator = load_pipeline(config.model_id)
        segment_durations = compute_segment_schedule(config.total_duration, config.crossfade)

        total_segments = len(segment_durations)
        if progress_callback:
            progress_callback("start", {"total_segments": total_segments})

        variations = max(1, config.batch_variations)
        output_paths: list[Path] = []
        base_output_path = config.output_path
        if config.audio_format == "mp3" and base_output_path.suffix.lower() != ".mp3":
            base_output_path = base_output_path.with_suffix(".mp3")

        for variation_idx in range(variations):
            variation_suffix = "" if variations == 1 else f"_{variation_idx+1}"
            current_output_path = (
                base_output_path
                if variation_suffix == ""
                else base_output_path.with_name(base_output_path.stem + variation_suffix + base_output_path.suffix)
            )
            if progress_callback:
                progress_callback("variation_start", {"index": variation_idx + 1, "total": variations})

            combined_audio: Optional[np.ndarray] = None
            sample_rate: Optional[int] = None
            streaming_writer: Optional[StreamingWriter] = None

            for idx, segment_duration in enumerate(segment_durations):
                if cancel_event and cancel_event.is_set():
                    if streaming_writer is not None:
                        streaming_writer.finalize()
                    if progress_callback:
                        progress_callback("cancelled", None)
                    return None

                audio, sr = run_segment_generation(
                    generator,
                    prompt=apply_tempo_to_prompt(config.prompt, config.tempo_preset, variation_idx),
                    segment_duration=segment_duration,
                    guidance=config.guidance,
                    negative_prompt=config.negative_prompt,
                )

                if cancel_event and cancel_event.is_set():
                    if streaming_writer is not None:
                        streaming_writer.finalize()
                    if progress_callback:
                        progress_callback("cancelled", None)
                    return None

                if config.stream:
                    if streaming_writer is None:
                        streaming_writer = StreamingWriter(
                            current_output_path,
                            sample_rate=sr,
                            channels=audio.shape[1],
                            crossfade_seconds=config.crossfade,
                        )
                    else:
                        if streaming_writer.sample_rate != sr:
                            raise ValueError(
                                f"[nightdrive] Segment sample rate mismatch: expected {streaming_writer.sample_rate}, got {sr}."
                            )
                    streaming_writer.add_segment(audio)
                else:
                    if sample_rate is None:
                        sample_rate = sr
                        combined_audio = audio
                    else:
                        if sample_rate != sr:
                            raise ValueError(
                                f"[nightdrive] Segment sample rate mismatch: expected {sample_rate}, got {sr}."
                            )
                        assert combined_audio is not None
                        combined_audio = crossfade_audio(combined_audio, audio, config.crossfade, sample_rate)

                if progress_callback:
                    progress_callback(
                        "segment_complete",
                        {"completed": idx + 1, "total_segments": total_segments, "variation": variation_idx + 1},
                    )

            if config.stream:
                if streaming_writer is not None:
                    streaming_writer.finalize()
                if config.normalize:
                    print("[nightdrive] Streaming mode skips normalization. Use --no-normalize to disable this warning.")
                output_paths.append(current_output_path)
                if progress_callback:
                    progress_callback("variation_finished", {"path": current_output_path, "variation": variation_idx + 1})
                continue

            assert combined_audio is not None and sample_rate is not None

            target_samples = int(round(config.total_duration * sample_rate))
            if combined_audio.shape[0] > target_samples:
                combined_audio = combined_audio[:target_samples]
            elif combined_audio.shape[0] < target_samples:
                pad_length = target_samples - combined_audio.shape[0]
                pad = np.zeros((pad_length, combined_audio.shape[1]), dtype=combined_audio.dtype)
                combined_audio = np.concatenate([combined_audio, pad], axis=0)

            if config.loopable:
                combined_audio = enforce_loopable(combined_audio, sample_rate)

            if config.normalize and not config.stream:
                combined_audio = apply_peak_normalization(combined_audio)

            write_audio_file(combined_audio, sample_rate, current_output_path, config.audio_format)
            output_paths.append(current_output_path)
            if progress_callback:
                progress_callback("variation_finished", {"path": current_output_path, "variation": variation_idx + 1})

        final_path = output_paths[0] if output_paths else None
        if final_path and progress_callback:
            progress_callback("finished", final_path)
        return final_path
    except Exception as exc:  # pragma: no cover - runtime feedback
        if progress_callback:
            progress_callback("error", exc)
        raise


def main() -> None:
    config = parse_args()
    output_path = generate_song(config)
    if output_path is None:
        print("[nightdrive] Generation cancelled.")
    else:
        print(f"[nightdrive] Generated song saved to {output_path}")


if __name__ == "__main__":
    main()
