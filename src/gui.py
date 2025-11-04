"""Tkinter GUI for Nightdrive AI Song Generator."""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import soundfile as sf

from song_generator import (
    DEFAULT_CROSSFADE,
    DEFAULT_GUIDANCE,
    DEFAULT_TOTAL_DURATION,
    GenerationConfig,
    MAX_SEGMENT_DURATION,
    MAX_TOTAL_DURATION,
    MIN_TOTAL_DURATION,
    MODEL_CHOICES,
    TEMPO_PRESETS,
    compute_segment_schedule,
    generate_song,
)


DEFAULT_OUTPUT_PATH = Path("outputs/gui_song.wav")


class NightdriveGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Nightdrive AI Song Generator")
        self.root.geometry("640x520")

        self._busy = False

        self.model_var = tk.StringVar(value="small")
        self.model_id_var = tk.StringVar()
        self.duration_var = tk.StringVar(value=str(int(DEFAULT_TOTAL_DURATION)))
        self.guidance_var = tk.StringVar(value=str(DEFAULT_GUIDANCE))
        self.crossfade_var = tk.StringVar(value=str(DEFAULT_CROSSFADE))
        self.output_var = tk.StringVar(value=str(DEFAULT_OUTPUT_PATH))
        self.negative_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Idle")
        self.normalize_var = tk.BooleanVar(value=True)
        self.stream_var = tk.BooleanVar(value=False)
        self.format_var = tk.StringVar(value="wav")
        self.tempo_var = tk.StringVar(value="default")
        self.loop_var = tk.BooleanVar(value=False)
        self.variations_var = tk.StringVar(value="1")

        self.history: list[str] = []
        self.cancel_event = threading.Event()
        self.result_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._audio_cache: dict[Path, tuple[int, np.ndarray]] = {}

        self._build_ui()
        self.root.after(200, self._poll_worker)

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root, padding=16)
        main_frame.pack(fill=tk.BOTH, expand=True)

        prompt_label = ttk.Label(main_frame, text="Prompt")
        prompt_label.pack(anchor=tk.W)

        self.prompt_text = tk.Text(main_frame, height=6, wrap=tk.WORD)
        self.prompt_text.pack(fill=tk.BOTH, expand=True, pady=(4, 12))

        form_frame = ttk.Frame(main_frame)
        form_frame.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(form_frame, text="Model alias").grid(row=0, column=0, sticky=tk.W)
        model_combo = ttk.Combobox(
            form_frame,
            textvariable=self.model_var,
            values=sorted(MODEL_CHOICES.keys()),
            state="readonly",
            width=20,
        )
        model_combo.grid(row=0, column=1, sticky=tk.W, padx=(8, 16))

        ttk.Label(form_frame, text="Custom model ID").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(form_frame, textvariable=self.model_id_var, width=30).grid(
            row=0, column=3, sticky=tk.W, padx=(8, 0)
        )

        ttk.Label(form_frame, text="Duration (sec)").grid(row=1, column=0, sticky=tk.W, pady=(12, 0))
        ttk.Entry(form_frame, textvariable=self.duration_var, width=10).grid(
            row=1, column=1, sticky=tk.W, padx=(8, 16), pady=(12, 0)
        )

        ttk.Label(form_frame, text="Guidance").grid(row=1, column=2, sticky=tk.W, pady=(12, 0))
        ttk.Entry(form_frame, textvariable=self.guidance_var, width=10).grid(
            row=1, column=3, sticky=tk.W, padx=(8, 0), pady=(12, 0)
        )

        ttk.Label(form_frame, text="Crossfade (sec)").grid(row=2, column=0, sticky=tk.W, pady=(12, 0))
        ttk.Entry(form_frame, textvariable=self.crossfade_var, width=10).grid(
            row=2, column=1, sticky=tk.W, padx=(8, 16), pady=(12, 0)
        )

        ttk.Label(form_frame, text="Negative prompt").grid(row=2, column=2, sticky=tk.W, pady=(12, 0))
        ttk.Entry(form_frame, textvariable=self.negative_var, width=30).grid(
            row=2, column=3, sticky=tk.W, padx=(8, 0), pady=(12, 0)
        )

        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(output_frame, text="Output file").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(output_frame, textvariable=self.output_var, width=45).grid(
            row=0, column=1, sticky=tk.W, padx=(8, 8)
        )
        ttk.Button(output_frame, text="Browse", command=self._browse_output).grid(row=0, column=2, sticky=tk.W)
        ttk.Label(output_frame, text="Format").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        format_combo = ttk.Combobox(output_frame, textvariable=self.format_var, values=("wav", "mp3"), state="readonly")
        format_combo.grid(row=1, column=1, sticky=tk.W, padx=(8, 8), pady=(8, 0))

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        self.generate_button = ttk.Button(button_frame, text="Generate Song", command=self._on_generate)
        self.generate_button.pack(side=tk.LEFT)

        ttk.Button(button_frame, text="List Models", command=self._show_models).pack(side=tk.LEFT, padx=(8, 0))
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self._on_cancel, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_frame, text="Playback", command=self._on_playback).pack(side=tk.LEFT, padx=(8, 0))

        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(12, 0))

        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=(6, 0))

        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(8, 0))
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X)

        options_frame = ttk.Frame(main_frame)
        options_frame.pack(fill=tk.X, pady=(8, 0))

        ttk.Checkbutton(options_frame, text="Normalize", variable=self.normalize_var).pack(side=tk.LEFT)
        ttk.Checkbutton(options_frame, text="Stream", variable=self.stream_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Checkbutton(options_frame, text="Loopable", variable=self.loop_var).pack(side=tk.LEFT, padx=(12, 0))

        tempo_frame = ttk.Frame(main_frame)
        tempo_frame.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(tempo_frame, text="Tempo").pack(side=tk.LEFT)
        ttk.Combobox(
            tempo_frame,
            textvariable=self.tempo_var,
            values=tuple(TEMPO_PRESETS.keys()),
            state="readonly",
            width=12,
        ).pack(side=tk.LEFT, padx=(8, 12))

        ttk.Label(tempo_frame, text="Variations").pack(side=tk.LEFT)
        ttk.Spinbox(
            tempo_frame,
            from_=1,
            to=12,
            textvariable=self.variations_var,
            width=5,
        ).pack(side=tk.LEFT, padx=(8, 0))

        history_frame = ttk.LabelFrame(main_frame, text="Prompt History")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        self.history_list = tk.Listbox(history_frame, height=5)
        self.history_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.history_list.bind("<<ListboxSelect>>", self._on_history_select)

        history_scroll = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_list.yview)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_list.config(yscrollcommand=history_scroll.set)

        log_frame = ttk.LabelFrame(main_frame, text="Event Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.log_text = tk.Text(log_frame, height=6, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _log(self, message: str) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _browse_output(self) -> None:
        initial = Path(self.output_var.get()).expanduser().resolve()
        initial_dir = initial.parent if initial.parent.exists() else Path.cwd()
        filename = filedialog.asksaveasfilename(
            title="Select output WAV file",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialdir=initial_dir,
            initialfile=initial.name,
        )
        if filename:
            self.output_var.set(filename)

    def _show_models(self) -> None:
        lines = ["Available model aliases:"]
        for alias, identifier in sorted(MODEL_CHOICES.items()):
            lines.append(f"  {alias:8s} -> {identifier}")
        messagebox.showinfo("Nightdrive Models", "\n".join(lines))

    def _on_generate(self) -> None:
        if self._busy:
            return
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Nightdrive", "Please enter a prompt before generating.")
            return

        try:
            config = self._build_config(prompt)
        except ValueError as exc:
            messagebox.showerror("Nightdrive", str(exc))
            return

        self._busy = True
        self.status_var.set("Generating...")
        self.generate_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.progress_var.set(0.0)
        self.cancel_event.clear()
        self._log(f"[start] {prompt[:60]}...")

        self._worker_thread = threading.Thread(target=self._run_generation, args=(config,), daemon=True)
        self._worker_thread.start()

    def _build_config(self, prompt: str) -> GenerationConfig:
        try:
            duration = float(self.duration_var.get())
        except ValueError as exc:
            raise ValueError("Duration must be a number.") from exc

        duration = max(MIN_TOTAL_DURATION, min(MAX_TOTAL_DURATION, duration))

        try:
            guidance = float(self.guidance_var.get())
        except ValueError as exc:
            raise ValueError("Guidance must be a number.") from exc

        try:
            crossfade = float(self.crossfade_var.get())
        except ValueError as exc:
            raise ValueError("Crossfade must be a number.") from exc

        crossfade = max(0.0, min(MAX_SEGMENT_DURATION / 2, crossfade))

        output_path = Path(self.output_var.get()).expanduser().resolve()
        selected_format = self.format_var.get()
        expected_suffix = f".{selected_format}"
        if output_path.suffix.lower() != expected_suffix:
            output_path = output_path.with_suffix(expected_suffix)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_choice = self.model_var.get()
        model_id = self.model_id_var.get().strip() or MODEL_CHOICES[model_choice]

        if self.stream_var.get() and selected_format != "wav":
            raise ValueError("Streaming mode currently supports only WAV output.")

        tempo_choice = self.tempo_var.get()
        if tempo_choice not in TEMPO_PRESETS:
            tempo_choice = "default"

        try:
            variations = int(self.variations_var.get())
        except ValueError as exc:
            raise ValueError("Variations must be an integer.") from exc

        variations = max(1, min(variations, 12))

        return GenerationConfig(
            prompt=prompt,
            model_choice=model_choice,
            model_id=model_id,
            total_duration=duration,
            guidance=guidance,
            negative_prompt=self.negative_var.get().strip() or None,
            output_path=output_path,
            crossfade=crossfade,
            normalize=self.normalize_var.get(),
            stream=self.stream_var.get(),
            tempo_preset=tempo_choice,
            batch_variations=variations,
            loopable=self.loop_var.get(),
            audio_format=selected_format,
        )

    def _run_generation(self, config: GenerationConfig) -> None:
        total_segments = max(1, len(compute_segment_schedule(config.total_duration, config.crossfade)))

        def progress_hook(event: str, payload: object) -> None:
            if event == "start" and isinstance(payload, dict):
                self.result_queue.put(("start", payload))
            elif event == "variation_start" and isinstance(payload, dict):
                self.result_queue.put(("variation_start", payload))
            elif event == "variation_finished" and isinstance(payload, dict):
                self.result_queue.put(("variation_finished", payload))
            elif event == "segment_complete" and isinstance(payload, dict):
                self.result_queue.put(("progress", payload))
            elif event == "finished" and isinstance(payload, Path):
                self.result_queue.put(("success", payload))
            elif event == "cancelled":
                self.result_queue.put(("cancelled", None))
            elif event == "error" and isinstance(payload, Exception):
                self.result_queue.put(("error", payload))

        try:
            generate_song(config, progress_callback=progress_hook, cancel_event=self.cancel_event)
        except Exception as exc:  # pragma: no cover - GUI runtime
            self.result_queue.put(("error", exc))

    def _handle_success(self, path: Path) -> None:
        self.status_var.set(f"Done: {path}")
        messagebox.showinfo("Nightdrive", f"Song generated at:\n{path}")
        self._reset_state()
        self._log(f"[done] {path}")
        prompt_text = self.prompt_text.get("1.0", tk.END).strip()
        if prompt_text:
            self.history.insert(0, prompt_text)
            self.history_list.insert(0, prompt_text)
            if len(self.history) > 50:
                self.history.pop()
                self.history_list.delete(tk.END)

    def _handle_failure(self, exc: Exception) -> None:
        self.status_var.set("Failed")
        messagebox.showerror("Nightdrive", f"Generation failed:\n{exc}")
        self._reset_state()
        self._log(f"[error] {exc}")

    def _reset_state(self) -> None:
        self._busy = False
        self.generate_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
        self.progress_var.set(0.0)
        self.cancel_event.clear()

    def _on_cancel(self) -> None:
        if self._busy:
            self.cancel_event.set()
            self._log("[cancel] Requested")

    def _poll_worker(self) -> None:
        try:
            while True:
                event, payload = self.result_queue.get_nowait()
                if event == "start" and isinstance(payload, dict):
                    total = payload.get("total_segments", 0)
                    self._log(f"[progress] 0/{total} segments")
                    self.progress_var.set(0.0)
                elif event == "variation_start" and isinstance(payload, dict):
                    idx = payload.get("index", 1)
                    total = payload.get("total", 1)
                    self.status_var.set(f"Variation {idx}/{total}")
                    self._log(f"[variation] start {idx}/{total}")
                    self.progress_var.set(0.0)
                elif event == "variation_finished" and isinstance(payload, dict):
                    path = payload.get("path")
                    idx = payload.get("variation", 1)
                    msg = f"[variation] done {idx}" + (f" -> {path}" if path else "")
                    self._log(msg)
                elif event == "progress" and isinstance(payload, dict):
                    completed = payload.get("completed", 0)
                    total = payload.get("total_segments", 1)
                    percent = 0.0 if total <= 0 else min(completed / total * 100, 100)
                    self.progress_var.set(percent)
                    self.status_var.set(f"Segment {completed}/{total}")
                elif event == "success" and isinstance(payload, Path):
                    self._handle_success(payload)
                elif event == "cancelled":
                    self._handle_cancel()
                elif event == "error" and isinstance(payload, Exception):
                    self._handle_failure(payload)
        except queue.Empty:
            pass
        finally:
            self.root.after(200, self._poll_worker)

    def _handle_cancel(self) -> None:
        self.status_var.set("Cancelled")
        messagebox.showinfo("Nightdrive", "Generation cancelled.")
        self._reset_state()
        self._log("[cancelled]")

    def _on_history_select(self, event: tk.Event[tk.Listbox]) -> None:
        selection = self.history_list.curselection()
        if not selection:
            return
        index = selection[0]
        prompt = self.history_list.get(index)
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert(tk.END, prompt)

    def _on_playback(self) -> None:
        path = Path(self.output_var.get()).expanduser().resolve()
        if not path.exists():
            messagebox.showwarning("Nightdrive", "Output file does not exist yet.")
            return
        try:
            audio, sr = self._load_audio(path)
            self._play_audio_preview(audio, sr)
        except Exception as exc:
            messagebox.showerror("Nightdrive", f"Unable to play audio:\n{exc}")

    def _load_audio(self, path: Path) -> tuple[np.ndarray, int]:
        if path in self._audio_cache:
            return self._audio_cache[path]
        audio, sr = sf.read(path)
        self._audio_cache[path] = (audio, sr)
        return audio, sr

    def _play_audio_preview(self, audio: np.ndarray, sr: int) -> None:
        try:
            import simpleaudio as sa
        except ImportError:
            messagebox.showinfo("Nightdrive", "Install simpleaudio for playback: pip install simpleaudio")
            return

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
        play_obj = sa.play_buffer(audio_int16, 1, 2, sr)
        threading.Thread(target=play_obj.wait_done, daemon=True).start()


def main() -> None:
    root = tk.Tk()
    NightdriveGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
