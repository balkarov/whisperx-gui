import gc
import sys
import threading
import time
import traceback
from typing import Callable, Optional

from core.utils import get_device, save_result


class TranscriptionTask:
    """Выполняет транскрипцию WhisperX в отдельном потоке."""

    def __init__(
        self,
        file_path: str,
        model_name: str = "medium",
        language: str = "auto",
        device: str = "auto",
        compute_type: str = "float16",
        batch_size: int = 16,
        task: str = "transcribe",
        diarize: bool = False,
        hf_token: str = "",
        min_speakers: int = 1,
        max_speakers: int = 10,
        highlight_words: bool = False,
        output_dir: str = "",
        output_formats: list = None,
        on_progress: Optional[Callable[[str, float], None]] = None,
        on_complete: Optional[Callable[[dict, list], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_speakers_found: Optional[Callable[[list], dict]] = None,
    ):
        self.file_path = file_path
        self.model_name = model_name
        self.language = language if language != "auto" else None
        self.device = get_device(device)
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.task = task
        self.diarize = diarize
        self.hf_token = hf_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.highlight_words = highlight_words
        self.output_dir = output_dir
        self.output_formats = output_formats or ["srt"]
        self.on_progress = on_progress or (lambda msg, pct: None)
        self.on_complete = on_complete or (lambda result, files: None)
        self.on_error = on_error or (lambda msg: None)
        self.on_speakers_found = on_speakers_found
        self._cancelled = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._cancelled = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self):
        self._cancelled = True
        # Разблокировать ожидание маппинга если отменено
        if hasattr(self, '_speaker_mapping_event'):
            self._speaker_mapping_event.set()

    def set_speaker_mapping(self, mapping: dict):
        """Вызывается из UI после заполнения имён спикеров."""
        self._speaker_mapping_result = mapping
        self._speaker_mapping_event.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _log(self, msg: str, pct: float = -1):
        if not self._cancelled:
            self.on_progress(msg, pct)

    def _run(self):
        try:
            import whisperx
            import torch

            device = self.device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
                self._log("GPU не найден, используем CPU", 0)

            compute_type = self.compute_type
            if device == "cpu" and compute_type == "float16":
                compute_type = "float32"
                self._log("CPU не поддерживает float16, переключаемся на float32", 0)

            # 1. Загрузка аудио
            self._log("Загрузка аудио...", 5)
            t0 = time.time()
            audio = whisperx.load_audio(self.file_path)
            duration_sec = len(audio) / 16000
            self._log(f"Аудио загружено ({duration_sec:.0f} сек) за {time.time() - t0:.1f}с", 8)
            if self._cancelled:
                return

            # 2. Загрузка модели (с перехватом прогресса скачивания)
            self._log(f"Загрузка модели {self.model_name}...", 10)
            t0 = time.time()

            stderr_monitor = _StderrMonitor(self, phase="model", pct_range=(10, 20))
            stderr_monitor.start()

            model = whisperx.load_model(
                self.model_name,
                device,
                compute_type=compute_type,
                language=self.language,
                task=self.task,
            )

            stderr_monitor.stop()
            self._log(f"Модель загружена за {time.time() - t0:.1f}с", 20)
            if self._cancelled:
                return

            # 3. Транскрипция с прогрессом через перехват tqdm
            self._log("Транскрипция...", 20)
            t0 = time.time()

            stderr_monitor = _StderrMonitor(self, phase="transcribe", pct_range=(20, 45))
            stderr_monitor.start()

            result = model.transcribe(
                audio,
                batch_size=self.batch_size,
                language=self.language,
            )

            stderr_monitor.stop()
            detected_language = result.get("language", self.language or "en")
            n_segments = len(result.get("segments", []))
            self._log(f"Транскрипция: {n_segments} сегментов, "
                      f"язык: {detected_language}, за {time.time() - t0:.1f}с", 45)
            if self._cancelled:
                return

            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            # 4. Выравнивание
            self._log("Выравнивание по словам...", 50)
            t0 = time.time()
            try:
                stderr_monitor = _StderrMonitor(self, phase="align", pct_range=(50, 65))
                stderr_monitor.start()

                align_model, align_metadata = whisperx.load_align_model(
                    language_code=detected_language, device=device
                )
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    audio,
                    device,
                    return_char_alignments=False,
                )
                del align_model
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

                stderr_monitor.stop()
                self._log(f"Выравнивание завершено за {time.time() - t0:.1f}с", 65)
            except Exception as e:
                stderr_monitor.stop()
                self._log(f"Выравнивание не удалось ({e}), продолжаем без него", 65)

            if self._cancelled:
                return

            # 5. Диаризация
            if self.diarize and self.hf_token:
                self._log("Диаризация (определение спикеров)...", 68)
                t0 = time.time()
                try:
                    from whisperx.diarize import DiarizationPipeline

                    stderr_monitor = _StderrMonitor(self, phase="diarize", pct_range=(68, 88))
                    stderr_monitor.start()

                    diarize_model = DiarizationPipeline(
                        token=self.hf_token, device=device
                    )
                    diarize_segments = diarize_model(
                        audio,
                        min_speakers=self.min_speakers,
                        max_speakers=self.max_speakers,
                    )
                    result = whisperx.assign_word_speakers(diarize_segments, result)

                    stderr_monitor.stop()
                    self._log(f"Диаризация завершена за {time.time() - t0:.1f}с", 88)
                    del diarize_model
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
                except Exception as e:
                    stderr_monitor.stop()
                    self._log(f"Диаризация не удалась: {e}", 80)
                    self._log(f"Traceback: {traceback.format_exc()}", 80)
            elif self.diarize and not self.hf_token:
                self._log("Диаризация пропущена: не указан HuggingFace токен", 80)

            if self._cancelled:
                return

            # 5.1 Маппинг имён спикеров
            speakers = set()
            for seg in result.get("segments", []):
                if seg.get("speaker"):
                    speakers.add(seg["speaker"])

            if speakers and self.on_speakers_found:
                speakers_sorted = sorted(speakers)
                self._log(f"Найдено спикеров: {len(speakers_sorted)}", 90)

                # Запрашиваем маппинг у UI (блокирующий вызов через Event)
                self._speaker_mapping_event = threading.Event()
                self._speaker_mapping_result = {}
                self.on_speakers_found(speakers_sorted)
                # Ждём пока пользователь заполнит имена
                self._speaker_mapping_event.wait()

                if self._cancelled:
                    return

                mapping = self._speaker_mapping_result
                if mapping:
                    for seg in result.get("segments", []):
                        sp = seg.get("speaker", "")
                        if sp in mapping and mapping[sp]:
                            seg["speaker"] = mapping[sp]
                    self._log("Имена спикеров применены", 92)

            # 6. Сохранение
            self._log("Сохранение результатов...", 92)
            from pathlib import Path

            input_path = Path(self.file_path)
            if self.output_dir:
                out_dir = Path(self.output_dir)
            else:
                out_dir = input_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            base_path = str(out_dir / input_path.stem)

            saved_files = save_result(result, base_path, self.output_formats)

            self._log("Готово!", 100)
            self.on_complete(result, saved_files)

        except Exception as e:
            tb = traceback.format_exc()
            self.on_error(f"{e}\n\n{tb}")


class _StderrMonitor:
    """Перехватывает stderr для отслеживания прогресса tqdm (скачивание, батчи и т.д.)."""

    def __init__(self, task: TranscriptionTask, phase: str, pct_range: tuple):
        self.task = task
        self.phase = phase
        self.pct_min, self.pct_max = pct_range
        self._original_stderr = None
        self._last_pct_time = 0

    def start(self):
        self._original_stderr = sys.stderr
        sys.stderr = _TeeWriter(sys.stderr, self._on_output)

    def stop(self):
        if self._original_stderr:
            sys.stderr = self._original_stderr
            self._original_stderr = None

    def _on_output(self, text: str):
        if self.task._cancelled:
            return
        now = time.time()
        # Не спамим чаще чем раз в 0.5 сек
        if now - self._last_pct_time < 0.5:
            return

        # tqdm выводит прогресс вида "  45%|████      | 3/7"
        if "%" in text:
            try:
                pct_str = text.strip().split("%")[0].strip().split("\r")[-1].strip()
                # Может быть "  45" или просто "45"
                pct = int(pct_str)
                if 0 <= pct <= 100:
                    self._last_pct_time = now
                    mapped = self.pct_min + int((self.pct_max - self.pct_min) * pct / 100)
                    label = {
                        "model": f"Загрузка модели: {pct}%",
                        "transcribe": f"Транскрипция: {pct}%",
                        "align": f"Выравнивание: {pct}%",
                        "diarize": f"Диаризация: {pct}%",
                    }.get(self.phase, f"{pct}%")
                    self.task._log(label, mapped)
            except (ValueError, IndexError):
                pass


class _TeeWriter:
    """Перехватчик stderr: пишет в оригинал и вызывает callback."""

    def __init__(self, original, callback):
        self.original = original
        self.callback = callback

    def write(self, text):
        if self.original:
            self.original.write(text)
        if text.strip():
            try:
                self.callback(text)
            except Exception:
                pass

    def flush(self):
        if self.original:
            self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)
