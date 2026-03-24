import gc
import threading
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
        self._cancelled = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._cancelled = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self):
        self._cancelled = True

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

            # Подбираем compute_type для CPU
            compute_type = self.compute_type
            if device == "cpu" and compute_type == "float16":
                compute_type = "float32"
                self._log("CPU не поддерживает float16, переключаемся на float32", 0)

            # 1. Загрузка аудио
            self._log("Загрузка аудио...", 5)
            audio = whisperx.load_audio(self.file_path)
            if self._cancelled:
                return

            # 2. Загрузка модели
            self._log(f"Загрузка модели {self.model_name}...", 10)
            model = whisperx.load_model(
                self.model_name,
                device,
                compute_type=compute_type,
                language=self.language,
                task=self.task,
            )
            if self._cancelled:
                return

            # 3. Транскрипция
            self._log("Транскрипция...", 20)
            result = model.transcribe(
                audio,
                batch_size=self.batch_size,
                language=self.language,
            )
            detected_language = result.get("language", self.language or "en")
            self._log(f"Язык: {detected_language}", 40)
            if self._cancelled:
                return

            # Освобождаем память от модели ASR
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            # 4. Выравнивание (alignment)
            self._log("Выравнивание по словам...", 50)
            try:
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
            except Exception as e:
                self._log(f"Выравнивание не удалось ({e}), продолжаем без него", 55)

            if self._cancelled:
                return

            # 5. Диаризация
            if self.diarize and self.hf_token:
                self._log("Диаризация (определение спикеров)...", 65)
                try:
                    from whisperx.diarize import DiarizationPipeline

                    diarize_model = DiarizationPipeline(
                        token=self.hf_token, device=device
                    )
                    diarize_segments = diarize_model(
                        audio,
                        min_speakers=self.min_speakers,
                        max_speakers=self.max_speakers,
                    )
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    self._log("Диаризация завершена", 80)
                    del diarize_model
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
                except Exception as e:
                    self._log(f"Диаризация не удалась: {e}", 75)
                    self._log(f"Traceback: {traceback.format_exc()}", 75)
            elif self.diarize and not self.hf_token:
                self._log("Диаризация пропущена: не указан HuggingFace токен", 75)

            if self._cancelled:
                return

            # 6. Сохранение результатов
            self._log("Сохранение результатов...", 85)
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
