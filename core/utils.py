import json
import os
import sys
import subprocess
from pathlib import Path


def setup_bundled_paths():
    """Добавляет пути к встроенным бинарникам (FFmpeg и др.) в PATH.
    Вызывать в начале app.py до любых проверок."""
    if getattr(sys, 'frozen', False):
        # Запущено из PyInstaller exe
        exe_dir = Path(sys.executable).parent
        internal_dir = exe_dir / "_internal"
        for d in [exe_dir, internal_dir]:
            if d.exists():
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".opus",
}
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm", ".flv", ".ts", ".m4v",
}
SUPPORTED_EXTENSIONS = SUPPORTED_AUDIO_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

LANGUAGES = {
    "auto": "Авто-определение",
    "ru": "Русский",
    "en": "English",
    "de": "Deutsch",
    "fr": "Français",
    "es": "Español",
    "it": "Italiano",
    "ja": "日本語",
    "zh": "中文",
    "ko": "한국어",
    "pt": "Português",
    "nl": "Nederlands",
    "pl": "Polski",
    "uk": "Українська",
    "tr": "Türkçe",
    "ar": "العربية",
    "hi": "हिन्दी",
    "cs": "Čeština",
    "sv": "Svenska",
    "da": "Dansk",
    "fi": "Suomi",
    "el": "Ελληνικά",
    "he": "עברית",
    "hu": "Magyar",
    "no": "Norsk",
    "ro": "Română",
    "th": "ไทย",
    "vi": "Tiếng Việt",
}

OUTPUT_FORMATS = ["srt", "vtt", "txt", "tsv", "json", "ass"]

COMPUTE_TYPES = ["float16", "float32", "int8"]

CONFIG_DIR = Path(os.environ.get("APPDATA", Path.home())) / "WhisperX-UI"
CONFIG_FILE = CONFIG_DIR / "config.json"

def is_model_cached(model_name: str) -> bool:
    """Проверяет, есть ли модель в кеше HuggingFace."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--Systran--faster-whisper-{model_name}"
    return cache_dir.exists()


def get_config_dir() -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def load_config() -> dict:
    defaults = {
        "model": "medium",
        "language": "auto",
        "device": "auto",
        "compute_type": "float16",
        "batch_size": 16,
        "task": "transcribe",
        "diarize": True,
        "hf_token": "",
        "min_speakers": 1,
        "max_speakers": 10,
        "output_formats": ["srt"],
        "output_dir": "",
        "highlight_words": False,
    }
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            defaults.update(saved)
    except Exception:
        pass
    return defaults


def save_config(config: dict):
    get_config_dir()
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def check_gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return ""


def get_device(preference: str = "auto") -> str:
    if preference == "auto":
        return "cuda" if check_gpu_available() else "cpu"
    return preference


def check_ffmpeg() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return True
    except FileNotFoundError:
        return False


def format_timestamp_srt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_timestamp_ass(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int((seconds % 1) * 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def write_srt(segments: list, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp_srt(seg["start"])
            end = format_timestamp_srt(seg["end"])
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker", "")
            if speaker:
                text = f"[{speaker}] {text}"
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def write_vtt(segments: list, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, 1):
            start = format_timestamp_vtt(seg["start"])
            end = format_timestamp_vtt(seg["end"])
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker", "")
            if speaker:
                text = f"[{speaker}] {text}"
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def write_txt(segments: list, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        for seg in segments:
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker", "")
            if speaker:
                f.write(f"[{speaker}] {text}\n")
            else:
                f.write(f"{text}\n")


def write_tsv(segments: list, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("start\tend\ttext\tspeaker\n")
        for seg in segments:
            start = f"{seg['start']:.3f}"
            end = f"{seg['end']:.3f}"
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker", "")
            f.write(f"{start}\t{end}\t{text}\t{speaker}\n")


def write_json(result: dict, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def write_ass(segments: list, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("[Script Info]\nTitle: WhisperX Transcription\nScriptType: v4.00+\n")
        f.write("WrapStyle: 0\nPlayResX: 1920\nPlayResY: 1080\n\n")
        f.write("[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, "
                "SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
                "StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
                "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write("Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,"
                "&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,40,1\n\n")
        f.write("[Events]\nFormat: Layer, Start, End, Style, Name, "
                "MarginL, MarginR, MarginV, Effect, Text\n")
        for seg in segments:
            start = format_timestamp_ass(seg["start"])
            end = format_timestamp_ass(seg["end"])
            text = seg.get("text", "").strip().replace("\n", "\\N")
            speaker = seg.get("speaker", "")
            if speaker:
                text = f"[{speaker}] {text}"
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")


def save_result(result: dict, base_path: str, formats: list):
    segments = result.get("segments", [])
    saved_files = []
    for fmt in formats:
        filepath = f"{base_path}.{fmt}"
        if fmt == "srt":
            write_srt(segments, filepath)
        elif fmt == "vtt":
            write_vtt(segments, filepath)
        elif fmt == "txt":
            write_txt(segments, filepath)
        elif fmt == "tsv":
            write_tsv(segments, filepath)
        elif fmt == "json":
            write_json(result, filepath)
        elif fmt == "ass":
            write_ass(segments, filepath)
        saved_files.append(filepath)
    return saved_files
