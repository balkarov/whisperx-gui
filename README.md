# WhisperX UI

Windows GUI-приложение для транскрипции видео и аудио на базе [WhisperX](https://github.com/m-bain/whisperX).

Выбираете файл, настраиваете параметры, нажимаете "Старт" — получаете субтитры.

## Возможности

- **17 форматов** аудио/видео: mp3, wav, flac, ogg, m4a, wma, aac, opus, mp4, mkv, avi, mov, wmv, webm, flv, ts, m4v
- **27 языков** (включая русский, английский, украинский) + авто-определение
- **6 моделей** Whisper: tiny, base, small, medium, large-v2, large-v3
- **6 форматов вывода**: SRT, VTT, TXT, TSV, JSON, ASS
- **Диаризация** — определение спикеров (кто говорит)
- **Пакетная обработка** — выберите папку и обработайте все файлы разом
- **Перевод на английский** — встроенный режим translate
- **GPU-ускорение** — автоматическое определение NVIDIA GPU
- **Сохранение настроек** между сессиями

## Установка

### 1. Python

Установите [Python 3.10+](https://www.python.org/downloads/) (при установке поставьте галочку "Add to PATH").

### 2. FFmpeg

Установите [FFmpeg](https://www.gyan.dev/ffmpeg/builds/) и добавьте в PATH:

```bash
# Или через winget:
winget install Gyan.FFmpeg
```

### 3. Зависимости

```bash
cd whisperx-ui
pip install -r requirements.txt
```

### 4. PyTorch с CUDA (опционально, для GPU)

Если у вас NVIDIA GPU и вы хотите ускорение:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Без этого шага будет работать на CPU (медленнее, но работает).

## Запуск

```bash
python app.py
```

## Где хранятся модели

При первом запуске WhisperX скачивает модели. Это может занять время в зависимости от выбранной модели.

| Модель | Размер (примерно) |
|--------|-------------------|
| tiny | ~75 МБ |
| base | ~145 МБ |
| small | ~480 МБ |
| medium | ~1.5 ГБ |
| large-v2 | ~3 ГБ |
| large-v3 | ~3 ГБ |

### Расположение кеша

Все модели скачиваются в каталог HuggingFace:

```
Windows:  %USERPROFILE%\.cache\huggingface\hub\
Linux:    ~/.cache/huggingface/hub/
```

Сюда попадают:
- **Whisper-модели** (faster-whisper) — основная модель транскрипции
- **Alignment-модели** — модели выравнивания по словам (зависят от языка)
- **Pyannote-модели** — модели диаризации (определение спикеров)

### Переопределение пути

Если хотите хранить модели в другом месте, задайте переменную окружения:

```bash
set HF_HOME=D:\models\huggingface
```

## HuggingFace токен

Токен нужен **только для диаризации** (определения спикеров).

Как получить:
1. Зарегистрируйтесь на [huggingface.co](https://huggingface.co)
2. Перейдите в [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Создайте токен (Read)
4. Примите условия использования моделей:
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
5. Вставьте токен в поле "HuggingFace токен" в приложении

## Настройки приложения

Настройки сохраняются автоматически:

```
%APPDATA%\WhisperX-UI\config.json
```

## Структура проекта

```
whisperx-ui/
├── app.py                 # Точка входа
├── requirements.txt       # Зависимости
├── README.md
├── core/
│   ├── __init__.py
│   ├── transcriber.py     # Логика транскрипции (WhisperX pipeline)
│   └── utils.py           # Утилиты, константы, сохранение результатов
└── ui/
    ├── __init__.py
    └── main_window.py     # GUI (CustomTkinter)
```

## Лицензия

MIT
