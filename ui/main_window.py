import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import customtkinter as ctk

from core.utils import (
    SUPPORTED_EXTENSIONS,
    WHISPER_MODELS,
    LANGUAGES,
    OUTPUT_FORMATS,
    COMPUTE_TYPES,
    load_config,
    save_config,
    check_gpu_available,
    get_gpu_name,
    check_ffmpeg,
    is_model_cached,
)
from core.transcriber import TranscriptionTask


class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("WhisperX UI — Транскрипция видео и аудио")
        self.geometry("900x750")
        self.minsize(800, 650)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.config_data = load_config()
        self.current_task: TranscriptionTask | None = None
        self.file_list: list[str] = []

        self._build_ui()
        self._fix_keyboard_shortcuts()
        self._load_settings()
        self._check_environment()

    def _fix_keyboard_shortcuts(self):
        """Глобальный фикс Ctrl+V/C/A/X для работы в любой раскладке клавиатуры."""
        def _on_key(event):
            if not (event.state & 0x4):  # Ctrl не зажат
                return
            widget = self.focus_get()
            if widget is None:
                return

            # keycode 86=V, 67=C, 65=A, 88=X — работают в любой раскладке
            if event.keycode == 86:  # Ctrl+V (Paste)
                try:
                    text = self.clipboard_get()
                    if isinstance(widget, (tk.Entry, tk.Text)):
                        widget.insert("insert", text)
                        return "break"
                    elif hasattr(widget, "insert"):
                        widget.insert("insert", text)
                        return "break"
                except tk.TclError:
                    pass
            elif event.keycode == 67:  # Ctrl+C (Copy)
                try:
                    if isinstance(widget, tk.Text):
                        sel = widget.get("sel.first", "sel.last")
                        self.clipboard_clear()
                        self.clipboard_append(sel)
                        return "break"
                    elif isinstance(widget, tk.Entry):
                        if widget.selection_present():
                            sel = widget.selection_get()
                            self.clipboard_clear()
                            self.clipboard_append(sel)
                            return "break"
                except tk.TclError:
                    pass
            elif event.keycode == 65:  # Ctrl+A (Select All)
                if isinstance(widget, tk.Text):
                    widget.tag_add("sel", "1.0", "end")
                    return "break"
                elif isinstance(widget, tk.Entry):
                    widget.select_range(0, "end")
                    widget.icursor("end")
                    return "break"
            elif event.keycode == 88:  # Ctrl+X (Cut)
                try:
                    if isinstance(widget, tk.Text):
                        sel = widget.get("sel.first", "sel.last")
                        self.clipboard_clear()
                        self.clipboard_append(sel)
                        widget.delete("sel.first", "sel.last")
                        return "break"
                    elif isinstance(widget, tk.Entry):
                        if widget.selection_present():
                            sel = widget.selection_get()
                            self.clipboard_clear()
                            self.clipboard_append(sel)
                            widget.delete("sel.first", "sel.last")
                            return "break"
                except tk.TclError:
                    pass

        self.bind_all("<Key>", _on_key)

    def _build_ui(self):
        # Основной скролл-фрейм
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        main_frame = ctk.CTkScrollableFrame(self)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_columnconfigure(0, weight=1)

        row = 0

        # === ФАЙЛЫ ===
        file_frame = ctk.CTkFrame(main_frame)
        file_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        file_frame.grid_columnconfigure(1, weight=1)
        row += 1

        ctk.CTkLabel(file_frame, text="Файлы", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 5)
        )

        self.file_entry = ctk.CTkEntry(file_frame, placeholder_text="Выберите файл(ы)...")
        self.file_entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

        btn_frame = ctk.CTkFrame(file_frame, fg_color="transparent")
        btn_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

        ctk.CTkButton(btn_frame, text="Выбрать файлы", command=self._select_files, width=150).pack(
            side="left", padx=(0, 5)
        )
        ctk.CTkButton(btn_frame, text="Выбрать папку", command=self._select_folder, width=150).pack(
            side="left", padx=(0, 5)
        )
        ctk.CTkButton(btn_frame, text="Очистить", command=self._clear_files, width=100,
                       fg_color="gray30").pack(side="left")

        self.file_count_label = ctk.CTkLabel(file_frame, text="Файлов: 0")
        self.file_count_label.grid(row=3, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))

        # === ПАРАМЕТРЫ МОДЕЛИ ===
        model_frame = ctk.CTkFrame(main_frame)
        model_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        model_frame.grid_columnconfigure(1, weight=1)
        model_frame.grid_columnconfigure(3, weight=1)
        row += 1

        ctk.CTkLabel(model_frame, text="Параметры модели", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=10, pady=(10, 5)
        )

        # Модель
        ctk.CTkLabel(model_frame, text="Модель:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.model_var = ctk.StringVar(value="medium")
        self.model_combo = ctk.CTkComboBox(
            model_frame, values=WHISPER_MODELS, variable=self.model_var,
            width=160, command=self._on_model_changed
        )
        self.model_combo.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        self.model_hint_label = ctk.CTkLabel(
            model_frame, text="(встроена)", text_color="green", font=ctk.CTkFont(size=11)
        )
        self.model_hint_label.grid(row=1, column=2, sticky="w", padx=5, pady=5)

        # Язык
        ctk.CTkLabel(model_frame, text="Язык:").grid(row=1, column=3, sticky="w", padx=10, pady=5)
        lang_display = [f"{code} — {name}" for code, name in LANGUAGES.items()]
        self.lang_var = ctk.StringVar(value="auto — Авто-определение")
        self.lang_combo = ctk.CTkComboBox(model_frame, values=lang_display, variable=self.lang_var, width=220)
        self.lang_combo.grid(row=1, column=4, sticky="w", padx=5, pady=5)

        # Устройство
        ctk.CTkLabel(model_frame, text="Устройство:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.device_var = ctk.StringVar(value="auto")
        self.device_combo = ctk.CTkComboBox(
            model_frame, values=["auto", "cuda", "cpu"], variable=self.device_var, width=160
        )
        self.device_combo.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        self.gpu_label = ctk.CTkLabel(model_frame, text="", text_color="gray60", font=ctk.CTkFont(size=11))
        self.gpu_label.grid(row=2, column=2, columnspan=2, sticky="w", padx=10, pady=5)

        # Тип вычислений
        ctk.CTkLabel(model_frame, text="Compute type:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.compute_var = ctk.StringVar(value="float16")
        self.compute_combo = ctk.CTkComboBox(model_frame, values=COMPUTE_TYPES, variable=self.compute_var, width=160)
        self.compute_combo.grid(row=3, column=1, sticky="w", padx=5, pady=5)

        # Batch size
        ctk.CTkLabel(model_frame, text="Batch size:").grid(row=3, column=2, sticky="w", padx=10, pady=5)
        self.batch_var = ctk.IntVar(value=16)
        self.batch_slider = ctk.CTkSlider(model_frame, from_=1, to=32, number_of_steps=31,
                                           variable=self.batch_var, width=160)
        self.batch_slider.grid(row=3, column=3, sticky="w", padx=5, pady=5)
        self.batch_label = ctk.CTkLabel(model_frame, text="16")
        self.batch_label.grid(row=3, column=4, sticky="w", padx=5, pady=5)
        self.batch_var.trace_add("write", lambda *_: self.batch_label.configure(text=str(self.batch_var.get())))

        # Задача
        ctk.CTkLabel(model_frame, text="Задача:").grid(row=4, column=0, sticky="w", padx=10, pady=(5, 10))
        self.task_var = ctk.StringVar(value="transcribe")
        ctk.CTkRadioButton(model_frame, text="Транскрипция", variable=self.task_var, value="transcribe").grid(
            row=4, column=1, sticky="w", padx=5, pady=(5, 10)
        )
        ctk.CTkRadioButton(model_frame, text="Перевод на английский", variable=self.task_var, value="translate").grid(
            row=4, column=2, columnspan=2, sticky="w", padx=5, pady=(5, 10)
        )

        # === ДИАРИЗАЦИЯ ===
        diarize_frame = ctk.CTkFrame(main_frame)
        diarize_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        diarize_frame.grid_columnconfigure(1, weight=1)
        self.diarize_frame = diarize_frame
        row += 1

        ctk.CTkLabel(diarize_frame, text="Диаризация (определение спикеров)",
                      font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=10, pady=(10, 5)
        )

        self.diarize_var = ctk.BooleanVar(value=True)
        self.diarize_check = ctk.CTkCheckBox(
            diarize_frame, text="Включить диаризацию", variable=self.diarize_var,
            command=self._toggle_diarize
        )
        self.diarize_check.grid(row=1, column=0, columnspan=4, sticky="w", padx=10, pady=5)

        # Подсказка про токен
        self.token_info_label = ctk.CTkLabel(
            diarize_frame,
            text="ℹ Укажите токен HuggingFace — он нужен для скачивания моделей (транскрипции и диаризации).",
            font=ctk.CTkFont(size=12), text_color="gray60", wraplength=700, justify="left"
        )
        self.token_info_label.grid(row=2, column=0, columnspan=4, sticky="w", padx=10, pady=(5, 0))

        # Токен
        self.token_label = ctk.CTkLabel(diarize_frame, text="HuggingFace токен:")
        self.token_label.grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.hf_token_entry = ctk.CTkEntry(diarize_frame, placeholder_text="hf_...", show="*", width=300)
        self.hf_token_entry.grid(row=3, column=1, columnspan=2, sticky="ew", padx=(5, 10), pady=5)

        self.show_token_var = ctk.BooleanVar(value=False)
        self.show_token_check = ctk.CTkCheckBox(diarize_frame, text="Показать", variable=self.show_token_var,
                         command=self._toggle_token_visibility, width=80)
        self.show_token_check.grid(row=3, column=3, sticky="w", padx=5, pady=5)

        # Спикеры
        self.min_speakers_label = ctk.CTkLabel(diarize_frame, text="Мин. спикеров:")
        self.min_speakers_label.grid(row=4, column=0, sticky="w", padx=10, pady=5)
        self.min_speakers_var = ctk.StringVar(value="1")
        self.min_speakers_entry = ctk.CTkEntry(diarize_frame, textvariable=self.min_speakers_var, width=60)
        self.min_speakers_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)

        self.max_speakers_label = ctk.CTkLabel(diarize_frame, text="Макс. спикеров:")
        self.max_speakers_label.grid(row=4, column=2, sticky="w", padx=10, pady=5)
        self.max_speakers_var = ctk.StringVar(value="10")
        self.max_speakers_entry = ctk.CTkEntry(diarize_frame, textvariable=self.max_speakers_var, width=60)
        self.max_speakers_entry.grid(row=4, column=3, sticky="w", padx=5, pady=(5, 10))

        # Список виджетов, которые скрываются при выключении диаризации
        self._diarize_detail_widgets = [
            self.token_info_label, self.token_label, self.hf_token_entry,
            self.show_token_check, self.min_speakers_label, self.min_speakers_entry,
            self.max_speakers_label, self.max_speakers_entry,
        ]

        self._toggle_diarize()

        # === ВЫХОД ===
        output_frame = ctk.CTkFrame(main_frame)
        output_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        output_frame.grid_columnconfigure(1, weight=1)
        row += 1

        ctk.CTkLabel(output_frame, text="Результат", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=10, pady=(10, 5)
        )

        ctk.CTkLabel(output_frame, text="Форматы:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        fmt_frame = ctk.CTkFrame(output_frame, fg_color="transparent")
        fmt_frame.grid(row=1, column=1, columnspan=3, sticky="w", padx=5, pady=5)

        self.format_vars = {}
        for i, fmt in enumerate(OUTPUT_FORMATS):
            var = ctk.BooleanVar(value=(fmt == "srt"))
            cb = ctk.CTkCheckBox(fmt_frame, text=fmt.upper(), variable=var, width=70)
            cb.grid(row=0, column=i, padx=5)
            self.format_vars[fmt] = var

        self.highlight_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(output_frame, text="Подсветка слов в субтитрах", variable=self.highlight_var).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=10, pady=5
        )

        ctk.CTkLabel(output_frame, text="Папка:").grid(row=3, column=0, sticky="w", padx=10, pady=(5, 10))
        self.output_dir_entry = ctk.CTkEntry(output_frame, placeholder_text="Рядом с исходным файлом")
        self.output_dir_entry.grid(row=3, column=1, columnspan=2, sticky="ew", padx=5, pady=(5, 10))
        ctk.CTkButton(output_frame, text="Обзор...", command=self._select_output_dir, width=100).grid(
            row=3, column=3, sticky="w", padx=(5, 10), pady=(5, 10)
        )

        # === УПРАВЛЕНИЕ ===
        control_frame = ctk.CTkFrame(main_frame)
        control_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        control_frame.grid_columnconfigure(0, weight=1)
        row += 1

        btn_row = ctk.CTkFrame(control_frame, fg_color="transparent")
        btn_row.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.start_btn = ctk.CTkButton(
            btn_row, text="Старт", command=self._start_transcription,
            font=ctk.CTkFont(size=15, weight="bold"), height=40, width=200,
            fg_color="#2d8a4e", hover_color="#236b3e"
        )
        self.start_btn.pack(side="left", padx=(0, 10))

        self.stop_btn = ctk.CTkButton(
            btn_row, text="Стоп", command=self._stop_transcription,
            font=ctk.CTkFont(size=15), height=40, width=120,
            fg_color="#8a2d2d", hover_color="#6b2323", state="disabled"
        )
        self.stop_btn.pack(side="left")

        self.progress_bar = ctk.CTkProgressBar(control_frame, width=400)
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(control_frame, text="Готов к работе", font=ctk.CTkFont(size=13))
        self.status_label.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 5))

        # === ЛОГ ===
        log_frame = ctk.CTkFrame(main_frame)
        log_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        log_frame.grid_columnconfigure(0, weight=1)
        row += 1

        log_header = ctk.CTkFrame(log_frame, fg_color="transparent")
        log_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        log_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(log_header, text="Лог", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, sticky="w"
        )
        ctk.CTkButton(log_header, text="Копировать лог", command=self._copy_log,
                       width=120, height=28, fg_color="gray30", hover_color="gray40").grid(
            row=0, column=1, sticky="e", padx=(5, 0)
        )
        ctk.CTkButton(log_header, text="Очистить", command=self._clear_log,
                       width=90, height=28, fg_color="gray30", hover_color="gray40").grid(
            row=0, column=2, sticky="e", padx=(5, 0)
        )

        # Используем обычный tk.Text — он надёжно поддерживает выделение и Ctrl+C
        self.log_text = tk.Text(
            log_frame, height=10, wrap="word",
            bg="#1a1a1a", fg="#d4d4d4", insertbackground="#d4d4d4",
            selectbackground="#264f78", selectforeground="#ffffff",
            font=("Consolas", 10), relief="flat", borderwidth=0,
            padx=8, pady=8,
        )
        self.log_text.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

        # Скроллбар
        log_scroll = tk.Scrollbar(log_frame, command=self.log_text.yview)
        log_scroll.grid(row=1, column=1, sticky="ns", pady=(0, 10))
        self.log_text.configure(yscrollcommand=log_scroll.set)

        # Только чтение: блокируем ввод, но разрешаем выделение/копирование/навигацию
        def _on_log_key(event):
            # Ctrl+комбинации — разрешаем (копировать, выделить, вставить не повредит)
            if event.state & 0x4:
                return
            # Навигация — разрешаем
            if event.keysym in ("Up", "Down", "Left", "Right", "Home", "End",
                                "Prior", "Next", "Shift_L", "Shift_R",
                                "Control_L", "Control_R"):
                return
            # Shift+стрелки для выделения — разрешаем
            if event.state & 0x1 and event.keysym in ("Up", "Down", "Left", "Right", "Home", "End"):
                return
            return "break"
        self.log_text.bind("<Key>", _on_log_key)

        # Контекстное меню (правая кнопка мыши)
        self._log_context_menu = tk.Menu(self.log_text, tearoff=0)
        self._log_context_menu.add_command(label="Копировать выделенное", command=self._copy_log_selection)
        self._log_context_menu.add_command(label="Копировать весь лог", command=self._copy_log)
        self._log_context_menu.add_separator()
        self._log_context_menu.add_command(label="Очистить лог", command=self._clear_log)
        self.log_text.bind("<Button-3>", self._show_log_context_menu)

    def _copy_log(self):
        text = self.log_text.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(text)
        self._log("[ Лог скопирован в буфер обмена ]")

    def _copy_log_selection(self):
        try:
            text = self.log_text.get("sel.first", "sel.last")
            self.clipboard_clear()
            self.clipboard_append(text)
        except tk.TclError:
            self._copy_log()

    def _clear_log(self):
        self.log_text.delete("1.0", "end")

    def _show_log_context_menu(self, event):
        self._log_context_menu.tk_popup(event.x_root, event.y_root)

    def _log(self, message: str):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")

    def _check_environment(self):
        """Проверяет окружение при запуске."""
        # GPU
        if check_gpu_available():
            gpu = get_gpu_name()
            self.gpu_label.configure(text=f"GPU: {gpu}", text_color="green")
            self._log(f"GPU обнаружен: {gpu}")
        else:
            self.gpu_label.configure(text="GPU не найден — будет использован CPU", text_color="orange")
            self._log("GPU не обнаружен, будет использоваться CPU")

        # FFmpeg
        if check_ffmpeg():
            self._log("FFmpeg найден")
        else:
            self._log("ВНИМАНИЕ: FFmpeg не найден! Установите FFmpeg для работы с видео.")

        # WhisperX
        try:
            import whisperx
            self._log("WhisperX загружен")
        except ImportError:
            self._log("ОШИБКА: WhisperX не установлен! pip install git+https://github.com/m-bain/whisperX.git")

    def _load_settings(self):
        c = self.config_data
        self.model_var.set(c.get("model", "medium"))
        lang = c.get("language", "auto")
        lang_name = LANGUAGES.get(lang, lang)
        self.lang_var.set(f"{lang} — {lang_name}")
        self.device_var.set(c.get("device", "auto"))
        self.compute_var.set(c.get("compute_type", "float16"))
        self.batch_var.set(c.get("batch_size", 16))
        self.task_var.set(c.get("task", "transcribe"))
        self.diarize_var.set(c.get("diarize", False))
        if c.get("hf_token"):
            self.hf_token_entry.insert(0, c["hf_token"])
        self.min_speakers_var.set(str(c.get("min_speakers", 1)))
        self.max_speakers_var.set(str(c.get("max_speakers", 10)))
        for fmt in OUTPUT_FORMATS:
            self.format_vars[fmt].set(fmt in c.get("output_formats", ["srt"]))
        self.highlight_var.set(c.get("highlight_words", False))
        if c.get("output_dir"):
            self.output_dir_entry.insert(0, c["output_dir"])
        self._toggle_diarize()
        self._on_model_changed()

    def _save_settings(self):
        lang_str = self.lang_var.get()
        lang_code = lang_str.split(" — ")[0] if " — " in lang_str else lang_str
        config = {
            "model": self.model_var.get(),
            "language": lang_code,
            "device": self.device_var.get(),
            "compute_type": self.compute_var.get(),
            "batch_size": self.batch_var.get(),
            "task": self.task_var.get(),
            "diarize": self.diarize_var.get(),
            "hf_token": self.hf_token_entry.get(),
            "min_speakers": int(self.min_speakers_var.get() or 1),
            "max_speakers": int(self.max_speakers_var.get() or 10),
            "output_formats": [fmt for fmt, var in self.format_vars.items() if var.get()],
            "highlight_words": self.highlight_var.get(),
            "output_dir": self.output_dir_entry.get(),
        }
        save_config(config)

    def _on_model_changed(self, model_name: str = None):
        model = model_name or self.model_var.get()

        if is_model_cached(model):
            self.model_hint_label.configure(text="(в кеше)", text_color="green")
        else:
            self.model_hint_label.configure(text="(будет скачана)", text_color="orange")

    def _toggle_diarize(self):
        if self.diarize_var.get():
            for w in self._diarize_detail_widgets:
                w.grid()
        else:
            for w in self._diarize_detail_widgets:
                w.grid_remove()

    def _toggle_token_visibility(self):
        show = "" if self.show_token_var.get() else "*"
        self.hf_token_entry.configure(show=show)

    def _select_files(self):
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
        files = filedialog.askopenfilenames(
            title="Выберите видео/аудио файлы",
            filetypes=[("Медиа файлы", exts), ("Все файлы", "*.*")],
        )
        if files:
            self.file_list = list(files)
            self._update_file_display()

    def _select_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку с медиа файлами")
        if folder:
            files = []
            for f in Path(folder).iterdir():
                if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(str(f))
            if files:
                self.file_list = sorted(files)
                self._update_file_display()
            else:
                messagebox.showwarning("Нет файлов", "В выбранной папке нет поддерживаемых медиа файлов.")

    def _clear_files(self):
        self.file_list = []
        self._update_file_display()

    def _update_file_display(self):
        self.file_entry.delete(0, "end")
        if len(self.file_list) == 1:
            self.file_entry.insert(0, self.file_list[0])
        elif len(self.file_list) > 1:
            self.file_entry.insert(0, f"{len(self.file_list)} файлов выбрано")
        self.file_count_label.configure(text=f"Файлов: {len(self.file_list)}")

    def _select_output_dir(self):
        folder = filedialog.askdirectory(title="Выберите папку для сохранения")
        if folder:
            self.output_dir_entry.delete(0, "end")
            self.output_dir_entry.insert(0, folder)

    def _get_selected_formats(self) -> list:
        return [fmt for fmt, var in self.format_vars.items() if var.get()]

    def _start_transcription(self):
        if not self.file_list:
            messagebox.showwarning("Нет файлов", "Выберите файл(ы) для транскрипции.")
            return

        formats = self._get_selected_formats()
        if not formats:
            messagebox.showwarning("Нет формата", "Выберите хотя бы один формат вывода.")
            return

        self._save_settings()

        lang_str = self.lang_var.get()
        lang_code = lang_str.split(" — ")[0] if " — " in lang_str else lang_str

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress_bar.set(0)

        self._current_file_index = 0
        self._process_next_file()

    def _process_next_file(self):
        if self._current_file_index >= len(self.file_list):
            self._log("=== Все файлы обработаны ===")
            self.status_label.configure(text="Все файлы обработаны!")
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.progress_bar.set(1)
            return

        file_path = self.file_list[self._current_file_index]
        total = len(self.file_list)
        idx = self._current_file_index + 1

        self._log(f"\n=== Файл {idx}/{total}: {Path(file_path).name} ===")

        lang_str = self.lang_var.get()
        lang_code = lang_str.split(" — ")[0] if " — " in lang_str else lang_str

        self.current_task = TranscriptionTask(
            file_path=file_path,
            model_name=self.model_var.get(),
            language=lang_code,
            device=self.device_var.get(),
            compute_type=self.compute_var.get(),
            batch_size=self.batch_var.get(),
            task=self.task_var.get(),
            diarize=self.diarize_var.get(),
            hf_token=self.hf_token_entry.get(),
            min_speakers=int(self.min_speakers_var.get() or 1),
            max_speakers=int(self.max_speakers_var.get() or 10),
            highlight_words=self.highlight_var.get(),
            output_dir=self.output_dir_entry.get(),
            output_formats=self._get_selected_formats(),
            on_progress=lambda msg, pct: self.after(0, self._on_progress, msg, pct),
            on_complete=lambda result, files: self.after(0, self._on_file_complete, result, files),
            on_error=lambda msg: self.after(0, self._on_error, msg),
            on_speakers_found=lambda speakers: self.after(0, self._on_speakers_found, speakers),
        )
        self.current_task.start()

    def _on_progress(self, message: str, pct: float):
        self._log(message)
        if pct >= 0:
            self.progress_bar.set(pct / 100)
        self.status_label.configure(text=message)

    def _on_file_complete(self, result: dict, saved_files: list):
        for f in saved_files:
            self._log(f"  Сохранено: {f}")
        self._current_file_index += 1
        self._process_next_file()

    def _on_speakers_found(self, speakers: list):
        """Показывает диалог для ввода имён спикеров."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Имена спикеров")
        dialog.geometry("450x400")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)

        ctk.CTkLabel(
            dialog, text="Назначьте имена спикерам",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(padx=20, pady=(20, 5))

        ctk.CTkLabel(
            dialog,
            text="Файл уже сохранён — откройте его, посмотрите кто что говорил,\n"
                 "затем введите имена. Оставьте пустым, чтобы не менять.",
            font=ctk.CTkFont(size=12), text_color="gray60", justify="left"
        ).pack(padx=20, pady=(0, 15))

        # Скролл-фрейм для спикеров
        scroll = ctk.CTkScrollableFrame(dialog, height=220)
        scroll.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        scroll.grid_columnconfigure(1, weight=1)

        entries = {}
        for i, speaker in enumerate(speakers):
            ctk.CTkLabel(scroll, text=f"{speaker}  →").grid(
                row=i, column=0, sticky="w", padx=(10, 5), pady=5
            )
            entry = ctk.CTkEntry(scroll, placeholder_text="Имя спикера...")
            entry.grid(row=i, column=1, sticky="ew", padx=(5, 10), pady=5)
            entries[speaker] = entry

        def on_apply():
            mapping = {}
            for speaker, entry in entries.items():
                name = entry.get().strip()
                if name:
                    mapping[speaker] = name
            self.current_task.set_speaker_mapping(mapping)
            dialog.destroy()

        def on_skip():
            self.current_task.set_speaker_mapping({})
            dialog.destroy()

        def on_close():
            on_skip()

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(0, 20))

        ctk.CTkButton(
            btn_frame, text="Применить", command=on_apply,
            fg_color="#2d8a4e", hover_color="#236b3e", width=150
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            btn_frame, text="Пропустить", command=on_skip,
            fg_color="gray30", hover_color="gray40", width=150
        ).pack(side="left")

        dialog.protocol("WM_DELETE_WINDOW", on_close)

    def _on_error(self, message: str):
        self._log(f"ОШИБКА: {message}")
        self.status_label.configure(text="Ошибка!")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        messagebox.showerror("Ошибка", message[:500])

    def _stop_transcription(self):
        if self.current_task:
            self.current_task.cancel()
            self.current_task = None
        self._log("Остановлено пользователем")
        self.status_label.configure(text="Остановлено")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def destroy(self):
        self._save_settings()
        super().destroy()
