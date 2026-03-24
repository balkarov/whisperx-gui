#!/usr/bin/env python3
"""WhisperX UI — GUI приложение для транскрипции видео и аудио."""

import sys
import os

# Добавляем корень проекта в path (для работы как из IDE, так и из exe)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    # Настраиваем пути к встроенным бинарникам (FFmpeg и др.)
    from core.utils import setup_bundled_paths
    setup_bundled_paths()

    # Проверяем зависимости перед запуском GUI
    missing = []
    try:
        import customtkinter
    except ImportError:
        missing.append("customtkinter")

    if missing:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Отсутствуют зависимости",
            f"Не установлены пакеты: {', '.join(missing)}\n\n"
            "Установите их командой:\n"
            "pip install customtkinter"
        )
        sys.exit(1)

    from ui.main_window import MainWindow
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
