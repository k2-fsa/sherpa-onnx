# Copyright (c)  2025  Xiaomi Corporation
import os
from time import localtime, strftime


def get_current_time():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


class Display:
    def __init__(self):
        self.sentences = []
        self.currentText = ""

    def update_text(self, text):
        self.currentText = text

    def finalize_current_sentence(self):
        if self.currentText.strip():
            self.sentences.append((get_current_time(), self.currentText))

        self.currentText = ""

    def display(self):
        clear_console()
        print("=== Speech Recognition with Next-gen Kaldi ===")
        print("Time:", get_current_time())
        print("-" * 30)

        # display history sentences
        if self.sentences:
            for i, (when, text) in enumerate(self.sentences):
                print(f"[{when}] {i + 1}. {text}")
            print("-" * 30)

        if self.currentText.strip():
            print("Recognizing:", self.currentText)
