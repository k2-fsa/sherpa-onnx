# Copyright (c)  2025  Xiaomi Corporation
import os
from time import localtime, strftime


def get_current_time():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


class Display:
    """A class for displaying real-time speech recognition results.

    It maintains a history of finalized sentences and the current
    in-progress text, and renders them to the console.
    """

    def __init__(self):
        """Initialize the display with empty history and current text."""
        self.sentences = []
        self.currentText = ""

    def update_text(self, text):
        """Update the current in-progress text.

        Args:
          text:
            The new text to display as the current recognition result.
        """
        self.currentText = text

    def finalize_current_sentence(self):
        """Finalize the current sentence and add it to the history.

        If the current text is non-empty (after stripping whitespace), it is
        appended to the sentence history with a timestamp. The current text
        is then cleared.
        """
        if self.currentText.strip():
            self.sentences.append((get_current_time(), self.currentText))

        self.currentText = ""

    def display(self):
        """Render the display to the console.

        Clears the console and prints the header, finalized sentence history,
        and the current in-progress recognition text.
        """
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
