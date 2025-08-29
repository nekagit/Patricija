# components/training/training_progress.py

from __future__ import annotations
import queue
import time
import streamlit as st
from threading import Thread

def drive_progress(
    thread: Thread,
    q: queue.Queue,
    progress_bar: st.progress,
    status_text: st.empty
):
    """
    Monitors a training thread via a queue and updates the Streamlit UI.
    This function now handles dictionary-based messages.
    """
    while thread.is_alive():
        try:
            # Get a message from the queue
            message_dict = q.get_nowait()

            if message_dict["type"] == "progress":
                # Handle progress updates
                progress = message_dict.get("progress", 0.0)
                message = message_dict.get("message", "...")
                progress_bar.progress(float(progress), text=message)
                status_text.text(message)

            elif message_dict["type"] == "result":
                # Handle the final result
                # This key is set in the `launch_training_and_saving` function's 'result' dict
                # Although we don't use it here, this logic handles the final message
                break # Exit the loop once the final result is sent

        except queue.Empty:
            # If the queue is empty, wait a bit before checking again
            time.sleep(0.1)

    # Ensure the progress bar reaches 100% on completion
    progress_bar.progress(1.0, text="Fertig!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()