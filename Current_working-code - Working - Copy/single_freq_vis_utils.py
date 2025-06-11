# single_freq_vis_utils.py

import subprocess
import os
import threading
from visual_1freq import visualize_eeg_feedback  # Assuming this is your provided function

# Default path to the game executable
DEFAULT_GAME_PATH = r"C:\Users\Admin\Documents\BCI\Neurofeedback Varsha\BuildGames\EEGCubeGame.exe"

# === Simple Mode Visualization ===
def run_simple_visualization(power_change_stream=None, band_name="Alpha (8-12 Hz)", monitor_name="TestMonitor"):
    """
    Launch simple PsychoPy visualization in a separate thread so GUI stays open.
    """
    def visual_thread():
        try:
            visualize_eeg_feedback(power_change_stream, monitor_name=monitor_name, freq_band=band_name)
        except Exception as e:
            print(f"[Visualization Thread Error] {e}")

    thread = threading.Thread(target=visual_thread, daemon=True)
    thread.start()

# === Game Mode Launcher ===
def launch_game_mode(game_path):
    """
    Launch a prebuilt game from its executable path.

    Args:
        game_path (str): Absolute path to the game executable.
    """
    if not os.path.isfile(game_path):
        raise FileNotFoundError(f"Game executable not found at: {game_path}")

    try:
        subprocess.Popen([game_path], shell=True)
        print(f"Game launched from: {game_path}")
    except Exception as e:
        print(f"Error launching game: {e}")
