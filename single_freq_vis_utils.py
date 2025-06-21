# single_freq_vis_utils.py

import subprocess
import os

# Default path to the game executable
DEFAULT_GAME_PATH = r"C:\Users\varsh\NFBCubeGame\EEGCubeGame.exe"
#r"C:\Users\Admin\Documents\BCI\Neurofeedback Varsha\BuildGames\EEGCubeGame.exe"

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
