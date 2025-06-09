import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from mne_lsl.lsl import StreamInlet, resolve_streams

class StreamThread(QThread):
    new_epoch = pyqtSignal(np.ndarray)  # emits epoch with shape (channels, samples)

    def __init__(self, lsl_stream_name, info, epoch_duration=1.0, parent=None):
        super().__init__(parent)
        self.lsl_stream_name = lsl_stream_name
        self.info = info
        self.epoch_duration = epoch_duration
        self.running = False
        self.inlet = None

        self.buffer = np.empty((0, len(self.info['ch_names'])))  # rows=samples, cols=channels
        self.epoch_samples = int(self.info['sfreq'] * self.epoch_duration)

    def run(self):
        try:
            streams = resolve_streams(timeout=5.0)
            for stream in streams:
                if stream.name == self.lsl_stream_name:
                    self.inlet = StreamInlet(stream)
                    break
            if self.inlet is None:
                print(f"Stream {self.lsl_stream_name} not found!")
                return
        except Exception as e:
            print(f"Error resolving LSL streams: {e}")
            return

        self.running = True
        while self.running:
            samples, _ = self.inlet.pull_chunk(timeout=1.0)
            if samples is not None and len(samples) > 0:
                samples = np.array(samples)  # shape: (n_samples, n_channels)
                # Accumulate samples
                self.buffer = np.vstack([self.buffer, samples]) if self.buffer.size else samples

                # Emit epochs while buffer has enough samples
                while self.buffer.shape[0] >= self.epoch_samples:
                    epoch = self.buffer[:self.epoch_samples, :].T  # shape: (channels, samples)
                    self.new_epoch.emit(epoch)
                    self.buffer = self.buffer[self.epoch_samples:, :]

    def stop(self):
        self.running = False
        self.wait()
